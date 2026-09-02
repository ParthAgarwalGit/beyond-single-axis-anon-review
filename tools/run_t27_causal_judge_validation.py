#!/usr/bin/env python
"""T27 — validate the T23 causal judge against the T26 human labels, then branch.

Applies the gate frozen in ``configs/t27_causal_judge_frozen.json`` and pre-registered
in ``docs/DEVIATION_2026-08-18_T27_CAUSAL_JUDGE_GATE.md``. The thresholds were fixed
before the T26 sample was labelled, so this run cannot tune anything: it reads the
gate, computes the metrics, and reports which branch each outcome lands in.

Harmfulness and identity are gated INDEPENDENTLY - one may pass while the other fails.

    python tools/run_t27_causal_judge_validation.py run \
        --labels-1 annotations/t26_causal/labels_labeller1.csv \
        --labels-2 annotations/t26_causal/labels_labeller2.csv \
        --key      annotations/t26_causal/sampling_key_PRIVATE.json \
        --sample-manifest annotations/t26_causal/sample_manifest.json \
        --out      results/t27/causal_judge_validation.json

Labeller CSVs need: item_id, harmfulness, identity. Where the two labellers disagree,
an adjudication file (item_id, harmfulness, identity) supplies the consensus; raw
agreement is always reported BEFORE adjudication, as in T06.

PRIVACY: the emitted report is aggregate only - counts, rates, gate outcomes,
confusion matrices. It never contains item_ids, uids, or per-item labels joined to a
condition, so committing it cannot unblind the sample.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.causal_validation import (  # noqa: E402
    HARMFULNESS_LABELS, IDENTITY_LABELS, cohen_kappa, validate_outcome,
)
from src.provenance import stamp_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "t27_causal_judge_frozen.json"


def _read_labels(path):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            iid = (r.get("item_id") or "").strip()
            if not iid:
                continue
            h = (r.get("harmfulness") or "").strip() or None
            i = (r.get("identity") or "").strip() or None
            if h is not None and h not in HARMFULNESS_LABELS:
                raise SystemExit(f"{path}: {iid} harmfulness {h!r} outside the frozen set")
            if i is not None and i not in IDENTITY_LABELS:
                raise SystemExit(f"{path}: {iid} identity {i!r} outside the frozen set")
            out[iid] = {"harmfulness": h, "identity": i}
    return out


def _inter_labeller(l1, l2):
    """Raw agreement and kappa BEFORE adjudication - reported unconditionally."""
    common = sorted(set(l1) & set(l2))
    report = {"n_common": len(common)}
    for field, labels in (("harmfulness", HARMFULNESS_LABELS), ("identity", IDENTITY_LABELS)):
        pairs = [(l1[i][field], l2[i][field]) for i in common
                 if l1[i][field] is not None and l2[i][field] is not None]
        agree = sum(1 for a, b in pairs if a == b)
        report[field] = {
            "n": len(pairs),
            "raw_agreement": agree / len(pairs) if pairs else None,
            "kappa": cohen_kappa(pairs, labels),
            "n_disagreements": len(pairs) - agree,
        }
        if field == "identity":       # the estimand collapses 7 -> 2
            from src.causal_validation import NON_ASSISTANT
            bp = [(a in NON_ASSISTANT, b in NON_ASSISTANT) for a, b in pairs]
            report[field]["non_assistant_binary"] = {
                "raw_agreement": sum(1 for a, b in bp if a == b) / len(bp) if bp else None,
                "kappa": cohen_kappa(bp, (True, False)),
            }
    return report


def _consensus(l1, l2, adjudication):
    """gold = agreement where the labellers agree, adjudicated label where they do not."""
    gold, unresolved = {}, []
    for iid in sorted(set(l1) & set(l2)):
        rec, needs = {}, False
        for field in ("harmfulness", "identity"):
            a, b = l1[iid][field], l2[iid][field]
            if a is not None and a == b:
                rec[field] = a
            else:
                adj = adjudication.get(iid, {}).get(field)
                if adj is None:
                    needs = True
                    rec[field] = None
                else:
                    rec[field] = adj
        if needs:
            unresolved.append(iid)
        gold[iid] = rec
    return gold, unresolved


def _require_complete(key, l1, l2, gold, unresolved, manifest, branch_conditions):
    """Refuse to emit a gate result from a partial annotation.

    A gate computed on whatever happens to be labelled is not the pre-registered gate:
    the missing items are not missing at random - annotators skip the hard ones, which
    are exactly the boundary cases the thresholds are about.
    """
    problems = []
    expected = set(key)
    for name, got in (("labeller 1", set(l1)), ("labeller 2", set(l2))):
        missing = expected - got
        extra = got - expected
        if missing:
            problems.append(f"{name} is missing {len(missing)} of {len(expected)} sampled items")
        if extra:
            problems.append(f"{name} has {len(extra)} items not in the frozen sample")
    both = set(l1) & set(l2)
    for iid in sorted(expected & both):
        for field in ("harmfulness", "identity"):
            if l1[iid][field] is None or l2[iid][field] is None:
                problems.append(f"item {iid} has a blank {field} label")
                break
    if unresolved:
        problems.append(f"{len(unresolved)} item(s) still need adjudication")
    per_cond = {}
    for iid in gold:
        per_cond[key[iid]["condition"]] = per_cond.get(key[iid]["condition"], 0) + 1
    if manifest is not None:
        exp_pc = manifest["per_condition_target"]
        for c in branch_conditions:
            if c not in per_cond:
                problems.append(f"condition {c} has no resolved items")
            elif per_cond[c] != exp_pc:
                problems.append(f"condition {c} has {per_cond[c]} resolved items, expected {exp_pc}")
    return problems, per_cond


def _validate_sample_manifest(cfg, manifest, key_path, key):
    """Bind a gating run to one exact, well-formed T26 draw.

    The manifest is not advisory metadata. It is the authority that says which branch
    was drawn, what the frozen condition inventory and sample size were, and which
    private key belongs to that draw. Any missing or self-inconsistent field makes the
    gate non-reproducible, so production gating fails closed rather than guessing.
    """
    if manifest.get("task") != "T26_causal_human_validation_sample":
        raise SystemExit("sample manifest is not a T26 causal human-validation manifest")

    branches = cfg["scope"]["branches"]
    branch = manifest.get("branch")
    if branch not in branches:
        raise SystemExit(
            f"sample manifest branch {branch!r} is not in the frozen T27 scope; "
            f"expected one of {sorted(branches)}")

    frozen_conditions = list(branches[branch]["conditions"])
    if manifest.get("conditions") != frozen_conditions:
        raise SystemExit(
            "sample manifest condition list does not exactly match the frozen branch "
            f"inventory for {branch}: expected {frozen_conditions!r}, got "
            f"{manifest.get('conditions')!r}")

    if manifest.get("source_git_dirty") is not False:
        raise SystemExit(
            "sample manifest does not certify source_git_dirty: false; a gating run "
            "cannot bind to a T26 draw whose provenance is dirty or unrecorded. Draw "
            "again from a clean checkout, or run this validation with "
            "--allow-incomplete for a non-gating diagnostic."
        )

    per_condition = int(cfg["sample"]["per_condition"])
    expected_n = per_condition * len(frozen_conditions)
    if manifest.get("per_condition_target") != per_condition:
        raise SystemExit(
            f"sample manifest per_condition_target={manifest.get('per_condition_target')!r}; "
            f"frozen target is {per_condition}")
    if manifest.get("n_items") != expected_n:
        raise SystemExit(
            f"sample manifest n_items={manifest.get('n_items')!r}; expected exactly "
            f"{expected_n} for {len(frozen_conditions)} conditions x {per_condition}")
    if len(key) != expected_n:
        raise SystemExit(
            f"private key contains {len(key)} items; frozen {branch} sample requires "
            f"exactly {expected_n}")

    man_key_sha = manifest.get("private_key_sha256")
    if not isinstance(man_key_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", man_key_sha):
        raise SystemExit(
            "sample manifest must carry a non-empty 64-hex private_key_sha256; "
            "a gating run cannot proceed without binding to the exact private key")
    actual = hashlib.sha256(Path(key_path).read_bytes()).hexdigest()
    if man_key_sha != actual:
        raise SystemExit(
            f"private key does not match the T26 sample manifest "
            f"({actual[:12]} vs {man_key_sha[:12]}); this run would validate a "
            f"different sample than the one that was drawn")

    per_key = {c: 0 for c in frozen_conditions}
    for iid, rec in key.items():
        cond = rec.get("condition")
        if cond not in per_key:
            raise SystemExit(
                f"private key item {iid} has condition {cond!r} outside the frozen "
                f"{branch} inventory")
        per_key[cond] += 1
    bad = {c: n for c, n in per_key.items() if n != per_condition}
    if bad:
        raise SystemExit(
            "private key does not have the frozen per-condition sample size: "
            + ", ".join(f"{c}={n}" for c, n in bad.items())
            + f"; expected {per_condition} each")

    return branch, frozen_conditions


def run(labels_1, labels_2, key_path, out_path, adjudication_path=None,
        sample_manifest=None, allow_incomplete=False, allow_dirty=False):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("outcomes_inspected_before_freeze") is not False:
        raise SystemExit("gate config does not assert pre-registration; refusing to run")
    thresholds = cfg["gate"]

    if not sample_manifest and not allow_incomplete:
        raise SystemExit(
            "--sample-manifest is required for a gating run. Without it the "
            "private-key hash and per-condition counts are never checked, so the "
            "gate would not be bound to the T26 sample that was actually drawn. "
            "Pass --allow-incomplete to emit a non-gating diagnostic instead.")

    key_doc = json.loads(Path(key_path).read_text(encoding="utf-8"))
    if not isinstance(key_doc, dict) or not isinstance(key_doc.get("items"), dict):
        raise SystemExit("private key must be an object containing an 'items' mapping")
    key = key_doc["items"]
    l1, l2 = _read_labels(labels_1), _read_labels(labels_2)
    adjudication = _read_labels(adjudication_path) if adjudication_path else {}

    manifest = None
    branch = None
    branch_conditions = []
    if sample_manifest:
        manifest = json.loads(Path(sample_manifest).read_text(encoding="utf-8"))
        branch, branch_conditions = _validate_sample_manifest(cfg, manifest, key_path, key)

    inter = _inter_labeller(l1, l2)
    gold, unresolved = _consensus(l1, l2, adjudication)

    problems, per_cond = _require_complete(key, l1, l2, gold, unresolved,
                                           manifest, branch_conditions)
    if problems and not allow_incomplete:
        raise SystemExit(
            "T27 refuses to emit a gate result from an incomplete annotation:\n  - "
            + ("\n  - ").join(problems)
            + "\nComplete the labelling and adjudication, or pass "
              "--allow-incomplete to produce a clearly-marked NON-GATING diagnostic.")

    rows = []
    for iid, g in gold.items():
        k = key.get(iid)
        if k is None:
            raise SystemExit(f"item {iid} missing from the private key")
        rows.append({
            "condition": k["condition"],
            "ip_weight": float(k.get("ip_weight", 1.0)),
            "judge_coverage_failure": bool(k.get("judge_coverage_failure")),
            "auto_harmfulness": k.get("auto_harmfulness"),
            "auto_identity": k.get("auto_identity"),
            "human_harmfulness": g["harmfulness"],
            "human_identity": g["identity"],
        })

    outcomes = {}
    for outcome in ("harmfulness", "identity"):
        try:
            outcomes[outcome] = validate_outcome(rows, outcome, thresholds)
        except ValueError as e:
            outcomes[outcome] = {"outcome": outcome, "error": str(e),
                                 "gate": {"branch": "MEASUREMENT_LIMITED",
                                          "reason": "insufficient scored rows"}}

    gating = bool(sample_manifest) and not allow_incomplete and not problems and not allow_dirty
    report = stamp_report({
        "task": "T27_causal_judge_validation",
        "branch": branch,
        "gating": gating,
        "completeness_problems": problems or None,
        "resolved_per_condition": per_cond,
        "t26_sample_manifest": sample_manifest,
        "t26_sample_manifest_sha256": (
            hashlib.sha256(Path(sample_manifest).read_bytes()).hexdigest()
            if sample_manifest else None),
        "authority": "docs/DEVIATION_2026-08-18_T27_CAUSAL_JUDGE_GATE.md",
        "gate_config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "gate_frozen_utc": cfg.get("frozen_utc"),
        "scope": cfg["scope"],
        "judged_instrument": {
            k: cfg["judged_instrument"][k]
            for k in ("judge_model", "provider", "reproducibility_limitation")
            if k in cfg["judged_instrument"]},
        "n_items_in_key": len(key),
        "n_items_labelled_by_both": inter["n_common"],
        "n_unresolved_after_adjudication": len(unresolved),
        "inter_labeller_before_adjudication": inter,
        "outcomes": outcomes,
        "branches": {o: r["gate"]["branch"] for o, r in outcomes.items()},
        "privacy": "aggregate only; no item_ids, uids, or per-item labels joined to condition",
    }, allow_dirty=allow_dirty)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[T27] inter-labeller kappa: harmfulness="
          f"{inter['harmfulness']['kappa']}, identity={inter['identity']['kappa']}")
    if not gating:
        print("[T27] NON-GATING diagnostic: no production gate claim may be made from this report")
    for o, r in outcomes.items():
        g = r["gate"]
        if "error" in r:
            print(f"[T27] {o}: {r['error']} -> {g['branch']}")
        else:
            print(f"[T27] {o}: partA={'PASS' if g['part_a_pass'] else 'FAIL'} "
                  f"bias_detected={g['part_b_differential_bias_detected']} "
                  f"-> {g['branch']} (primary: {g['primary_evidence']})")
    print(f"-> {out_path}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--labels-1", required=True)
    r.add_argument("--labels-2", required=True)
    r.add_argument("--key", required=True)
    r.add_argument("--adjudication", default=None)
    r.add_argument("--sample-manifest", default=None,
                   help="T26 sample_manifest.json. REQUIRED for a gating run: it binds "
                        "the run to the drawn sample and verifies the private-key hash, "
                        "branch, condition inventory, and frozen sample sizes. Only "
                        "omissible with --allow-incomplete, which produces a non-gating "
                        "diagnostic.")
    r.add_argument(
        "--allow-dirty", action="store_true",
        help="DEVELOPMENT ONLY: stamp the gate report from a dirty working tree. "
             "A production gating run must come from a clean checkout.")
    r.add_argument("--allow-incomplete", action="store_true",
                   help="emit a NON-GATING diagnostic from a partial annotation; the "
                        "result is always marked gating=false and must not be reported "
                        "as the gate")
    r.add_argument("--out", default=str(ROOT / "results" / "t27" / "causal_judge_validation.json"))
    a = ap.parse_args(argv)
    run(a.labels_1, a.labels_2, a.key, a.out, a.adjudication,
        sample_manifest=a.sample_manifest,
        allow_incomplete=a.allow_incomplete,
        allow_dirty=a.allow_dirty)


if __name__ == "__main__":
    main()
