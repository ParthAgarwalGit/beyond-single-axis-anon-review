#!/usr/bin/env python
"""Run the T29-H human-primary Qwen causal-specificity analysis.

This runner is activated only by the frozen T27 Qwen harmfulness
``MEASUREMENT_LIMITED`` branch. It privately joins the adjudicated T26 human
labels to the frozen sampling key, computes the design-aware human-primary
estimator, verifies that its condition-level strict-harm rates reproduce the
aggregate T27 human rates, and writes aggregate-only output.

The private sampling key and item-to-condition human labels must never be
committed. The output contains hashes and aggregate statistics only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.causal_validation import HARMFULNESS_LABELS  # noqa: E402
from src.provenance import stamp_report  # noqa: E402
from src.t29_human_primary import REQUIRED_CONDITIONS, human_primary_report, validate_design  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS_1 = ROOT / "annotations/t26_causal/labels_round2_labeller1.csv"
DEFAULT_LABELS_2 = ROOT / "annotations/t26_causal/labels_round2_labeller2.csv"
DEFAULT_ADJUDICATION = ROOT / "annotations/t26_causal/round2/adjudication_record.csv"
DEFAULT_MANIFEST = ROOT / "annotations/t26_causal/sample_manifest.json"
DEFAULT_T27 = ROOT / "results/t27/qwen_causal_judge_validation.json"
DEFAULT_METHOD_NOTE = ROOT / "docs/DEVIATION_2026-08-25_T29_HUMAN_PRIMARY.md"
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


def die(msg: str):
    raise SystemExit(f"T29-H ABORTED (fail-closed): {msg}")


def sha256_file(path) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        die(f"required file missing: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path, label: str) -> dict:
    p = Path(path)
    if not p.exists():
        die(f"{label} missing: {p}")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        die(f"cannot parse {label} {p}: {exc}")
    if not isinstance(doc, dict):
        die(f"{label} must be a JSON object")
    return doc


def read_harmfulness_labels(path) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            iid = (row.get("item_id") or "").strip()
            if not iid:
                continue
            if iid in out:
                die(f"duplicate item_id {iid!r} in {path}")
            lab = (row.get("harmfulness") or "").strip() or None
            if lab is not None and lab not in HARMFULNESS_LABELS:
                die(f"{path}: {iid} harmfulness {lab!r} outside frozen label set")
            out[iid] = lab
    return out


def consensus_harmfulness(l1: dict, l2: dict, adjudication: dict) -> dict[str, str]:
    if set(l1) != set(l2):
        die(
            f"round-2 labeller item sets differ: labeller1={len(l1)}, labeller2={len(l2)}, "
            f"only1={len(set(l1)-set(l2))}, only2={len(set(l2)-set(l1))}"
        )
    gold = {}
    unresolved = []
    for iid in sorted(l1):
        a, b = l1[iid], l2[iid]
        if a is not None and a == b:
            gold[iid] = a
            continue
        adj = adjudication.get(iid)
        if adj is None:
            unresolved.append(iid)
        else:
            gold[iid] = adj
    if unresolved:
        die(f"{len(unresolved)} harmfulness labels remain unresolved after adjudication")
    return gold


def validate_manifest(manifest: dict, manifest_path, key_path, key: dict) -> dict:
    if manifest.get("task") != "T26_causal_human_validation_sample":
        die("sample manifest is not a T26 causal human-validation manifest")
    if manifest.get("branch") != "qwen_capping":
        die(f"sample manifest branch must be 'qwen_capping', got {manifest.get('branch')!r}")
    if manifest.get("conditions") != list(REQUIRED_CONDITIONS):
        die(
            "sample manifest condition inventory does not match frozen T29-H Qwen conditions: "
            f"expected {list(REQUIRED_CONDITIONS)!r}, got {manifest.get('conditions')!r}"
        )
    if manifest.get("n_items") != 120 or manifest.get("per_condition_target") != 30:
        die(
            "T29-H requires the frozen 120-item Qwen draw with exactly 30 items/condition; "
            f"manifest has n_items={manifest.get('n_items')!r}, "
            f"per_condition_target={manifest.get('per_condition_target')!r}"
        )
    if manifest.get("private_key_committed") is not False:
        die("sample manifest does not certify private_key_committed: false")
    if manifest.get("source_git_dirty") is not False:
        die("sample manifest does not certify a clean source tree")

    expected_key_sha = str(manifest.get("private_key_sha256") or "").lower()
    if not _SHA64.fullmatch(expected_key_sha):
        die("sample manifest lacks a full private_key_sha256")
    actual_key_sha = sha256_file(key_path)
    if actual_key_sha != expected_key_sha:
        die(
            "private sampling key does not match the committed T26 manifest "
            f"({actual_key_sha[:12]} vs {expected_key_sha[:12]})"
        )
    if len(key) != 120:
        die(f"private key contains {len(key)} items; expected exactly 120")

    try:
        design = validate_design(manifest.get("strata"))
    except ValueError as exc:
        die(f"invalid frozen T26 strata: {exc}")
    for condition in REQUIRED_CONDITIONS:
        if design["sampled_per_condition"][condition] != 30:
            die(
                f"frozen strata imply {design['sampled_per_condition'][condition]} sampled rows for "
                f"{condition}, expected 30"
            )
        if design["population_per_condition"][condition] != 100:
            die(
                f"frozen strata imply population {design['population_per_condition'][condition]} for "
                f"{condition}, expected 100"
            )

    draw = manifest.get("execution_status_at_draw") or {}
    if draw.get("all_frozen_conditions_generated_and_scored") is not True:
        die("sample manifest does not certify all four frozen conditions generated/scored at draw time")
    if draw.get("n_rows_scored") != 400:
        die(f"sample manifest draw population must be 400 rows, got {draw.get('n_rows_scored')!r}")
    if draw.get("rows_per_condition") != {c: 100 for c in REQUIRED_CONDITIONS}:
        die("sample manifest draw rows_per_condition is not exactly 100 for every frozen condition")
    if draw.get("n_coverage_failures") != 0:
        die("sample manifest reports causal-judge coverage failures at draw time")

    return {
        "manifest_sha256": sha256_file(manifest_path),
        "private_key_sha256": actual_key_sha,
        "design": design,
    }


def validate_t27(t27: dict, t27_path, manifest_sha: str) -> dict:
    if t27.get("task") != "T27_causal_judge_validation" or t27.get("branch") != "qwen_capping":
        die("T27 authority is not the qwen_capping causal-judge validation result")
    if t27.get("gating") is not True:
        die("T27 result is not a gating run")
    if t27.get("n_items_in_key") != 120 or t27.get("n_items_labelled_by_both") != 120:
        die("T27 result is not the complete 120-item Qwen validation")
    if t27.get("n_unresolved_after_adjudication") != 0:
        die("T27 still has unresolved human labels")
    if t27.get("t26_sample_manifest_sha256") != manifest_sha:
        die("T27 gate is bound to a different T26 sample manifest")
    resolved = t27.get("resolved_per_condition") or {}
    if resolved != {c: 30 for c in REQUIRED_CONDITIONS}:
        die(f"T27 resolved_per_condition is not exactly 30/condition: {resolved!r}")

    harmful = (t27.get("outcomes") or {}).get("harmfulness") or {}
    gate = harmful.get("gate") or {}
    if gate.get("branch") != "MEASUREMENT_LIMITED":
        die(
            "T29-H is only authorized by the harmfulness MEASUREMENT_LIMITED branch; "
            f"T27 reports {gate.get('branch')!r}"
        )
    if gate.get("primary_evidence") != "human_subset":
        die(f"T27 does not designate human_subset as primary evidence: {gate.get('primary_evidence')!r}")
    if gate.get("primary_evidence_is_full_corpus") is not False:
        die("T27 unexpectedly marks full-corpus evidence as primary")
    if gate.get("full_corpus_reportable_as") != "full_corpus_automatic":
        die("T27 full-corpus sensitivity branch is not the expected automatic-label diagnostic")

    per_condition = (harmful.get("part_b_non_differential_error") or {}).get("per_condition") or {}
    frozen_human_rates = {}
    for condition in REQUIRED_CONDITIONS:
        rec = per_condition.get(condition) or {}
        rate = rec.get("human_positive_rate_ip")
        if rate is None:
            die(f"T27 lacks frozen human_positive_rate_ip for {condition}")
        frozen_human_rates[condition] = float(rate)

    return {
        "t27_sha256": sha256_file(t27_path),
        "harmfulness_branch": gate["branch"],
        "primary_evidence": gate["primary_evidence"],
        "full_corpus_reportable_as": gate["full_corpus_reportable_as"],
        "frozen_human_strict_rates": frozen_human_rates,
    }


def build_joined_rows(key: dict, gold: dict, manifest: dict) -> list[dict]:
    if set(key) != set(gold):
        die(
            f"private key and resolved human label sets differ: key={len(key)}, gold={len(gold)}, "
            f"only_key={len(set(key)-set(gold))}, only_gold={len(set(gold)-set(key))}"
        )
    strata = manifest["strata"]
    rows = []
    seen_per_stratum = {}
    for iid in sorted(key):
        rec = key[iid]
        condition = rec.get("condition")
        auto = rec.get("auto_harmfulness")
        stratum = rec.get("stratum") or f"{condition}|{auto}"
        expected = strata.get(stratum)
        if expected is None:
            die(f"private key item maps to stratum absent from manifest: {stratum!r}")
        if condition != stratum.split("|", 1)[0] or auto != stratum.split("|", 1)[1]:
            die(f"private key condition/automatic label disagree with frozen stratum {stratum!r}")
        weight = float(rec.get("ip_weight", math.nan))
        if not math.isfinite(weight) or not math.isclose(
            weight, float(expected["ip_weight"]), rel_tol=1e-12, abs_tol=1e-12
        ):
            die(f"private key ip_weight disagrees with manifest for stratum {stratum!r}")
        seen_per_stratum[stratum] = seen_per_stratum.get(stratum, 0) + 1
        rows.append(
            {
                "condition": condition,
                "auto_harmfulness": auto,
                "human_harmfulness": gold[iid],
                "ip_weight": weight,
            }
        )
    for stratum, srec in strata.items():
        if seen_per_stratum.get(stratum, 0) != int(srec["sampled"]):
            die(
                f"private key has {seen_per_stratum.get(stratum, 0)} rows in {stratum}, "
                f"manifest requires {srec['sampled']}"
            )
    return rows


def verify_t27_rate_regression(report: dict, frozen_rates: dict, tol: float = 1e-12):
    observed = report["outcomes"]["strict"]["condition_rates"]
    for condition in REQUIRED_CONDITIONS:
        got = float(observed[condition]["rate"])
        want = float(frozen_rates[condition])
        if not math.isclose(got, want, rel_tol=tol, abs_tol=tol):
            die(
                f"human-primary strict rate for {condition} does not reproduce T27: "
                f"computed={got:.17g}, T27={want:.17g}"
            )


def assert_aggregate_only(obj):
    forbidden_keys = {
        "item_id", "item_ids", "uid", "uids", "items", "private_key_path",
        "condition_mapping", "per_item", "row_id", "opaque_id",
    }

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in forbidden_keys:
                    die(f"aggregate output contains forbidden private/item-level key {key!r}")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)


def run(
    *, key_path, out_path, labels_1=DEFAULT_LABELS_1, labels_2=DEFAULT_LABELS_2,
    adjudication=DEFAULT_ADJUDICATION, sample_manifest=DEFAULT_MANIFEST,
    t27_gate=DEFAULT_T27, method_note=DEFAULT_METHOD_NOTE, n_boot=10000,
    seed=290819, allow_dirty=False,
):
    manifest = load_json(sample_manifest, "T26 sample manifest")
    key_doc = load_json(key_path, "private T26 sampling key")
    if not isinstance(key_doc.get("items"), dict):
        die("private key must be a JSON object containing an 'items' mapping")
    key = key_doc["items"]

    manifest_info = validate_manifest(manifest, sample_manifest, key_path, key)
    t27 = load_json(t27_gate, "T27 Qwen causal-judge validation")
    t27_info = validate_t27(t27, t27_gate, manifest_info["manifest_sha256"])

    l1 = read_harmfulness_labels(labels_1)
    l2 = read_harmfulness_labels(labels_2)
    adj = read_harmfulness_labels(adjudication)
    gold = consensus_harmfulness(l1, l2, adj)
    if len(gold) != 120:
        die(f"resolved human harmfulness set contains {len(gold)} items; expected 120")

    joined = build_joined_rows(key, gold, manifest)
    try:
        report = human_primary_report(joined, manifest["strata"], n_boot=int(n_boot), seed=int(seed))
    except ValueError as exc:
        die(f"human-primary estimator rejected the joined sample: {exc}")
    verify_t27_rate_regression(report, t27_info["frozen_human_strict_rates"])

    out = {
        "task": "T29_QWEN_HUMAN_PRIMARY",
        "status": "HUMAN_PRIMARY_MEASUREMENT_LIMITED",
        "primary_result_eligible": True,
        "measurement_branch": {
            "t27_branch": t27_info["harmfulness_branch"],
            "primary_evidence": t27_info["primary_evidence"],
            "full_corpus_reportable_as": t27_info["full_corpus_reportable_as"],
            "automatic_400_row_primary_allowed": False,
        },
        "method": {
            "authority": "docs/DEVIATION_2026-08-25_T29_HUMAN_PRIMARY.md",
            "authority_sha256": sha256_file(method_note),
            "bootstrap_replicates": int(n_boot),
            "bootstrap_seed": int(seed),
            "paired": False,
            "confirmatory_p_value_added": False,
        },
        "provenance": {
            "t26_sample_manifest_sha256": manifest_info["manifest_sha256"],
            "t26_private_key_sha256": manifest_info["private_key_sha256"],
            "t27_gate_sha256": t27_info["t27_sha256"],
            "labels_round2_labeller1_sha256": sha256_file(labels_1),
            "labels_round2_labeller2_sha256": sha256_file(labels_2),
            "adjudication_sha256": sha256_file(adjudication),
            "private_key_committed": False,
        },
        "regression_check": {
            "target": "T27 human_positive_rate_ip by condition",
            "status": "PASS",
            "tolerance": 1e-12,
        },
        "report": report,
        "identity_claim_guard": (
            "Identity is not part of the T29-H primary estimator. The T27 Qwen identity branch "
            "has only two human-positive, adjudicator-dependent cases and must retain that caveat."
        ),
        "privacy": (
            "aggregate only; no item IDs, UIDs, item-to-condition mapping, or per-item human labels "
            "joined to condition are emitted"
        ),
    }
    assert_aggregate_only(out)
    out = stamp_report(out, allow_dirty=allow_dirty)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    primary = out["report"]["primary_specificity"]
    print("[T29-H] HUMAN PRIMARY COMPLETE")
    print(f"[T29-H] axis-random strict RD = {primary['risk_difference']:.6f}")
    print(f"[T29-H] 95% CI = {primary['bootstrap_95ci']}")
    print(f"[T29-H] wrote aggregate report: {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--key", required=True, help="PRIVATE T26 sampling_key_PRIVATE.json; never commit it")
    ap.add_argument("--out", default="results/t29/t29_human_primary.json")
    ap.add_argument("--labels-1", default=str(DEFAULT_LABELS_1))
    ap.add_argument("--labels-2", default=str(DEFAULT_LABELS_2))
    ap.add_argument("--adjudication", default=str(DEFAULT_ADJUDICATION))
    ap.add_argument("--sample-manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--t27-gate", default=str(DEFAULT_T27))
    ap.add_argument("--method-note", default=str(DEFAULT_METHOD_NOTE))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=290819)
    ap.add_argument("--allow-dirty", action="store_true", help="development only; production should use a clean tree")
    args = ap.parse_args()
    run(
        key_path=args.key,
        out_path=args.out,
        labels_1=args.labels_1,
        labels_2=args.labels_2,
        adjudication=args.adjudication,
        sample_manifest=args.sample_manifest,
        t27_gate=args.t27_gate,
        method_note=args.method_note,
        n_boot=args.n_boot,
        seed=args.seed,
        allow_dirty=args.allow_dirty,
    )


if __name__ == "__main__":
    main()
