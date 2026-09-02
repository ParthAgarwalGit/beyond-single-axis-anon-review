#!/usr/bin/env python
"""T26 — draw a blinded human-validation sample from a frozen scored corpus.

The draw is allowed only from the exact scored artifact named by a production,
drawable causal-judge scoring report. This prevents T26 from sampling a partial run,
a hash-mismatched file, an unavailable instrument run, or an instrument run whose
start/end canary drifted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.causal_validation import HARMFULNESS_LABELS  # noqa: E402
from src.provenance import git_dirty, stamp_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "t27_causal_judge_frozen.json"
KEY_NAME = "sampling_key_PRIVATE.json"
WORKSHEET_FIELDS = ("item_id", "role", "role_description", "question", "completion")

PRODUCTION_POPULATIONS = {
    "deepseek_steering": {"total": 5000, "per_condition": 1000},
    "qwen_capping": {"total": 400, "per_condition": 100},
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_cfg(branch):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    branches = cfg["scope"]["branches"]
    if branch not in branches:
        raise SystemExit(f"unknown --branch {branch!r}; expected one of {sorted(branches)}")
    return cfg, cfg["sample"], list(branches[branch]["conditions"]), branches[branch]


def _get(row, *names, default=None):
    for name in names:
        if row.get(name) not in (None, ""):
            return row[name]
    return default


def _validate_scoring_report(report_path, outputs_path, branch, conditions):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("task") != "causal_judge_scoring":
        raise SystemExit("--scoring-report is not a causal_judge_scoring report")
    if report.get("branch") != branch:
        raise SystemExit(
            f"scoring report branch {report.get('branch')!r} does not match {branch!r}")
    if report.get("production") is not True or report.get("drawable_by_t26") is not True:
        raise SystemExit("scoring report is not production=true and drawable_by_t26=true")
    if report.get("source_git_dirty") is not False:
        raise SystemExit(
            "scoring report does not certify source_git_dirty: false; a missing or "
            "true value is rejected even if production=true, so an older report "
            "predating this field or a hand-edited one cannot be accepted on trust "
            "in a field it never independently certified.")
    if (report.get("canary") or {}).get("drift_detected") is not False:
        raise SystemExit("scoring report does not certify a stable start/end judge canary")

    source_sha = report.get("source_git_sha")
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise SystemExit("scoring report lacks a valid 40-hex source_git_sha")

    scored_sha = ((report.get("artifact_sha256") or {}).get("scored"))
    actual_sha = _sha256(outputs_path)
    if not isinstance(scored_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", scored_sha):
        raise SystemExit("scoring report lacks a valid scored-artifact SHA-256")
    if scored_sha != actual_sha:
        raise SystemExit(
            f"scored artifact hash mismatch ({actual_sha[:12]} vs {scored_sha[:12]})")

    spec = PRODUCTION_POPULATIONS[branch]
    if report.get("n_rows") != spec["total"]:
        raise SystemExit(
            f"scoring report has n_rows={report.get('n_rows')!r}; expected {spec['total']}")

    # Defense in depth for reports produced by the pre-fix scorer. That scorer
    # could stamp production=true even when every API call failed and every row
    # was a coverage failure. T26 must reject such an artifact regardless of the
    # booleans in the report.
    n_coverage = report.get("n_coverage_failures")
    if isinstance(n_coverage, int) and n_coverage >= spec["total"]:
        raise SystemExit(
            "scoring report has no usable automatic labels: all production rows are "
            "coverage failures; T26 cannot draw from an unavailable judge instrument")
    if report.get("all_production_coverage_failed") is True:
        raise SystemExit("scoring report certifies all production rows failed judge coverage")
    canary = report.get("canary") or {}
    if canary.get("start_usable") is False or canary.get("end_usable") is False:
        raise SystemExit("scoring report certifies an unusable start/end judge canary")

    per = report.get("per_condition") or {}
    for condition in conditions:
        if (per.get(condition) or {}).get("n") != spec["per_condition"]:
            raise SystemExit(
                f"scoring report condition {condition!r} is not exactly "
                f"{spec['per_condition']} rows")
    if set(per) != set(conditions):
        raise SystemExit("scoring report condition inventory differs from the frozen branch")
    return report


def draw(outputs_path, out_dir, seed, branch, scoring_report_path,
         allow_dirty=False):
    pre_write_dirty = git_dirty()
    if pre_write_dirty and not allow_dirty:
        raise RuntimeError(
            "refusing to draw a production sample from a dirty working tree; "
            "commit first or pass allow_dirty=True (development only). Checked "
            "before writing any worksheet/key output so a clean run cannot "
            "dirty its own tree and fail on the later provenance stamp."
        )
    cfg, sample_cfg, conditions, branch_cfg = _load_cfg(branch)
    scoring = _validate_scoring_report(scoring_report_path, outputs_path, branch, conditions)
    per_condition_target = int(sample_cfg["per_condition"])
    rng = random.Random(seed)

    rows = [json.loads(line) for line in Path(outputs_path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise SystemExit(f"no rows in {outputs_path}")

    spec = PRODUCTION_POPULATIONS[branch]
    if len(rows) != spec["total"]:
        raise SystemExit(f"scored corpus has {len(rows)} rows; expected {spec['total']}")
    uids = [_get(row, "uid", "rollout_id") for row in rows]
    if any(uid is None for uid in uids) or len(set(uids)) != len(uids):
        raise SystemExit("scored corpus must have exactly one unique uid per row")

    present = {_get(row, "condition", "steering_condition") for row in rows}
    if present != set(conditions):
        raise SystemExit(
            f"scored corpus condition inventory {sorted(present)} does not match frozen "
            f"inventory {conditions}")

    per_population = {condition: 0 for condition in conditions}
    strata = {}
    for row in rows:
        condition = _get(row, "condition", "steering_condition")
        per_population[condition] += 1
        auto_h = _get(row, "auto_harmfulness", "harmfulness")
        if auto_h is not None and auto_h not in HARMFULNESS_LABELS:
            raise SystemExit(f"automatic harmfulness {auto_h!r} outside the frozen set")
        stratum_h = auto_h if auto_h is not None else "PARSE_FAILURE"
        strata.setdefault((condition, stratum_h), []).append(row)
    for condition, count in per_population.items():
        if count != spec["per_condition"]:
            raise SystemExit(
                f"condition {condition} has {count} rows; expected {spec['per_condition']}")

    picked, allocation = [], {}
    for condition in conditions:
        cond_strata = {key: value for key, value in strata.items() if key[0] == condition}
        population = sum(len(value) for value in cond_strata.values())
        quota = min(per_condition_target, population)
        alloc = {key: 1 for key in cond_strata}
        left = quota - len(alloc)
        if left < 0:
            keep = sorted(cond_strata, key=lambda key: -len(cond_strata[key]))[:quota]
            alloc = {key: 1 for key in keep}
            left = 0
        order = sorted(alloc, key=lambda key: -len(cond_strata[key]))
        cursor = 0
        while left > 0:
            key = order[cursor % len(order)]
            if alloc[key] < len(cond_strata[key]):
                alloc[key] += 1
                left -= 1
            elif all(alloc[x] >= len(cond_strata[x]) for x in order):
                break
            cursor += 1
        for key, n in alloc.items():
            pool = list(cond_strata[key])
            rng.shuffle(pool)
            chosen = pool[:n]
            allocation[f"{key[0]}|{key[1]}"] = {
                "population": len(pool),
                "sampled": len(chosen),
                "ip_weight": len(pool) / len(chosen),
            }
            picked.extend((key, row) for row in chosen)

    rng.shuffle(picked)
    worksheet, private_key = [], {}
    for index, ((condition, stratum_h), row) in enumerate(picked, 1):
        item_id = f"C{index:04d}"
        worksheet.append({
            "item_id": item_id,
            "role": _get(row, "role", "role_id", default=""),
            "role_description": _get(row, "role_description", "description", default=""),
            "question": _get(row, "question", "question_text", default=""),
            "completion": _get(row, "completion", "output_text", default=""),
        })
        auto_h = _get(row, "auto_harmfulness", "harmfulness")
        private_key[item_id] = {
            "uid": _get(row, "uid", "rollout_id"),
            "condition": condition,
            "stratum": f"{condition}|{stratum_h}",
            "ip_weight": allocation[f"{condition}|{stratum_h}"]["ip_weight"],
            "auto_harmfulness": auto_h,
            "auto_identity": _get(row, "auto_identity", "identity"),
            "judge_coverage_failure": bool(row.get("judge_coverage_failure", auto_h is None)),
        }

    expected_sample_n = per_condition_target * len(conditions)
    if len(worksheet) != expected_sample_n:
        raise SystemExit(
            f"draw produced {len(worksheet)} items; frozen branch requires {expected_sample_n}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    worksheet_path = out / "worksheet.csv"
    with open(worksheet_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(WORKSHEET_FIELDS))
        writer.writeheader()
        writer.writerows(worksheet)
    for labeller in (1, 2):
        (out / f"worksheet_labeller{labeller}.csv").write_text(
            worksheet_path.read_text(encoding="utf-8"), encoding="utf-8")

    key_path = out / KEY_NAME
    key_path.write_text(
        json.dumps({"items": private_key, "seed": seed}, indent=1), encoding="utf-8")

    manifest = stamp_report({
        "task": "T26_causal_human_validation_sample",
        "branch": branch,
        "judge_model": scoring["judge_model"],
        "judge_provider": scoring["provider"],
        "judge_provider_model_id": scoring.get("provider_model_id"),
        "judge_endpoint": scoring.get("endpoint"),
        "judge_source_git_sha": scoring["source_git_sha"],
        "source_scoring_report_sha256": _sha256(scoring_report_path),
        "source_scored_artifact_sha256": _sha256(outputs_path),
        "validation_window_constraint":
            cfg["judged_instrument"]["reproducibility_limitation"]["validation_window_constraint"],
        "authority": "docs/DEVIATION_2026-08-18_T27_CAUSAL_JUDGE_GATE.md",
        # Derived from the bound scoring report, NOT from the frozen config's
        # preregistration prose. That prose is written before execution and goes
        # stale the moment a stage completes; a field named "at_draw" must state
        # what was actually true at draw time. Every value below is only
        # reachable after _validate_scoring_report() has already enforced
        # production/drawable, the exact frozen condition inventory, and exactly
        # spec["per_condition"] rows per condition - so this is provable from the
        # bound artifact, not asserted.
        "execution_status_at_draw": {
            "derived_from": "bound scoring report (source_scoring_report_sha256)",
            "all_frozen_conditions_generated_and_scored": True,
            "n_rows_scored": scoring.get("n_rows"),
            "rows_per_condition": {
                condition: (scoring.get("per_condition") or {}).get(condition, {}).get("n")
                for condition in conditions
            },
            "n_coverage_failures": scoring.get("n_coverage_failures"),
            "scoring_production": scoring.get("production"),
            "scoring_drawable_by_t26": scoring.get("drawable_by_t26"),
        },
        "frozen_config_execution_status_text": {
            "text": branch_cfg["execution_status"],
            "status": "PREREGISTRATION_TEXT_MAY_PREDATE_EXECUTION",
            "note": (
                "Retained verbatim from configs/t27_causal_judge_frozen.json for audit. "
                "It is preregistration prose, not a statement about draw-time state, and "
                "the frozen config is deliberately not edited to keep it stale-proof. "
                "Use execution_status_at_draw for what was true when this sample was drawn."),
        },
        "seed": seed,
        "n_items": len(worksheet),
        "per_condition_target": per_condition_target,
        "conditions": conditions,
        "strata": allocation,
        "worksheet_sha256": _sha256(worksheet_path),
        "private_key_sha256": _sha256(key_path),
        "private_key_committed": False,
        "blinded_from_labellers": sorted(set(private_key[next(iter(private_key))]) - {"ip_weight"}),
    }, allow_dirty=allow_dirty, dirty=pre_write_dirty)
    (out / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[T26] {len(worksheet)} items across {len(conditions)} conditions -> {worksheet_path}")
    print(f"[T26] PRIVATE KEY {key_path} — DO NOT COMMIT")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    draw_parser = sub.add_parser("draw")
    draw_parser.add_argument("--outputs", required=True,
                             help="private scored JSONL produced by run_causal_judge_scoring.py")
    draw_parser.add_argument("--scoring-report", required=True,
                             help="committed provenance report for --outputs")
    draw_parser.add_argument("--out", default="annotations/t26_causal")
    draw_parser.add_argument(
        "--allow-dirty", action="store_true",
        help="DEVELOPMENT ONLY: stamp the sample manifest from a dirty working tree. "
             "A production draw must run from a clean checkout, because the manifest "
             "is the authority T27 binds its gating run to.")
    draw_parser.add_argument("--seed", type=int, default=20260818)
    draw_parser.add_argument("--branch", required=True,
                             choices=("deepseek_steering", "qwen_capping"))
    args = parser.parse_args(argv)
    draw(args.outputs, args.out, args.seed, args.branch, args.scoring_report,
         allow_dirty=args.allow_dirty)


if __name__ == "__main__":
    main()
