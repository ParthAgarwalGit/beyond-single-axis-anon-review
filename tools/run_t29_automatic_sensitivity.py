#!/usr/bin/env python
"""Run the T29 full-corpus automatic harmfulness sensitivity.

T27 harmfulness is MEASUREMENT_LIMITED. This runner therefore emits only a
secondary/sensitivity result and mechanically forbids promotion to primary evidence.
It consumes the exact private 400-row Qwen scorer artifact and writes aggregate-only
output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.t29_automatic_sensitivity import PRODUCTION_CONDITIONS, automatic_sensitivity_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SCORING_STATUS = "PRODUCTION_SCORING_PASS_DRAWABLE_BY_T26"
EXPECTED_MATCHED_STATUS = "MATCHED_CONTROLS_COMPLETE_IMMUTABLY_ARCHIVED"


def die(msg):
    raise SystemExit(f"T29 AUTO SENSITIVITY ABORTED (fail-closed): {msg}")


def sha256_file(path):
    p = Path(path)
    if not p.is_file():
        die(f"missing required file: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path, label):
    p = Path(path)
    if not p.is_file():
        die(f"missing {label}: {p}")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        die(f"cannot parse {label}: {exc}")
    if not isinstance(obj, dict):
        die(f"{label} must be a JSON object")
    return obj


def load_jsonl(path):
    p = Path(path)
    if not p.is_file():
        die(f"missing private scored artifact: {p}")
    try:
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        die(f"cannot parse private scored artifact: {exc}")


def validate_authorities(scored_path, receipt_path, manifest_path, t27_path, matched_path, human_path):
    scored_sha = sha256_file(scored_path)
    receipt = load_json(receipt_path, "Qwen scoring production receipt")
    manifest = load_json(manifest_path, "T26 sample manifest")
    t27 = load_json(t27_path, "T27 gate")
    matched = load_json(matched_path, "T29 matched-control execution")
    human = load_json(human_path, "T29 human-primary result")

    if receipt.get("status") != EXPECTED_SCORING_STATUS or receipt.get("source_git_dirty") is not False:
        die("wrong or dirty production scoring receipt")
    checks = receipt.get("report_checks") or {}
    canary = checks.get("canary") or {}
    if checks.get("production") is not True or checks.get("drawable_by_t26") is not True:
        die("scoring receipt is not production/drawable")
    if checks.get("n_rows") != 400 or checks.get("n_coverage_failures") != 0:
        die("production scoring receipt is not the exact 400-row zero-coverage-failure run")
    if canary.get("start_usable") is not True or canary.get("end_usable") is not True or canary.get("drift_detected") is not False:
        die("production scoring canary is not usable/stable")
    artifacts = receipt.get("artifacts") or {}
    if scored_sha != ((artifacts.get("scored") or {}).get("sha256")):
        die("private scored artifact SHA does not match production receipt")

    report_sha = ((artifacts.get("report") or {}).get("sha256"))
    if manifest.get("branch") != "qwen_capping" or manifest.get("n_items") != 120:
        die("wrong T26 sample manifest")
    if manifest.get("source_scored_artifact_sha256") != scored_sha:
        die("T26 manifest is bound to a different scored artifact")
    if manifest.get("source_scoring_report_sha256") != report_sha:
        die("T26 manifest scoring-report SHA differs from the production receipt")

    if t27.get("task") != "T27_causal_judge_validation" or t27.get("branch") != "qwen_capping" or t27.get("gating") is not True:
        die("wrong or non-gating T27 artifact")
    if t27.get("n_unresolved_after_adjudication") != 0:
        die("T27 has unresolved human labels")
    gate = (((t27.get("outcomes") or {}).get("harmfulness") or {}).get("gate") or {})
    if gate.get("branch") != "MEASUREMENT_LIMITED":
        die("T27 harmfulness is not MEASUREMENT_LIMITED")
    if gate.get("primary_evidence") != "human_subset" or gate.get("full_corpus_reportable_as") != "full_corpus_automatic":
        die("T27 does not authorize the automatic full corpus as secondary sensitivity")

    if matched.get("task") != "T29" or matched.get("status") != EXPECTED_MATCHED_STATUS:
        die("wrong/incomplete matched-control execution authority")
    population = matched.get("population") or {}
    if population.get("rows") != 400 or population.get("rows_per_condition") != 100:
        die("matched-control authority does not describe 400 rows / 100 per condition")
    if tuple(population.get("conditions") or ()) != PRODUCTION_CONDITIONS:
        die("matched-control condition inventory differs from production conditions")
    if population.get("sha256") != ((artifacts.get("input") or {}).get("sha256")):
        die("generation population SHA differs between execution and scoring receipt")

    if human.get("task") != "T29_QWEN_HUMAN_PRIMARY" or human.get("status") != "HUMAN_PRIMARY_MEASUREMENT_LIMITED":
        die("wrong/missing human-primary result")
    if human.get("primary_result_eligible") is not True:
        die("human-primary result is not eligible")
    if ((human.get("measurement_branch") or {}).get("automatic_400_row_primary_allowed")) is not False:
        die("human-primary result does not forbid automatic primary promotion")
    if ((human.get("provenance") or {}).get("t27_gate_sha256")) != sha256_file(t27_path):
        die("human-primary result and automatic sensitivity do not bind to the same T27 gate")

    frozen = matched.get("frozen_authorities") or {}
    return {
        "private_scored_sha256": scored_sha,
        "t26_sample_manifest_sha256": sha256_file(manifest_path),
        "t27_gate_sha256": sha256_file(t27_path),
        "human_primary_sha256": sha256_file(human_path),
        "source_scoring_report_sha256": report_sha,
        "generation_population_sha256": population.get("sha256"),
        "t24_controls_npz_sha256": frozen.get("t24_controls_npz_sha256"),
        "t24_freeze_sha256": frozen.get("t24_freeze_sha256"),
        "assistant_axis_sha256": frozen.get("assistant_axis_sha256"),
        "source_capping_config_sha256": frozen.get("source_capping_config_sha256"),
    }, t27


def analyze(args):
    provenance, t27 = validate_authorities(
        args.scored_private, args.production_receipt, args.t26_manifest,
        args.t27_gate, args.matched_execution, args.human_primary,
    )
    report = automatic_sensitivity_report(load_jsonl(args.scored_private), n_boot=args.n_boot, seed=args.seed)

    per_t27 = (((t27.get("outcomes") or {}).get("harmfulness") or {}).get("part_b_non_differential_error") or {}).get("per_condition") or {}
    for condition in PRODUCTION_CONDITIONS:
        got = report["condition_rates"][condition]["strict_harmful_rate"]
        want = (per_t27.get(condition) or {}).get("auto_positive_rate_ip")
        if want is None or abs(got - float(want)) > 1e-12:
            die(f"T27 automatic-rate regression failed for {condition}: {got} vs {want}")

    out = {
        "task": "T29_QWEN_AUTOMATIC_SENSITIVITY",
        "status": "SECONDARY_AUTOMATIC_MEASUREMENT_LIMITED",
        "primary_result_eligible": False,
        "measurement_branch": {
            "t27_branch": "MEASUREMENT_LIMITED",
            "primary_evidence": "human_subset",
            "full_corpus_reportable_as": "full_corpus_automatic",
            "automatic_400_row_primary_allowed": False,
        },
        "method": {
            "paired": True,
            "bootstrap_replicates": int(args.n_boot),
            "bootstrap_seed": int(args.seed),
            "mcnemar_role": "secondary automatic-label diagnostic only",
        },
        "provenance": provenance,
        "regression_check": {
            "target": "T27 IP-weighted automatic strict-harm rate by condition",
            "status": "PASS",
            "tolerance": 1e-12,
        },
        "report": report,
        "claim_guard": "T27 harmfulness is MEASUREMENT_LIMITED. These paired full-corpus automatic-label statistics are secondary sensitivity evidence only and cannot override the human-primary T29-H result.",
        "privacy": "aggregate only; no item IDs, UIDs, completions, rationales, or per-item automatic labels are emitted",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    primary = report["comparisons"]["axis_vs_random"]["strict"]
    print("[T29 AUTO SECONDARY] COMPLETE")
    print(f"[T29 AUTO SECONDARY] axis-random strict RD = {primary['risk_difference']:+.6f}")
    print(f"[T29 AUTO SECONDARY] 95% CI = {primary['bootstrap_95ci']}")
    print(f"[T29 AUTO SECONDARY] McNemar p = {primary['mcnemar_exact_two_sided_p']}")
    print(f"[T29 AUTO SECONDARY] wrote aggregate report: {out_path}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored-private", required=True)
    ap.add_argument("--production-receipt", default=str(ROOT / "results/t29/QWEN_CAUSAL_SCORING_PRODUCTION_RECEIPT.json"))
    ap.add_argument("--t26-manifest", default=str(ROOT / "annotations/t26_causal/sample_manifest.json"))
    ap.add_argument("--t27-gate", default=str(ROOT / "results/t27/qwen_causal_judge_validation.json"))
    ap.add_argument("--matched-execution", default=str(ROOT / "results/t29/qwen_matched_control_execution.json"))
    ap.add_argument("--human-primary", default=str(ROOT / "results/t29/t29_human_primary.json"))
    ap.add_argument("--out", default=str(ROOT / "results/t29/t29_automatic_sensitivity.json"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=290819)
    analyze(ap.parse_args(argv))


if __name__ == "__main__":
    main()
