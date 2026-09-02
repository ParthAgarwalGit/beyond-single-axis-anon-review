#!/usr/bin/env python
"""Export aggregate-only T29 source data for causal figures and tables.

The human T29-H report is primary under the T27 MEASUREMENT_LIMITED branch. The
400-row automatic report is secondary sensitivity only. This exporter preserves that
status in a compact comparison source and never reads private item-level data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def die(msg):
    raise SystemExit(f"T29 FIGURE EXPORT ABORTED: {msg}")


def load_json(path, label):
    p = Path(path)
    if not p.exists():
        die(f"missing {label}: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        die(f"{label} must be a JSON object")
    return doc


def load_human(path):
    doc = load_json(path, "T29-H aggregate report")
    if doc.get("task") != "T29_QWEN_HUMAN_PRIMARY":
        die(f"wrong human task identifier: {doc.get('task')!r}")
    if doc.get("status") != "HUMAN_PRIMARY_MEASUREMENT_LIMITED":
        die(f"human report is not frozen T29-H: {doc.get('status')!r}")
    if doc.get("primary_result_eligible") is not True:
        die("human report is not eligible as primary evidence")
    return doc


def load_automatic(path):
    doc = load_json(path, "T29 automatic sensitivity")
    if doc.get("task") != "T29_QWEN_AUTOMATIC_SENSITIVITY":
        die(f"wrong automatic task identifier: {doc.get('task')!r}")
    if doc.get("status") != "SECONDARY_AUTOMATIC_MEASUREMENT_LIMITED":
        die(f"automatic report has wrong status: {doc.get('status')!r}")
    if doc.get("primary_result_eligible") is not False:
        die("automatic report must remain secondary")
    if ((doc.get("measurement_branch") or {}).get("automatic_400_row_primary_allowed")) is not False:
        die("automatic report does not mechanically forbid primary promotion")
    return doc


def export(human_path, automatic_path, out_dir):
    human = load_human(human_path)
    automatic = load_automatic(automatic_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    human_report = human["report"]
    rates_path = out / "t29_human_primary_condition_rates.csv"
    with rates_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "outcome", "condition", "population_n", "human_sample_n",
            "human_sample_positive_n", "weighted_positive_estimate", "rate",
        ])
        w.writeheader()
        for outcome in ("strict", "inclusive"):
            for condition, rec in human_report["outcomes"][outcome]["condition_rates"].items():
                w.writerow({"outcome": outcome, "condition": condition, **rec})

    contrasts_path = out / "t29_human_primary_contrasts.csv"
    with contrasts_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "outcome", "comparison", "estimand", "rate_A", "rate_B",
            "risk_difference", "ci95_low", "ci95_high", "paired",
            "bootstrap_replicates", "bootstrap_seed",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for outcome in ("strict", "inclusive"):
            for comparison, rec in human_report["outcomes"][outcome]["comparisons"].items():
                lo, hi = rec["bootstrap_95ci"]
                w.writerow({
                    "outcome": outcome, "comparison": comparison, "estimand": rec["estimand"],
                    "rate_A": rec["rate_A"], "rate_B": rec["rate_B"],
                    "risk_difference": rec["risk_difference"], "ci95_low": lo, "ci95_high": hi,
                    "paired": rec["paired"], "bootstrap_replicates": rec["bootstrap_replicates"],
                    "bootstrap_seed": rec["bootstrap_seed"],
                })

    auto_report = automatic["report"]
    auto_rates_path = out / "t29_automatic_sensitivity_condition_rates.csv"
    with auto_rates_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "condition", "n", "strict_harmful_n", "strict_harmful_rate",
            "inclusive_harmful_n", "inclusive_harmful_rate", "refusal_n", "refusal_rate",
            "degenerate_n", "degenerate_rate", "coverage_failure_n",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for condition, rec in auto_report["condition_rates"].items():
            w.writerow({"condition": condition, **rec})

    auto_contrasts_path = out / "t29_automatic_sensitivity_contrasts.csv"
    with auto_contrasts_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "outcome", "comparison", "contrast", "rate_A", "rate_B", "risk_difference",
            "ci95_low", "ci95_high", "mcnemar_exact_two_sided_p", "discordant_A1_B0",
            "discordant_A0_B1", "n_pairs", "paired", "bootstrap_replicates", "bootstrap_seed",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for comparison, outcomes in auto_report["comparisons"].items():
            for outcome, rec in outcomes.items():
                lo, hi = rec["bootstrap_95ci"]
                w.writerow({
                    "outcome": outcome, "comparison": comparison, "contrast": rec["contrast"],
                    "rate_A": rec["rate_A"], "rate_B": rec["rate_B"],
                    "risk_difference": rec["risk_difference"], "ci95_low": lo, "ci95_high": hi,
                    "mcnemar_exact_two_sided_p": rec["mcnemar_exact_two_sided_p"],
                    "discordant_A1_B0": rec["discordant_A1_B0"],
                    "discordant_A0_B1": rec["discordant_A0_B1"], "n_pairs": rec["n_pairs"],
                    "paired": rec["paired"], "bootstrap_replicates": rec["bootstrap_replicates"],
                    "bootstrap_seed": rec["bootstrap_seed"],
                })

    compare_path = out / "t29_primary_vs_automatic_axis_random.csv"
    with compare_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["evidence", "outcome", "rate_assistant", "rate_random", "risk_difference", "ci95_low", "ci95_high", "primary"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        hp = human_report["outcomes"]["strict"]["comparisons"]["axis_vs_random"]
        lo, hi = hp["bootstrap_95ci"]
        w.writerow({"evidence": "human_design_weighted", "outcome": "strict", "rate_assistant": hp["rate_A"], "rate_random": hp["rate_B"], "risk_difference": hp["risk_difference"], "ci95_low": lo, "ci95_high": hi, "primary": True})
        ap = auto_report["comparisons"]["axis_vs_random"]["strict"]
        lo, hi = ap["bootstrap_95ci"]
        w.writerow({"evidence": "automatic_full_corpus", "outcome": "strict", "rate_assistant": ap["rate_A"], "rate_random": ap["rate_B"], "risk_difference": ap["risk_difference"], "ci95_low": lo, "ci95_high": hi, "primary": False})

    for path in (rates_path, contrasts_path, auto_rates_path, auto_contrasts_path, compare_path):
        print(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human", default="results/t29/t29_human_primary.json")
    ap.add_argument("--automatic", default="results/t29/t29_automatic_sensitivity.json")
    ap.add_argument("--out-dir", default="design/figures/sources/t29")
    args = ap.parse_args()
    export(args.human, args.automatic, args.out_dir)


if __name__ == "__main__":
    main()
