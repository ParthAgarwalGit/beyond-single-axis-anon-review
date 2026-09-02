#!/usr/bin/env python
"""Export the frozen T24 engagement-matching diagnostic used by T29.

This is outcome-blind calibration evidence. It records intervention incidence only;
it must not be described as matching intervention magnitude or geometry.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPECTED_LAYERS = tuple(range(46, 54))


def die(msg):
    raise SystemExit(f"T29 ENGAGEMENT EXPORT ABORTED: {msg}")


def export(freeze_path, out_path):
    p = Path(freeze_path)
    if not p.is_file():
        die(f"missing T24 freeze: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("task") != "T24" or doc.get("status") != "FROZEN_NUMERICAL_CONTROLS":
        die("T24 control artifact is not the frozen numerical authority")
    validation = doc.get("validation") or {}
    if validation.get("primary_controls_engagement_matched") != "PASS":
        die("T24 primary control engagement matching is not PASS")
    layers = doc.get("layers") or {}
    if tuple(sorted(int(k) for k in layers)) != EXPECTED_LAYERS:
        die("T24 layer inventory is not exactly 46-53")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "layer", "source_axis_engagement_rate", "random_engagement_rate",
            "sphere_engagement_rate", "random_minus_source", "sphere_minus_source",
        ])
        w.writeheader()
        for layer in EXPECTED_LAYERS:
            rec = layers[str(layer)]
            source = float(rec["source_axis_engagement_rate"])
            random = float(rec["random_engagement_rate"])
            sphere = float(rec["sphere_engagement_rate"])
            random_delta = float(rec["random_minus_source_engagement"])
            sphere_delta = float(rec["sphere_minus_source_engagement"])
            if abs(random - source) > 1e-12 or abs(sphere - source) > 1e-12:
                die(f"engagement mismatch at layer {layer}")
            if abs(random_delta) > 1e-12 or abs(sphere_delta) > 1e-12:
                die(f"recorded engagement delta is nonzero at layer {layer}")
            w.writerow({
                "layer": layer,
                "source_axis_engagement_rate": source,
                "random_engagement_rate": random,
                "sphere_engagement_rate": sphere,
                "random_minus_source": random_delta,
                "sphere_minus_source": sphere_delta,
            })
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t24-freeze", default="results/t24/source_control_freeze.json")
    ap.add_argument("--out", default="design/figures/sources/t29/t29_frozen_engagement_matching.csv")
    args = ap.parse_args()
    export(args.t24_freeze, args.out)


if __name__ == "__main__":
    main()
