#!/usr/bin/env python
"""Retention-boundary diagnostic supporting the 2026-08-17 membership deviation.

T07 rejected the automatic judge at the score-3 boundary. Human-human agreement is
itself moderate: four-class kappa is 0.556 and score-3-versus-rest kappa is 0.622, so
2-versus-3 is hard to measure even between annotators. This tool asks whether a more
permissive boundary (score >= 2, "expressed the role at all") would be reliable enough
to serve as a behavioural retention filter.

The comparison is EXPLORATORY. Collapsing the rubric at >= 2 makes macro-F1 and kappa
binary quantities, whereas the frozen gate's macro-F1 and kappa are four-class, so
those two are reported against the same numerical thresholds for orientation rather
than as a re-application of the gate. Precision and recall are directly comparable.

It reuses ``tools/run_t07_judge_validation.py``'s loader, so the gold/private-key
join is bit-identical to the committed T07 result.

PRIVACY: output is AGGREGATE ONLY - counts, rates, and a 4x4 confusion of counts.
No uids, no G-ids, no per-item lu_scores. The private sampling key is read but never
copied into the report. Only this aggregate report may be committed.

    python tools/run_boundary_diagnostic.py \
        --key annotations/t14_gold_v3/sampling_key_PRIVATE.json \
        --out results/t07/boundary_diagnostic.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.provenance import stamp_report  # noqa: E402
from tools.run_t07_judge_validation import GATE, _load_inputs  # noqa: E402

BOUNDARIES = {"score_eq_3": lambda s: s == 3, "score_ge_2": lambda s: s >= 2}


def _wkappa(pairs, weights):
    tot = sum(weights)
    if not tot:
        return None
    po = sum(w for (m, h), w in zip(pairs, weights) if m == h) / tot
    pe = 0.0
    for c in (True, False):
        pm = sum(w for (m, _), w in zip(pairs, weights) if m == c) / tot
        ph = sum(w for (_, h), w in zip(pairs, weights) if h == c) / tot
        pe += pm * ph
    return None if pe == 1 else (po - pe) / (1 - pe)


def _boundary(scored, name, meets):
    tp = fp = fn = tn = 0
    w_tp = w_fn = w_fp = w_tn = 0.0
    pairs, wts = [], []
    for r in scored:
        m, h, w = meets(r["lu_score"]), meets(r["human_label"]), r["ip_weight"]
        pairs.append((m, h)); wts.append(w)
        if m and h:
            tp += 1; w_tp += w
        elif m:
            fp += 1; w_fp += w
        elif h:
            fn += 1; w_fn += w
        else:
            tn += 1; w_tn += w
    # Precision is unweighted: sampling was stratified on the automatic score, so
    # conditioning on a machine-positive set is already representative. Recall and
    # macro-F1 are ip-weighted for corpus validity (frozen T07 weighting rule).
    prec = tp / (tp + fp) if tp + fp else None
    rec = w_tp / (w_tp + w_fn) if w_tp + w_fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    nprec = w_tn / (w_tn + w_fn) if w_tn + w_fn else None
    nrec = w_tn / (w_tn + w_fp) if w_tn + w_fp else None
    nf1 = 2 * nprec * nrec / (nprec + nrec) if nprec and nrec else None
    macro = (f1 + nf1) / 2 if f1 is not None and nf1 is not None else None
    kap = _wkappa(pairs, wts)
    # Comparison against the frozen NUMBERS, not a re-application of the frozen gate.
    # precision/recall are like-for-like at either boundary; macro-F1 and kappa are
    # four-class in the gate but binary once the rubric is collapsed at >= 2, so those
    # two are flagged not_comparable rather than reported as passes or failures.
    # This tool always collapses the rubric to two classes, at BOTH boundaries, so its
    # macro-F1 and kappa are binary quantities. The frozen gate's macro-F1 (0.409) and
    # four-class kappa (0.233) are four-class and are computed in
    # run_t07_judge_validation.py; they are not reproduced or re-tested here.
    # Precision and recall condition on the same positive class in both places and are
    # therefore the only directly comparable quantities.
    met = {
        "precision": (prec or 0) >= GATE["score3_precision_min"],
        "recall": (rec or 0) >= GATE["score3_recall_min"],
        "macro_f1": "not_comparable_binary_vs_four_class",
        "kappa": "not_comparable_binary_vs_four_class",
    }
    comparable = [v for v in met.values() if isinstance(v, bool)]
    return {
        "boundary": name,
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision_unweighted": prec,
        "recall_ip": rec,
        "recall_unweighted": tp / (tp + fn) if tp + fn else None,
        "macro_f1_ip": macro,
        "kappa_binary_ip": kap,
        "kappa_binary_unweighted": _wkappa(pairs, [1.0] * len(pairs)),
        "comparison_status": ("EXPLORATORY: this tool collapses the rubric to two "
                              "classes, so its macro-F1 and kappa are binary and are "
                              "NOT the frozen four-class criteria; see "
                              "results/t07/judge_validation.json for those"),
        "meets_frozen_threshold": met,
        "n_comparable_thresholds_met": sum(comparable),
        "n_comparable_thresholds": len(comparable),
    }


def run(consensus_path, key_path, out_path):
    rows, _ = _load_inputs(consensus_path, key_path)
    scored = [r for r in rows
              if r["human_validity"] == "ok" and r["lu_score"] is not None]
    conf = {}
    for r in scored:
        conf.setdefault(f"human_{r['human_label']}", {})
        k = f"machine_{r['lu_score']}"
        conf[f"human_{r['human_label']}"][k] = conf[f"human_{r['human_label']}"].get(k, 0) + 1
    report = {
        "task": "retention_boundary_diagnostic",
        "purpose": "supports docs/DEVIATION_2026-08-17_CONFIRMATORY_MEMBERSHIP.md",
        "population": "scored (human validity ok AND archived lu_score present)",
        "n_scored": len(scored),
        "gate_reference": GATE,
        "boundaries": {n: _boundary(scored, n, f) for n, f in BOUNDARIES.items()},
        "confusion_4x4_counts": conf,
        "privacy": "aggregate only; no uids, G-ids, or per-item scores",
    }
    report = stamp_report(report, allow_dirty=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for n, b in report["boundaries"].items():
        print(f"[{n}] precision={b['precision_unweighted']:.3f} "
              f"recall_ip={b['recall_ip']:.3f} macroF1={b['macro_f1_ip']:.3f} "
              f"kappa={b['kappa_binary_ip']:.3f} -> "
              f"{b['n_comparable_thresholds_met']}/{b['n_comparable_thresholds']} "
              f"comparable thresholds met")
    print(f"-> {out_path}")
    return report


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--consensus", default=str(root / "results" / "t06" / "gold_consensus.csv"))
    ap.add_argument("--key", default=str(root / "annotations" / "t14_gold_v3" / "sampling_key_PRIVATE.json"))
    ap.add_argument("--out", default=str(root / "results" / "t07" / "boundary_diagnostic.json"))
    a = ap.parse_args(argv)
    run(a.consensus, a.key, a.out)


if __name__ == "__main__":
    main()
