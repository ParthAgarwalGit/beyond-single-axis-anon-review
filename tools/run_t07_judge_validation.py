#!/usr/bin/env python3
"""T07 — validate and (when it passes) freeze the production role judge.

Implements the frozen T07_SCORING_SPEC gate against the adjudicated T06 gold
consensus, with no rubric tuning. Reports the three populations separately
(scored / abstained / unjudgeable), evaluates the predeclared gate on the scored
population, reports the abstention rate next to it, ip-weights recall and
macro-F1 for corpus validity, and reports each arm as well as pooled.

    scored      = human (adjudicated) validity ok AND lu_score is not None
    abstained   = human ok AND lu_score is None      (judge coverage failure)
    unjudgeable = human (adjudicated) validity unjudgeable

The gate (score-3 precision >= 0.85, recall >= 0.85, macro-F1 >= 0.75,
four-class kappa >= 0.70) is evaluated on `scored`. score-3 precision is
unbiased unweighted; recall and macro-F1 are ip-weighted for the corpus and the
unweighted sample values are reported alongside.

BLOCKED INPUTS (why real numbers are not produced yet)
------------------------------------------------------
This needs two things that are not in the repo:
  1. the FINAL adjudicated T06 consensus (a gold label per item) — the current
     results/t06/gold_consensus.PROVISIONAL.csv still has ADJUDICATION_REQUIRED
     rows and lacks labeller-1's round-2;
  2. the per-item automatic judge score + ip_weight + arm + sensitivity_id from
     `annotations/t14_gold_v3/sampling_key_PRIVATE.json`, which is never
     committed by design.
`--consensus`/`--key` fail closed until both exist. `--self-test` validates the
scorer on synthetic data now, with no private data, no judge, no labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import provenance

FORMAT_VERSION = "t07-judge-validation-v1"
GATE = {"score3_precision_min": 0.85, "score3_recall_min": 0.85,
        "macro_f1_min": 0.75, "four_class_kappa_min": 0.70}
LABELS = [0, 1, 2, 3]


def die(msg):
    sys.exit(f"\nT07 ABORTED (fail-closed): {msg}\n")


def _metrics(h, j, w):
    """h, j: int label arrays (human gold, judge). w: ip weights. Scored pop."""
    h = np.asarray(h, int); j = np.asarray(j, int); w = np.asarray(w, float)
    h3 = (h == 3).astype(int); j3 = (j == 3).astype(int)
    out = {
        "n": int(len(h)),
        # precision conditions on the judge's own label -> unbiased unweighted
        "score3_precision": _safe(lambda: precision_score(h3, j3, zero_division=0)),
        "score3_recall_unweighted": _safe(lambda: recall_score(h3, j3, zero_division=0)),
        "score3_recall_ipweighted": _safe(
            lambda: recall_score(h3, j3, sample_weight=w, zero_division=0)),
        "macro_f1_unweighted": _safe(
            lambda: f1_score(h, j, labels=LABELS, average="macro", zero_division=0)),
        "macro_f1_ipweighted": _safe(
            lambda: f1_score(h, j, labels=LABELS, average="macro",
                             sample_weight=w, zero_division=0)),
        "four_class_kappa": _safe(lambda: cohen_kappa_score(h, j, labels=LABELS)),
    }
    return out


def _safe(fn):
    try:
        v = float(fn())
        return round(v, 6) if np.isfinite(v) else None
    except Exception:
        return None


def _gate(m):
    """Gate on the ip-weighted recall/macro-F1 (corpus-valid) + precision + kappa."""
    checks = {
        "score3_precision": (m["score3_precision"], GATE["score3_precision_min"]),
        "score3_recall": (m["score3_recall_ipweighted"], GATE["score3_recall_min"]),
        "macro_f1": (m["macro_f1_ipweighted"], GATE["macro_f1_min"]),
        "four_class_kappa": (m["four_class_kappa"], GATE["four_class_kappa_min"]),
    }
    passed = {k: (v is not None and v >= thr) for k, (v, thr) in checks.items()}
    return {"per_metric": {k: {"value": v, "min": thr, "pass": passed[k]}
                           for k, (v, thr) in checks.items()},
            "all_pass": all(passed.values())}


def _crosstab(rows, field):
    from collections import Counter
    return dict(sorted(Counter(str(r.get(field)) for r in rows).items()))


def validate(rows, sensitivity_pairs=None):
    """rows: dicts with human_label(int), human_validity(str), lu_score(int|None),
    ip_weight(float), arm(str), machine_validity(str). sensitivity_pairs: optional
    list for the sensitivity-set analysis. Returns the report."""
    def partition(subset):
        scored = [r for r in subset
                  if r["human_validity"] == "ok" and r["lu_score"] is not None]
        abstained = [r for r in subset
                     if r["human_validity"] == "ok" and r["lu_score"] is None]
        unjudge = [r for r in subset if r["human_validity"] == "unjudgeable"]
        n_ok = len(scored) + len(abstained)
        block = {
            "n_total": len(subset),
            "n_scored": len(scored),
            "n_abstained": len(abstained),
            "n_unjudgeable": len(unjudge),
            "abstention_rate_over_human_ok": (round(len(abstained) / n_ok, 6)
                                              if n_ok else None),
            # T07_SCORING_SPEC 3: unjudgeable population cross-tabbed against the
            # machine validity flag (was the row machine-closed/unclosed).
            "unjudgeable_x_machine_validity": _crosstab(unjudge, "machine_validity"),
            "abstained_x_machine_validity": _crosstab(abstained, "machine_validity"),
        }
        if scored:
            m = _metrics([r["human_label"] for r in scored],
                         [r["lu_score"] for r in scored],
                         [r["ip_weight"] for r in scored])
            block["scored_metrics"] = m
            block["gate"] = _gate(m)
        else:
            block["scored_metrics"] = None
            block["gate"] = None
        return block

    arms = sorted({r["arm"] for r in rows})
    report = {
        "format_version": FORMAT_VERSION,
        "provisional": True,
        "gate_thresholds": GATE,
        "pooled": partition(rows),
        "by_arm": {arm: partition([r for r in rows if r["arm"] == arm]) for arm in arms},
        "sensitivity_set": sensitivity_paired_distribution(sensitivity_pairs or []),
    }
    report["selected_branch"] = _select_branch(report)
    return report


def sensitivity_paired_distribution(pairs):
    """T07_SCORING_SPEC 6. pairs: [{ao_label,fr_label,ao_validity,fr_validity}],
    ao = final-answer-only (primary), fr = full-response (reasoning trace shown),
    joined by sensitivity_id. Reports the paired 4x4 label distribution and how
    often seeing the reasoning trace flips the 2/3 (retained) decision."""
    if not pairs:
        return {"n": 0, "note": "no sensitivity pairs supplied"}
    cm = [[0] * 4 for _ in range(4)]
    flips_23 = 0
    for p in pairs:
        a, f = int(p["ao_label"]), int(p["fr_label"])
        cm[a][f] += 1
        if {a, f} == {2, 3}:
            flips_23 += 1
    return {
        "n": len(pairs),
        "paired_label_confusion_rows_answer_only_cols_full_response_0123": cm,
        "n_2_3_flips": flips_23,
        "n_validity_flips": sum(1 for p in pairs
                                if p.get("ao_validity") != p.get("fr_validity")),
        "note": ("answer-only (primary) vs full-response human labels on the "
                 "sensitivity subset. A 2<->3 flip means seeing the reasoning "
                 "trace changed the retained/not-retained call. Sensitivity "
                 "analysis, not part of the role-judge gate."),
    }


def _select_branch(report):
    """Narrowest supported branch. This is a suggested reading of the gate, not
    a substitute for the human decision the TODO requires."""
    pooled = report["pooled"]
    g = pooled.get("gate")
    if g is None:
        return {"branch": "measurement_limited_result",
                "why": "no scored items"}
    if g["all_pass"]:
        return {"branch": "PASS", "why": "all gate metrics pass on the pooled scored population"}
    pm = g["per_metric"]
    prec_ok = pm["score3_precision"]["pass"]
    kappa = pm["four_class_kappa"]["value"]
    if prec_ok and not pm["score3_recall"]["pass"]:
        return {"branch": "narrow_score3_use",
                "why": "score-3 precision passes but recall/coverage does not"}
    if kappa is not None and kappa < 0.5:
        return {"branch": "measurement_limited_result",
                "why": f"four-class judge-human kappa {kappa} is low; the gold label is "
                       "itself noisy (compare the T06 human-human kappa)"}
    return {"branch": "human_assisted_narrow_study",
            "why": "gate not met; a human-backed narrow subset is the supported claim"}


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------

def _load_inputs(consensus_path, key_path):
    if not Path(consensus_path).exists():
        die(f"consensus not found: {consensus_path}")
    if not Path(key_path).exists():
        die(f"private key not found: {key_path} (never committed; supply it locally)")
    gold = {}
    with open(consensus_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gl, gv = r.get("gold_label"), r.get("gold_validity")
            if gl == "ADJUDICATION_REQUIRED" or gv == "ADJUDICATION_REQUIRED":
                die("consensus still has ADJUDICATION_REQUIRED rows; adjudicate T06 first")
            gold[r["item_id"]] = {"label": int(gl), "validity": gv}
    key = json.loads(Path(key_path).read_text("utf-8")).get("items", {})
    rows = []
    for item_id, g in gold.items():
        if item_id not in key:
            die(f"item {item_id} missing from the private key")
        k = key[item_id]
        rows.append({
            "human_label": g["label"], "human_validity": g["validity"],
            "lu_score": k.get("lu_score"), "ip_weight": float(k.get("ip_weight", 1.0)),
            "arm": k.get("arm"),
            "machine_validity": ("closed" if k.get("machine_closed_think") else "unclosed"),
        })
    return rows, gold


def _load_sensitivity(sensitivity_path, key_path, gold):
    """Sensitivity pairs: answer-only (primary, from the adjudicated consensus)
    vs full-response (from the sensitivity labeller file), joined by the private
    key's sensitivity_id -> item_id link."""
    if not sensitivity_path or not Path(sensitivity_path).exists():
        return []
    key = json.loads(Path(key_path).read_text("utf-8")).get("items", {})
    s_to_g = {v["sensitivity_id"]: iid for iid, v in key.items() if v.get("sensitivity_id")}
    pairs = []
    with open(sensitivity_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gid = s_to_g.get(r["item_id"])
            if gid is None or gid not in gold:
                continue
            if not (r.get("label_0_to_3") or "").strip():
                continue                    # sensitivity row not labelled — skip
            g = gold[gid]
            pairs.append({"ao_label": g["label"], "ao_validity": g["validity"],
                          "fr_label": int(r["label_0_to_3"]), "fr_validity": r["human_validity"]})
    return pairs


def cmd_run(a):
    rows, gold = _load_inputs(a.consensus, a.key)
    pairs = _load_sensitivity(a.sensitivity, a.key, gold)
    report = provenance.stamp_report(validate(rows, pairs), allow_dirty=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "judge_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("T07 judge validation ->", out / "judge_validation.json")
    print("  branch:", report["selected_branch"]["branch"])


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _synthetic(n=400, seed=0, judge_quality=0.9):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        arm = "translated" if i % 2 == 0 else "wrapper"
        h = int(rng.integers(0, 4))
        validity = "ok" if rng.random() > 0.08 else "unjudgeable"
        if rng.random() < 0.9:                 # judge scores most rows
            j = h if rng.random() < judge_quality else int(rng.integers(0, 4))
        else:
            j = None                            # abstention
        mv = "closed" if rng.random() > 0.1 else "unclosed"
        rows.append({"human_label": h, "human_validity": validity, "machine_validity": mv,
                     "lu_score": j, "ip_weight": float(rng.uniform(0.5, 4.0)), "arm": arm})
    return rows


def _synthetic_sensitivity(n=60, seed=3):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        a = int(rng.integers(0, 4))
        f = a if rng.random() > 0.2 else int(rng.integers(0, 4))   # trace sometimes flips it
        pairs.append({"ao_label": a, "fr_label": f, "ao_validity": "ok",
                      "fr_validity": "ok" if rng.random() > 0.1 else "unjudgeable"})
    return pairs


def cmd_self_test(a):
    good = validate(_synthetic(judge_quality=0.97, seed=1), _synthetic_sensitivity())
    poor = validate(_synthetic(judge_quality=0.30, seed=2))
    s = good["sensitivity_set"]
    checks = {
        "sensitivity_reported": s["n"] == 60 and "n_2_3_flips" in s,
        "unjudgeable_crosstab_present": isinstance(
            good["pooled"]["unjudgeable_x_machine_validity"], dict),
        "good_judge_high_kappa": good["pooled"]["scored_metrics"]["four_class_kappa"] > 0.8,
        "poor_judge_low_kappa": poor["pooled"]["scored_metrics"]["four_class_kappa"] < 0.4,
        "poor_judge_not_pass": not poor["pooled"]["gate"]["all_pass"],
        "populations_partition": (
            good["pooled"]["n_scored"] + good["pooled"]["n_abstained"]
            + good["pooled"]["n_unjudgeable"] == good["pooled"]["n_total"]),
        "abstention_reported": good["pooled"]["abstention_rate_over_human_ok"] is not None,
        "per_arm_present": set(good["by_arm"]) == {"translated", "wrapper"},
        "poor_branch_is_limited_or_narrow": poor["selected_branch"]["branch"] in {
            "measurement_limited_result", "human_assisted_narrow_study", "narrow_score3_use"},
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if not all(checks.values()):
        die("self-test failed")
    print("\nself-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="validate against consensus + private key")
    r.add_argument("--consensus", default=str(REPO_ROOT / "results" / "t06" / "gold_consensus.csv"))
    r.add_argument("--key", default=str(REPO_ROOT / "annotations" / "t14_gold_v3" / "sampling_key_PRIVATE.json"))
    r.add_argument("--out", default=str(REPO_ROOT / "results" / "t07"))
    r.add_argument("--sensitivity", default=str(
        REPO_ROOT / "annotations" / "t14_gold_v3" / "sensitivity_labeller_1.csv"),
        help="full-response sensitivity labels (joined to the consensus by sensitivity_id)")
    r.set_defaults(func=cmd_run)
    st = sub.add_parser("self-test", help="validate the scorer on synthetic data")
    st.set_defaults(func=cmd_self_test)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
