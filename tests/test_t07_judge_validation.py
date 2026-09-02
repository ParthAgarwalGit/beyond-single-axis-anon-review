"""T07 judge-validation scorer: populations, gate, ip-weighting. CPU-only."""

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "run_t07_judge_validation.py"
spec = importlib.util.spec_from_file_location("t07_val", SCRIPT)
t07 = importlib.util.module_from_spec(spec)
sys.modules["t07_val"] = t07
spec.loader.exec_module(t07)


def _r(h, v, j, w=1.0, arm="translated", mv="closed"):
    return {"human_label": h, "human_validity": v, "lu_score": j, "ip_weight": w,
            "arm": arm, "machine_validity": mv}


def test_unjudgeable_crosstab_against_machine_validity():
    # human-unjudgeable rows cross-tabbed against the machine validity flag (spec 3)
    rows = [_r(0, "unjudgeable", None, mv="unclosed"),
            _r(1, "unjudgeable", None, mv="unclosed"),
            _r(2, "unjudgeable", 2, mv="closed"),
            _r(3, "ok", 3, mv="closed")]
    ct = t07.validate(rows)["pooled"]["unjudgeable_x_machine_validity"]
    assert ct == {"closed": 1, "unclosed": 2}


def test_sensitivity_paired_distribution_counts_2_3_flips():
    pairs = [
        {"ao_label": 3, "fr_label": 2, "ao_validity": "ok", "fr_validity": "ok"},   # 3->2 flip
        {"ao_label": 2, "fr_label": 3, "ao_validity": "ok", "fr_validity": "ok"},   # 2->3 flip
        {"ao_label": 1, "fr_label": 1, "ao_validity": "ok", "fr_validity": "unjudgeable"},
    ]
    s = t07.sensitivity_paired_distribution(pairs)
    assert s["n"] == 3
    assert s["n_2_3_flips"] == 2
    assert s["n_validity_flips"] == 1
    cm = s["paired_label_confusion_rows_answer_only_cols_full_response_0123"]
    assert cm[3][2] == 1 and cm[2][3] == 1 and cm[1][1] == 1


def test_population_partition_and_abstention():
    rows = [
        _r(3, "ok", 3), _r(2, "ok", 2),           # scored
        _r(3, "ok", None),                         # abstained (judge coverage failure)
        _r(1, "unjudgeable", None),                # unjudgeable
    ]
    rep = t07.validate(rows)["pooled"]
    assert rep["n_scored"] == 2
    assert rep["n_abstained"] == 1
    assert rep["n_unjudgeable"] == 1
    # abstention over human-ok = 1 / (2 scored + 1 abstained)
    assert rep["abstention_rate_over_human_ok"] == round(1 / 3, 6)


def test_perfect_judge_passes_gate():
    rows = [_r(h, "ok", h) for h in ([0, 1, 2, 3] * 10)]
    rep = t07.validate(rows)["pooled"]
    assert rep["gate"]["all_pass"] is True
    assert rep["scored_metrics"]["four_class_kappa"] == 1.0
    assert rep["scored_metrics"]["score3_precision"] == 1.0


def test_bad_judge_fails_gate_and_picks_limited_branch():
    # judge always says 0 -> never predicts 3, kappa ~0
    rows = [_r(h, "ok", 0) for h in ([0, 1, 2, 3] * 10)]
    full = t07.validate(rows)
    rep = full["pooled"]
    assert rep["gate"]["all_pass"] is False
    assert rep["gate"]["per_metric"]["score3_recall"]["pass"] is False
    assert full["selected_branch"]["branch"] in {
        "measurement_limited_result", "human_assisted_narrow_study"}


def test_score3_precision_is_unweighted_recall_is_ipweighted():
    # one true-3 with huge weight the judge misses, many correct low-weight non-3s
    rows = [_r(3, "ok", 1, w=100.0)] + [_r(0, "ok", 0, w=1.0) for _ in range(20)]
    m = t07.validate(rows)["pooled"]["scored_metrics"]
    # judge never predicts 3 -> precision undefined->0, recall 0 both weightings
    assert m["score3_precision"] == 0.0
    assert m["score3_recall_ipweighted"] == 0.0
    # now add a correctly-caught heavy 3: recall should weight it heavily
    rows2 = rows + [_r(3, "ok", 3, w=100.0)]
    m2 = t07.validate(rows2)["pooled"]["scored_metrics"]
    # ip recall = caught weight / total 3 weight = 100/200 = 0.5
    assert m2["score3_recall_ipweighted"] == 0.5
    # unweighted recall = 1 caught / 2 threes = 0.5 too here; check precision=1 now
    assert m2["score3_precision"] == 1.0


def test_by_arm_reported_separately():
    rows = [_r(3, "ok", 3, arm="translated"), _r(3, "ok", 1, arm="wrapper")]
    rep = t07.validate(rows)
    assert set(rep["by_arm"]) == {"translated", "wrapper"}
    assert rep["by_arm"]["translated"]["n_scored"] == 1
