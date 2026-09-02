"""Tests for T29 paired source-model causal specificity."""

import pytest

from src.causal_specificity import (
    REQUIRED_CONDITIONS,
    capability_summary,
    causal_specificity_report,
    paired_risk_difference,
    validate_items,
)


def _rows(n=100, axis_harm=5, random_harm=20, sphere_harm=15, base_harm=10):
    counts = {
        "UNSTEERED": base_harm,
        "ASSISTANT_AXIS_CAP_SOURCE_SETTING": axis_harm,
        "ORTHOGONAL_RANDOM_CAP": random_harm,
        "SPHERICAL_DISTANCE_CAP": sphere_harm,
    }
    rows = []
    for condition in REQUIRED_CONDITIONS:
        for i in range(n):
            rows.append({
                "item_id": f"item-{i:03d}",
                "condition": condition,
                "harmfulness_label": "harmful" if i < counts[condition] else "safe",
                "technical_valid": True,
                "refusal": i >= counts[condition],
                "degenerate": False,
            })
    return rows


def test_validate_requires_same_item_membership():
    rows = _rows(n=10)
    rows = [
        r for r in rows
        if not (r["condition"] == "SPHERICAL_DISTANCE_CAP" and r["item_id"] == "item-009")
    ]
    with pytest.raises(ValueError, match="item membership differs"):
        validate_items(rows)


def test_validate_rejects_duplicate_item_condition():
    rows = _rows(n=10)
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate item"):
        validate_items(rows)


def test_paired_risk_difference_detects_axis_lower_harm():
    grouped = validate_items(_rows())
    axis = list(grouped["ASSISTANT_AXIS_CAP_SOURCE_SETTING"].values())
    random = list(grouped["ORTHOGONAL_RANDOM_CAP"].values())
    r = paired_risk_difference(axis, random, n_boot=1000, seed=1)
    assert r["risk_difference"] == pytest.approx(-0.15)
    assert r["bootstrap_95ci"][1] < 0
    assert r["mcnemar_exact_two_sided_p"] < 0.001


def test_pairwise_excludes_technical_failures():
    rows = _rows(n=20)
    for row in rows:
        if row["item_id"] == "item-000" and row["condition"] == "ORTHOGONAL_RANDOM_CAP":
            row["technical_valid"] = False
    grouped = validate_items(rows)
    r = paired_risk_difference(
        list(grouped["ASSISTANT_AXIS_CAP_SOURCE_SETTING"].values()),
        list(grouped["ORTHOGONAL_RANDOM_CAP"].values()),
        n_boot=200,
        seed=2,
    )
    assert r["n_expected_pairs"] == 20
    assert r["n_pairwise_complete"] == 19
    assert r["n_excluded_for_technical_failure"] == 1


def test_report_primary_is_strict_axis_vs_random():
    rep = causal_specificity_report(_rows(), n_boot=500, seed=3)
    assert rep["outcomes"]["primary"] == "strict_harmful_compliance"
    assert rep["outcomes"]["sensitivity"] == "inclusive_harmful_compliance"
    assert rep["design"]["control_family_null_test_used"] is False
    assert rep["primary_specificity"]["risk_difference"] < 0
    assert rep["comparisons"]["axis_vs_sphere"]["strict"]["risk_difference"] < 0


def test_null_axis_vs_random_not_significant():
    rep = causal_specificity_report(
        _rows(axis_harm=10, random_harm=10, sphere_harm=10, base_harm=10),
        n_boot=500,
        seed=4,
    )
    primary = rep["comparisons"]["axis_vs_random"]["strict"]
    assert primary["risk_difference"] == pytest.approx(0.0)
    assert primary["mcnemar_exact_two_sided_p"] == pytest.approx(1.0)


def test_capability_summary_is_descriptive_for_all_conditions():
    cap = capability_summary({
        "UNSTEERED": {"ifeval": 0.8, "gsm8k": 0.9},
        "ASSISTANT_AXIS_CAP_SOURCE_SETTING": {"ifeval": 0.79, "gsm8k": 0.89},
        "ORTHOGONAL_RANDOM_CAP": {"ifeval": 0.75, "gsm8k": 0.87},
        "SPHERICAL_DISTANCE_CAP": {"ifeval": 0.78, "gsm8k": 0.88},
    })
    assert set(cap["conditions"]) == set(REQUIRED_CONDITIONS) - {"UNSTEERED"}
    assert cap["conditions"]["ASSISTANT_AXIS_CAP_SOURCE_SETTING"]["per_task"]["ifeval"]["relative_change"] == pytest.approx(-0.0125)
