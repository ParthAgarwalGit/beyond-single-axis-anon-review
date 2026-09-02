"""Tests for the T29-H design-aware human-primary harmfulness branch."""

import json
import math

import pytest

from src.t29_human_primary import human_primary_report, validate_design
from tools.run_t29_human_primary import (
    assert_aggregate_only,
    consensus_harmfulness,
    validate_t27,
    verify_t27_rate_regression,
)


def _strata():
    return {
        "unsteered_baseline|refused": {"population": 8, "sampled": 4, "ip_weight": 2.0},
        "unsteered_baseline|harmful": {"population": 2, "sampled": 2, "ip_weight": 1.0},
        "assistant_cap|refused": {"population": 8, "sampled": 4, "ip_weight": 2.0},
        "assistant_cap|harmful": {"population": 2, "sampled": 2, "ip_weight": 1.0},
        "random_direction_cap|refused": {"population": 8, "sampled": 4, "ip_weight": 2.0},
        "random_direction_cap|harmful": {"population": 2, "sampled": 2, "ip_weight": 1.0},
        "sphere_proximity_cap|refused": {"population": 8, "sampled": 4, "ip_weight": 2.0},
        "sphere_proximity_cap|harmful": {"population": 2, "sampled": 2, "ip_weight": 1.0},
    }


def _rows():
    rows = []
    labels = {
        "unsteered_baseline": (["refused"] * 4, ["harmful", "harmful"]),
        "assistant_cap": (["refused"] * 4, ["refused", "harmful"]),
        "random_direction_cap": (["refused"] * 4, ["harmful", "harmful"]),
        "sphere_proximity_cap": (["refused"] * 4, ["refused", "harmful"]),
    }
    for condition, (refused, rare) in labels.items():
        for lab in refused:
            rows.append({
                "condition": condition,
                "auto_harmfulness": "refused",
                "human_harmfulness": lab,
                "ip_weight": 2.0,
            })
        for lab in rare:
            rows.append({
                "condition": condition,
                "auto_harmfulness": "harmful",
                "human_harmfulness": lab,
                "ip_weight": 1.0,
            })
    return rows


def test_design_requires_exact_ip_weights():
    bad = _strata()
    bad["assistant_cap|refused"] = dict(bad["assistant_cap|refused"], ip_weight=3.0)
    with pytest.raises(ValueError, match="ip_weight"):
        validate_design(bad)


def test_human_primary_weighted_rates_and_primary_contrast():
    report = human_primary_report(_rows(), _strata(), n_boot=200, seed=123)
    strict = report["outcomes"]["strict"]["condition_rates"]
    assert math.isclose(strict["unsteered_baseline"]["rate"], 0.2)
    assert math.isclose(strict["assistant_cap"]["rate"], 0.1)
    assert math.isclose(strict["random_direction_cap"]["rate"], 0.2)
    assert math.isclose(strict["sphere_proximity_cap"]["rate"], 0.1)
    primary = report["primary_specificity"]
    assert math.isclose(primary["risk_difference"], -0.1)
    assert primary["paired"] is False


def test_fully_enumerated_rare_strata_do_not_add_bootstrap_sampling_noise():
    report = human_primary_report(_rows(), _strata(), n_boot=300, seed=7)
    # Assistant and sphere have identical observed strata, so their bootstrap difference is exactly zero.
    comp = report["outcomes"]["strict"]["comparisons"]["axis_vs_sphere"]
    assert comp["bootstrap_95ci"] == [0.0, 0.0]


def test_joined_rows_must_match_frozen_stratum_counts():
    rows = _rows()[:-1]
    with pytest.raises(ValueError, match="does not match frozen stratum counts"):
        human_primary_report(rows, _strata(), n_boot=200, seed=1)


def test_consensus_requires_adjudication_for_disagreement():
    with pytest.raises(SystemExit, match="remain unresolved"):
        consensus_harmfulness({"C1": "harmful"}, {"C1": "refused"}, {})
    assert consensus_harmfulness(
        {"C1": "harmful"}, {"C1": "refused"}, {"C1": "partial"}
    ) == {"C1": "partial"}


def _t27(tmp_path, branch="MEASUREMENT_LIMITED", primary="human_subset"):
    doc = {
        "task": "T27_causal_judge_validation",
        "branch": "qwen_capping",
        "gating": True,
        "n_items_in_key": 120,
        "n_items_labelled_by_both": 120,
        "n_unresolved_after_adjudication": 0,
        "t26_sample_manifest_sha256": "a" * 64,
        "resolved_per_condition": {
            "unsteered_baseline": 30,
            "assistant_cap": 30,
            "random_direction_cap": 30,
            "sphere_proximity_cap": 30,
        },
        "outcomes": {
            "harmfulness": {
                "part_b_non_differential_error": {
                    "per_condition": {
                        c: {"human_positive_rate_ip": 0.1}
                        for c in (
                            "unsteered_baseline", "assistant_cap",
                            "random_direction_cap", "sphere_proximity_cap"
                        )
                    }
                },
                "gate": {
                    "branch": branch,
                    "primary_evidence": primary,
                    "primary_evidence_is_full_corpus": False,
                    "full_corpus_reportable_as": "full_corpus_automatic",
                },
            }
        },
    }
    path = tmp_path / "t27.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return doc, path


def test_t29_h_requires_measurement_limited_human_primary_branch(tmp_path):
    doc, path = _t27(tmp_path, branch="VALIDATED", primary="full_corpus_automatic")
    with pytest.raises(SystemExit, match="MEASUREMENT_LIMITED"):
        validate_t27(doc, path, "a" * 64)


def test_t29_h_t27_binding_accepts_frozen_branch(tmp_path):
    doc, path = _t27(tmp_path)
    bound = validate_t27(doc, path, "a" * 64)
    assert bound["harmfulness_branch"] == "MEASUREMENT_LIMITED"
    assert bound["primary_evidence"] == "human_subset"


def test_regression_check_fails_when_human_rates_do_not_match_t27():
    report = human_primary_report(_rows(), _strata(), n_boot=200, seed=3)
    frozen = {
        "unsteered_baseline": 0.2,
        "assistant_cap": 0.1,
        "random_direction_cap": 0.2,
        "sphere_proximity_cap": 0.9,
    }
    with pytest.raises(SystemExit, match="does not reproduce T27"):
        verify_t27_rate_regression(report, frozen)


def test_aggregate_privacy_guard_rejects_item_level_keys():
    assert_aggregate_only({"condition_rates": {"assistant_cap": 0.1}})
    with pytest.raises(SystemExit, match="forbidden private/item-level key"):
        assert_aggregate_only({"items": [{"condition": "assistant_cap"}]})
