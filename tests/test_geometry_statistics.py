"""Smoke tests for canonical geometry, statistics, and provenance stamping."""

import numpy as np
import pytest

from src.geometry import assistant_axis, confirmatory_role_set, cross_projections, default_mean, pearson_r
from src.statistics import holm_adjust, outcome_rates
from src.provenance import stamp_report


def test_axis_orientation_and_sign_check():
    mu_default = np.array([2.0, 0.0])
    role_means = {"pirate": np.array([0.0, 0.0]), "alien": np.array([0.0, 2.0])}
    unit = assistant_axis(mu_default, role_means)
    assert np.isclose(np.linalg.norm(unit), 1.0)
    # Positive direction is toward the default Assistant.
    assert np.dot(mu_default, unit) > np.mean([np.dot(m, unit) for m in role_means.values()])

    with pytest.raises(ValueError, match="degenerate"):
        # A default mean equal to the grand role mean cannot define an axis.
        assistant_axis(np.array([0.0, 0.0]), {"r": np.array([0.0, 0.0])})


def test_default_mean_weights_conditions_equally():
    # Condition 0 has many outputs; equal 0.2 weighting must ignore that.
    conditions = [[[10.0]] * 100, [[0.0]], [[0.0]], [[0.0]], [[0.0]]]
    assert np.isclose(default_mean(conditions)[0], 2.0)


def test_confirmatory_eligibility_is_intersection():
    a = {"pirate": 12, "alien": 9, "scientist": 10}
    b = {"pirate": 10, "alien": 11, "scientist": 3}
    assert confirmatory_role_set(a, b, threshold=10) == {"pirate"}


def test_cross_projections_and_pearson():
    means_a = {"r1": np.array([1.0, 0.0]), "r2": np.array([0.0, 1.0])}
    means_b = {"r1": np.array([2.0, 0.0]), "r2": np.array([0.0, 2.0])}
    axis = np.array([1.0, 0.0])
    roles, s_a, s_b = cross_projections(means_a, means_b, axis, axis)
    assert roles == ["r1", "r2"]
    assert np.isclose(pearson_r(s_a, s_b), 1.0)


def test_outcome_rates_carry_counts_and_denominators():
    rates = outcome_rates(
        ["harmful", "partial", "refused", "refused"],
        ["assistant", "human_role", "weird_or_mystical_role", "assistant"],
    )
    assert rates["strict_harmful_compliance"] == 0.25
    assert rates["inclusive_harmful_compliance"] == 0.5
    assert rates["non_assistant_rate"] == 0.5
    assert rates["n_completed_nontechnical"] == 4
    assert rates["harm_counts"]["harmful"] == 1


def test_outcome_labels_must_be_paired():
    with pytest.raises(ValueError, match="unpaired"):
        outcome_rates(["harmful"], ["assistant", "assistant"])


def test_pearson_rejects_nonfinite_and_constant_input():
    with pytest.raises(ValueError, match="non-finite"):
        pearson_r([1.0, float("nan"), 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="constant"):
        pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


def test_holm_two_coprimaries():
    assert holm_adjust([0.01, 0.04]) == [0.02, 0.04]


def test_reports_are_stamped_with_source_sha_and_config_hash():
    from src.provenance import git_dirty
    report = stamp_report({"result": 1}, allow_dirty=True)
    assert len(report["source_git_sha"]) == 40
    assert len(report["config_sha256"]) == 64
    if report["source_git_dirty"]:
        assert len(report["working_tree_diff_sha256"]) == 64
    with pytest.raises(ValueError):
        stamp_report({"source_git_sha": "0" * 40}, allow_dirty=True)
    if git_dirty():
        with pytest.raises(RuntimeError, match="dirty"):
            stamp_report({"result": 1})
