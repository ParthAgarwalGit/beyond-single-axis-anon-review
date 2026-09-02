"""T17 C160 merge, PCA, and response-region geometry tests."""

import numpy as np
import pytest

from src.t17_geometry import (
    REGIONS,
    compare_regions,
    default_vector,
    role_space_pca,
    weighted_merge_matched,
    weighted_merge_single,
)


def test_weighted_merge_single_equals_row_union_mean():
    a = {"r": np.array([1.0, 3.0])}
    b = {"r": np.array([5.0, 7.0])}
    means, counts = weighted_merge_single(a, {"r": 2}, b, {"r": 3})
    assert counts == {"r": 5}
    assert np.allclose(means["r"], (2 * a["r"] + 3 * b["r"]) / 5)


def test_weighted_merge_matched_preserves_shared_cohort_counts():
    ma = {"r": {p: np.array([1.0]) for p in REGIONS}}
    mb = {"r": {p: np.array([4.0]) for p in REGIONS}}
    out, counts = weighted_merge_matched(ma, {"r": 2}, mb, {"r": 1}, REGIONS)
    assert counts["r"] == 3
    assert all(np.allclose(out["r"][p], [2.0]) for p in REGIONS)


def test_weighted_merge_matched_is_true_c160_union_when_one_block_has_no_rows():
    ma = {"a": {p: np.array([1.0]) for p in REGIONS}}
    mb = {"b": {p: np.array([2.0]) for p in REGIONS}}
    out, counts = weighted_merge_matched(ma, {"a": 3}, mb, {"b": 4}, REGIONS)
    assert set(out) == {"a", "b"}
    assert counts == {"a": 3, "b": 4}
    assert all(np.allclose(out["a"][p], [1.0]) for p in REGIONS)
    assert all(np.allclose(out["b"][p], [2.0]) for p in REGIONS)


def test_weighted_merge_can_require_support_in_both_blocks_when_estimand_needs_it():
    ma = {"a": {p: np.array([1.0]) for p in REGIONS}}
    mb = {"b": {p: np.array([1.0]) for p in REGIONS}}
    with pytest.raises(ValueError, match="role sets differ"):
        weighted_merge_matched(ma, {"a": 1}, mb, {"b": 1}, REGIONS,
                               require_both_blocks=True)


def test_default_vector_equal_weights_conditions_after_floor():
    means = {str(i): np.array([float(i)]) for i in range(5)}
    counts = {str(i): {"total": 100, "eligible": 100} for i in range(5)}
    assert np.allclose(default_vector(means, counts), [2.0])
    counts["0"] = {"total": 100, "eligible": 94}
    with pytest.raises(ValueError, match="below frozen 0.95"):
        default_vector(means, counts)


def _low_rank_roles(n=40, d=12, seed=4):
    rng = np.random.default_rng(seed)
    e1 = np.zeros(d); e1[0] = 1.0
    e2 = np.zeros(d); e2[1] = 1.0
    coords = np.linspace(-4, 4, n)
    roles = {
        f"r{i:02d}": coords[i] * e1 + 0.15 * rng.normal(size=d) + 0.3 * np.sin(i) * e2
        for i in range(n)
    }
    default = 7.0 * e1
    return roles, default


def test_role_space_pca_is_centered_unstandardized_and_axis_independent():
    roles, default = _low_rank_roles()
    summary, curve, axis, pc1 = role_space_pca(roles, default)
    assert summary["n_roles"] == 40
    assert summary["centering"] == "center role means across roles"
    assert summary["standardization"] == "none"
    assert summary["pc1_variance_explained"] > 0.7
    assert summary["components_to_70pct"] == 1
    assert 0.0 <= summary["axis_pc1_alignment_abs"] <= 1.0
    assert curve[-1]["cumulative_explained_variance"] == pytest.approx(1.0)
    assert np.linalg.norm(axis) == pytest.approx(1.0)
    assert np.linalg.norm(pc1) == pytest.approx(1.0)


def test_compare_regions_reports_all_three_pairs_and_role_correlations():
    roles, default = _low_rank_roles()
    rng = np.random.default_rng(9)
    region_means = {
        "all_response": roles,
        "answer": {r: v + 0.01 * rng.normal(size=v.shape) for r, v in roles.items()},
        "reasoning": {r: 0.8 * v + 0.01 * rng.normal(size=v.shape) for r, v in roles.items()},
    }
    defaults = {p: default.copy() for p in REGIONS}
    pairs, rows, axes = compare_regions(region_means, defaults)
    assert len(pairs) == 3
    assert len(rows) == 40
    assert set(axes) == set(REGIONS)
    assert all(p["n_roles"] == 40 for p in pairs)
    assert all(p["axis_cosine"] > 0.95 for p in pairs)
    assert all(p["same_role_all_response_reference_projection_pearson"] > 0.95
               for p in pairs)
