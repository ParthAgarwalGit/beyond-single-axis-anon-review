"""Serial-vs-parallel equivalence tests for T18 outer-fold scheduling."""

import numpy as np
import pytest

from src import t18_cached as cached
from src import t18_readouts as legacy
from src.t18_parallel import analyse_parallel_cached, run_families_cached_parallel
from tests.test_t18_readouts import (
    FAST_AXIS,
    FAST_FULL,
    FAST_LINEAR,
    FAST_NONLINEAR,
    one_dimensional_world,
    multidimensional_linear_world,
)

BUILDERS = {
    "axis_1d": FAST_AXIS,
    "pca_linear": FAST_LINEAR,
    "full_linear": FAST_FULL,
    "small_nonlinear": FAST_NONLINEAR,
}
FAMILIES = ("pca_linear", "full_linear", "small_nonlinear")


def _compare_selections(a, b):
    assert list(a) == list(b)
    for family in a:
        assert len(a[family]) == len(b[family])
        for x, y in zip(a[family], b[family]):
            assert x.keys() == y.keys()
            for key in x:
                if key == "inner_auroc":
                    assert np.isclose(x[key], y[key], rtol=0.0, atol=1e-10)
                else:
                    assert x[key] == y[key]


@pytest.mark.parametrize(
    "world", [one_dimensional_world, multidimensional_linear_world]
)
def test_parallel_folds_match_serial_predictions(world):
    rows = world()
    order = ("axis_1d", *FAMILIES)

    s_pred, s_fold, s_sel = cached.run_families_cached(
        rows, order, folds=(4, 3), inner_folds=(2, 2),
        seed=legacy.SPLIT_SEED, builders_by_family=BUILDERS,
    )
    p_pred, p_fold, p_sel = run_families_cached_parallel(
        rows, order, folds=(4, 3), inner_folds=(2, 2),
        seed=legacy.SPLIT_SEED, builders_by_family=BUILDERS, outer_workers=2,
    )

    for family in order:
        np.testing.assert_array_equal(s_fold[family], p_fold[family])
        np.testing.assert_array_equal(np.isnan(s_pred[family]), np.isnan(p_pred[family]))
        m = ~np.isnan(s_pred[family])
        np.testing.assert_allclose(
            s_pred[family][m], p_pred[family][m], rtol=1e-9, atol=1e-10
        )
    _compare_selections(s_sel, p_sel)


def test_parallel_analysis_matches_serial_verdict():
    rows = one_dimensional_world(seed=31)

    serial = cached.analyse_cached(
        rows, replicates=60, seed=legacy.SPLIT_SEED, families=FAMILIES,
        builders_by_family=BUILDERS, folds=(4, 3), inner_folds=(2, 2),
        permutations=2,
    )
    parallel = analyse_parallel_cached(
        rows, replicates=60, seed=legacy.SPLIT_SEED, families=FAMILIES,
        builders_by_family=BUILDERS, folds=(4, 3), inner_folds=(2, 2),
        permutations=2, outer_workers=2,
    )

    assert serial["verdict"] == parallel["verdict"]
    for family in ("axis_1d", *FAMILIES):
        for key in (
            "point", "ci_low", "ci_high", "replicates_used",
            "replicates_skipped", "n_scored_rows", "n_folds", "cluster",
            "estimand",
        ):
            a = serial["performance"][family][key]
            b = parallel["performance"][family][key]
            if isinstance(a, float):
                assert np.isclose(a, b, rtol=0.0, atol=1e-10)
            else:
                assert a == b
    _compare_selections(serial["selections"], parallel["selections"])
    assert parallel["compute_execution"]["outer_workers"] == 2


def test_one_worker_delegates_to_serial():
    rows = one_dimensional_world(seed=41)
    order = ("axis_1d", "pca_linear")
    builders = {"axis_1d": FAST_AXIS, "pca_linear": FAST_LINEAR}
    serial = cached.run_families_cached(
        rows, order, folds=(2, 2), inner_folds=(2, 2),
        seed=legacy.SPLIT_SEED, builders_by_family=builders,
    )
    one = run_families_cached_parallel(
        rows, order, folds=(2, 2), inner_folds=(2, 2),
        seed=legacy.SPLIT_SEED, builders_by_family=builders, outer_workers=1,
    )
    for family in order:
        np.testing.assert_allclose(serial[0][family], one[0][family], rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(serial[1][family], one[1][family])
    assert serial[2] == one[2]
