"""Reference-vs-cache equivalence tests for T18."""

import numpy as np
import pytest

from src import t18_readouts as legacy
from src.t18_cached import analyse_cached, run_family_cached
from tests.test_t18_readouts import (
    FAST_AXIS,
    FAST_FULL,
    FAST_LINEAR,
    FAST_NONLINEAR,
    multidimensional_linear_world,
    one_dimensional_world,
    radial_world,
)

FAST_BY_FAMILY = {
    "axis_1d": FAST_AXIS,
    "pca_linear": FAST_LINEAR,
    "full_linear": FAST_FULL,
    "small_nonlinear": FAST_NONLINEAR,
}


def _assert_selection_equivalent(old, new, atol=1e-9):
    assert len(old) == len(new)
    for a, b in zip(old, new):
        for key in (
            "fold_id", "role_fold", "question_fold", "n_train", "n_test",
            "n_dropped", "n_train_roles", "selected", "n_params",
        ):
            assert a[key] == b[key]
        assert np.isclose(a["inner_auroc"], b["inner_auroc"], rtol=0.0, atol=atol)


@pytest.mark.parametrize(
    "family", ["axis_1d", "pca_linear", "full_linear", "small_nonlinear"]
)
def test_cached_family_matches_reference_oof(family):
    rows = one_dimensional_world(seed=17)
    old_pred, old_fold, old_sel = legacy.run_family(
        rows,
        family,
        folds=(4, 3),
        inner_folds=(2, 2),
        seed=legacy.SPLIT_SEED,
        builders=FAST_BY_FAMILY[family],
    )
    new_pred, new_fold, new_sel = run_family_cached(
        rows,
        family,
        folds=(4, 3),
        inner_folds=(2, 2),
        seed=legacy.SPLIT_SEED,
        builders=FAST_BY_FAMILY[family],
    )
    np.testing.assert_array_equal(old_fold, new_fold)
    np.testing.assert_array_equal(np.isnan(old_pred), np.isnan(new_pred))
    scored = ~np.isnan(old_pred)
    np.testing.assert_allclose(old_pred[scored], new_pred[scored], rtol=1e-8, atol=1e-10)
    _assert_selection_equivalent(old_sel, new_sel)


def _reference_analyse(*args, **kwargs):
    from tools.run_t18_dimensional_adequacy import analyse
    return analyse(*args, **kwargs)


@pytest.mark.parametrize(
    "world",
    [
        pytest.param(one_dimensional_world, id="one_dimensional"),
        pytest.param(multidimensional_linear_world, id="multidimensional_linear"),
        pytest.param(radial_world, id="radial_nonlinear"),
    ],
)
def test_cached_analysis_matches_reference(world):
    rows = world()
    builders = {
        "axis_1d": FAST_AXIS,
        "pca_linear": FAST_LINEAR,
        "full_linear": FAST_FULL,
        "small_nonlinear": FAST_NONLINEAR,
    }
    families = ("pca_linear", "full_linear", "small_nonlinear")
    common = dict(
        replicates=80,
        seed=legacy.SPLIT_SEED,
        families=families,
        builders_by_family=builders,
        folds=(4, 3),
        inner_folds=(2, 2),
        permutations=2,
    )
    old = _reference_analyse(rows, **common)
    new = analyse_cached(rows, **common)

    assert old["verdict"]["verdict"] == new["verdict"]["verdict"]
    assert old["verdict"]["best_alternative"] == new["verdict"]["best_alternative"]

    for family in ("axis_1d", *families):
        np.testing.assert_allclose(
            old["performance"][family]["point"],
            new["performance"][family]["point"],
            rtol=0.0,
            atol=1e-9,
        )
        _assert_selection_equivalent(old["selections"][family], new["selections"][family])

    for key in ("max_family_difference", "ci_low", "ci_high"):
        np.testing.assert_allclose(
            old["simultaneous_vs_axis"][key],
            new["simultaneous_vs_axis"][key],
            rtol=0.0,
            atol=1e-9,
        )


def test_cache_reduces_svd_calls(monkeypatch):
    rows = one_dimensional_world(seed=23)
    calls = 0
    original = np.linalg.svd

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counted)
    builders = {
        "axis_1d": FAST_AXIS,
        "pca_linear": FAST_LINEAR,
        "full_linear": FAST_FULL,
        "small_nonlinear": FAST_NONLINEAR,
    }
    analyse_cached(
        rows,
        replicates=5,
        seed=legacy.SPLIT_SEED,
        families=("pca_linear", "full_linear", "small_nonlinear"),
        builders_by_family=builders,
        # Three outer role folds leave ~40 genuine training roles;
        # the nested two-fold role split then leaves ~20, enough for
        # the smallest frozen FAST nonlinear candidate (17 params).
        folds=(3, 2),
        inner_folds=(2, 2),
        permutations=0,
    )
    # 4 outer folds, <=4 valid inner splits each, plus one outer SVD per fold.
    assert 0 < calls <= 30
