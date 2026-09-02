"""T15/T16 confirmatory-geometry unit tests."""
import numpy as np
import pytest

from src.confirmatory_geometry import confirmatory_reliability, _spearman
from src.geometry import confirmatory_role_set
from tools.run_t15_confirmatory import _frozen_params, _require_immutable_hf_revision


def _planted(d=128, n=18, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(d); axis /= np.linalg.norm(axis)
    coord = rng.uniform(-3, 3, n)
    a, b, ca, cb = {}, {}, {}, {}
    for i in range(n):
        name = f"r{i:02d}"
        base = coord[i] * axis
        a[name] = base + noise * rng.standard_normal(d)
        b[name] = base + noise * rng.standard_normal(d)
        ca[name] = cb[name] = 12
    mu = 6.0 * axis + 0.1 * rng.standard_normal(d)
    return mu, a, b, ca, cb


def test_frozen_params_load_active_v4():
    p = _frozen_params()
    assert p["threshold"] == 10
    assert p["block_index"] == 16
    assert p["arm"] == "USER_TRANSLATED_LU"
    assert p["pool"] == "ALL_RESPONSE_TOKENS"
    assert p["hidden_size"] == 4096
    assert p["default_valid_floor"] == pytest.approx(0.95)


def test_require_immutable_hf_revision_accepts_only_40_hex():
    sha = "e1885e0865d1e6d7a39a9c2585b224b30e28b3f1"
    assert _require_immutable_hf_revision(sha) == sha
    assert _require_immutable_hf_revision(sha.upper()) == sha
    for bad in (None, "", "main", "e1885e0", "g" * 40, sha + "0"):
        with pytest.raises(ValueError, match="40-hex"):
            _require_immutable_hf_revision(bad)


def test_reliable_geometry_recovered():
    mu, a, b, ca, cb = _planted()
    retained = confirmatory_role_set(ca, cb, 10)
    res = confirmatory_reliability(mu, a, mu, b, retained, n_perm=800, n_boot=400, seed=1)
    assert res["cross_axis_pearson_r"] > 0.9
    assert res["secondary"]["spearman"] > 0.9
    assert res["nulls"]["role_correspondence_permutation"]["p_ge_observed"] < 0.02
    assert res["nulls"]["isotropic_random_direction"]["p_ge_observed"] < 0.02
    assert res["nulls"]["common_orientation_sign_flip"]["p_ge_observed"] < 0.2
    lo, hi = res["bootstrap"]["ci95"]
    assert lo < res["cross_axis_pearson_r"] <= hi + 1e-9
    assert res["n_retained_roles"] == 18


def test_scrambled_block_collapses_correlation():
    mu, a, b, ca, cb = _planted(seed=3)
    names = sorted(b)
    order = np.random.default_rng(11).permutation(len(names))
    scrambled = {names[i]: b[names[order[i]]] for i in range(len(names))}
    retained = confirmatory_role_set(ca, cb, 10)
    res = confirmatory_reliability(mu, a, mu, scrambled, retained, n_perm=400, n_boot=200, seed=4)
    assert abs(res["cross_axis_pearson_r"]) < 0.5
    assert res["nulls"]["role_correspondence_permutation"]["p_ge_observed"] > 0.02


def test_threshold_intersection_controls_retained_set():
    mu, a, b, ca, cb = _planted(n=12, seed=5)
    cb["r00"] = 4
    cb["r01"] = 9
    retained = confirmatory_role_set(ca, cb, 10)
    assert "r00" not in retained and "r01" not in retained
    res = confirmatory_reliability(mu, a, mu, b, retained, n_perm=200, n_boot=200, seed=6)
    assert res["n_retained_roles"] == 10


def test_too_few_roles_raises():
    mu, a, b, _, _ = _planted(n=6)
    with pytest.raises(ValueError):
        confirmatory_reliability(mu, a, mu, b, {"r00", "r01"}, n_perm=10, n_boot=10)


def test_spearman_monotone_invariance():
    x = np.array([1.0, 2, 3, 4, 5])
    assert _spearman(x, np.exp(x)) == pytest.approx(1.0)
    assert _spearman(x, -x) == pytest.approx(-1.0)
