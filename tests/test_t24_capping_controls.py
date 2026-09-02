import numpy as np

from src.capping_controls import (
    calibrate_lower_tail_threshold,
    calibrate_sphere_radius,
    cosine,
    default_centroid,
    engagement_rate,
    lower_tail_cap,
    orthogonal_random_direction,
    source_upper_tail_engagement,
    spherical_distance_cap,
)


def test_random_direction_is_unit_and_orthogonal():
    axis = np.arange(1, 65, dtype=np.float64)
    u = orthogonal_random_direction(axis, base_seed=20260818, layer=46)
    assert np.isclose(np.linalg.norm(u), 1.0, atol=1e-12)
    assert abs(cosine(axis, u)) < 1e-12


def test_random_direction_is_deterministic_per_layer():
    axis = np.arange(1, 65, dtype=np.float64)
    a = orthogonal_random_direction(axis, base_seed=20260818, layer=46)
    b = orthogonal_random_direction(axis, base_seed=20260818, layer=46)
    c = orthogonal_random_direction(axis, base_seed=20260818, layer=47)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_source_upper_tail_engagement_matches_source_capping_semantics():
    direction = np.array([2.0, 0.0])  # function must unit-normalize, as source does
    x = np.array([[0.49, 3.0], [0.50, -1.0], [0.51, 8.0], [2.0, 0.0]])
    engaged = source_upper_tail_engagement(x, direction, threshold=0.50)
    # Source capping uses excess=(proj-tau).clamp(min=0): equality is a no-op.
    assert engaged.tolist() == [False, False, True, True]
    assert np.isclose(engagement_rate(engaged), 0.5)


def test_lower_tail_cap_is_noop_above_threshold_and_clamps_below():
    direction = np.array([1.0, 0.0])
    x = np.array([[2.0, 3.0], [-1.0, 4.0], [0.5, -2.0]])
    out, engaged = lower_tail_cap(x, direction, threshold=0.5)
    assert np.array_equal(out[0], x[0])
    assert np.array_equal(out[2], x[2])
    assert engaged.tolist() == [False, True, False]
    assert np.isclose(out[1] @ direction, 0.5)
    assert np.isclose(out[1, 1], x[1, 1])


def test_random_threshold_targets_lower_quartile_on_continuous_data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20000, 16))
    u = rng.normal(size=16)
    threshold = calibrate_lower_tail_threshold(x, u, quantile=0.25)
    _, engaged = lower_tail_cap(x, u, threshold)
    assert abs(engagement_rate(engaged) - 0.25) < 0.01


def test_sphere_noop_and_radial_projection():
    center = np.array([1.0, -1.0])
    x = np.array([[1.5, -1.0], [4.0, -1.0], [1.0, -5.0]])
    out, engaged = spherical_distance_cap(x, center, radius=2.0)
    assert engaged.tolist() == [False, True, True]
    assert np.array_equal(out[0], x[0])
    d = np.linalg.norm(out[engaged] - center, axis=1)
    assert np.allclose(d, 2.0)
    # Direction from center is preserved for modified rows.
    before = x[engaged] - center
    after = out[engaged] - center
    assert np.allclose(before[:, 0] * after[:, 1], before[:, 1] * after[:, 0])


def test_sphere_radius_targets_outer_quartile_about_default_centroid():
    rng = np.random.default_rng(11)
    default = rng.normal(loc=1.0, scale=0.1, size=(1000, 12))
    natural = rng.normal(loc=0.0, scale=1.0, size=(20000, 12))
    center = default_centroid(default)
    radius = calibrate_sphere_radius(natural, center, quantile=0.75)
    _, engaged = spherical_distance_cap(natural, center, radius)
    assert abs(engagement_rate(engaged) - 0.25) < 0.01


def test_default_centroid_uses_default_reference_rows():
    default = np.array([[10.0, 0.0], [12.0, 2.0]])
    assert np.allclose(default_centroid(default), np.array([11.0, 1.0]))
