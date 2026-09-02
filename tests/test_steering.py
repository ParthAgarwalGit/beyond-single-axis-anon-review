"""Required T03 test: the duplicate zero control collapses correctly."""

import numpy as np
import pytest

from src.steering import (
    AXIS_AWAY, AXIS_TOWARD, CAUSAL_CONDITIONS, RANDOM_NEGATIVE,
    RANDOM_POSITIVE, SHARED_ZERO, apply_steering, build_causal_conditions,
    dose_columns, random_control_direction,
)


def test_zero_control_collapses_to_single_shared_condition():
    conditions = build_causal_conditions(alpha=8.0)
    # Both dose families contribute a zero; exactly five unique conditions remain.
    assert set(conditions) == set(CAUSAL_CONDITIONS)
    assert len(conditions) == 5
    assert conditions[SHARED_ZERO] == {"direction": None, "alpha": 0.0}
    zero_conditions = [c for c, spec in conditions.items() if spec["alpha"] == 0.0]
    assert zero_conditions == [SHARED_ZERO]


def test_dose_columns_encode_shared_zero_once():
    a = 8.0
    assert dose_columns(SHARED_ZERO, a) == (0.0, 0.0)
    assert dose_columns(AXIS_TOWARD, a) == (a, 0.0)
    assert dose_columns(AXIS_AWAY, a) == (-a, 0.0)
    assert dose_columns(RANDOM_POSITIVE, a) == (0.0, a)
    assert dose_columns(RANDOM_NEGATIVE, a) == (0.0, -a)


def test_apply_steering_formula():
    h = np.zeros(4)
    unit = np.array([1.0, 0.0, 0.0, 0.0])
    steered = apply_steering(h, alpha=2.0, mean_residual_norm=3.0, unit_direction=unit)
    assert np.allclose(steered, [6.0, 0.0, 0.0, 0.0])
    # Zero condition reproduces the baseline exactly.
    assert np.allclose(apply_steering(h, 0.0, 3.0, unit), h)


def test_condition_names_come_from_frozen_yaml(cfg):
    assert tuple(CAUSAL_CONDITIONS) == tuple(cfg["steering"]["unique_conditions"])
    assert AXIS_TOWARD == "assistant_axis_toward"
    assert AXIS_AWAY == "assistant_axis_away"


def test_random_control_is_seeded_and_orthogonal(cfg):
    unit_axis = np.zeros(64)
    unit_axis[0] = 1.0
    r1 = random_control_direction(cfg, unit_axis)
    r2 = random_control_direction(cfg, unit_axis)
    assert np.array_equal(r1, r2), "random control must be reproducible from the frozen seed"
    tolerance = cfg["steering"]["random_control"]["orthogonality_tolerance_absolute_cosine"]
    assert abs(np.dot(r1, unit_axis)) <= tolerance
    assert np.isclose(np.linalg.norm(r1), 1.0)
    with pytest.raises(ValueError, match="unit-norm"):
        random_control_direction(cfg, unit_axis * 3.0)


def test_nonpositive_alpha_rejected():
    with pytest.raises(ValueError):
        build_causal_conditions(alpha=0.0)
