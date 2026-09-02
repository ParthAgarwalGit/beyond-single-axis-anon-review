"""Canonical steering conditions and intervention math (METHOD_FREEZE §13).

Five unique causal conditions are generated: one SHARED zero, Assistant Axis
toward/away, random positive/negative. The zero control belongs to both dose
families, so naive per-family grids produce two zero rows; they must collapse
to the single shared condition before generation.

Condition names are taken from the frozen YAML (``steering.unique_conditions``),
never redefined here.
"""

import numpy as np

from .config import load_frozen_config

_CFG, _ = load_frozen_config()

CAUSAL_CONDITIONS = _CFG["steering"]["unique_conditions"]

SHARED_ZERO = "shared_zero"
AXIS_TOWARD = "assistant_axis_toward"
AXIS_AWAY = "assistant_axis_away"
RANDOM_POSITIVE = "random_positive"
RANDOM_NEGATIVE = "random_negative"
assert tuple(CAUSAL_CONDITIONS) == (
    SHARED_ZERO, AXIS_TOWARD, AXIS_AWAY, RANDOM_POSITIVE, RANDOM_NEGATIVE
), "code drifted from the frozen steering condition names"


def build_causal_conditions(alpha):
    """The five unique conditions with their signed coefficients.

    Both the Axis family and the random family contain a zero point; the
    duplicates collapse to one shared zero so the design has exactly five
    unique generation conditions and the shared zero is generated once.
    """
    if alpha <= 0:
        raise ValueError("nonzero coefficient magnitude must be positive")
    families = {
        "axis": [-alpha, 0.0, alpha],
        "random": [-alpha, 0.0, alpha],
    }
    conditions = {}
    for family, coefficients in families.items():
        for coefficient in coefficients:
            if coefficient == 0.0:
                conditions[SHARED_ZERO] = {"direction": None, "alpha": 0.0}
            elif family == "axis":
                name = AXIS_TOWARD if coefficient > 0 else AXIS_AWAY
                conditions[name] = {"direction": "axis", "alpha": coefficient}
            else:
                name = RANDOM_POSITIVE if coefficient > 0 else RANDOM_NEGATIVE
                conditions[name] = {"direction": "random", "alpha": coefficient}
    return conditions


def dose_columns(condition, alpha):
    """Signed slope encoding: (axis_dose, random_dose).

    The shared zero is 0 in both columns; it is never duplicated per family.
    """
    return {
        SHARED_ZERO: (0.0, 0.0),
        AXIS_TOWARD: (alpha, 0.0),
        AXIS_AWAY: (-alpha, 0.0),
        RANDOM_POSITIVE: (0.0, alpha),
        RANDOM_NEGATIVE: (0.0, -alpha),
    }[condition]


def apply_steering(h, alpha, mean_residual_norm, unit_direction):
    """``h' = h + alpha * mean_residual_norm * unit_direction``.

    Applied at block-16 output on generated response-token positions only;
    prompt-prefill positions are never modified (enforced by the hook, not
    here).
    """
    h = np.asarray(h, dtype=np.float64)
    return h + alpha * mean_residual_norm * np.asarray(unit_direction, dtype=np.float64)


def mean_residual_norm(residual_vectors):
    """Mean L2 norm over generated response content tokens from frozen,
    valid, unsteered default-calibration outputs (prompt/padding/EOS
    positions already excluded upstream)."""
    arr = np.asarray(residual_vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("expected a non-empty (n_tokens, hidden) array")
    return float(np.linalg.norm(arr, axis=1).mean())


def random_control_direction(cfg, unit_axis):
    """Frozen random control: seeded draw, orthogonalized to the unit Axis,
    normalized. Sampled once with the frozen seed; the result is verified
    against the frozen orthogonality tolerance before it is returned."""
    control = cfg["steering"]["random_control"]
    unit_axis = np.asarray(unit_axis, dtype=np.float64)
    if not np.isclose(np.linalg.norm(unit_axis), 1.0):
        raise ValueError("unit_axis is not unit-norm")

    rng = np.random.default_rng(control["seed"])
    g = rng.standard_normal(len(unit_axis))
    u = g - np.dot(g, unit_axis) * unit_axis
    unit_random = u / np.linalg.norm(u)

    if not np.isclose(np.linalg.norm(unit_random), 1.0):
        raise ValueError("random control failed normalization check")
    if abs(np.dot(unit_random, unit_axis)) > control["orthogonality_tolerance_absolute_cosine"]:
        raise ValueError("random control violates the frozen orthogonality tolerance")
    return unit_random
