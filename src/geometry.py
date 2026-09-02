"""Canonical role vectors, eligibility, and Assistant-Axis geometry
(METHOD_FREEZE §10–§11).

The Axis is ``v = mu_default − equal_weighted_mean_r(mu_role_r)``; positive
movement is toward the default Assistant. PCA is descriptive only and never
defines the Axis.

Output membership is supplied upstream. Historical/E80 analyses may use score-3
membership, while the primary C80 confirmatory path uses the label-independent
technical-validity rule authorized by the 2026-08-17 deviation record.
"""

import numpy as np


def role_mean(vectors):
    """Equal-weighted mean of a role's eligible response vectors."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("expected a non-empty (n_outputs, hidden) array")
    return arr.mean(axis=0)


def default_mean(condition_vectors, min_valid_fraction_ok=True):
    """Block default vector: mean within each of the five conditions first,
    then equal 0.2 weighting of the condition means, so no condition gains
    weight from having more valid outputs.
    """
    if not min_valid_fraction_ok:
        raise ValueError("a default condition fell below the frozen 95% valid floor")
    if len(condition_vectors) != 5:
        raise ValueError(f"expected 5 default conditions, got {len(condition_vectors)}")
    means = [role_mean(v) for v in condition_vectors]
    return np.mean(means, axis=0)


def eligible_roles(eligible_counts, threshold):
    """Roles with at least ``threshold`` outputs under the upstream membership rule."""
    return {r for r, n in eligible_counts.items() if n >= threshold}


def confirmatory_role_set(counts_a, counts_b, threshold):
    """Frozen split-half eligibility: intersection of roles meeting the
    threshold in both C80-A and C80-B. The same set builds both Axes.

    ``counts_a``/``counts_b`` are membership-agnostic; the caller is responsible
    for constructing them under the declared output-membership rule.
    """
    return eligible_roles(counts_a, threshold) & eligible_roles(counts_b, threshold)


def assistant_axis(mu_default, role_means):
    """Unit Assistant Axis with the frozen sign check.

    ``role_means`` maps role -> mean vector for the frozen eligible set.
    Raises if the sign check fails; sign is never chosen after causal
    results are known.
    """
    if not role_means:
        raise ValueError("no eligible roles")
    grand_role_mean = np.mean(list(role_means.values()), axis=0)
    v = np.asarray(mu_default, dtype=np.float64) - grand_role_mean
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("degenerate axis: default mean equals grand role mean")
    unit = v / norm

    default_proj = float(np.dot(mu_default, unit))
    mean_role_proj = float(np.mean([np.dot(m, unit) for m in role_means.values()]))
    if default_proj <= mean_role_proj:
        raise ValueError(
            "sign check failed: default projection does not exceed mean role projection"
        )
    return unit


def cross_projections(role_means_a, role_means_b, unit_axis_a, unit_axis_b):
    """Split-half cross-axis projections for the common role set.

    Returns (roles, s_A, s_B) where ``s_A[i] = mu_r,A · unit(v_B)`` and
    ``s_B[i] = mu_r,B · unit(v_A)``.
    """
    roles = sorted(set(role_means_a) & set(role_means_b))
    s_a = np.array([np.dot(role_means_a[r], unit_axis_b) for r in roles])
    s_b = np.array([np.dot(role_means_b[r], unit_axis_a) for r in roles])
    return roles, s_a, s_b


def pearson_r(x, y):
    """Primary reliability statistic between the two projection sets."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need two aligned samples of length >= 2")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("projections contain non-finite values")
    if np.std(x) == 0 or np.std(y) == 0:
        raise ValueError("constant projections: correlation undefined")
    return float(np.corrcoef(x, y)[0, 1])
