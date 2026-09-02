"""Deterministic T22 P50 causal-role selection.

Consumes frozen exploratory E80 role means and a default-Assistant mean, computes
the predeclared Assistant-side score, excludes the literal Assistant role, and
selects exactly 50 roles with lexical tie-breaking.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable, Sequence

import numpy as np

P50_SIZE = 50
LITERAL_ASSISTANT_ROLE = "assistant"


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vector_sha256(v: np.ndarray) -> str:
    arr = np.asarray(v, dtype="<f8")
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def assistant_axis(mu_default: np.ndarray, role_means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (axis, unit_axis) using v = mu_default - mean_r(mu_role_r)."""
    rm = np.asarray(role_means, dtype=np.float64)
    md = np.asarray(mu_default, dtype=np.float64)
    if rm.ndim != 2 or rm.shape[0] < 2:
        raise ValueError("role_means must be [n_roles, hidden] with n_roles >= 2")
    if md.shape != (rm.shape[1],):
        raise ValueError("mu_default shape must match role hidden size")
    if not np.isfinite(rm).all() or not np.isfinite(md).all():
        raise ValueError("non-finite geometry input")
    axis = md - rm.mean(axis=0)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("degenerate Assistant Axis")
    unit = axis / norm
    if float(md @ unit) <= float(rm.mean(axis=0) @ unit):
        raise ValueError("Assistant Axis orientation check failed")
    return axis, unit


def assistant_side_scores(
    role_ids: Sequence[str], role_means: np.ndarray, unit_axis: np.ndarray
) -> dict[str, float]:
    """Frozen T22 score: (mu_role - mean_role) dot unit_axis."""
    ids = [str(x) for x in role_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate role_id in E80 geometry")
    rm = np.asarray(role_means, dtype=np.float64)
    ua = np.asarray(unit_axis, dtype=np.float64)
    if rm.ndim != 2 or rm.shape[0] != len(ids) or ua.shape != (rm.shape[1],):
        raise ValueError("role geometry shape mismatch")
    centered = rm - rm.mean(axis=0, keepdims=True)
    vals = centered @ ua
    return {rid: float(vals[i]) for i, rid in enumerate(ids)}


def select_p50(
    role_ids: Sequence[str],
    role_means: np.ndarray,
    mu_default: np.ndarray,
    *,
    literal_assistant_role_id: str = LITERAL_ASSISTANT_ROLE,
    p50_size: int = P50_SIZE,
) -> dict:
    """Select the Assistant-proximal causal roles under the frozen rule."""
    ids = [str(x) for x in role_ids]
    if p50_size <= 0:
        raise ValueError("p50_size must be positive")
    if literal_assistant_role_id not in ids:
        raise ValueError("literal Assistant role is absent; refusing to guess its ID")
    axis, unit = assistant_axis(mu_default, role_means)
    scores = assistant_side_scores(ids, role_means, unit)

    candidates = [rid for rid in ids if rid != literal_assistant_role_id]
    if len(candidates) < p50_size:
        raise ValueError(f"need at least {p50_size} non-Assistant eligible roles")
    ordered = sorted(candidates, key=lambda rid: (-scores[rid], rid))
    selected = ordered[:p50_size]
    boundary = {
        "selected_min_score": scores[selected[-1]],
        "first_excluded_role": ordered[p50_size] if len(ordered) > p50_size else None,
        "first_excluded_score": scores[ordered[p50_size]] if len(ordered) > p50_size else None,
    }
    return {
        "p50_size": p50_size,
        "literal_assistant_role_id": literal_assistant_role_id,
        "n_eligible_roles_including_assistant": len(ids),
        "n_candidates_after_assistant_exclusion": len(candidates),
        "axis_sha256_f64le": _vector_sha256(axis),
        "unit_axis_sha256_f64le": _vector_sha256(unit),
        "membership_sha256": _canonical_sha256(selected),
        "selected_roles": [
            {"rank": i + 1, "role_id": rid, "assistant_side_score": scores[rid]}
            for i, rid in enumerate(selected)
        ],
        "boundary": boundary,
    }


def reconcile_observed_roles(expected_roles: Iterable[str], observed_roles: Iterable[str]) -> dict:
    """Compare a T25 output role set to the frozen P50 membership."""
    expected = {str(x) for x in expected_roles}
    observed = {str(x) for x in observed_roles}
    return {
        "match": expected == observed,
        "expected_count": len(expected),
        "observed_count": len(observed),
        "missing": sorted(expected - observed),
        "unexpected": sorted(observed - expected),
    }
