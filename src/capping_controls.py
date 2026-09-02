"""Matched controls for the T24 source-model capping specificity design.

The source Assistant-Axis cap is kept fixed by T21. This module implements only
pre-outcome controls: an orthogonal random-direction lower-tail cap and a
spherical distance-to-default cap. Calibration always uses each control's own
natural activation distribution. It also provides the source upper-tail
engagement calculation needed to verify that the project controls are actually
comparable to the fixed T21 treatment on the same natural rows.
"""
from __future__ import annotations

import hashlib

import numpy as np


def unit_vector(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("direction must be a finite 1-D vector")
    n = float(np.linalg.norm(x))
    if n <= 0:
        raise ValueError("zero direction")
    return x / n


def layer_seed(base_seed: int, layer: int) -> int:
    """Stable per-layer seed independent of Python hash randomization."""
    payload = f"t24|{int(base_seed)}|layer={int(layer)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def orthogonal_random_direction(axis: np.ndarray, *, base_seed: int, layer: int) -> np.ndarray:
    """Draw N(0,I), remove the Axis component, then unit-normalize."""
    a = unit_vector(axis)
    rng = np.random.default_rng(layer_seed(base_seed, layer))
    g = rng.standard_normal(a.shape[0])
    u = g - float(g @ a) * a
    return unit_vector(u)


def directional_projection(x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    u = unit_vector(direction)
    if arr.shape[-1] != u.shape[0]:
        raise ValueError("activation hidden size does not match direction")
    return arr @ u


def source_upper_tail_engagement(
    natural_activations: np.ndarray, direction: np.ndarray, threshold: float
) -> np.ndarray:
    """Return the engagement mask for Lu's released source capping operator.

    The source implementation unit-normalizes the stored capping vector and
    engages only when ``projection > threshold``. T24 uses this function for
    measurement only; the T21 treatment itself is never recomputed or retuned.
    """
    if not np.isfinite(threshold):
        raise ValueError("source cap threshold must be finite")
    p = directional_projection(natural_activations, direction)
    if p.ndim != 1 or p.size == 0 or not np.isfinite(p).all():
        raise ValueError("natural calibration projections must be finite and non-empty")
    return p > float(threshold)


def calibrate_lower_tail_threshold(
    natural_activations: np.ndarray, direction: np.ndarray, *, quantile: float = 0.25
) -> float:
    """Calibrate a directional cap on its own natural projection distribution."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between 0 and 1")
    p = directional_projection(natural_activations, direction)
    if p.ndim != 1 or p.size == 0 or not np.isfinite(p).all():
        raise ValueError("natural calibration projections must be finite and non-empty")
    return float(np.quantile(p, quantile, method="linear"))


def lower_tail_cap(
    activations: np.ndarray, direction: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Clamp projection values below ``threshold`` up to the boundary.

    Returns ``(capped, engaged_mask)``. Points already at/above threshold are
    bitwise-equivalent up to the float64 conversion performed here.
    """
    x = np.asarray(activations, dtype=np.float64)
    u = unit_vector(direction)
    if x.shape[-1] != u.shape[0] or not np.isfinite(x).all():
        raise ValueError("invalid activation array")
    proj = x @ u
    delta = np.maximum(float(threshold) - proj, 0.0)
    out = x + delta[..., None] * u
    return out, delta > 0


def default_centroid(natural_activations: np.ndarray) -> np.ndarray:
    x = np.asarray(natural_activations, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or not np.isfinite(x).all():
        raise ValueError("natural activations must be finite [n, hidden]")
    return x.mean(axis=0)


def calibrate_sphere_radius(
    natural_activations: np.ndarray, center: np.ndarray, *, quantile: float = 0.75
) -> float:
    """Radius at the natural distance quantile; q=.75 targets outer 25%."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between 0 and 1")
    x = np.asarray(natural_activations, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    if x.ndim != 2 or c.shape != (x.shape[1],) or x.shape[0] == 0:
        raise ValueError("sphere calibration shape mismatch")
    d = np.linalg.norm(x - c, axis=1)
    if not np.isfinite(d).all():
        raise ValueError("non-finite sphere distances")
    return float(np.quantile(d, quantile, method="linear"))


def spherical_distance_cap(
    activations: np.ndarray, center: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project points outside the radius radially to the sphere boundary."""
    x = np.asarray(activations, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    if x.shape[-1] != c.shape[0] or not np.isfinite(x).all() or not np.isfinite(c).all():
        raise ValueError("invalid activation/center")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    diff = x - c
    dist = np.linalg.norm(diff, axis=-1)
    engaged = dist > radius
    out = x.copy()
    if np.any(engaged):
        scale = radius / dist[engaged]
        out[engaged] = c + diff[engaged] * scale[..., None]
    return out, engaged


def engagement_rate(mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        raise ValueError("empty engagement mask")
    return float(m.mean())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(unit_vector(a) @ unit_vector(b))
