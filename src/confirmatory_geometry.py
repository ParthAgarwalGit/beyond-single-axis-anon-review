"""T15/T16 — confirmatory split-half Axis reliability (METHOD_FREEZE
``confirmatory_geometry``).

Build the C80-A and C80-B Assistant-related directions independently on the frozen
retained role set, cross-project each block's role means onto the OTHER block's unit
Axis, and test whether a role's position is reproducible across the two decoding-
matched halves. The primary statistic is the cross-axis role-projection Pearson r;
reliability is judged against three frozen null models and a role bootstrap, never
against a bare significance test.

This module is pure numpy on the small per-role means (275 x 4096 at most); the
heavy pooling is upstream (T12). Axes reuse ``src.geometry`` so the sign check and
formula are identical to the exploratory path.
"""
from __future__ import annotations

import numpy as np

from .geometry import assistant_axis, cross_projections, pearson_r

_BATCH = 500


def _cosine(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _rank(x):
    order = np.argsort(np.argsort(np.asarray(x, float)))
    return order.astype(float)


def _spearman(x, y):
    return pearson_r(_rank(x), _rank(y))


def _axis(mu_default, role_means, roles):
    return assistant_axis(mu_default, {r: role_means[r] for r in roles})


def confirmatory_reliability(mu_default_a, role_means_a, mu_default_b, role_means_b,
                             retained_roles, n_perm=5000, n_boot=2000, seed=0):
    """Cross-axis reliability on the frozen retained role set."""
    roles = sorted(set(retained_roles) & set(role_means_a) & set(role_means_b))
    if len(roles) < 3:
        raise ValueError(f"only {len(roles)} retained roles present in both blocks")
    rng = np.random.default_rng(seed)
    d = len(role_means_a[roles[0]])

    role_means_a = {r: np.asarray(role_means_a[r], float) for r in roles}
    role_means_b = {r: np.asarray(role_means_b[r], float) for r in roles}

    unit_a = _axis(mu_default_a, role_means_a, roles)
    unit_b = _axis(mu_default_b, role_means_b, roles)
    common, s_a, s_b = cross_projections(role_means_a, role_means_b, unit_a, unit_b)
    r_primary = pearson_r(s_a, s_b)

    same_a = np.array([np.dot(role_means_a[r], unit_a) for r in common])
    same_b = np.array([np.dot(role_means_b[r], unit_b) for r in common])

    secondary = {
        "spearman": round(_spearman(s_a, s_b), 6),
        "cosine_vA_vB": round(_cosine(unit_a, unit_b), 6),
        "same_axis_projection_pearson": round(pearson_r(same_a, same_b), 6),
    }

    M_a = np.stack([role_means_a[r] for r in common])
    M_b = np.stack([role_means_b[r] for r in common])
    n_r = len(common)

    def _batched_corr(Fa, Fb):
        fa = Fa - Fa.mean(0)
        fb = Fb - Fb.mean(0)
        na = np.linalg.norm(fa, axis=0)
        nb = np.linalg.norm(fb, axis=0)
        out = np.zeros(Fa.shape[1])
        ok = (na > 0) & (nb > 0)
        out[ok] = (fa[:, ok] * fb[:, ok]).sum(0) / (na[ok] * nb[ok])
        return out

    # Null 1: destroy same-role correspondence while preserving marginals.
    perm = np.array([pearson_r(s_a, rng.permutation(s_b)) for _ in range(n_perm)])

    # Null 2: common sign flips on matched default-minus-role contrasts.
    C_a = np.asarray(mu_default_a, float) - M_a
    C_b = np.asarray(mu_default_b, float) - M_b
    flip = np.empty(n_perm)
    for i in range(0, n_perm, _BATCH):
        k = min(_BATCH, n_perm - i)
        sgn = rng.choice((-1.0, 1.0), size=(k, n_r))
        Va, Vb = sgn @ C_a, sgn @ C_b
        Va /= np.maximum(np.linalg.norm(Va, axis=1, keepdims=True), 1e-12)
        Vb /= np.maximum(np.linalg.norm(Vb, axis=1, keepdims=True), 1e-12)
        flip[i:i + k] = _batched_corr(M_a @ Vb.T, M_b @ Va.T)

    # Null 3: independent isotropic random directions in the frozen hidden size.
    iso = np.empty(n_perm)
    for i in range(0, n_perm, _BATCH):
        k = min(_BATCH, n_perm - i)
        Ua = rng.standard_normal((k, d)); Ua /= np.linalg.norm(Ua, axis=1, keepdims=True)
        Ub = rng.standard_normal((k, d)); Ub /= np.linalg.norm(Ub, axis=1, keepdims=True)
        iso[i:i + k] = _batched_corr(M_a @ Ub.T, M_b @ Ua.T)

    def _null(dist):
        n_ge = int(np.sum(dist >= r_primary))
        out = {
            "mean": float(dist.mean()),
            "p95": float(np.percentile(dist, 95)),
            "p_ge_observed": float(n_ge / len(dist)),
            "n_draws": int(len(dist)),
        }
        if n_ge == 0:
            out["reporting_note"] = (
                f"0 of {len(dist)} null draws reached or exceeded the observed statistic"
            )
        return out

    idx = rng.integers(0, len(common), (n_boot, len(common)))
    boot = np.array([
        pearson_r(s_a[j], s_b[j]) if np.std(s_a[j]) and np.std(s_b[j]) else 0.0
        for j in idx
    ])

    loo = []
    for i in range(len(common)):
        m = np.ones(len(common), bool); m[i] = False
        loo.append(pearson_r(s_a[m], s_b[m]))

    return {
        "primary_statistic": "cross_axis_role_projection_pearson_r",
        "cross_axis_pearson_r": round(r_primary, 6),
        "n_retained_roles": len(common),
        "secondary": secondary,
        "nulls": {
            "role_correspondence_permutation": _null(perm),
            "common_orientation_sign_flip": _null(flip),
            "isotropic_random_direction": _null(iso),
        },
        "bootstrap": {
            "replicates": n_boot,
            "ci95": [round(float(np.percentile(boot, 2.5)), 6),
                     round(float(np.percentile(boot, 97.5)), 6)],
        },
        "leave_one_role_out": {
            "min": round(float(np.min(loo)), 6),
            "max": round(float(np.max(loo)), 6),
        },
    }
