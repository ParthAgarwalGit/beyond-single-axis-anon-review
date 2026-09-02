"""T17 C160 descriptive role-space geometry.

PCA is descriptive: role means are centered across roles with no standardization.
It never defines or reorients the Assistant Axis. PC1 sign is arbitrary; only the
absolute Axis-PC1 alignment is scientifically meaningful, while a display-aligned
PC1 may be emitted for plotting.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .geometry import assistant_axis, pearson_r

REGIONS = ("all_response", "answer", "reasoning")


def weighted_merge_single(means_a, counts_a, means_b, counts_b, *, require_same_keys=True):
    """Merge per-key means exactly as if the underlying rows had been concatenated."""
    ka, kb = set(means_a), set(means_b)
    if require_same_keys and ka != kb:
        missing_a, missing_b = sorted(kb - ka), sorted(ka - kb)
        raise ValueError(
            f"mean-key mismatch; only B={missing_a[:3]} only A={missing_b[:3]}"
        )
    out_means, out_counts = {}, {}
    for key in sorted(ka & kb if require_same_keys else ka | kb):
        na, nb = int(counts_a.get(key, 0)), int(counts_b.get(key, 0))
        if na < 0 or nb < 0 or na + nb == 0:
            raise ValueError(f"invalid counts for {key!r}: {na}/{nb}")
        if na and key not in means_a:
            raise ValueError(f"{key!r}: positive A count without A mean")
        if nb and key not in means_b:
            raise ValueError(f"{key!r}: positive B count without B mean")
        if na and nb:
            a = np.asarray(means_a[key], dtype=np.float64)
            b = np.asarray(means_b[key], dtype=np.float64)
            if a.shape != b.shape:
                raise ValueError(f"{key!r}: shape mismatch {a.shape}/{b.shape}")
            out_means[key] = (na * a + nb * b) / (na + nb)
        elif na:
            out_means[key] = np.asarray(means_a[key], dtype=np.float64).copy()
        else:
            out_means[key] = np.asarray(means_b[key], dtype=np.float64).copy()
        out_counts[key] = na + nb
    return out_means, out_counts


def weighted_merge_matched(means_a, counts_a, means_b, counts_b, pools,
                           *, require_both_blocks=False):
    """Merge matched-cohort means exactly as a C160 row union.

    A role may contribute matched rows from only one C80 block; that is still a
    valid C160 matched cohort. Set ``require_both_blocks=True`` only for analyses
    whose estimand explicitly requires within-block support in both halves.
    """
    ka, kb = set(means_a), set(means_b)
    if require_both_blocks and ka != kb:
        raise ValueError("C80-A/B matched role sets differ")
    roles = ka & kb if require_both_blocks else ka | kb
    out, out_counts = {}, {}
    for role in sorted(roles):
        na, nb = int(counts_a.get(role, 0)), int(counts_b.get(role, 0))
        if na < 0 or nb < 0 or na + nb == 0:
            raise ValueError(f"{role}: invalid matched counts {na}/{nb}")
        if na and role not in means_a:
            raise ValueError(f"{role}: positive A matched count without mean")
        if nb and role not in means_b:
            raise ValueError(f"{role}: positive B matched count without mean")
        out[role] = {}
        for pool in pools:
            if na and nb:
                a = np.asarray(means_a[role][pool], dtype=np.float64)
                b = np.asarray(means_b[role][pool], dtype=np.float64)
                if a.shape != b.shape:
                    raise ValueError(f"{role}/{pool}: shape mismatch {a.shape}/{b.shape}")
                out[role][pool] = (na * a + nb * b) / (na + nb)
            elif na:
                out[role][pool] = np.asarray(means_a[role][pool], dtype=np.float64).copy()
            else:
                out[role][pool] = np.asarray(means_b[role][pool], dtype=np.float64).copy()
        out_counts[role] = na + nb
    return out, out_counts


def default_vector(condition_means, condition_counts, floor=0.95):
    """Equal-weight five condition means after enforcing pool-specific coverage."""
    if len(condition_means) != 5:
        raise ValueError(f"expected five default conditions, got {sorted(condition_means)}")
    if set(condition_means) != set(condition_counts):
        raise ValueError("default condition means/counts use different condition keys")
    ordered = sorted(condition_means, key=lambda x: int(x) if str(x).isdigit() else str(x))
    vals = []
    for cond in ordered:
        c = condition_counts[cond]
        total = int(c.get("total", 0))
        eligible = int(c.get("eligible", c.get("valid", 0)))
        frac = eligible / total if total else 0.0
        if frac < floor:
            raise ValueError(
                f"default condition {cond!r}: pool-eligible fraction {frac:.4f} "
                f"below frozen {floor} ({eligible}/{total})"
            )
        vals.append(np.asarray(condition_means[cond], dtype=np.float64))
    return np.mean(vals, axis=0)


def role_space_pca(role_means, default_vec, variance_target=0.70):
    """Centered, unstandardized PCA over role means plus Axis-PC1 alignment."""
    roles = sorted(role_means)
    if len(roles) < 3:
        raise ValueError("role-space PCA needs at least three roles")
    M = np.stack([np.asarray(role_means[r], dtype=np.float64) for r in roles])
    if not np.isfinite(M).all():
        raise ValueError("role means contain non-finite values")
    centered = M - M.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    eig = s ** 2
    total = float(eig.sum())
    if total <= 0:
        raise ValueError("degenerate role-space PCA: zero total variance")
    ratio = eig / total
    cumulative = np.cumsum(ratio)
    k = int(np.searchsorted(cumulative, variance_target, side="left") + 1)

    unit_axis = assistant_axis(np.asarray(default_vec, dtype=np.float64),
                               {r: M[i] for i, r in enumerate(roles)})
    raw_pc1 = vt[0]
    raw_cos = float(np.dot(unit_axis, raw_pc1) /
                    (np.linalg.norm(unit_axis) * np.linalg.norm(raw_pc1)))
    display_pc1 = raw_pc1 if raw_cos >= 0 else -raw_pc1
    aligned_cos = abs(raw_cos)

    curve = [
        {
            "component": i + 1,
            "explained_variance_ratio": float(ratio[i]),
            "cumulative_explained_variance": float(cumulative[i]),
        }
        for i in range(len(ratio))
    ]
    summary = {
        "n_roles": len(roles),
        "hidden_dim": int(M.shape[1]),
        "centering": "center role means across roles",
        "standardization": "none",
        "pc1_variance_explained": float(ratio[0]),
        "components_to_70pct": k,
        "variance_target": float(variance_target),
        "axis_pc1_alignment_abs": float(aligned_cos),
        "pc1_sign_policy": "sign aligned to Assistant Axis for display only",
    }
    return summary, curve, unit_axis, display_pc1


def compare_regions(region_role_means, region_defaults):
    """Compare regional Axes and same-role projections on one common role set."""
    missing = [p for p in REGIONS if p not in region_role_means or p not in region_defaults]
    if missing:
        raise ValueError(f"missing regions: {missing}")
    common = sorted(set.intersection(*(set(region_role_means[p]) for p in REGIONS)))
    if len(common) < 3:
        raise ValueError("region comparison needs at least three common roles")

    rm = {p: {r: np.asarray(region_role_means[p][r], dtype=np.float64) for r in common}
          for p in REGIONS}
    axes = {p: assistant_axis(region_defaults[p], rm[p]) for p in REGIONS}
    own_proj = {p: np.array([np.dot(rm[p][r], axes[p]) for r in common]) for p in REGIONS}
    ref = axes["all_response"]
    ref_proj = {p: np.array([np.dot(rm[p][r], ref) for r in common]) for p in REGIONS}

    pairs = []
    for a, b in combinations(REGIONS, 2):
        pairs.append({
            "region_a": a,
            "region_b": b,
            "axis_cosine": float(np.dot(axes[a], axes[b])),
            "same_role_own_axis_projection_pearson": pearson_r(own_proj[a], own_proj[b]),
            "same_role_all_response_reference_projection_pearson": pearson_r(ref_proj[a], ref_proj[b]),
            "n_roles": len(common),
        })

    rows = []
    for r in common:
        row = {"role_id": r}
        for p in REGIONS:
            row[f"proj_{p}_own_axis"] = float(np.dot(rm[p][r], axes[p]))
            row[f"proj_{p}_all_response_axis"] = float(np.dot(rm[p][r], ref))
        rows.append(row)
    return pairs, rows, axes
