"""All-response vs answer-only vs reasoning-only pool geometry (T17).

For the primary C80 measurement-limited branch, output membership is label-independent:
technically valid outputs with the required pool present, under the 2026-08-17
confirmatory-membership deviation. A score-3 path remains available only as an
explicitly unvalidated sensitivity.

Direct region comparisons use matched output cohorts so a pool difference reflects
removing a token region rather than silently changing which outputs contribute.
Available-case summaries remain sensitivity-only.
"""
from __future__ import annotations

import numpy as np

from .geometry import assistant_axis, pearson_r

POOLS = ("all_response", "answer", "reasoning")
REFERENCE_POOL = "all_response"
F5_POOLS = ("all_response", "answer")
TRIPLE_POOLS = ("all_response", "answer", "reasoning")
DEFAULT_VALID_FLOOR = 0.95
N_DEFAULT_CONDITIONS = 5


class _RunningMeans:
    def __init__(self):
        self._sum = {}
        self._n = {}

    def add(self, key, vec):
        v = np.asarray(vec, dtype=np.float64)
        if key in self._sum:
            self._sum[key] += v
        else:
            self._sum[key] = v.copy()
        self._n[key] = self._n.get(key, 0) + 1

    def counts(self):
        return dict(self._n)

    def means(self):
        return {k: self._sum[k] / self._n[k] for k in self._sum}


class _MatchedMeans:
    def __init__(self, pools):
        self.pools = tuple(pools)
        self._sum = {}
        self._n = {}

    def add(self, role, vecs):
        if role not in self._sum:
            self._sum[role] = {p: np.asarray(vecs[p], dtype=np.float64).copy()
                               for p in self.pools}
            self._n[role] = 1
        else:
            for p in self.pools:
                self._sum[role][p] += np.asarray(vecs[p], dtype=np.float64)
            self._n[role] += 1

    def counts(self):
        return dict(self._n)

    def means(self):
        return {r: {p: self._sum[r][p] / self._n[r] for p in self.pools}
                for r in self._sum}


def _pool_available(rec, pool):
    """Use T12 has_*_pool metadata as authority and fail on tensor disagreement."""
    has = bool(rec.get("has", {}).get(pool))
    vec = rec.get("pools", {}).get(pool)
    uid = rec.get("uid")
    if has and vec is None:
        raise ValueError(f"uid {uid}: metadata has_{pool}_pool=true but no {pool} tensor")
    if (not has) and vec is not None:
        raise ValueError(f"uid {uid}: stale {pool} tensor present but has_{pool}_pool=false")
    return has and vec is not None


def reduce_records(records, membership_ok, retained_roles=None,
                   arms=("translated", "wrapper"), default_mode="default"):
    """Reduce response pools to per-role means and default-condition means.

    Role outputs obey ``membership_ok`` and optional retained-role membership.
    Defaults always use technical validity only. For defaults we retain per-pool
    means and per-condition pool availability so regional Axes can be constructed
    only when their own default pool clears the frozen 95% floor.
    """
    f5 = {a: _MatchedMeans(F5_POOLS) for a in arms}
    tri = {a: _MatchedMeans(TRIPLE_POOLS) for a in arms}
    avail = {a: {p: _RunningMeans() for p in POOLS} for a in arms}

    default_means = {p: _RunningMeans() for p in POOLS}
    default_counts = {}
    default_pool_counts = {p: {} for p in POOLS}

    for rec in records:
        mode = rec["mode"]
        present = {p: _pool_available(rec, p) for p in POOLS}

        if mode == default_mode:
            cond = rec["condition"]
            dc = default_counts.setdefault(cond, {"total": 0, "valid": 0})
            dc["total"] += 1
            is_valid = rec.get("validity") == "valid"
            if is_valid:
                dc["valid"] += 1
            for p in POOLS:
                pc = default_pool_counts[p].setdefault(cond, {"total": 0, "eligible": 0})
                pc["total"] += 1
                if is_valid and present[p]:
                    pc["eligible"] += 1
                    default_means[p].add(cond, rec["pools"][p])
            continue

        if mode not in f5 or not membership_ok(rec):
            continue
        role = rec.get("role")
        if role is None:
            continue
        if retained_roles is not None and role not in retained_roles:
            continue

        for p in POOLS:
            if present[p]:
                avail[mode][p].add(role, rec["pools"][p])
        if all(present[p] for p in F5_POOLS):
            f5[mode].add(role, {p: rec["pools"][p] for p in F5_POOLS})
        if all(present[p] for p in TRIPLE_POOLS):
            tri[mode].add(role, {p: rec["pools"][p] for p in TRIPLE_POOLS})

    arms_out = {}
    for a in arms:
        arms_out[a] = {
            "matched_f5": {"means": f5[a].means(), "counts": f5[a].counts()},
            "matched_triple": {"means": tri[a].means(), "counts": tri[a].counts()},
            "available": {"means": {p: avail[a][p].means() for p in POOLS},
                          "counts": {p: avail[a][p].counts() for p in POOLS}},
        }

    by_pool = {p: default_means[p].means() for p in POOLS}
    return {
        "arms": arms_out,
        "default": {
            "all_response_condition_means": by_pool["all_response"],
            "condition_counts": default_counts,
            "condition_means_by_pool": by_pool,
            "condition_counts_by_pool": default_pool_counts,
        },
    }


def default_mean_enforced(condition_means, condition_counts, floor=DEFAULT_VALID_FLOOR,
                          eligible_key="valid"):
    """Equal-weight five condition means after enforcing a per-condition floor."""
    if len(condition_means) != N_DEFAULT_CONDITIONS:
        raise ValueError(f"expected {N_DEFAULT_CONDITIONS} default conditions, "
                         f"got {len(condition_means)}: {sorted(condition_means)}")
    for cond in sorted(condition_means):
        c = condition_counts.get(cond, {})
        total = c.get("total", 0)
        eligible = c.get(eligible_key, 0)
        frac = (eligible / total) if total else 0.0
        if frac < floor:
            raise ValueError(
                f"default condition {cond!r}: {eligible_key} fraction {frac:.4f} below the "
                f"frozen {floor} floor ({eligible}/{total})")
    means = [np.asarray(condition_means[c], dtype=np.float64)
             for c in sorted(condition_means)]
    return np.mean(means, axis=0)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("cosine undefined for a zero vector")
    return float(np.dot(a, b) / (na * nb))


def _pearson(xs, ys):
    if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return round(pearson_r(xs, ys), 6)


def _reference_axis(default_vec, cohort_means):
    role_all = {r: cohort_means[r][REFERENCE_POOL] for r in cohort_means}
    return assistant_axis(default_vec, role_all)


def _project(cohort_means, pool, unit):
    return {r: float(np.dot(np.asarray(cohort_means[r][pool], dtype=np.float64), unit))
            for r in cohort_means}


def _own_axis(default_vec, cohort_means, pool):
    roles = {r: cohort_means[r][pool] for r in cohort_means}
    grand = np.mean(list(roles.values()), axis=0)
    v = np.asarray(default_vec, dtype=np.float64) - grand
    norm = np.linalg.norm(v)
    if norm == 0:
        return None, False, "degenerate: default equals grand role mean"
    unit = v / norm
    ok = float(np.dot(default_vec, unit)) > float(np.mean(
        [np.dot(m, unit) for m in roles.values()]))
    return unit, ok, ("sign check passed" if ok else
                      "sign check FAILED: this pool does not place the default at the Assistant end")


def matched_f5(default_vec, f5_means):
    if not f5_means:
        raise ValueError("no roles in the matched all/answer cohort")
    unit = _reference_axis(default_vec, f5_means)
    proj = {p: _project(f5_means, p, unit) for p in F5_POOLS}
    roles = sorted(f5_means)
    rows = [{"role_id": r,
             "proj_all_response": round(proj["all_response"][r], 6),
             "proj_answer_only": round(proj["answer"][r], 6)} for r in roles]
    xs = [proj["all_response"][r] for r in roles]
    ys = [proj["answer"][r] for r in roles]
    return {
        "cohort": "outputs with both all-response and answer pools (matched)",
        "n_roles": len(roles),
        "rows": rows,
        "pearson_all_vs_answer": _pearson(xs, ys),
        "mean_abs_shift_all_vs_answer": (round(float(np.mean(
            [abs(a - b) for a, b in zip(xs, ys)])), 6) if roles else None),
    }


def matched_triple(default_vec, tri_means):
    if not tri_means:
        return {"cohort": "outputs with all three pools (matched)",
                "n_roles": 0, "rows": [], "note": "empty triple cohort"}
    unit = _reference_axis(default_vec, tri_means)
    proj = {p: _project(tri_means, p, unit) for p in TRIPLE_POOLS}
    roles = sorted(tri_means)
    rows = [{"role_id": r,
             "proj_all_response": round(proj["all_response"][r], 6),
             "proj_answer_only": round(proj["answer"][r], 6),
             "proj_reasoning_only": round(proj["reasoning"][r], 6)} for r in roles]

    def col(p):
        return [proj[p][r] for r in roles]

    own, cos_to_ref = {}, {}
    for p in TRIPLE_POOLS:
        u, ok, note = _own_axis(default_vec, tri_means, p)
        own[p] = {"sign_check_passed": ok, "note": note}
        if u is not None:
            cos_to_ref[p] = round(cosine(u, unit), 6)
    return {
        "cohort": "outputs with all-response, answer, and reasoning pools (matched)",
        "n_roles": len(roles),
        "rows": rows,
        "pearson_all_vs_answer": _pearson(col("all_response"), col("answer")),
        "pearson_all_vs_reasoning": _pearson(col("all_response"), col("reasoning")),
        "pearson_answer_vs_reasoning": _pearson(col("answer"), col("reasoning")),
        "own_axis": own,
        "cosine_own_axis_to_reference": cos_to_ref,
    }


def available_case_sensitivity(default_vec, available_means):
    ref = available_means.get(REFERENCE_POOL, {})
    if not ref:
        return {"n_roles": 0, "note": "no all-response available means"}
    unit = assistant_axis(default_vec, ref)
    out = {"cohort": "available-case (per-pool, unmatched) — SENSITIVITY ONLY",
           "n_roles_all_response": len(ref)}
    for a, b in (("all_response", "answer"), ("all_response", "reasoning")):
        common = sorted(set(available_means.get(a, {})) & set(available_means.get(b, {})))
        if len(common) >= 2:
            xs = [float(np.dot(available_means[a][r], unit)) for r in common]
            ys = [float(np.dot(available_means[b][r], unit)) for r in common]
            out[f"pearson_{a}_vs_{b}"] = _pearson(xs, ys)
            out[f"n_roles_{a}_vs_{b}"] = len(common)
    return out


INTERPRETATION = (
    "Primary C80 membership is label-independent technical validity under the "
    "2026-08-17 deviation record. Direct region comparisons use matched output "
    "cohorts. A low all-vs-reasoning correlation means the reasoning trace does "
    "not preserve the same role ordering along the all-response Assistant Axis; "
    "it does not show that persona information is absent from reasoning. PCA in "
    "T17 is role-space variance structure, not intrinsic dimensionality of one persona."
)
