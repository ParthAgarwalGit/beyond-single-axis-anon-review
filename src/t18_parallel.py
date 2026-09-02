"""Bounded outer-fold process parallelism for cached T18.

This module changes only execution scheduling. Each outer fold is still
constructed by the frozen doubly-grouped splitter and fitted with the reviewed
fold-local PCA cache. The full set of folds, seeds, candidate grids, fitting
rules, AUROC, bootstrap, multiplicity adjustment, adequacy margin, and verdict
rule are unchanged.

Parallelism uses Linux ``fork`` so the large read-only activation matrix is
shared copy-on-write between workers rather than serialized once per task.
Each worker receives only an outer-fold descriptor and returns predictions and
selection metadata for that fold. The parent restores results in fold-id order.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import numpy as np

from src import t18_cached as cached
from src import t18_readouts as legacy

_FORK_STATE = None


def _resolved_builders(rows, family_order, builders_by_family):
    n_roles = len(rows.role_roles)
    out = {
        family: cached._resolve_builders(family, n_roles, builders_by_family)
        for family in family_order
    }
    for family in family_order:
        if not out[family]:
            raise RuntimeError(f"family {family} has no candidate within the budget")
    return out


def _run_outer_fold_worker(fold):
    if _FORK_STATE is None:
        raise RuntimeError("parallel T18 worker started without inherited fork state")
    rows, family_order, resolved_builders, inner_folds, seed = _FORK_STATE

    legacy.check_fold_isolation(rows, fold)
    tr, te = fold["train_idx"], fold["test_idx"]
    if len(te) == 0 or len(tr) == 0 or len(set(rows.y[tr])) < 2:
        return {"fold_id": int(fold["fold_id"]), "skipped": True}

    selected = cached._select_families_on_training(
        rows, tr, family_order, resolved_builders, inner_folds, seed
    )

    selected_builders = [(selected[f][0], selected[f][1]) for f in family_order]
    max_k = cached._max_pca_k(selected_builders, seed)
    feature_cache = cached.SplitFeatureCache(rows.X[tr], rows.X[te], max_pca_k=max_k)

    family_payload = {}
    for family in family_order:
        label, build, inner_score = selected[family]
        model = build(seed)
        model, pred = cached._fit_predict_cached(
            model,
            feature_cache,
            rows.y[tr],
            rows.roles[tr],
            rows.questions[tr],
            rows.weights[tr],
        )
        family_payload[family] = {
            "pred": np.asarray(pred, dtype=np.float64),
            "selection": {
                "fold_id": int(fold["fold_id"]),
                "role_fold": int(fold["role_fold"]),
                "question_fold": int(fold["question_fold"]),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "n_dropped": int(fold["n_dropped"]),
                "n_train_roles": len({str(r) for r in rows.roles[tr]}),
                "selected": label,
                "inner_auroc": float(inner_score),
                "n_params": int(model.n_params),
            },
        }

    return {
        "fold_id": int(fold["fold_id"]),
        "skipped": False,
        "test_idx": np.asarray(te, dtype=np.int64),
        "families": family_payload,
    }


def run_families_cached_parallel(
    rows,
    family_order,
    folds=None,
    inner_folds=None,
    seed=legacy.SPLIT_SEED,
    builders_by_family=None,
    outer_workers=1,
):
    outer_workers = int(outer_workers)
    if outer_workers < 1:
        raise ValueError("outer_workers must be >= 1")
    if outer_workers == 1:
        return cached.run_families_cached(
            rows, family_order, folds=folds, inner_folds=inner_folds,
            seed=seed, builders_by_family=builders_by_family,
        )

    if os.name != "posix" or "fork" not in mp.get_all_start_methods():
        raise RuntimeError("parallel T18 requires POSIX/Linux fork semantics")

    folds = legacy.OUTER_FOLDS if folds is None else folds
    inner_folds = legacy.INNER_FOLDS if inner_folds is None else inner_folds
    builders_by_family = builders_by_family or {}
    family_order = tuple(family_order)

    resolved = _resolved_builders(rows, family_order, builders_by_family)
    outer = legacy.doubly_grouped_folds(rows, *folds, seed=seed)

    predictions = {f: np.full(len(rows), np.nan) for f in family_order}
    fold_ids = {f: np.full(len(rows), -1, dtype=int) for f in family_order}
    selections = {f: [] for f in family_order}

    global _FORK_STATE
    if _FORK_STATE is not None:
        raise RuntimeError("parallel T18 fork state is already populated")
    _FORK_STATE = (rows, family_order, resolved, inner_folds, seed)
    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=outer_workers) as pool:
            results = pool.map(_run_outer_fold_worker, outer, chunksize=1)
    finally:
        _FORK_STATE = None

    for result in sorted(results, key=lambda x: x["fold_id"]):
        if result["skipped"]:
            continue
        te = result["test_idx"]
        fold_id = result["fold_id"]
        for family in family_order:
            payload = result["families"][family]
            predictions[family][te] = payload["pred"]
            fold_ids[family][te] = fold_id
            selections[family].append(payload["selection"])

    return predictions, fold_ids, selections


def analyse_parallel_cached(
    rows,
    replicates=legacy.BOOTSTRAP_REPLICATES,
    seed=legacy.SPLIT_SEED,
    families=None,
    builders_by_family=None,
    folds=None,
    inner_folds=None,
    permutations=legacy.PERMUTATION_REPLICATES,
    outer_workers=1,
):
    families = tuple(families or legacy.VERDICT_FAMILIES)
    builders_by_family = builders_by_family or {}
    folds = folds or legacy.OUTER_FOLDS
    inner_folds = inner_folds or legacy.INNER_FOLDS

    family_order = (cached.BASELINE_FAMILY, *families)
    predictions, fold_ids, selections = run_families_cached_parallel(
        rows, family_order, folds=folds, inner_folds=inner_folds, seed=seed,
        builders_by_family=builders_by_family, outer_workers=outer_workers,
    )

    fold_of_row = fold_ids[cached.BASELINE_FAMILY]
    performance = {
        family: {
            **legacy.cluster_bootstrap(
                rows, pred, fold_ids[family], replicates, seed,
                legacy.INFERENCE_CLUSTER
            ),
            "question_clustered_diagnostic": legacy.cluster_bootstrap(
                rows, pred, fold_ids[family], replicates, seed, "question"
            ),
        }
        for family, pred in predictions.items()
    }

    simultaneous = legacy.simultaneous_family_bootstrap(
        rows,
        {f: predictions[f] for f in families},
        predictions[cached.BASELINE_FAMILY],
        fold_of_row,
        replicates,
        seed,
        legacy.INFERENCE_CLUSTER,
    )

    nulls = []
    for i in range(permutations):
        permuted = legacy.permute_labels_by_role(rows, seed + 1000 + i)
        pred, permuted_folds, _ = legacy.run_family(
            permuted, cached.BASELINE_FAMILY, folds=folds,
            inner_folds=inner_folds, seed=seed,
        )
        scored = ~np.isnan(pred)
        nulls.append(
            legacy.fold_aggregate_auroc(
                permuted.y[scored], pred[scored], permuted_folds[scored],
                permuted.weights[scored],
            )
        )

    return {
        "primary_metric": legacy.PRIMARY_METRIC,
        "estimand": "mean within-fold role-balanced AUROC",
        "inference_cluster": legacy.INFERENCE_CLUSTER,
        "adequacy_margin": legacy.ADEQUACY_MARGIN,
        "t18_config_sha256": legacy.T18_CONFIG_SHA256,
        "performance": performance,
        "simultaneous_vs_axis": simultaneous,
        "verdict": legacy.adequacy_verdict(simultaneous),
        "selections": selections,
        "permutation_null_diagnostic": {
            "status": "descriptive only; not an inferential test",
            "replicates": nulls,
            "mean": float(np.mean(nulls)) if nulls else float("nan"),
            "max_abs_deviation_from_chance": (
                float(np.max(np.abs(np.asarray(nulls) - 0.5))) if nulls
                else float("nan")
            ),
        },
        "compute_execution": {
            "outer_workers": int(outer_workers),
            "backend": "multiprocessing_fork",
        },
    }
