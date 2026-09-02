"""Computation-equivalent cache for the frozen T18 read-out analysis.

This module is deliberately separate from ``src.t18_readouts``, which remains
the frozen/reference implementation. It changes only computation:

* one SVD is fitted per unique training split and shared across PCA-based
  candidate families;
* each requested k-dimensional projection is computed once per split and
  reused across C / hidden-size / alpha candidates;
* the full-dimensional centring/scaling transform is computed once per split
  and reused across C values.

It does NOT change T18's rows, folds, seeds, candidate grids, role balancing,
model classes, solvers, max_iter values, AUROC, bootstrap, multiplicity
adjustment, adequacy margin, or verdict rule.

Unlike ``src.t18_readouts``'s duck-typed candidate contract (any object with
``.fit``/``.predict_proba``/``.n_params``), the cached fitting path
(``_fit_predict_cached``) only knows how to share transforms for exactly the
four frozen readout classes - ``AxisReadout``, ``PCALinearReadout``,
``FullLinearReadout``, ``SmallNonlinearReadout`` - since sharing a fold-local
SVD requires knowing each class's projection logic. A ``builders_by_family``
override supplying any other candidate class raises ``TypeError`` here; it
would run unmodified against the reference ``legacy.run_family``/``analyse``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier

from src import t18_readouts as legacy


BASELINE_FAMILY = "axis_1d"


class PcaCacheInvariantError(Exception):
    """A cache bookkeeping bug (an underestimated ``max_pca_k``), not a
    per-candidate fit failure. Deliberately NOT a ValueError/RuntimeError
    subclass, so it is never caught by the ``except (ValueError,
    RuntimeError)`` around per-candidate fitting - which exists to catch the
    reference's known unfittable-candidate cases (exceeded parameter budget,
    degenerate Axis), not to silently disqualify a candidate because of an
    internal cache defect."""


class SplitFeatureCache:
    """Fold-local feature transforms fitted from training rows only."""

    def __init__(self, X_train, X_eval, max_pca_k=0):
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.X_eval = np.asarray(X_eval, dtype=np.float64)
        self.max_pca_k = int(max_pca_k)
        self._mean = None
        self._centered_train = None
        self._centered_eval = None
        self._components = None
        self._pca_by_k = {}
        self._full_scaled = None

    @property
    def mean(self):
        if self._mean is None:
            self._mean = self.X_train.mean(axis=0)
        return self._mean

    @property
    def centered_train(self):
        if self._centered_train is None:
            self._centered_train = self.X_train - self.mean
        return self._centered_train

    @property
    def centered_eval(self):
        if self._centered_eval is None:
            self._centered_eval = self.X_eval - self.mean
        return self._centered_eval

    def _ensure_components(self, k):
        """Fit the same full SVD as the reference code, once per split."""
        needed = max(int(k), self.max_pca_k)
        if self._components is None:
            _, _, vt = np.linalg.svd(self.centered_train, full_matrices=False)
            self._components = vt[:needed]
        elif self._components.shape[0] < min(needed, self.X_train.shape[0], self.X_train.shape[1]):
            raise PcaCacheInvariantError(
                f"PCA cache retained only {self._components.shape[0]} components "
                f"but candidate requested k={k} (cache max={self.max_pca_k})"
            )

    def pca_scaled(self, k):
        """Return exactly-k train/eval projections, cached once per k.

        We intentionally perform the k-column matrix multiply separately for
        each requested k, matching the reference implementation's GEMM shape
        as closely as possible. The expensive SVD is still shared.
        """
        k = int(k)
        if k not in self._pca_by_k:
            self._ensure_components(k)
            comps = self._components[:k]
            z_train = self.centered_train @ comps.T
            z_eval = self.centered_eval @ comps.T
            scale = legacy._train_scale(z_train)
            self._pca_by_k[k] = (
                self.mean,
                comps,
                scale,
                z_train / scale,
                z_eval / scale,
            )
        return self._pca_by_k[k]

    def full_scaled(self):
        if self._full_scaled is None:
            scale = legacy._train_scale(self.centered_train)
            self._full_scaled = (
                self.mean,
                scale,
                self.centered_train / scale,
                self.centered_eval / scale,
            )
        return self._full_scaled


def _max_pca_k(builders, seed):
    out = 0
    for _, build in builders:
        model = build(seed)
        if isinstance(model, (legacy.PCALinearReadout, legacy.SmallNonlinearReadout)):
            out = max(out, int(model.k))
    return out


def _fit_predict_cached(model, cache, y_train, roles_train, questions_train, weights_train):
    """Fit one frozen candidate with cached fold-local transforms."""
    y_train = np.asarray(y_train, dtype=int)
    roles_train = np.asarray(roles_train, dtype=object)

    if isinstance(model, legacy.AxisReadout):
        model.axis = legacy._fold_axis(cache.X_train, y_train, roles_train)
        if model.axis is None:
            raise RuntimeError("fold cannot support an Axis (missing a class)")
        model.clf = legacy._logistic(1.0, model.seed).fit(
            (cache.X_train @ model.axis)[:, None],
            y_train,
            sample_weight=weights_train,
        )
        pred = model.clf.predict_proba((cache.X_eval @ model.axis)[:, None])[:, 1]
        return model, pred

    if isinstance(model, legacy.PCALinearReadout):
        mean, comps, scale, z_train, z_eval = cache.pca_scaled(model.k)
        model.mean_ = mean
        model.comps_ = comps
        model.scale_ = scale
        model.clf = legacy._logistic(model.C, model.seed).fit(
            z_train, y_train, sample_weight=weights_train
        )
        return model, model.clf.predict_proba(z_eval)[:, 1]

    if isinstance(model, legacy.FullLinearReadout):
        mean, scale, x_train, x_eval = cache.full_scaled()
        model.mean_ = mean
        model.scale_ = scale
        model.clf = legacy._logistic(model.C, model.seed).fit(
            x_train, y_train, sample_weight=weights_train
        )
        return model, model.clf.predict_proba(x_eval)[:, 1]

    if isinstance(model, legacy.SmallNonlinearReadout):
        budget = len({str(r) for r in roles_train[y_train == 0]})
        if legacy.nonlinear_param_count(model.k, model.hidden) >= budget:
            raise ValueError(
                f"parameter budget exceeded: "
                f"{legacy.nonlinear_param_count(model.k, model.hidden)} params "
                f"vs {budget} training roles"
            )
        mean, comps, scale, z_train, z_eval = cache.pca_scaled(model.k)
        model.mean_ = mean
        model.comps_ = comps
        model.scale_ = scale
        bal = legacy._balanced_index(y_train, roles_train)
        model.clf = MLPClassifier(
            hidden_layer_sizes=(model.hidden,),
            alpha=model.alpha,
            solver="lbfgs",
            max_iter=2000,
            random_state=model.seed,
        ).fit(z_train[bal], y_train[bal])
        return model, model.clf.predict_proba(z_eval)[:, 1]

    raise TypeError(
        "Cached T18 only supports the four frozen readout classes; "
        f"got {type(model).__name__}"
    )


@dataclass
class _CandidateState:
    label: str
    build: object
    scores: list
    valid: bool = True


def _resolve_builders(family, n_train_roles, builders_by_family):
    custom = builders_by_family.get(family)
    if custom is not None:
        return list(custom)
    if family not in legacy.FAMILIES:
        raise RuntimeError(
            f"family {family!r} is unknown and no explicit builders were supplied"
        )
    return list(legacy.FAMILIES[family](n_train_roles))


def _select_families_on_training(
    rows, train_idx, family_order, resolved_builders, inner_folds, seed
):
    """Select one setting per family while sharing one SVD per inner split.

    ``resolved_builders`` is constructed once from the full input role count,
    exactly like ``legacy.run_family``. Inner-fold capacity failures are still
    enforced by the model fit, so the frozen candidate grid is not narrowed
    early by an outer-fold role count.
    """
    sub = legacy.ReadoutRows(
        rows.X[train_idx],
        rows.roles[train_idx],
        rows.questions[train_idx],
        rows.y[train_idx],
    )
    inner = legacy.doubly_grouped_folds(sub, *inner_folds, seed=seed + 977)

    states = {}
    all_builders = []
    for family in family_order:
        builders = list(resolved_builders[family])
        if not builders:
            raise RuntimeError(f"family {family} has no candidate within the budget")
        states[family] = [_CandidateState(label, build, []) for label, build in builders]
        all_builders.extend(builders)

    max_k = _max_pca_k(all_builders, seed)

    for f in inner:
        tr, te = f["train_idx"], f["test_idx"]
        if len(te) == 0 or len(tr) == 0:
            continue
        if len(set(sub.y[tr])) < 2 or len(set(sub.y[te])) < 2:
            continue

        cache = SplitFeatureCache(sub.X[tr], sub.X[te], max_pca_k=max_k)

        for family in family_order:
            for state in states[family]:
                if not state.valid:
                    continue
                try:
                    model = state.build(seed)
                    _, pred = _fit_predict_cached(
                        model,
                        cache,
                        sub.y[tr],
                        sub.roles[tr],
                        sub.questions[tr],
                        sub.weights[tr],
                    )
                    state.scores.append(legacy.auroc(sub.y[te], pred, sub.weights[te]))
                except (ValueError, RuntimeError):
                    # Exact reference semantics: one unfittable inner fold
                    # invalidates this candidate for the outer fold.
                    state.valid = False
                    state.scores = []

    selected = {}
    for family in family_order:
        best = None
        best_score = -np.inf
        for state in states[family]:
            if not state.valid or not state.scores:
                continue
            score = float(np.mean(state.scores))
            if score > best_score:  # strict > preserves first-on-tie behavior
                best = state
                best_score = score
        if best is None:
            raise RuntimeError("no candidate in this family could be fitted")
        selected[family] = (best.label, best.build, best_score)

    return selected


def run_families_cached(
    rows,
    family_order,
    folds=None,
    inner_folds=None,
    seed=legacy.SPLIT_SEED,
    builders_by_family=None,
):
    """Run several frozen families together, sharing transforms by split."""
    folds = legacy.OUTER_FOLDS if folds is None else folds
    inner_folds = legacy.INNER_FOLDS if inner_folds is None else inner_folds
    builders_by_family = builders_by_family or {}
    family_order = tuple(family_order)

    predictions = {f: np.full(len(rows), np.nan) for f in family_order}
    fold_ids = {f: np.full(len(rows), -1, dtype=int) for f in family_order}
    selections = {f: [] for f in family_order}

    # Match legacy.run_family: resolve each family grid from the full input
    # role count once, before entering the outer folds.
    n_roles = len(rows.role_roles)
    resolved_builders = {
        family: _resolve_builders(family, n_roles, builders_by_family)
        for family in family_order
    }
    # Match legacy.run_family: raise unconditionally on an empty grid, not
    # only when an outer fold happens to survive the skip guard below.
    for family in family_order:
        if not resolved_builders[family]:
            raise RuntimeError(f"family {family} has no candidate within the budget")

    for fold in legacy.doubly_grouped_folds(rows, *folds, seed=seed):
        legacy.check_fold_isolation(rows, fold)
        tr, te = fold["train_idx"], fold["test_idx"]
        if len(te) == 0 or len(tr) == 0 or len(set(rows.y[tr])) < 2:
            continue

        selected = _select_families_on_training(
            rows, tr, family_order, resolved_builders, inner_folds, seed
        )

        selected_builders = [(selected[f][0], selected[f][1]) for f in family_order]
        max_k = _max_pca_k(selected_builders, seed)
        cache = SplitFeatureCache(rows.X[tr], rows.X[te], max_pca_k=max_k)

        for family in family_order:
            label, build, inner_score = selected[family]
            model = build(seed)
            model, pred = _fit_predict_cached(
                model,
                cache,
                rows.y[tr],
                rows.roles[tr],
                rows.questions[tr],
                rows.weights[tr],
            )
            predictions[family][te] = pred
            fold_ids[family][te] = fold["fold_id"]
            selections[family].append(
                {
                    "fold_id": fold["fold_id"],
                    "role_fold": fold["role_fold"],
                    "question_fold": fold["question_fold"],
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "n_dropped": int(fold["n_dropped"]),
                    "n_train_roles": len({str(r) for r in rows.roles[tr]}),
                    "selected": label,
                    "inner_auroc": inner_score,
                    "n_params": int(model.n_params),
                }
            )

    return predictions, fold_ids, selections


def run_family_cached(
    rows,
    family,
    folds=None,
    inner_folds=None,
    seed=legacy.SPLIT_SEED,
    builders=None,
):
    """Cached drop-in analogue of ``legacy.run_family`` for tests/audits."""
    builders_by_family = {} if builders is None else {family: builders}
    predictions, fold_ids, selections = run_families_cached(
        rows,
        (family,),
        folds=folds,
        inner_folds=inner_folds,
        seed=seed,
        builders_by_family=builders_by_family,
    )
    return predictions[family], fold_ids[family], selections[family]


def analyse_cached(
    rows,
    replicates=legacy.BOOTSTRAP_REPLICATES,
    seed=legacy.SPLIT_SEED,
    families=None,
    builders_by_family=None,
    folds=None,
    inner_folds=None,
    permutations=legacy.PERMUTATION_REPLICATES,
):
    """Computation-equivalent replacement for the reference ``analyse``."""
    families = tuple(families or legacy.VERDICT_FAMILIES)
    builders_by_family = builders_by_family or {}
    folds = folds or legacy.OUTER_FOLDS
    inner_folds = inner_folds or legacy.INNER_FOLDS

    family_order = (BASELINE_FAMILY, *families)
    predictions, fold_ids, selections = run_families_cached(
        rows,
        family_order,
        folds=folds,
        inner_folds=inner_folds,
        seed=seed,
        builders_by_family=builders_by_family,
    )

    fold_of_row = fold_ids[BASELINE_FAMILY]
    performance = {
        family: {
            **legacy.cluster_bootstrap(
                rows, pred, fold_ids[family], replicates, seed, legacy.INFERENCE_CLUSTER
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
        predictions[BASELINE_FAMILY],
        fold_of_row,
        replicates,
        seed,
        legacy.INFERENCE_CLUSTER,
    )

    # Keep the descriptive permutation diagnostic on the reference Axis path.
    nulls = []
    for i in range(permutations):
        permuted = legacy.permute_labels_by_role(rows, seed + 1000 + i)
        pred, permuted_folds, _ = legacy.run_family(
            permuted,
            BASELINE_FAMILY,
            folds=folds,
            inner_folds=inner_folds,
            seed=seed,
        )
        scored = ~np.isnan(pred)
        nulls.append(
            legacy.fold_aggregate_auroc(
                permuted.y[scored],
                pred[scored],
                permuted_folds[scored],
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
                float(np.max(np.abs(np.asarray(nulls) - 0.5)))
                if nulls
                else float("nan")
            ),
        },
    }
