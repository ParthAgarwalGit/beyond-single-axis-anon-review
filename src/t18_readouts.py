"""T18 dimensional-adequacy and nonlinearity stress test.

Registered in the frozen method YAML as ``analysis_registry.sensitivity[13]``,
"grouped multidimensional linear and restrained nonlinear readouts". The
question is narrow: **is one direction an adequate summary?** We answer it by
comparing held-out predictive performance of the fixed one-direction
Assistant-Axis score against multidimensional linear alternatives and a
deliberately small nonlinear read-out.

This is diagnostic only. It does not define, replace, or re-sign the Axis
(``role_vectors_and_axis.PCA.descriptive_only``), and a win for a richer
read-out is evidence about *summary adequacy*, not about the causal claims.

Every analysis parameter comes from ``configs/t18_frozen.yaml`` — grids, fold
counts, the estimand, the resampling unit, the multiplicity rule, and the
equivalence margin — so none of them can be chosen after seeing a result.

Four properties are load-bearing, each covered by a test in
``tests/test_t18_readouts.py``:

1. **Grouped evaluation.** Every fold holds out whole roles *and* whole
   questions simultaneously. Rows sharing a role or a question with the test
   set are dropped from that fold's training set rather than allowed to leak.
2. **Everything is fitted inside the fold.** The Axis, the PCA basis, the
   centring and scaling statistics, and every hyper-parameter come from that
   fold's training rows alone.
3. **Fold-aggregated estimand.** Outer folds are fitted independently, so
   their probability scales are not necessarily rank-comparable. Scores are
   never pooled across folds; the statistic is the mean of within-fold AUROC,
   and every bootstrap replicate recomputes that same statistic.
4. **Role-balanced weighting.** The Axis equal-weights retained roles, so the
   read-outs are fitted and scored with weights that equalise each role's
   total contribution within its class. Otherwise a high-retention role would
   carry more influence in the comparison than it carries in the geometry.

The PCA used here is a predictive feature basis fitted on training response
vectors. It is **not** the descriptive role-space PCA of
``role_vectors_and_axis.PCA``, which is fitted on role means over the whole
retained set. The two answer different questions and their component counts
are not interchangeable.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
import yaml

DEFAULT_ROLE_PREFIX = "__default__"
T18_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "t18_frozen.yaml"

_t18_cache = None


def load_t18_config():
    """Return (frozen T18 parameters, sha256 hex) for ``configs/t18_frozen.yaml``.

    Held separately from ``method_frozen.yaml`` so that freezing T18 did not
    change the method hash already stamped into issued reports; see
    ``docs/deviations/2026-08-13-t18-analysis-parameters.md``.
    """
    global _t18_cache
    if _t18_cache is None:
        raw = T18_CONFIG_PATH.read_bytes()
        _t18_cache = (yaml.safe_load(raw), hashlib.sha256(raw).hexdigest())
    return _t18_cache


_CFG, T18_CONFIG_SHA256 = load_t18_config()

PCA_K_GRID = tuple(_CFG["candidates"]["pca_k_grid"])
LINEAR_C_GRID = tuple(_CFG["candidates"]["linear_c_grid"])
NONLINEAR_HIDDEN_GRID = tuple(_CFG["candidates"]["nonlinear_hidden_grid"])
NONLINEAR_ALPHA_GRID = tuple(_CFG["candidates"]["nonlinear_alpha_grid"])
VERDICT_FAMILIES = tuple(_CFG["candidates"]["families_that_may_determine_the_verdict"])

PRIMARY_METRIC = _CFG["estimand"]["metric"]
INFERENCE_CLUSTER = _CFG["inference"]["cluster"]
ADEQUACY_MARGIN = _CFG["inference"]["adequacy_margin_auroc"]
BOOTSTRAP_REPLICATES = _CFG["inference"]["bootstrap_replicates"]

OUTER_FOLDS = (_CFG["splitting"]["outer_role_folds"],
               _CFG["splitting"]["outer_question_folds"])
INNER_FOLDS = (_CFG["splitting"]["inner_role_folds"],
               _CFG["splitting"]["inner_question_folds"])
SPLIT_SEED = _CFG["splitting"]["seed"]
PERMUTATION_REPLICATES = _CFG["diagnostics"]["group_label_permutation"]["replicates"]
REQUIRED_DEFAULT_CONDITIONS = _CFG["target"]["default_conditions_required"]
PRIMARY_JUDGE_REGION = _CFG["target"]["judge_region"]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

class ReadoutRows:
    """Pooled response vectors with the group keys the split needs.

    ``label`` is 1 for default-Assistant responses and 0 for retained
    role-adopted responses, so the read-out predicts Assistant status and the
    Axis (signed toward the default) is a natural one-dimensional score for it.

    Default responses carry no role, so each default condition is given the
    synthetic role name ``__default__<condition_index>``. That keeps the
    grouping honest: a fold that holds out a default condition really has
    never seen it, and the conditions cannot be memorised individually.
    """

    def __init__(self, vectors, roles, question_ids, labels):
        self.X = np.asarray(vectors, dtype=np.float64)
        self.roles = np.asarray(roles, dtype=object)
        self.questions = np.asarray(question_ids)
        self.y = np.asarray(labels, dtype=int)

        n = self.X.shape[0]
        if self.X.ndim != 2 or n == 0:
            raise ValueError("expected a non-empty (n_rows, hidden) matrix")
        if not (len(self.roles) == len(self.questions) == len(self.y) == n):
            raise ValueError("vectors, roles, question_ids, and labels must align")
        if not np.isfinite(self.X).all():
            raise ValueError("activation matrix contains non-finite values")
        if set(np.unique(self.y)) - {0, 1}:
            raise ValueError("labels must be 0 (role) or 1 (default Assistant)")

        is_default_role = np.array(
            [str(r).startswith(DEFAULT_ROLE_PREFIX) for r in self.roles]
        )
        if not np.array_equal(is_default_role, self.y == 1):
            raise ValueError(
                "label 1 must correspond exactly to synthetic "
                f"{DEFAULT_ROLE_PREFIX}* role names"
            )
        self.weights = role_balance_weights(self.roles, self.y)

    def __len__(self):
        return self.X.shape[0]

    @property
    def role_roles(self):
        """Sorted distinct genuine (non-default) role names."""
        return sorted({str(r) for r in self.roles[self.y == 0]})

    @property
    def default_roles(self):
        """Sorted distinct synthetic default-condition group names."""
        return sorted({str(r) for r in self.roles[self.y == 1]})


def role_balance_weights(roles, y):
    """Weights that equalise each role's contribution within its class.

    The Axis equal-weights retained roles. Row-level fitting and scoring would
    instead let a role with many retained responses dominate, so the read-outs
    would be judged on a different estimand from the geometry they assess.
    Within each class every role receives the same total weight, and the two
    classes receive the same total weight; the sum is the row count, so the
    penalty grids mean roughly what they would on unweighted data.
    """
    roles = np.asarray(roles, dtype=object)
    y = np.asarray(y, dtype=int)
    w = np.zeros(len(y), dtype=np.float64)
    classes = [c for c in (0, 1) if (y == c).any()]
    for cls in classes:
        in_class = y == cls
        for role in {str(r) for r in roles[in_class]}:
            sel = in_class & (roles == role)
            w[sel] = 1.0 / int(sel.sum())
        w[in_class] *= len(y) / (len(classes) * w[in_class].sum())
    return w


# ---------------------------------------------------------------------------
# Doubly grouped splitting
# ---------------------------------------------------------------------------

def _assign_folds(keys, n_folds, seed):
    """Deterministic balanced assignment of distinct keys to folds.

    Sorted first so the result depends only on the key *set* and the seed,
    never on the order rows happen to arrive in.
    """
    keys = sorted(keys)
    if n_folds < 2:
        raise ValueError("need at least 2 folds")
    if len(keys) < n_folds:
        raise ValueError(f"{len(keys)} distinct keys cannot fill {n_folds} folds")
    order = np.random.default_rng(seed).permutation(len(keys))
    return {keys[int(k)]: int(i % n_folds) for i, k in enumerate(order)}


def _assign_default_folds(default_roles, n_role_folds):
    """Spread default conditions across role folds, one per fold when the
    counts match, so every fold holds out default rows and no fold is left
    with a single-class test set."""
    return {name: i % n_role_folds for i, name in enumerate(sorted(default_roles))}


def doubly_grouped_folds(rows, n_role_folds=None, n_question_folds=None,
                         seed=SPLIT_SEED):
    """Outer folds holding out whole roles and whole questions at once.

    Fold ``(i, j)`` tests rows whose role is in role-fold ``i`` *and* whose
    question is in question-fold ``j``. It trains on rows whose role is
    outside fold ``i`` *and* whose question is outside fold ``j``. Rows
    matching exactly one of the two conditions are used by neither side —
    dropping them is the price of testing on unseen roles and unseen
    questions simultaneously.

    Every row is tested exactly once, so each row belongs to exactly one fold
    and the fold-aggregated statistic partitions the data cleanly.
    """
    n_role_folds = OUTER_FOLDS[0] if n_role_folds is None else n_role_folds
    n_question_folds = OUTER_FOLDS[1] if n_question_folds is None else n_question_folds

    role_fold = _assign_folds(rows.role_roles, n_role_folds, seed)
    role_fold.update(_assign_default_folds(rows.default_roles, n_role_folds))
    question_fold = _assign_folds(
        {int(q) for q in rows.questions}, n_question_folds, seed + 1
    )

    r_of_row = np.array([role_fold[str(r)] for r in rows.roles])
    q_of_row = np.array([question_fold[int(q)] for q in rows.questions])

    folds = []
    for i in range(n_role_folds):
        for j in range(n_question_folds):
            test = np.flatnonzero((r_of_row == i) & (q_of_row == j))
            train = np.flatnonzero((r_of_row != i) & (q_of_row != j))
            folds.append(
                {
                    "fold_id": i * n_question_folds + j,
                    "role_fold": i,
                    "question_fold": j,
                    "train_idx": train,
                    "test_idx": test,
                    "n_dropped": len(rows) - len(train) - len(test),
                }
            )
    return folds


def check_fold_isolation(rows, fold):
    """Raise unless the fold's training and test rows share no role and no
    question. Called on every fold of every run; a leak is a hard stop, not a
    warning, because a leaked fold silently inflates every candidate."""
    tr, te = fold["train_idx"], fold["test_idx"]
    shared_roles = set(rows.roles[tr]) & set(rows.roles[te])
    shared_questions = set(rows.questions[tr]) & set(rows.questions[te])
    if shared_roles:
        raise RuntimeError(f"role leakage in fold {fold['role_fold']}/"
                           f"{fold['question_fold']}: {sorted(shared_roles)[:5]}")
    if shared_questions:
        raise RuntimeError(f"question leakage in fold {fold['role_fold']}/"
                           f"{fold['question_fold']}: {sorted(shared_questions)[:5]}")


# ---------------------------------------------------------------------------
# Candidate read-outs
# ---------------------------------------------------------------------------

def _fold_axis(X, y, roles):
    """Unit Assistant Axis from this fold's training rows only.

    Follows the frozen construction: mean within each default condition
    first, then equal weights across conditions; equal weights across roles
    for the role side. Returns ``None`` when the fold cannot support an Axis.
    """
    default_names = sorted({str(r) for r in roles[y == 1]})
    role_names = sorted({str(r) for r in roles[y == 0]})
    if not default_names or not role_names:
        return None
    mu_default = np.mean(
        [X[roles == n].mean(axis=0) for n in default_names], axis=0
    )
    grand_role = np.mean([X[roles == n].mean(axis=0) for n in role_names], axis=0)
    v = mu_default - grand_role
    norm = float(np.linalg.norm(v))
    if norm == 0:
        return None
    return v / norm


def _train_pca(X_train, k):
    """Centre on training rows and return (mean, components) for k PCs."""
    mean = X_train.mean(axis=0)
    centred = X_train - mean
    # full_matrices=False keeps this cheap at hidden=4096 with n_rows << 4096.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return mean, vt[:k]


def _train_scale(Z):
    """Per-feature training standard deviations, with zeros neutralised.

    Principal components differ in scale by orders of magnitude, and an L2
    penalty applied to raw components would penalise a low-variance direction
    far more than a high-variance one — the penalty strength would then mean
    something different for every k. Standardising first makes the frozen C
    grid comparable across candidates. Scales come from training rows only.
    """
    scale = np.asarray(Z).std(axis=0)
    return np.where(scale == 0, 1.0, scale)


def _logistic(C, seed):
    from sklearn.linear_model import LogisticRegression

    # L2 is the solver default across the pinned scikit-learn range; naming it
    # explicitly is deprecated in later versions, so it is left implicit.
    # Class balance comes from the role-balanced sample weights, which already
    # equalise the two classes, so class_weight would double-count it.
    return LogisticRegression(C=C, solver="lbfgs", max_iter=2000,
                              random_state=seed)


def nonlinear_param_count(k, hidden):
    """Trainable parameters of the one-hidden-layer read-out on k inputs."""
    return k * hidden + hidden + hidden + 1


def _balanced_index(y, roles):
    """Row indices realising the role-balanced design by deterministic repetition.

    ``MLPClassifier`` accepts neither sample weights nor class weights, so the
    weighting used by the linear candidates is realised explicitly in the row set.
    Every role/default condition receives exactly the same number of selected rows
    within its class, and the two classes receive exactly the same total number of
    rows. The shared class total is the smallest value at least as large as either
    raw class size that is divisible by both classes' numbers of groups; this avoids
    the partial-prefix repetition bug caused by resizing a whole class vector.
    """
    y = np.asarray(y, dtype=int)
    roles = np.asarray(roles, dtype=object)

    groups = {}
    for cls in (0, 1):
        in_class = np.flatnonzero(y == cls)
        if len(in_class) == 0:
            continue
        names = sorted({str(r) for r in roles[in_class]})
        groups[cls] = [np.flatnonzero((y == cls) & (roles == name)) for name in names]

    if not groups:
        return np.arange(len(y))
    if len(groups) < 2:
        only = next(iter(groups.values()))
        target_per_group = max(len(v) for v in only)
        return np.concatenate([np.resize(v, target_per_group) for v in only])

    n_groups = {cls: len(v) for cls, v in groups.items()}
    class_sizes = {cls: int((y == cls).sum()) for cls in groups}
    common_multiple = math.lcm(*n_groups.values())
    raw_target = max(class_sizes.values())
    class_target = ((raw_target + common_multiple - 1) // common_multiple) * common_multiple

    balanced = []
    for cls in sorted(groups):
        per_group = class_target // n_groups[cls]
        balanced.extend(np.resize(idx, per_group) for idx in groups[cls])
    return np.concatenate(balanced)


class AxisReadout:
    """The fixed one-direction candidate: project onto the fold's Axis.

    AUROC of a monotone one-dimensional score does not depend on the
    calibration fitted on top of it, so this is the honest single-direction
    baseline; the logistic layer exists only so the probability-scale outputs
    are comparable in form with the other candidates.
    """

    name = "axis_1d"

    def __init__(self, seed=0):
        self.seed = seed
        self.axis = None
        self.clf = None

    def fit(self, X, y, roles, questions, weights=None):
        self.axis = _fold_axis(X, y, roles)
        if self.axis is None:
            raise RuntimeError("fold cannot support an Axis (missing a class)")
        self.clf = _logistic(1.0, self.seed).fit(
            (X @ self.axis)[:, None], y, sample_weight=weights)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba((X @ self.axis)[:, None])[:, 1]

    @property
    def n_params(self):
        return 2


class PCALinearReadout:
    """Linear read-out on the leading k principal components, fitted on the
    fold's training rows only."""

    def __init__(self, k, C, seed=0):
        self.k, self.C, self.seed = k, C, seed
        self.name = f"pca{k}_linear"

    def fit(self, X, y, roles, questions, weights=None):
        self.mean_, self.comps_ = _train_pca(X, self.k)
        self.scale_ = _train_scale((X - self.mean_) @ self.comps_.T)
        self.clf = _logistic(self.C, self.seed).fit(
            self._project(X), y, sample_weight=weights)
        return self

    def _project(self, X):
        return ((X - self.mean_) @ self.comps_.T) / self.scale_

    def predict_proba(self, X):
        return self.clf.predict_proba(self._project(X))[:, 1]

    @property
    def n_params(self):
        return self.k + 1


class FullLinearReadout:
    """Regularised linear read-out on the full activation vector. Its
    capacity is controlled by the penalty strength, chosen on training folds,
    not by a parameter budget."""

    def __init__(self, C, seed=0):
        self.C, self.seed = C, seed
        self.name = "full_linear"

    def fit(self, X, y, roles, questions, weights=None):
        self.mean_ = X.mean(axis=0)
        self.scale_ = _train_scale(X - self.mean_)
        self.clf = _logistic(self.C, self.seed).fit(
            self._project(X), y, sample_weight=weights)
        return self

    def _project(self, X):
        return (X - self.mean_) / self.scale_

    def predict_proba(self, X):
        return self.clf.predict_proba(self._project(X))[:, 1]

    @property
    def n_params(self):
        return int(self.clf.coef_.size) + 1


class SmallNonlinearReadout:
    """One hidden layer on the leading k training-fitted PCs.

    Kept deliberately small: ``nonlinear_param_count`` must stay under the
    number of training roles, and the weight-decay grid is coarse and strong.
    This bounds capacity relative to the grouping unit — it is a capacity
    control, not a proof that role-specific information cannot be represented.

    Fitted with L-BFGS rather than Adam. On data this small the stochastic
    solver's early-stopping heuristic sometimes halts on a degenerate
    inverted solution, and that heuristic would in any case be a second,
    undeclared capacity control. With L-BFGS the capacity story is exactly
    the declared one: the parameter budget and ``alpha``.
    """

    def __init__(self, k, hidden, alpha, seed=0):
        self.k, self.hidden, self.alpha, self.seed = k, hidden, alpha, seed
        self.name = f"pca{k}_mlp{hidden}"

    def fit(self, X, y, roles, questions, weights=None):
        from sklearn.neural_network import MLPClassifier

        budget = len({str(r) for r in roles[y == 0]})
        if nonlinear_param_count(self.k, self.hidden) >= budget:
            raise ValueError(
                f"parameter budget exceeded: {nonlinear_param_count(self.k, self.hidden)} "
                f"params vs {budget} training roles"
            )
        self.mean_, self.comps_ = _train_pca(X, self.k)
        self.scale_ = _train_scale((X - self.mean_) @ self.comps_.T)
        bal = _balanced_index(y, roles)
        self.clf = MLPClassifier(
            hidden_layer_sizes=(self.hidden,),
            alpha=self.alpha,
            solver="lbfgs",
            max_iter=2000,
            random_state=self.seed,
        ).fit(self._project(X)[bal], y[bal])
        return self

    def _project(self, X):
        return ((X - self.mean_) @ self.comps_.T) / self.scale_

    def predict_proba(self, X):
        return self.clf.predict_proba(self._project(X))[:, 1]

    @property
    def n_params(self):
        return nonlinear_param_count(self.k, self.hidden)


FAMILIES = {
    "axis_1d": lambda n: [("axis_1d", lambda s: AxisReadout(seed=s))],
    "pca_linear": lambda n: [
        (f"pca{k}_linear_C{C}", lambda s, k=k, C=C: PCALinearReadout(k, C, seed=s))
        for k in PCA_K_GRID for C in LINEAR_C_GRID
    ],
    "full_linear": lambda n: [
        (f"full_linear_C{C}", lambda s, C=C: FullLinearReadout(C, seed=s))
        for C in LINEAR_C_GRID
    ],
    "small_nonlinear": lambda n: [
        (f"pca{k}_mlp{h}_a{a}",
         lambda s, k=k, h=h, a=a: SmallNonlinearReadout(k, h, a, seed=s))
        for k in PCA_K_GRID for h in NONLINEAR_HIDDEN_GRID
        for a in NONLINEAR_ALPHA_GRID
        if nonlinear_param_count(k, h) < n
    ],
}


def candidate_grid(n_train_roles):
    """Every frozen candidate that fits the capacity rule.

    The nonlinear grid shrinks as training roles get scarcer, which is the
    intended behaviour: a fold with few roles is not allowed a bigger model
    than a fold with many.
    """
    grid = []
    for family in ("axis_1d", *VERDICT_FAMILIES):
        grid.extend(FAMILIES[family](n_train_roles))
    return grid


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc(y, score, weights=None):
    """Weighted AUROC, ties counted as half, computed by rank sums.

    Weights let a role contribute the same total mass regardless of how many
    retained responses it has, matching the Axis's equal-weighting of roles.
    Vectorised over tie groups so the bootstrap stays affordable.
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)

    pos, neg = y == 1, y == 0
    w_pos, w_neg = w[pos].sum(), w[neg].sum()
    if not pos.any() or not neg.any() or w_pos == 0 or w_neg == 0:
        return float("nan")

    _, group = np.unique(score, return_inverse=True)
    pos_per_group = np.bincount(group, weights=np.where(pos, w, 0.0))
    neg_per_group = np.bincount(group, weights=np.where(neg, w, 0.0))
    neg_below = np.cumsum(neg_per_group) - neg_per_group
    concordant = float(
        (pos_per_group * (neg_below + 0.5 * neg_per_group)).sum())
    return concordant / float(w_pos * w_neg)


def fold_aggregate_auroc(y, score, fold_id, weights=None):
    """Mean of within-fold AUROC over folds that contain both classes.

    Outer folds are fitted independently, so their probability outputs are not
    on a common scale and an AUROC over pooled scores would compare
    incomparable numbers. Ranking within a fold and then averaging keeps every
    comparison inside a single fitted model. Folds are weighted equally.
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    fold_id = np.asarray(fold_id)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)

    values = []
    for f in np.unique(fold_id):
        m = fold_id == f
        value = auroc(y[m], score[m], w[m])
        if not np.isnan(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def balanced_accuracy(y, score, threshold=0.5):
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(score) >= threshold).astype(int)
    tpr = float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan")
    tnr = float((pred[y == 0] == 0).mean()) if (y == 0).any() else float("nan")
    return (tpr + tnr) / 2.0


# ---------------------------------------------------------------------------
# Fitting: inner selection, outer evaluation
# ---------------------------------------------------------------------------

def _select_on_training(rows, train_idx, family_builders, inner_folds, seed):
    """Choose one hyper-parameter setting using only the fold's training rows.

    The inner split is doubly grouped in the same way as the outer one, so a
    setting cannot be chosen because it exploits a role or question that the
    outer test set will later reuse. Selection uses the same fold-aggregated,
    role-balanced statistic as the primary estimand.
    """
    sub = ReadoutRows(rows.X[train_idx], rows.roles[train_idx],
                      rows.questions[train_idx], rows.y[train_idx])
    inner = doubly_grouped_folds(sub, *inner_folds, seed=seed + 977)

    best, best_score = None, -np.inf
    for label, build in family_builders:
        scores = []
        for f in inner:
            tr, te = f["train_idx"], f["test_idx"]
            if len(te) == 0 or len(tr) == 0:
                continue
            if len(set(sub.y[tr])) < 2 or len(set(sub.y[te])) < 2:
                continue
            try:
                model = build(seed).fit(sub.X[tr], sub.y[tr], sub.roles[tr],
                                        sub.questions[tr], sub.weights[tr])
                scores.append(auroc(sub.y[te], model.predict_proba(sub.X[te]),
                                    sub.weights[te]))
            except (ValueError, RuntimeError):
                # A setting that cannot be fitted on an inner fold — an
                # exceeded budget, a degenerate Axis — is not eligible.
                scores = []
                break
        if scores and np.mean(scores) > best_score:
            best, best_score = (label, build), float(np.mean(scores))
    if best is None:
        raise RuntimeError("no candidate in this family could be fitted")
    return best[0], best[1], best_score


def run_family(rows, family, folds=None, inner_folds=None, seed=SPLIT_SEED,
               builders=None):
    """Out-of-fold predictions and fold identities for one candidate family.

    Returns ``(predictions, fold_of_row, selections)``. ``fold_of_row`` is
    required: predictions from different folds come from independently fitted
    models and may only be compared within a fold.

    ``builders`` overrides the frozen grid. Production runs leave it unset — a
    run that narrowed its own grid would no longer be predeclared — but the
    validation suite passes reduced grids to keep synthetic checks fast.
    """
    folds = OUTER_FOLDS if folds is None else folds
    inner_folds = INNER_FOLDS if inner_folds is None else inner_folds
    n_roles = len(rows.role_roles)
    builders = FAMILIES[family](n_roles) if builders is None else builders
    if not builders:
        raise RuntimeError(f"family {family} has no candidate within the budget")

    oof = np.full(len(rows), np.nan)
    fold_of_row = np.full(len(rows), -1, dtype=int)
    selections = []
    for fold in doubly_grouped_folds(rows, *folds, seed=seed):
        check_fold_isolation(rows, fold)
        tr, te = fold["train_idx"], fold["test_idx"]
        if len(te) == 0 or len(tr) == 0 or len(set(rows.y[tr])) < 2:
            continue
        label, build, inner_score = _select_on_training(
            rows, tr, builders, inner_folds, seed)
        model = build(seed).fit(rows.X[tr], rows.y[tr], rows.roles[tr],
                                rows.questions[tr], rows.weights[tr])
        oof[te] = model.predict_proba(rows.X[te])
        fold_of_row[te] = fold["fold_id"]
        selections.append({
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
        })
    return oof, fold_of_row, selections


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

def _cluster_index(rows, scored, cluster):
    """Group the scored rows by resampling unit, split into class strata.

    Clusters are resampled within class so every replicate keeps both classes
    and the statistic stays defined. Questions appear under both defaults and
    roles, so they do not split by class and are resampled as one pool.
    """
    keys = (rows.roles[scored] if cluster == "role"
            else rows.questions[scored].astype(object))
    by_key = {}
    for pos, k in enumerate(keys):
        by_key.setdefault(k, []).append(pos)
    by_key = {k: np.asarray(v) for k, v in by_key.items()}

    y = rows.y[scored]
    if cluster != "role":
        return by_key, list(by_key), []
    pos_keys = [k for k, v in by_key.items() if y[v][0] == 1]
    neg_keys = [k for k, v in by_key.items() if y[v][0] != 1]
    return by_key, pos_keys, neg_keys


def _resamples(by_key, pos_keys, neg_keys, replicates, seed):
    """Yield row-index arrays for each cluster bootstrap replicate."""
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        picks = list(rng.choice(pos_keys, len(pos_keys), replace=True))
        if neg_keys:
            picks += list(rng.choice(neg_keys, len(neg_keys), replace=True))
        yield np.concatenate([by_key[k] for k in picks])


def cluster_bootstrap(rows, predictions, fold_of_row,
                      replicates=BOOTSTRAP_REPLICATES, seed=SPLIT_SEED,
                      cluster=INFERENCE_CLUSTER):
    """Percentile CI for the fold-aggregated AUROC, resampling whole groups.

    Rows within a role (or a question) are not independent, so the resampling
    unit is the group, not the row. Every replicate recomputes the same
    fold-aggregated, role-balanced statistic as the point estimate.
    """
    scored = np.flatnonzero(~np.isnan(predictions))
    y, p = rows.y[scored], predictions[scored]
    f, w = fold_of_row[scored], rows.weights[scored]
    by_key, pos_keys, neg_keys = _cluster_index(rows, scored, cluster)

    draws, skipped = [], 0
    for idx in _resamples(by_key, pos_keys, neg_keys, replicates, seed):
        value = fold_aggregate_auroc(y[idx], p[idx], f[idx], w[idx])
        if np.isnan(value):
            skipped += 1
            continue
        draws.append(value)

    draws = np.asarray(draws)
    return {
        "point": fold_aggregate_auroc(y, p, f, w),
        "ci_low": float(np.percentile(draws, 2.5)) if len(draws) else float("nan"),
        "ci_high": float(np.percentile(draws, 97.5)) if len(draws) else float("nan"),
        "replicates_used": int(len(draws)),
        "replicates_skipped": int(skipped),
        "n_scored_rows": int(len(scored)),
        "n_folds": int(len(np.unique(f))),
        "cluster": cluster,
        "estimand": "mean within-fold role-balanced AUROC",
    }


def simultaneous_family_bootstrap(rows, family_predictions, axis_predictions,
                                  fold_of_row,
                                  replicates=BOOTSTRAP_REPLICATES,
                                  seed=SPLIT_SEED, cluster=INFERENCE_CLUSTER):
    """Simultaneous interval for ``max_family (AUROC_family − AUROC_axis)``.

    Taking the maximum across families *inside* each replicate is what makes
    the interval simultaneous. Reporting a per-family interval and then acting
    on whichever family looked best would select on the data and understate
    the chance of a spurious `INADEQUATE`, since any one of several
    alternatives could trigger it.

    Pairing also earns its keep here: all families see the same default
    conditions in a replicate, so that large shared variance component cancels
    in the difference.
    """
    if not family_predictions:
        raise ValueError("no candidate families to compare against the Axis")

    finite = ~np.isnan(axis_predictions)
    for pred in family_predictions.values():
        finite &= ~np.isnan(pred)
    scored = np.flatnonzero(finite)
    y, f, w = rows.y[scored], fold_of_row[scored], rows.weights[scored]
    axis = axis_predictions[scored]
    families = {k: v[scored] for k, v in family_predictions.items()}
    by_key, pos_keys, neg_keys = _cluster_index(rows, scored, cluster)

    def max_gain(idx):
        base = fold_aggregate_auroc(y[idx], axis[idx], f[idx], w[idx])
        gains = {
            name: fold_aggregate_auroc(y[idx], pred[idx], f[idx], w[idx]) - base
            for name, pred in families.items()
        }
        best = max(gains, key=gains.get)
        return gains[best], best, gains

    point, best_family, per_family_point = max_gain(np.arange(len(scored)))

    draws, per_family_draws = [], {name: [] for name in families}
    for idx in _resamples(by_key, pos_keys, neg_keys, replicates, seed):
        value, _, gains = max_gain(idx)
        if np.isnan(value):
            continue
        draws.append(value)
        for name, gain in gains.items():
            per_family_draws[name].append(gain)

    draws = np.asarray(draws)

    def interval(values):
        values = np.asarray(values)
        if not len(values):
            return float("nan"), float("nan")
        return (float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)))

    low, high = interval(draws)
    return {
        "max_family_difference": point,
        "ci_low": low,
        "ci_high": high,
        "best_family": best_family,
        "replicates_used": int(len(draws)),
        "cluster": cluster,
        "simultaneous_over": sorted(families),
        "per_family": {
            name: {
                "difference": per_family_point[name],
                "ci_low": interval(per_family_draws[name])[0],
                "ci_high": interval(per_family_draws[name])[1],
                "note": "marginal interval, not simultaneous; "
                        "the verdict uses the max-family interval",
            }
            for name in families
        },
    }


def adequacy_verdict(simultaneous, margin=ADEQUACY_MARGIN):
    """Turn the simultaneous max-family interval into the T18 verdict.

    The decision is an equivalence one:

    * ``ADEQUATE``     — the simultaneous interval lies entirely below
      ``margin``, so no alternative gains materially and the data are
      consistent with one direction being a sufficient summary;
    * ``INADEQUATE``   — the interval lies entirely above ``margin``, so the
      best alternative is materially better even after accounting for having
      selected it;
    * ``INCONCLUSIVE`` — the interval straddles ``margin``; the study cannot
      separate the two, which with only five default conditions is a real
      possibility and must be reported as such rather than rounded down to
      "adequate".
    """
    if simultaneous.get("cluster") != INFERENCE_CLUSTER:
        raise ValueError(
            f"interval was summarised with cluster="
            f"{simultaneous.get('cluster')!r}; only {INFERENCE_CLUSTER!r} "
            "clustering may support a verdict"
        )
    low, high = simultaneous["ci_low"], simultaneous["ci_high"]
    if low > margin:
        verdict = "INADEQUATE"
    elif high < margin:
        verdict = "ADEQUATE"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "margin": margin,
        "best_alternative": simultaneous["best_family"],
        "max_family_difference": simultaneous["max_family_difference"],
        "simultaneous_ci": [low, high],
        "simultaneous_over": simultaneous["simultaneous_over"],
        "cluster": INFERENCE_CLUSTER,
        "t18_config_sha256": T18_CONFIG_SHA256,
    }


# ---------------------------------------------------------------------------
# Null calibration
# ---------------------------------------------------------------------------

def permute_labels_by_role(rows, seed):
    """Reassign the Assistant/role label at the level of whole groups.

    Permuting row labels would leave a role's rows split across both classes
    and let any read-out score above chance from role identity alone. Moving
    whole roles keeps the group structure intact, so a correctly grouped
    pipeline must land at chance — which is what makes this a usable leakage
    diagnostic rather than a formality.
    """
    names = np.array(rows.role_roles + rows.default_roles, dtype=object)
    labels = np.array([0] * len(rows.role_roles) + [1] * len(rows.default_roles))
    shuffled = np.random.default_rng(seed).permutation(labels)
    mapping = dict(zip(names, shuffled))

    new_y = np.array([mapping[str(r)] for r in rows.roles])
    # Rename so the container's label/name invariant still holds under the
    # permuted assignment.
    new_roles = np.array(
        [f"{DEFAULT_ROLE_PREFIX}perm_{r}" if mapping[str(r)] == 1
         else str(r).replace(DEFAULT_ROLE_PREFIX, "permrole_")
         for r in rows.roles],
        dtype=object,
    )
    return ReadoutRows(rows.X, new_roles, rows.questions, new_y)
