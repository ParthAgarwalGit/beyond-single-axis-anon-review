"""Validation of the T18 read-out pipeline on synthetic data with known
answers.

The point of T18 is to say whether one direction is enough. That verdict is
only worth reporting if the pipeline would have said something different had
the truth been different, and would have said "nothing here" when nothing was
there. These tests pin all four corners:

* a genuinely one-dimensional world -> extra dimensions buy nothing;
* a multidimensional linear world  -> the linear read-outs beat the Axis;
* a nonlinear (radial) world       -> only the nonlinear read-out succeeds;
* a role-identity-only world       -> everything lands at chance, and the
  grouped split is demonstrably what forces that.

They also pin the four properties that make those verdicts mean anything:
grouped folds, fold-local fitting, a fold-aggregated estimand that never
compares scores across independently fitted models, and role-balanced
weighting that matches the Axis's equal weighting of roles.

The synthetic dimensionality and role counts are small so the suite stays on
CPU in seconds; the reduced candidate grids passed via ``builders`` are a
test-speed device only and production runs use the frozen grids.
"""

import numpy as np
import pytest

from src.t18_readouts import (
    ADEQUACY_MARGIN,
    DEFAULT_ROLE_PREFIX,
    INFERENCE_CLUSTER,
    AxisReadout,
    FullLinearReadout,
    PCALinearReadout,
    ReadoutRows,
    SmallNonlinearReadout,
    adequacy_verdict,
    auroc,
    candidate_grid,
    check_fold_isolation,
    cluster_bootstrap,
    doubly_grouped_folds,
    fold_aggregate_auroc,
    nonlinear_param_count,
    permute_labels_by_role,
    role_balance_weights,
    run_family,
    simultaneous_family_bootstrap,
)

N_ROLES = 60
N_QUESTIONS = 12
N_DEFAULT_CONDITIONS = 5
REPS_PER_DEFAULT = 4


def _assemble(role_rows, default_rows, dim):
    """Stack (vector, role, question) triples into a validated container."""
    X = np.vstack([r[0] for r in role_rows] + [d[0] for d in default_rows])
    roles = np.array([r[1] for r in role_rows] + [d[1] for d in default_rows],
                     dtype=object)
    qs = np.array([r[2] for r in role_rows] + [d[2] for d in default_rows])
    y = np.array([0] * len(role_rows) + [1] * len(default_rows))
    assert X.shape[1] == dim
    return ReadoutRows(X, roles, qs, y)


def _make(dim, role_centre, default_centre, noise, seed, reps_per_role=None):
    """Common scaffold: every role answers every question; every default
    condition answers every question several times.

    ``reps_per_role`` may vary retention per role, which is how the
    role-balancing tests create the imbalance they check for.
    """
    rng = np.random.default_rng(seed)
    role_rows, default_rows = [], []
    for r in range(N_ROLES):
        centre = role_centre(rng, r)
        reps = 1 if reps_per_role is None else reps_per_role(r)
        for q in range(N_QUESTIONS):
            for _ in range(reps):
                role_rows.append((centre + noise(rng), f"role_{r:03d}", q))
    for c in range(N_DEFAULT_CONDITIONS):
        centre = default_centre(rng, c)
        for q in range(N_QUESTIONS):
            for _ in range(REPS_PER_DEFAULT):
                default_rows.append(
                    (centre + noise(rng), f"{DEFAULT_ROLE_PREFIX}{c}", q))
    return _assemble(role_rows, default_rows, dim)


# ---------------------------------------------------------------------------
# Synthetic regimes
# ---------------------------------------------------------------------------

def one_dimensional_world(seed=1, dim=16, gap=3.0, reps_per_role=None):
    """Class is decided by position along a single axis; every other
    coordinate is pure noise. One direction is genuinely sufficient."""
    u = np.zeros(dim)
    u[0] = 1.0
    return _make(
        dim,
        role_centre=lambda rng, r: -gap / 2 * u + 0.3 * rng.normal(size=dim),
        default_centre=lambda rng, c: gap / 2 * u + 0.3 * rng.normal(size=dim),
        noise=lambda rng: rng.normal(scale=1.0, size=dim),
        seed=seed,
        reps_per_role=reps_per_role,
    )


def multidimensional_linear_world(seed=2, dim=16):
    """A wide, noisy separation on coordinate 0 and four moderate, clean
    separations on coordinates 1-4.

    The difference in means is dominated by coordinate 0 because that gap is
    the largest, but coordinate 0 also carries by far the worst
    signal-to-noise. The optimal linear rule weights each quiet coordinate
    about six times more heavily, so the fixed one-direction score is
    provably suboptimal even though the truth is perfectly linear. The
    informative coordinates are also the highest-variance ones, so they
    survive the unsupervised PCA step: this regime isolates *dimensionality*,
    not PCA's well-known blindness to low-variance discriminative directions.
    """
    gaps = np.zeros(dim)
    gaps[0] = 3.0
    gaps[1:5] = 1.2
    sd = np.full(dim, 0.5)   # nuisance coordinates, ranked below the signal
    sd[0] = 3.0
    sd[1:5] = 0.8

    return _make(
        dim,
        role_centre=lambda rng, r: -gaps / 2 + 0.15 * sd * rng.normal(size=dim),
        default_centre=lambda rng, c: gaps / 2 + 0.15 * sd * rng.normal(size=dim),
        noise=lambda rng: sd * rng.normal(size=dim),
        seed=seed,
    )


def radial_world(seed=3, dim=8, radius=4.0):
    """Defaults sit in a ball at the origin; roles sit on a shell around it.

    Role centres are spread over the shell, so the mean role vector is close
    to the default mean and the difference in means is near-degenerate. No
    linear rule separates inside from outside; the distance from the origin
    does. This is the only regime where the nonlinear read-out should win.
    """
    def role_centre(rng, r):
        g = rng.normal(size=2)
        c = np.zeros(dim)
        c[:2] = radius * g / np.linalg.norm(g)
        return c

    return _make(
        dim,
        role_centre=role_centre,
        default_centre=lambda rng, c: np.zeros(dim),
        noise=lambda rng: np.concatenate(
            [rng.normal(scale=0.6, size=2), rng.normal(scale=0.6, size=dim - 2)]),
        seed=seed,
    )


def role_identity_only_world(seed=4, dim=16):
    """Class is decided by role identity alone: every role has a random
    centre and the label is attached to the role, not to any geometry that
    transfers. A pipeline that generalises across roles must score chance."""
    rng0 = np.random.default_rng(seed)
    centres = {r: rng0.normal(scale=2.0, size=dim) for r in range(N_ROLES)}
    d_centres = {c: rng0.normal(scale=2.0, size=dim)
                 for c in range(N_DEFAULT_CONDITIONS)}
    return _make(
        dim,
        role_centre=lambda rng, r: centres[r],
        default_centre=lambda rng, c: d_centres[c],
        noise=lambda rng: rng.normal(scale=0.5, size=dim),
        seed=seed,
    )


# Reduced grids: enough spread to distinguish the regimes, small enough to run
# in seconds. Production uses the frozen grids in the module.
FAST_LINEAR = [(f"pca{k}_linear_C{c}",
                lambda s, k=k, c=c: PCALinearReadout(k, c, seed=s))
               for k in (1, 3, 8) for c in (0.01, 1.0)]
FAST_FULL = [(f"full_linear_C{c}", lambda s, c=c: FullLinearReadout(c, seed=s))
             for c in (0.01, 1.0)]
FAST_NONLINEAR = [(f"pca{k}_mlp{h}_a{a}",
                   lambda s, k=k, h=h, a=a: SmallNonlinearReadout(k, h, a, seed=s))
                  for k in (2, 3) for h in (4, 8) for a in (0.1, 1.0)]
FAST_AXIS = [("axis_1d", lambda s: AxisReadout(seed=s))]
FAST = dict(folds=(4, 3), inner_folds=(2, 2))
VERDICT_GRIDS = {"pca_linear": FAST_LINEAR, "full_linear": FAST_FULL,
                 "small_nonlinear": FAST_NONLINEAR}


def _oof(rows, builders):
    """Out-of-fold predictions, fold identities, and per-fold selections."""
    return run_family(rows, "unused", builders=builders, **FAST)


def _score(rows, builders):
    """The primary estimand for one family: fold-aggregated, role-balanced."""
    pred, fold_of_row, _ = _oof(rows, builders)
    return fold_aggregate_auroc(rows.y, pred, fold_of_row, rows.weights)


# ---------------------------------------------------------------------------
# Splitting and leakage
# ---------------------------------------------------------------------------

def test_every_row_is_tested_exactly_once():
    rows = one_dimensional_world()
    seen = np.zeros(len(rows), dtype=int)
    for fold in doubly_grouped_folds(rows, 5, 4):
        seen[fold["test_idx"]] += 1
    assert (seen == 1).all()


def test_folds_hold_out_roles_and_questions_together():
    rows = one_dimensional_world()
    for fold in doubly_grouped_folds(rows, 5, 4):
        check_fold_isolation(rows, fold)
        assert len(fold["test_idx"]) > 0
        # Dropping the roles-only and questions-only remainders is the cost of
        # holding both out at once; it must be real, not zero.
        assert fold["n_dropped"] > 0


def test_fold_isolation_detects_a_planted_leak():
    rows = one_dimensional_world()
    fold = doubly_grouped_folds(rows, 5, 4)[0]
    leaky = dict(fold, train_idx=np.append(fold["train_idx"], fold["test_idx"][0]))
    with pytest.raises(RuntimeError, match="role leakage"):
        check_fold_isolation(rows, leaky)


def test_fold_assignment_ignores_row_order():
    rows = one_dimensional_world()
    perm = np.random.default_rng(0).permutation(len(rows))
    shuffled = ReadoutRows(rows.X[perm], rows.roles[perm], rows.questions[perm],
                           rows.y[perm])
    a = {tuple(sorted(rows.roles[f["test_idx"]]))
         for f in doubly_grouped_folds(rows, 5, 4)}
    b = {tuple(sorted(shuffled.roles[f["test_idx"]]))
         for f in doubly_grouped_folds(shuffled, 5, 4)}
    assert a == b


def test_default_conditions_are_spread_across_role_folds():
    rows = one_dimensional_world()
    folds = doubly_grouped_folds(rows, N_DEFAULT_CONDITIONS, 4)
    held_out = {f["role_fold"]: set(rows.roles[f["test_idx"]]) for f in folds}
    for i in range(N_DEFAULT_CONDITIONS):
        assert any(str(r).startswith(DEFAULT_ROLE_PREFIX) for r in held_out[i]), (
            "a fold with no held-out default rows cannot test Assistant status"
        )


# ---------------------------------------------------------------------------
# Container, weighting, and capacity invariants
# ---------------------------------------------------------------------------

def test_rows_reject_label_and_name_disagreement():
    X = np.zeros((4, 3))
    with pytest.raises(ValueError, match="synthetic"):
        ReadoutRows(X, ["role_a"] * 4, [0, 1, 2, 3], [0, 0, 1, 0])


def test_rows_reject_non_finite_activations():
    X = np.zeros((2, 3))
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ReadoutRows(X, ["role_a", f"{DEFAULT_ROLE_PREFIX}0"], [0, 1], [0, 1])


def test_role_balance_weights_equalise_roles_and_classes():
    """A role with ten retained responses must carry the same total weight as
    a role with one, because the Axis equal-weights roles."""
    roles = np.array(["a"] * 10 + ["b"] + [f"{DEFAULT_ROLE_PREFIX}0"] * 3,
                     dtype=object)
    y = np.array([0] * 11 + [1] * 3)
    w = role_balance_weights(roles, y)

    assert np.isclose(w[roles == "a"].sum(), w[roles == "b"].sum())
    assert np.isclose(w[y == 0].sum(), w[y == 1].sum())
    assert np.isclose(w.sum(), len(y))


def test_role_balancing_stops_one_prolific_role_dominating():
    """A single role with forty times the retention must not be able to drag
    the estimand around, which is what row-level weighting would allow."""
    rows = one_dimensional_world(
        reps_per_role=lambda r: 40 if r == 0 else 1)
    prolific = rows.roles == "role_000"
    assert prolific.sum() / (rows.y == 0).sum() > 0.35, "fixture is not skewed"

    share = rows.weights[prolific].sum() / rows.weights[rows.y == 0].sum()
    assert np.isclose(share, 1 / N_ROLES, atol=1e-9), (
        f"the prolific role holds {share:.3f} of the role-class weight; "
        f"equal weighting requires {1 / N_ROLES:.3f}"
    )


def test_capacity_rule_excludes_models_larger_than_the_training_role_count():
    small = {n for n, _ in candidate_grid(n_train_roles=30) if "mlp" in n}
    large = {n for n, _ in candidate_grid(n_train_roles=300) if "mlp" in n}
    assert small < large, "a smaller role budget must permit strictly fewer models"
    for name in small:
        k = int(name.split("_")[0][3:])
        h = int(name.split("_")[1][3:])
        assert nonlinear_param_count(k, h) < 30


def test_nonlinear_readout_refuses_to_fit_over_budget():
    rows = one_dimensional_world()
    idx = np.arange(len(rows))
    # 32 PCs x 16 hidden is 545 parameters against 60 roles.
    with pytest.raises(ValueError, match="parameter budget"):
        SmallNonlinearReadout(32, 16, 1.0).fit(
            rows.X[idx], rows.y[idx], rows.roles[idx], rows.questions[idx],
            rows.weights[idx])


def test_axis_weights_default_conditions_equally():
    from src.t18_readouts import _fold_axis

    # One condition supplies 100 rows, the others one each. Equal condition
    # weighting must stop the crowded condition from dominating the default
    # mean, matching the frozen default-vector rule.
    X = np.vstack([np.tile([10.0, 0.0], (100, 1)),
                   np.zeros((4, 2)),
                   np.array([[0.0, -1.0]])])
    roles = np.array([f"{DEFAULT_ROLE_PREFIX}0"] * 100
                     + [f"{DEFAULT_ROLE_PREFIX}{i}" for i in range(1, 5)]
                     + ["role_a"], dtype=object)
    y = np.array([1] * 104 + [0])
    axis = _fold_axis(X, y, roles)
    assert np.isclose(axis @ np.array([1.0, 0.0]), 2.0 / np.linalg.norm([2.0, 1.0]))


# ---------------------------------------------------------------------------
# Metric and estimand
# ---------------------------------------------------------------------------

def test_auroc_matches_sklearn_including_ties():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    for _ in range(5):
        y = rng.integers(0, 2, 200)
        s = np.round(rng.normal(size=200), 1)  # rounding forces ties
        assert np.isclose(auroc(y, s), roc_auc_score(y, s))


def test_weighted_auroc_matches_sklearn_sample_weights():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    for _ in range(5):
        y = rng.integers(0, 2, 200)
        s = np.round(rng.normal(size=200), 1)
        w = rng.uniform(0.1, 5.0, 200)
        assert np.isclose(auroc(y, s, w), roc_auc_score(y, s, sample_weight=w))


def test_auroc_is_undefined_for_a_single_class():
    assert np.isnan(auroc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_fold_aggregation_does_not_compare_scores_across_folds():
    """Why the estimand is fold-aggregated rather than pooled.

    Each fold's model is fitted and calibrated independently, so the same
    probability can mean different things in different folds. Here every fold
    ranks its own rows perfectly, but the fold's probability band is shifted,
    so pooling the scores destroys the ordering. Fold aggregation reports the
    truth; pooling does not.
    """
    y = np.array([0, 0, 1, 1] * 3)
    fold_id = np.repeat([0, 1, 2], 4)
    # Within every fold the positives outrank the negatives, but fold 0's
    # scores sit entirely above fold 2's positives.
    score = np.concatenate([[0.90, 0.91, 0.92, 0.93],
                            [0.50, 0.51, 0.52, 0.53],
                            [0.10, 0.11, 0.12, 0.13]])

    assert fold_aggregate_auroc(y, score, fold_id) == 1.0
    assert auroc(y, score) < 0.7, (
        "the pooled statistic should be badly misleading here, which is the "
        "whole reason pooling is forbidden"
    )


def test_fold_aggregation_ignores_single_class_folds():
    y = np.array([0, 1, 1, 1])
    fold_id = np.array([0, 0, 1, 1])   # fold 1 has no negatives
    score = np.array([0.1, 0.9, 0.4, 0.6])
    assert fold_aggregate_auroc(y, score, fold_id) == 1.0


# ---------------------------------------------------------------------------
# The four regimes
# ---------------------------------------------------------------------------

def _verdict_for(rows, grids=None):
    """Run each verdict family and take the simultaneous verdict."""
    grids = grids or VERDICT_GRIDS
    axis, fold_of_row, _ = _oof(rows, FAST_AXIS)
    preds = {name: _oof(rows, builders)[0] for name, builders in grids.items()}
    simultaneous = simultaneous_family_bootstrap(
        rows, preds, axis, fold_of_row, replicates=400)
    return adequacy_verdict(simultaneous), simultaneous


def test_one_dimensional_world_is_judged_adequate():
    rows = one_dimensional_world()
    verdict, simultaneous = _verdict_for(rows)

    assert _score(rows, FAST_AXIS) > 0.85, "a real 1-D separation must be recovered"
    assert verdict["verdict"] == "ADEQUATE", (
        f"one direction is genuinely sufficient here; got {verdict} "
        f"from {simultaneous['per_family']}"
    )


def test_multidimensional_linear_world_is_judged_inadequate():
    rows = multidimensional_linear_world()
    verdict, simultaneous = _verdict_for(rows)

    assert verdict["verdict"] == "INADEQUATE", (
        f"genuine multidimensional linear structure must be detected; "
        f"got {simultaneous['per_family']}"
    )
    assert simultaneous["max_family_difference"] > 0.05


def test_radial_world_is_found_only_by_the_nonlinear_readout():
    rows = radial_world()
    assert _score(rows, FAST_AXIS) < 0.75, "a radial truth has no good linear summary"
    assert _score(rows, FAST_LINEAR) < 0.75
    assert _score(rows, FAST_FULL) < 0.75
    assert _score(rows, FAST_NONLINEAR) > 0.90, "the nonlinear read-out must find it"

    verdict, simultaneous = _verdict_for(rows)
    assert verdict["verdict"] == "INADEQUATE"
    assert verdict["best_alternative"] == "small_nonlinear", (
        "the nonlinear read-out, not a linear one, must be named as the "
        f"candidate that beats the Axis; got {verdict}"
    )
    per_family = simultaneous["per_family"]
    assert (per_family["small_nonlinear"]["difference"]
            > per_family["pca_linear"]["difference"])


def test_every_verdict_family_is_covered_by_the_known_answer_worlds():
    """No family may determine a real verdict without having been validated.

    The frozen config lists the families that can move a verdict; this fails
    if one of them is ever added to production without a synthetic world
    exercising it.
    """
    from src.t18_readouts import VERDICT_FAMILIES

    assert set(VERDICT_FAMILIES) == set(VERDICT_GRIDS), (
        f"frozen verdict families {sorted(VERDICT_FAMILIES)} are not all "
        f"covered by the self-test grids {sorted(VERDICT_GRIDS)}"
    )


NULL_SEEDS = [4 + 11 * i for i in range(6)]


def test_role_identity_only_world_scores_chance_under_grouping():
    """Under the null the estimand must be *centred* on chance.

    Asserting that one 95% interval covers chance would be a coin flip with a
    one-in-twenty failure rate, and with only five default groups the
    per-replicate estimate genuinely swings between about 0.2 and 0.75. The
    testable claim is the unbiasedness of the estimator across independent
    null datasets, not its value on any single one. Interval *coverage* is
    tested separately, over many datasets, in
    ``test_question_clustering_is_anticonservative``.
    """
    for name, builders in (("axis", FAST_AXIS), ("pca_linear", FAST_LINEAR),
                           ("full_linear", FAST_FULL),
                           ("small_nonlinear", FAST_NONLINEAR)):
        scores = [_score(role_identity_only_world(seed=s), builders)
                  for s in NULL_SEEDS]
        assert abs(float(np.mean(scores)) - 0.5) < 0.10, (
            f"{name} is biased away from chance on data whose labels carry "
            f"only role identity: mean {np.mean(scores):.3f} over {scores}"
        )


def test_grouping_is_what_forces_chance_on_role_identity_data():
    """The same data scored with a row-random split, which is what a careless
    implementation would do. It must succeed there and collapse under
    grouping — otherwise the result above proves nothing about the split."""
    grouped, ungrouped = [], []
    for seed in NULL_SEEDS:
        rows = role_identity_only_world(seed=seed)
        grouped.append(_score(rows, FAST_LINEAR))

        rng = np.random.default_rng(0)
        idx = rng.permutation(len(rows))
        cut = len(rows) // 2
        tr, te = idx[:cut], idx[cut:]
        model = PCALinearReadout(8, 1.0).fit(
            rows.X[tr], rows.y[tr], rows.roles[tr], rows.questions[tr],
            rows.weights[tr])
        ungrouped.append(
            auroc(rows.y[te], model.predict_proba(rows.X[te]), rows.weights[te]))

    mean_grouped, mean_ungrouped = float(np.mean(grouped)), float(np.mean(ungrouped))
    assert mean_ungrouped > 0.80, (
        f"row-random splitting should leak role identity; got {mean_ungrouped:.3f}")
    assert abs(mean_grouped - 0.5) < 0.10, (
        f"grouped splitting should leave nothing; got {mean_grouped:.3f}")
    assert mean_ungrouped - mean_grouped > 0.25, (
        f"the grouped split must remove the leak the row split exploits "
        f"(ungrouped {mean_ungrouped:.3f}, grouped {mean_grouped:.3f})"
    )


# ---------------------------------------------------------------------------
# Multiplicity and the verdict rule
# ---------------------------------------------------------------------------

def test_simultaneous_interval_is_wider_than_any_single_family():
    """Selecting the best of several alternatives must cost something.

    The verdict acts on whichever family looks best, so the interval it uses
    has to account for that choice. A simultaneous interval no wider than the
    marginal ones would mean the maximum was being taken for free.
    """
    rows = multidimensional_linear_world()
    axis, fold_of_row, _ = _oof(rows, FAST_AXIS)
    preds = {name: _oof(rows, b)[0] for name, b in VERDICT_GRIDS.items()}
    simultaneous = simultaneous_family_bootstrap(
        rows, preds, axis, fold_of_row, replicates=400)

    for name, marginal in simultaneous["per_family"].items():
        assert simultaneous["ci_high"] >= marginal["ci_high"] - 1e-9, (
            f"simultaneous upper bound {simultaneous['ci_high']:.4f} is below "
            f"{name}'s marginal {marginal['ci_high']:.4f}"
        )
    assert simultaneous["max_family_difference"] >= max(
        m["difference"] for m in simultaneous["per_family"].values()) - 1e-9
    assert set(simultaneous["simultaneous_over"]) == set(VERDICT_GRIDS)


def test_verdict_refuses_a_non_role_clustered_interval():
    fake = {"max_family_difference": 0.3, "ci_low": 0.25, "ci_high": 0.35,
            "best_family": "pca_linear", "cluster": "question",
            "simultaneous_over": ["pca_linear"]}
    with pytest.raises(ValueError, match="only 'role'"):
        adequacy_verdict(fake)


def _interval(low, high, best="pca_linear"):
    return {"max_family_difference": (low + high) / 2, "ci_low": low,
            "ci_high": high, "best_family": best, "cluster": INFERENCE_CLUSTER,
            "simultaneous_over": ["pca_linear", "full_linear"]}


def test_verdict_is_inconclusive_when_the_interval_straddles_the_margin():
    assert adequacy_verdict(_interval(-0.01, 0.06))["verdict"] == "INCONCLUSIVE"


def test_verdict_does_not_call_a_trivial_but_certain_gain_inadequate():
    # A 0.005 AUROC gain with a tight interval excludes zero, but is far below
    # the frozen margin and must not overturn one-direction adequacy.
    assert adequacy_verdict(_interval(0.003, 0.008))["verdict"] == "ADEQUATE"


def test_verdict_uses_the_frozen_margin():
    just_over = _interval(ADEQUACY_MARGIN + 0.001, ADEQUACY_MARGIN + 0.05)
    just_under = _interval(ADEQUACY_MARGIN - 0.05, ADEQUACY_MARGIN - 0.001)
    assert adequacy_verdict(just_over)["verdict"] == "INADEQUATE"
    assert adequacy_verdict(just_under)["verdict"] == "ADEQUATE"
    assert adequacy_verdict(just_over)["margin"] == ADEQUACY_MARGIN


# ---------------------------------------------------------------------------
# Null calibration and uncertainty
# ---------------------------------------------------------------------------

def test_group_level_label_permutation_lands_at_chance():
    rows = one_dimensional_world()
    permuted = permute_labels_by_role(rows, seed=7)
    pred, fold_of_row, _ = _oof(permuted, FAST_LINEAR)
    boot = cluster_bootstrap(permuted, pred, fold_of_row, replicates=400)
    assert boot["ci_low"] < 0.5 < boot["ci_high"], (
        f"a permuted-label run must be indistinguishable from chance; got {boot}"
    )


def test_permutation_preserves_group_structure():
    rows = one_dimensional_world()
    permuted = permute_labels_by_role(rows, seed=7)
    # Each original group keeps all of its rows and gets exactly one label.
    for original in set(rows.roles):
        mask = rows.roles == original
        assert len(set(permuted.y[mask])) == 1
    assert len(set(permuted.roles)) == len(set(rows.roles))


def test_bootstrap_resamples_groups_not_rows():
    rows = one_dimensional_world()
    pred, fold_of_row, _ = _oof(rows, FAST_AXIS)
    by_role = cluster_bootstrap(rows, pred, fold_of_row, 400, cluster="role")
    by_question = cluster_bootstrap(rows, pred, fold_of_row, 400,
                                    cluster="question")

    for boot in (by_role, by_question):
        assert boot["replicates_used"] > 300
        assert boot["ci_low"] <= boot["point"] <= boot["ci_high"]
        assert boot["n_scored_rows"] == len(rows)
        assert boot["n_folds"] == FAST["folds"][0] * FAST["folds"][1]
        assert boot["estimand"] == "mean within-fold role-balanced AUROC"
    assert by_role["point"] == by_question["point"], (
        "the clustering choice must affect only the interval, never the "
        "point estimate"
    )
    # Interval *width* on a signal-carrying world is not a stable invariant,
    # so the claim that question clustering is unsafe is made where it can be
    # measured — coverage under a null, in
    # test_question_clustering_is_anticonservative.


def test_question_clustering_is_anticonservative():
    """Why ``INFERENCE_CLUSTER`` is role, recorded as a test so nobody later
    swaps in the far tighter question-clustered interval.

    On data where the label carries no transferable information, a calibrated
    95% interval should cover chance almost every time. Role clustering does.
    Question clustering does not come close: rows sharing a role stay together
    inside every question cluster, so the dependence that actually drives the
    estimate is never resampled and the interval collapses.
    """
    role_cover = question_cover = 0
    trials = 8
    for seed in range(trials):
        rows = role_identity_only_world(seed=40 + seed)
        pred, fold_of_row, _ = _oof(rows, FAST_LINEAR)
        r = cluster_bootstrap(rows, pred, fold_of_row, 400, cluster="role")
        q = cluster_bootstrap(rows, pred, fold_of_row, 400, cluster="question")
        role_cover += r["ci_low"] < 0.5 < r["ci_high"]
        question_cover += q["ci_low"] < 0.5 < q["ci_high"]

    assert role_cover >= trials - 1, (
        f"role-clustered intervals must cover chance under the null; "
        f"got {role_cover}/{trials}"
    )
    assert question_cover <= trials // 2, (
        f"question clustering is expected to under-cover; got "
        f"{question_cover}/{trials}, which would make it safe to use and "
        f"invalidate the reason INFERENCE_CLUSTER is 'role'"
    )


def test_selection_records_what_each_fold_chose():
    rows = multidimensional_linear_world()
    _, _, selections = _oof(rows, FAST_LINEAR)
    assert len(selections) == FAST["folds"][0] * FAST["folds"][1]
    for s in selections:
        assert s["selected"].startswith("pca")
        assert s["n_train_roles"] > 0
        assert s["n_test"] > 0
        assert 0.0 <= s["inner_auroc"] <= 1.0
    assert len({s["fold_id"] for s in selections}) == len(selections)
