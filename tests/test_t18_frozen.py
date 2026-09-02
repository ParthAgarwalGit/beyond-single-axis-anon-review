"""The T18 analysis parameters must come from the frozen file, not from code.

Review of PR #30 correctly noted that the equivalence margin was not part of
any frozen specification. It now lives in ``configs/t18_frozen.yaml`` with the
rest of the T18 analysis parameters, and these tests fail if a value in the
module ever drifts from the file — which is the only thing that makes
"predeclared" mean anything.
"""

import numpy as np
import pytest
import yaml

from src import t18_readouts as t18


@pytest.fixture(scope="module")
def frozen():
    return yaml.safe_load(t18.T18_CONFIG_PATH.read_bytes())


def test_the_frozen_file_is_frozen(frozen):
    assert frozen["freeze_status"] == "FROZEN"
    assert frozen["frozen_before_any_real_activations_examined"] is True
    assert frozen["registry_entry"] == "analysis_registry.sensitivity[13]"


def test_module_constants_match_the_frozen_file(frozen):
    assert t18.ADEQUACY_MARGIN == frozen["inference"]["adequacy_margin_auroc"]
    assert t18.INFERENCE_CLUSTER == frozen["inference"]["cluster"]
    assert t18.BOOTSTRAP_REPLICATES == frozen["inference"]["bootstrap_replicates"]
    assert t18.PRIMARY_METRIC == frozen["estimand"]["metric"]
    assert t18.PCA_K_GRID == tuple(frozen["candidates"]["pca_k_grid"])
    assert t18.LINEAR_C_GRID == tuple(frozen["candidates"]["linear_c_grid"])
    assert t18.NONLINEAR_HIDDEN_GRID == tuple(
        frozen["candidates"]["nonlinear_hidden_grid"])
    assert t18.NONLINEAR_ALPHA_GRID == tuple(
        frozen["candidates"]["nonlinear_alpha_grid"])
    assert t18.VERDICT_FAMILIES == tuple(
        frozen["candidates"]["families_that_may_determine_the_verdict"])
    assert t18.OUTER_FOLDS == (frozen["splitting"]["outer_role_folds"],
                               frozen["splitting"]["outer_question_folds"])
    assert t18.INNER_FOLDS == (frozen["splitting"]["inner_role_folds"],
                               frozen["splitting"]["inner_question_folds"])
    assert t18.SPLIT_SEED == frozen["splitting"]["seed"]
    assert t18.PRIMARY_JUDGE_REGION == frozen["target"]["judge_region"]
    assert t18.REQUIRED_DEFAULT_CONDITIONS == frozen["target"][
        "default_conditions_required"]
    assert t18.PERMUTATION_REPLICATES == frozen["diagnostics"][
        "group_label_permutation"]["replicates"]


def test_config_hash_is_bound_and_stable():
    import hashlib

    expected = hashlib.sha256(t18.T18_CONFIG_PATH.read_bytes()).hexdigest()
    assert t18.T18_CONFIG_SHA256 == expected
    assert t18.load_t18_config()[1] == expected


def test_the_verdict_carries_the_config_hash():
    verdict = t18.adequacy_verdict({
        "max_family_difference": 0.0, "ci_low": -0.01, "ci_high": 0.01,
        "best_family": "pca_linear", "cluster": t18.INFERENCE_CLUSTER,
        "simultaneous_over": ["pca_linear"],
    })
    assert verdict["t18_config_sha256"] == t18.T18_CONFIG_SHA256


def test_pooling_predictions_across_folds_is_declared_forbidden(frozen):
    """The frozen file must keep saying this, because the whole estimand
    depends on it and a later edit that quietly re-enabled pooling would
    otherwise pass every other test."""
    assert frozen["estimand"]["pooling_predictions_across_folds_forbidden"] is True
    assert frozen["estimand"]["row_weighting"] == "role-balanced within class"
    assert "max-over-families" in frozen["inference"]["multiplicity"]


def test_every_verdict_family_has_an_implementation():
    for family in t18.VERDICT_FAMILIES:
        assert family in t18.FAMILIES
        assert t18.FAMILIES[family](300), f"{family} yields no candidates"


def test_frozen_grids_are_actually_used_by_the_families():
    """A grid that the family builders ignore would be frozen in name only."""
    names = {n for n, _ in t18.FAMILIES["pca_linear"](300)}
    for k in t18.PCA_K_GRID:
        assert any(n.startswith(f"pca{k}_linear_C") for n in names)
    for c in t18.LINEAR_C_GRID:
        assert any(n.endswith(f"_C{c}") for n in names)

    nonlinear = {n for n, _ in t18.FAMILIES["small_nonlinear"](10_000)}
    for h in t18.NONLINEAR_HIDDEN_GRID:
        assert any(f"_mlp{h}_" in n for n in nonlinear)
    for a in t18.NONLINEAR_ALPHA_GRID:
        assert any(n.endswith(f"_a{a}") for n in nonlinear)


def test_default_fold_counts_give_every_fold_a_default_condition():
    """Five role folds and five default conditions is not a coincidence: it is
    what lets every outer fold hold out exactly one default condition and
    still have both classes in its test set."""
    assert t18.OUTER_FOLDS[0] == t18.REQUIRED_DEFAULT_CONDITIONS

    roles = np.array([f"role_{i:03d}" for i in range(20)]
                     + [f"{t18.DEFAULT_ROLE_PREFIX}{c}" for c in range(5)],
                     dtype=object)
    y = np.array([0] * 20 + [1] * 5)
    rows = t18.ReadoutRows(np.random.default_rng(0).normal(size=(25, 3)),
                           roles, np.arange(25) % 8, y)
    folds = t18.doubly_grouped_folds(rows, *t18.OUTER_FOLDS)
    per_role_fold = {}
    for f in folds:
        per_role_fold.setdefault(f["role_fold"], set()).update(
            rows.roles[f["test_idx"]])
    for role_fold, held in per_role_fold.items():
        n_defaults = sum(str(r).startswith(t18.DEFAULT_ROLE_PREFIX) for r in held)
        assert n_defaults == 1, (
            f"role fold {role_fold} holds out {n_defaults} default conditions; "
            "exactly one keeps every fold two-class and every condition tested"
        )
