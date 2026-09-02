"""Focused tests for the T18 response-length nuisance diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_t18_length_nuisance.py"

spec = importlib.util.spec_from_file_location("t18_length_nuisance", SCRIPT)
m = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m)


def test_feature_specs_are_exactly_the_two_frozen_diagnostics():
    assert m.FEATURE_SPECS == {
        "total_generated_tokens": ("all_response",),
        "reasoning_plus_answer_tokens": ("reasoning", "final_answer"),
    }
    assert all(len(v) < 3 for v in m.FEATURE_SPECS.values())


def test_extract_token_counts_requires_exact_partition():
    row = {
        "uid": "x",
        "token_region_counts": {
            "all_response": 13,
            "reasoning": 9,
            "final_answer": 4,
        },
    }
    assert m.extract_token_counts(row, "synthetic") == (13, 9, 4)

    bad = {
        "uid": "x",
        "token_region_counts": {
            "all_response": 13,
            "reasoning": 8,
            "final_answer": 4,
        },
    }
    with pytest.raises(SystemExit):
        m.extract_token_counts(bad, "synthetic")

    missing = {
        "uid": "x",
        "token_region_counts": {
            "all_response": 13,
            "reasoning": 9,
            "final_answer": None,
        },
    }
    with pytest.raises(SystemExit):
        m.extract_token_counts(missing, "synthetic")


def _records():
    return [
        {
            "uid": "r1",
            "block": "C80-A",
            "group": "role_a",
            "question_id": 1,
            "label": 0,
            "all_response": 10,
            "reasoning": 7,
            "final_answer": 3,
        },
        {
            "uid": "r2",
            "block": "C80-B",
            "group": "role_b",
            "question_id": 2,
            "label": 0,
            "all_response": 20,
            "reasoning": 12,
            "final_answer": 8,
        },
        {
            "uid": "d1",
            "block": "C80-A",
            "group": f"{m.DEFAULT_ROLE_PREFIX}0",
            "question_id": 1,
            "label": 1,
            "all_response": 15,
            "reasoning": 10,
            "final_answer": 5,
        },
    ]


def test_make_rows_has_one_and_two_feature_models_only():
    one = m.make_rows(_records(), "total_generated_tokens")
    two = m.make_rows(_records(), "reasoning_plus_answer_tokens")
    assert one.X.shape == (3, 1)
    assert two.X.shape == (3, 2)
    assert np.array_equal(one.y, [0, 0, 1])
    assert one.roles[-1] == f"{m.DEFAULT_ROLE_PREFIX}0"
    assert np.array_equal(one.X[:, 0], [10.0, 20.0, 15.0])
    assert np.array_equal(two.X, [[7.0, 3.0], [12.0, 8.0], [10.0, 5.0]])


def test_fit_reuses_frozen_t18_family_and_cv(monkeypatch):
    rows = m.make_rows(_records(), "total_generated_tokens")
    called = {}

    def fake_run_family(rows_arg, family, folds, inner_folds, seed):
        called["run"] = (family, folds, inner_folds, seed)
        pred = np.array([0.1, 0.2, 0.8])
        fold = np.array([0, 1, 2])
        sel = [
            {"fold_id": 0, "selected": "full_linear_C0.1"},
            {"fold_id": 1, "selected": "full_linear_C0.1"},
            {"fold_id": 2, "selected": "full_linear_C0.01"},
        ]
        return pred, fold, sel

    def fake_boot(rows_arg, pred, fold, replicates, seed, cluster):
        called["boot"] = (replicates, seed, cluster)
        return {
            "point": 0.7,
            "ci_low": 0.6,
            "ci_high": 0.8,
            "replicates_used": replicates,
            "replicates_skipped": 0,
            "n_scored_rows": len(rows_arg),
            "n_folds": 3,
            "cluster": cluster,
            "estimand": "synthetic",
        }

    monkeypatch.setattr(m, "run_family", fake_run_family)
    monkeypatch.setattr(m, "cluster_bootstrap", fake_boot)

    perf, *_ = m.fit_nuisance_baseline(rows, "total_generated_tokens")
    assert called["run"] == (
        "full_linear", m.OUTER_FOLDS, m.INNER_FOLDS, m.SPLIT_SEED
    )
    assert called["boot"] == (
        m.BOOTSTRAP_REPLICATES, m.SPLIT_SEED, m.INFERENCE_CLUSTER
    )
    assert perf["n_features"] == 1
    assert perf["selection_counts"] == {
        "full_linear_C0.01": 1,
        "full_linear_C0.1": 2,
    }
