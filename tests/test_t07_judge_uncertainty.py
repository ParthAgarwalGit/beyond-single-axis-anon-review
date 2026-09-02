from __future__ import annotations

import copy
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "run_t07_judge_uncertainty.py"

spec = importlib.util.spec_from_file_location("t07_uncertainty_under_test", MODULE_PATH)
assert spec is not None and spec.loader is not None
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


def synthetic_rows(n_each: int = 6):
    rows = []
    for arm in ("extraction", "translated"):
        for sc in (0, 1, 2, 3, None):
            for i in range(n_each):
                rows.append(
                    {
                        "arm": arm,
                        "lu_score": sc,
                        "ip_weight": float(10 + i + (0 if sc is None else sc)),
                        "human_validity": "unjudgeable" if i == 0 else "ok",
                        "human_label": int((i + (0 if sc is None else sc)) % 4),
                        "machine_validity": "closed",
                    }
                )
    return rows


def test_historical_six_decimal_ip_weight_serialization():
    # Regression for the original T05 builder: it wrote round(N/n, 6).
    assert round(8211 / 45, 6) == 182.466667
    assert (8211 / 45) != 182.466667


def test_fixed_seed_is_deterministic():
    rows = synthetic_rows()
    expected = Counter(u.stratum_key(r) for r in rows)
    d1 = u.bootstrap_distributions(
        rows, n_replicates=30, seed=12345, expected_stratum_sizes=dict(expected)
    )
    d2 = u.bootstrap_distributions(
        rows, n_replicates=30, seed=12345, expected_stratum_sizes=dict(expected)
    )
    assert d1 == d2


def test_stratum_sizes_preserved_in_each_replicate():
    rows = synthetic_rows()
    expected = Counter(u.stratum_key(r) for r in rows)
    strata = u.make_strata(rows, dict(expected))
    rep = u.resample_within_strata(rows, strata, np.random.default_rng(7))
    assert Counter(u.stratum_key(r) for r in rep) == expected
    assert len(rep) == len(rows)


def test_ip_weights_are_retained_from_sampled_items():
    rows = synthetic_rows()
    expected = Counter(u.stratum_key(r) for r in rows)
    strata = u.make_strata(rows, dict(expected))
    rep = u.resample_within_strata(rows, strata, np.random.default_rng(11))
    for sk in expected:
        original_weights = {r["ip_weight"] for r in rows if u.stratum_key(r) == sk}
        drawn_weights = {r["ip_weight"] for r in rep if u.stratum_key(r) == sk}
        assert drawn_weights <= original_weights


def test_missing_required_stratum_information_fails_but_explicit_none_is_valid():
    with pytest.raises(ValueError, match="missing required sampling-stratum field 'lu_score'"):
        u.stratum_key({"arm": "translated"})
    assert u.stratum_key({"arm": "translated", "lu_score": None}) == "translated|None"


def test_undefined_metrics_are_counted_not_zero_filled():
    out = u.summarize_one(
        [0.2, None, float("nan"), 0.4],
        level=0.95,
        percentile_method="linear",
    )
    assert out["valid_replicates"] == 2
    assert out["undefined_or_degenerate_replicates"] == 2
    assert out["ci95_percentile"] is not None


def test_gate_fields_are_point_estimate_based_only():
    frozen = {
        "pooled": {
            "scored_metrics": {
                "score3_precision": 0.555556,
                "score3_recall_ipweighted": 0.797853,
                "macro_f1_ipweighted": 0.409206,
                "four_class_kappa": 0.232956,
            }
        },
        "by_arm": {
            "extraction": {
                "scored_metrics": {
                    "score3_precision": 0.622222,
                    "score3_recall_ipweighted": 0.814675,
                    "macro_f1_ipweighted": 0.446891,
                    "four_class_kappa": 0.219375,
                }
            },
            "translated": {
                "scored_metrics": {
                    "score3_precision": 0.488889,
                    "score3_recall_ipweighted": 0.776224,
                    "macro_f1_ipweighted": 0.384997,
                    "four_class_kappa": 0.246403,
                }
            },
        },
    }
    fake_summary = {
        "pooled": {
            k: {
                "ci95_percentile": [0.0, 1.0],  # deliberately overlaps every threshold
                "valid_replicates": 100,
                "undefined_or_degenerate_replicates": 0,
            }
            for k in u.METRIC_KEYS
        },
        "by_arm:extraction": {
            k: {
                "ci95_percentile": [0.0, 1.0],
                "valid_replicates": 100,
                "undefined_or_degenerate_replicates": 0,
            }
            for k in u.METRIC_KEYS
        },
        "by_arm:translated": {
            k: {
                "ci95_percentile": [0.0, 1.0],
                "valid_replicates": 100,
                "undefined_or_degenerate_replicates": 0,
            }
            for k in u.METRIC_KEYS
        },
    }
    attached = u.attach_estimates_and_gate(fake_summary, frozen)
    assert all(
        attached["pooled"][k]["frozen_point_estimate_pass"] is False
        for k in u.METRIC_KEYS
    )


def test_point_estimate_guard_rejects_mismatch(monkeypatch, tmp_path):
    # Exercise the hard-frozen mismatch branch without needing the private production key.
    rows = synthetic_rows()
    fake_current = {
        "pooled": {
            "n_total": 300,
            "n_scored": 278,
            "n_abstained": 10,
            "n_unjudgeable": 12,
            "scored_metrics": {
                "score3_precision": 0.555556,
                "score3_recall_ipweighted": 0.797853,
                "macro_f1_ipweighted": 0.409206,
                "four_class_kappa": 0.232956,
            },
            "gate": {"all_pass": False},
        },
        "by_arm": {},
        "selected_branch": {"branch": "measurement_limited_result"},
    }
    frozen = copy.deepcopy(fake_current)

    # Deliberately make the frozen by-arm block disagree with the reproduced one.
    # This exercises the fail-closed point-estimate reproduction guard.
    frozen["by_arm"] = {
        "translated": {"sentinel": "intentional_test_mismatch"}
    }

    # The function calls canonical _load_inputs and validate. Replace only IO here.
    monkeypatch.setattr(u.T07, "_load_inputs", lambda *_args, **_kwargs: (rows, {}))
    monkeypatch.setattr(u.T07, "validate", lambda *_args, **_kwargs: fake_current)

    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")

    # First this fails on the stronger by-arm equality, which proves the guard is active.
    with pytest.raises(SystemExit):
        u.reproduce_frozen_point_estimate(
            tmp_path / "consensus.csv",
            tmp_path / "key.json",
            frozen_path,
        )



def test_private_key_hash_binding_requires_frozen_config_and_optional_manifest(tmp_path):
    actual = "a" * 64
    cfg = {"sampling": {"private_key_sha256": actual}}
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps({"private_sampling_key": {"sha256": None}}), encoding="utf-8")

    # Pre-bootstrap reproduction may run before the manifest hash is filled.
    u.verify_private_key_hash_binding(
        actual, cfg, prov, require_manifest_hash=False
    )

    # Production bootstrap must not run until the provenance manifest is bound.
    with pytest.raises(SystemExit):
        u.verify_private_key_hash_binding(
            actual, cfg, prov, require_manifest_hash=True
        )

    prov.write_text(
        json.dumps({"private_sampling_key": {"sha256": actual}}),
        encoding="utf-8",
    )
    u.verify_private_key_hash_binding(
        actual, cfg, prov, require_manifest_hash=True
    )

    bad_cfg = {"sampling": {"private_key_sha256": "b" * 64}}
    with pytest.raises(SystemExit):
        u.verify_private_key_hash_binding(
            actual, bad_cfg, prov, require_manifest_hash=False
        )



def test_build_result_preserves_existing_frozen_fields():
    frozen = {
        "format_version": "t07-judge-validation-v1",
        "pooled": {
            "n_total": 300,
            "n_scored": 278,
            "n_abstained": 10,
            "n_unjudgeable": 12,
            "scored_metrics": {
                "score3_precision": 0.555556,
                "score3_recall_ipweighted": 0.797853,
                "macro_f1_ipweighted": 0.409206,
                "four_class_kappa": 0.232956,
            },
            "gate": {"all_pass": False},
        },
        "by_arm": {
            "extraction": {
                "scored_metrics": {
                    "score3_precision": 0.622222,
                    "score3_recall_ipweighted": 0.814675,
                    "macro_f1_ipweighted": 0.446891,
                    "four_class_kappa": 0.219375,
                }
            },
            "translated": {
                "scored_metrics": {
                    "score3_precision": 0.488889,
                    "score3_recall_ipweighted": 0.776224,
                    "macro_f1_ipweighted": 0.384997,
                    "four_class_kappa": 0.246403,
                }
            },
        },
        "selected_branch": {"branch": "measurement_limited_result", "why": "frozen"},
    }
    summary = {
        block: {
            k: {
                "estimate": u.frozen_estimates_for_block(frozen, block)[k],
                "ci95_percentile": [0.1, 0.9],
                "valid_replicates": 10,
                "undefined_or_degenerate_replicates": 0,
                "frozen_threshold": u.GATE_THRESHOLDS_BY_METRIC[k],
                "frozen_point_estimate_pass": False,
                "gate_note": "x",
            }
            for k in u.METRIC_KEYS
        }
        for block in ("pooled", "by_arm:extraction", "by_arm:translated")
    }
    config = {
        "_sha256": "cfg",
        "bootstrap": {
            "replicates": 10,
            "seed": 1,
            "level": 0.95,
            "ci_method": "percentile",
            "percentile_method": "linear",
        },
    }
    result = u.build_result(
        frozen_report=frozen,
        config=config,
        sampling_authority={"aggregate_strata": []},
        interval_summary=summary,
        public_hashes={
            "consensus_sha256": "c",
            "frozen_report_sha256": "r",
            "canonical_t07_scorer_sha256": "s",
        },
        key_sha256="k",
        source_sha="g",
    )
    stripped = copy.deepcopy(result)
    stripped.pop("uncertainty")
    assert stripped == frozen
