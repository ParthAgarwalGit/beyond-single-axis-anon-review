"""Fail-closed provenance tests for T29 production inference."""

import hashlib
import json
import random

import pytest

from src.causal_specificity import REQUIRED_CONDITIONS, causal_specificity_report
from tools.run_t29_specificity import (
    EXPECTED_AXIS_SHA256,
    EXPECTED_CAPPING_SHA256,
    EXPECTED_MODEL_ID,
    EXPECTED_MODEL_REVISION,
    EXPECTED_T24_SETTING,
    validate_judge_binding,
    validate_t24_freeze,
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _t21_binding():
    return {
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "assistant_axis_sha256": EXPECTED_AXIS_SHA256,
        "capping_config_sha256": EXPECTED_CAPPING_SHA256,
        "setting": EXPECTED_T24_SETTING,
    }


def _write_t24(tmp_path, controls_bytes=b"real-frozen-controls", **overrides):
    controls = tmp_path / "source_control_parameters.npz"
    controls.write_bytes(controls_bytes)
    freeze = {
        "task": "T24",
        "status": "FROZEN_NUMERICAL_CONTROLS",
        "source_git_sha": "1" * 40,
        "model": {"id": EXPECTED_MODEL_ID, "revision": EXPECTED_MODEL_REVISION},
        "source_axis_condition": {
            "assistant_axis_sha256": EXPECTED_AXIS_SHA256,
            "capping_config_sha256": EXPECTED_CAPPING_SHA256,
            "setting": EXPECTED_T24_SETTING,
        },
        "calibration_input_sha256": "2" * 64,
        "numerical_controls_npz_sha256": _sha_bytes(controls_bytes),
        "validation": {
            "released_axis_sha256_check": "PASS",
            "released_capping_config_sha256_check": "PASS",
            "calibration_axis_vs_released_axis_check": "PASS",
            "source_capping_vector_orientation_check": "PASS",
            "source_axis_engagement_measured": "PASS",
        },
    }
    freeze.update(overrides)
    path = tmp_path / "source_control_freeze.json"
    path.write_text(json.dumps(freeze), encoding="utf-8")
    return path, controls


def _write_judge_artifacts(tmp_path, *, t27_branch="VALIDATED", gating=True):
    scored_sha = "a" * 64
    t27 = {
        "task": "T27_causal_judge_validation",
        "branch": "qwen_capping",
        "gating": gating,
        "judged_instrument": {"judge_model": "judge/model", "provider": "provider"},
        "outcomes": {
            "harmfulness": {
                "gate": {
                    "branch": t27_branch,
                    "primary_evidence": (
                        "full_corpus_automatic" if t27_branch == "VALIDATED" else "human_subset"
                    ),
                }
            }
        },
    }
    scoring = {
        "task": "causal_judge_scoring",
        "branch": "qwen_capping",
        "judge_model": "judge/model",
        "provider": "provider",
        "endpoint": "https://judge.invalid/v1",
        "production": True,
        "drawable_by_t26": True,
        "source_git_sha": "b" * 40,
        "prompt_sha256": "c" * 64,
        "prompt_file_sha256": "d" * 64,
        "config_sha256": "e" * 64,
        "artifact_sha256": {"scored": scored_sha, "raw": "f" * 64},
        "canary": {
            "n": 50,
            "start_sha256": "1" * 64,
            "end_sha256": "1" * 64,
            "drift_detected": False,
        },
    }
    t27_path = tmp_path / "t27.json"
    score_path = tmp_path / "scoring.json"
    t27_path.write_text(json.dumps(t27), encoding="utf-8")
    score_path.write_text(json.dumps(scoring), encoding="utf-8")
    data = {
        "label_source": "automatic_causal_judge",
        "source_scored_artifact_sha256": scored_sha,
    }
    return t27_path, score_path, data


def _rows(n=40):
    rows = []
    for condition in REQUIRED_CONDITIONS:
        for i in range(n):
            rows.append(
                {
                    "item_id": f"i-{i:03d}",
                    "condition": condition,
                    "harmfulness_label": "harmful" if i < 5 else "safe",
                    "technical_valid": True,
                }
            )
    return rows


def test_t24_requires_real_numerical_freeze_status(tmp_path):
    freeze, controls = _write_t24(tmp_path, status="PROTOCOL_FROZEN_NUMERICAL_CALIBRATION_PENDING")
    with pytest.raises(SystemExit, match="not numerically frozen"):
        validate_t24_freeze(freeze, controls, _t21_binding())


def test_t24_rejects_controls_bytes_not_bound_by_freeze(tmp_path):
    freeze, controls = _write_t24(tmp_path)
    controls.write_bytes(b"different-bytes")
    with pytest.raises(SystemExit, match="controls NPZ hash mismatch"):
        validate_t24_freeze(freeze, controls, _t21_binding())


def test_t24_rejects_t21_axis_hash_mismatch(tmp_path):
    freeze, controls = _write_t24(tmp_path)
    doc = json.loads(freeze.read_text())
    doc["source_axis_condition"]["assistant_axis_sha256"] = "0" * 64
    freeze.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit, match="Assistant-Axis SHA"):
        validate_t24_freeze(freeze, controls, _t21_binding())


def test_unvalidated_judge_cannot_pass_as_clean_primary(tmp_path):
    t27, scoring, data = _write_judge_artifacts(
        tmp_path, t27_branch="MEASUREMENT_LIMITED", gating=True
    )
    with pytest.raises(SystemExit, match="Refusing a clean T29 primary result"):
        validate_judge_binding(t27, scoring, data)


def test_unvalidated_override_is_explicitly_not_primary(tmp_path):
    t27, scoring, data = _write_judge_artifacts(
        tmp_path, t27_branch="MEASUREMENT_LIMITED", gating=True
    )
    bound = validate_judge_binding(t27, scoring, data, allow_unvalidated=True)
    assert bound["validated_for_primary"] is False
    assert bound["t27_harmfulness_branch"] == "MEASUREMENT_LIMITED"


def test_judge_provenance_must_match_scored_artifact(tmp_path):
    t27, scoring, data = _write_judge_artifacts(tmp_path)
    data["source_scored_artifact_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="not bound to the scored artifact"):
        validate_judge_binding(t27, scoring, data)


def test_judge_drift_is_fail_closed(tmp_path):
    t27, scoring, data = _write_judge_artifacts(tmp_path)
    doc = json.loads(scoring.read_text())
    doc["canary"]["drift_detected"] = True
    scoring.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit, match="canary drift"):
        validate_judge_binding(t27, scoring, data)


def test_seed_determinism_and_shuffled_input_order():
    rows = _rows()
    a = causal_specificity_report(rows, n_boot=200, seed=123)
    b = causal_specificity_report(rows, n_boot=200, seed=123)
    assert a == b

    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    c = causal_specificity_report(shuffled, n_boot=200, seed=123)
    assert a == c
