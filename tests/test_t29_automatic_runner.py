"""Provenance-gate tests for the T29 automatic sensitivity runner."""

import hashlib
import json

import pytest

from tools.run_t29_automatic_sensitivity import validate_authorities


def _write_json(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authorities(tmp_path):
    scored = tmp_path / "scored.jsonl"
    scored.write_text("{}\n", encoding="utf-8")
    scored_sha = hashlib.sha256(scored.read_bytes()).hexdigest()

    t27 = tmp_path / "t27.json"
    t27_obj = {
        "task": "T27_causal_judge_validation",
        "branch": "qwen_capping",
        "gating": True,
        "n_unresolved_after_adjudication": 0,
        "outcomes": {
            "harmfulness": {
                "gate": {
                    "branch": "MEASUREMENT_LIMITED",
                    "primary_evidence": "human_subset",
                    "full_corpus_reportable_as": "full_corpus_automatic",
                }
            }
        },
    }
    t27_sha = _write_json(t27, t27_obj)

    population_sha = "a" * 64
    report_sha = "b" * 64
    receipt = tmp_path / "receipt.json"
    _write_json(receipt, {
        "status": "PRODUCTION_SCORING_PASS_DRAWABLE_BY_T26",
        "source_git_dirty": False,
        "report_checks": {
            "production": True,
            "drawable_by_t26": True,
            "n_rows": 400,
            "n_coverage_failures": 0,
            "canary": {"start_usable": True, "end_usable": True, "drift_detected": False},
        },
        "artifacts": {
            "input": {"sha256": population_sha},
            "scored": {"sha256": scored_sha},
            "report": {"sha256": report_sha},
        },
    })

    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {
        "branch": "qwen_capping",
        "n_items": 120,
        "source_scored_artifact_sha256": scored_sha,
        "source_scoring_report_sha256": report_sha,
    })

    matched = tmp_path / "matched.json"
    _write_json(matched, {
        "task": "T29",
        "status": "MATCHED_CONTROLS_COMPLETE_IMMUTABLY_ARCHIVED",
        "population": {
            "rows": 400,
            "rows_per_condition": 100,
            "conditions": [
                "unsteered_baseline", "assistant_cap",
                "random_direction_cap", "sphere_proximity_cap",
            ],
            "sha256": population_sha,
        },
        "frozen_authorities": {
            "t24_controls_npz_sha256": "c" * 64,
            "t24_freeze_sha256": "d" * 64,
            "assistant_axis_sha256": "e" * 64,
            "source_capping_config_sha256": "f" * 64,
        },
    })

    human = tmp_path / "human.json"
    _write_json(human, {
        "task": "T29_QWEN_HUMAN_PRIMARY",
        "status": "HUMAN_PRIMARY_MEASUREMENT_LIMITED",
        "primary_result_eligible": True,
        "measurement_branch": {"automatic_400_row_primary_allowed": False},
        "provenance": {"t27_gate_sha256": t27_sha},
    })

    return {
        "scored": scored, "receipt": receipt, "manifest": manifest,
        "t27": t27, "matched": matched, "human": human,
    }


def _validate(paths):
    return validate_authorities(
        paths["scored"], paths["receipt"], paths["manifest"],
        paths["t27"], paths["matched"], paths["human"],
    )


def test_valid_authorities_pass(tmp_path):
    paths = _authorities(tmp_path)
    provenance, t27 = _validate(paths)
    assert provenance["private_scored_sha256"] == hashlib.sha256(paths["scored"].read_bytes()).hexdigest()
    assert t27["outcomes"]["harmfulness"]["gate"]["branch"] == "MEASUREMENT_LIMITED"


def test_rejects_scored_artifact_hash_mismatch(tmp_path):
    paths = _authorities(tmp_path)
    receipt = json.loads(paths["receipt"].read_text())
    receipt["artifacts"]["scored"]["sha256"] = "0" * 64
    _write_json(paths["receipt"], receipt)
    with pytest.raises(SystemExit, match="private scored artifact SHA"):
        _validate(paths)


def test_rejects_wrong_t27_branch(tmp_path):
    paths = _authorities(tmp_path)
    t27 = json.loads(paths["t27"].read_text())
    t27["outcomes"]["harmfulness"]["gate"]["branch"] = "VALIDATED"
    new_t27_sha = _write_json(paths["t27"], t27)
    human = json.loads(paths["human"].read_text())
    human["provenance"]["t27_gate_sha256"] = new_t27_sha
    _write_json(paths["human"], human)
    with pytest.raises(SystemExit, match="MEASUREMENT_LIMITED"):
        _validate(paths)


def test_rejects_automatic_primary_promotion(tmp_path):
    paths = _authorities(tmp_path)
    human = json.loads(paths["human"].read_text())
    human["measurement_branch"]["automatic_400_row_primary_allowed"] = True
    _write_json(paths["human"], human)
    with pytest.raises(SystemExit, match="forbid automatic primary promotion"):
        _validate(paths)
