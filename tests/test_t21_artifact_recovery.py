import hashlib
import json
from pathlib import Path

import importlib.util

import pytest

VERIFY_PATH = Path(__file__).resolve().parents[1] / "tools" / "verify_t21_recovered_artifacts.py"
spec = importlib.util.spec_from_file_location("verify_t21_recovered_artifacts", VERIFY_PATH)
verify = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(verify)

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / "results" / "t21" / "T21_ARTIFACT_RECOVERY.json"
ARTIFACT_MANIFEST = ROOT / "results" / "t21" / "ARTIFACT_MANIFEST.json"
EXPECTED_MANIFEST_SHA = "d611e904f3b5d47c71e5ab1f3fa1a84ead6cfd5d94dba337f5c3b71aafef7793"
EXPECTED_HF_REVISION = "050445e47a0d29d0e9de1c21c81a5da8dccf6fc4"


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recovery_record_preserves_exact_historical_manifest_hash():
    receipt = json.loads(RECOVERY.read_text(encoding="utf-8"))
    manifest = receipt["files"]["t21_eval_manifest_frozen.jsonl"]
    assert manifest["sha256_now"] == EXPECTED_MANIFEST_SHA
    assert manifest["prior_recorded_sha256"] == EXPECTED_MANIFEST_SHA
    assert manifest["raw_rows"] == 100
    assert manifest["hash_originally_verified"] is True
    assert manifest["hf_roundtrip_verified"] is True


def test_recovery_record_pins_roundtripped_immutable_hf_bytes():
    receipt = json.loads(RECOVERY.read_text(encoding="utf-8"))
    assert receipt["hf_repo_id"] == "[Author-B-HF]/[anonymized-repository-name]-artifacts"
    assert receipt["hf_revision"] == EXPECTED_HF_REVISION
    assert receipt["files"]["t21_eval_manifest_frozen.jsonl"]["prior_recorded_sha256"] == EXPECTED_MANIFEST_SHA
    assert receipt["files"]["t21_eval_manifest_frozen.jsonl"]["hash_originally_verified"] is True
    assert all(record["hf_roundtrip_verified"] is True for record in receipt["files"].values())


def test_original_null_hashes_are_preserved_separately_from_recovery_attestations():
    manifest = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["recovery_pin"]["hf_revision"] == EXPECTED_HF_REVISION
    assert manifest["frozen_manifests"]["safety_manifest_sha256"] == EXPECTED_MANIFEST_SHA
    for record in manifest["external_raw_artifacts"]:
        assert record["sha256"] is None
        assert record["hash_status"] == "NOT_CAPTURED_IN_UPLOADED_NOTEBOOK_OUTPUT"
        assert len(record["recovered_sha256"]) == 64
        assert record["recovery_status"] == "RECOVERED_AND_ATTESTED_NOT_ORIGINALLY_HASH_VERIFIED"
        assert record["hf_revision"] == EXPECTED_HF_REVISION


def _synthetic_recovery_fixture(tmp_path):
    manifest_rows = [{"item_id": f"item-{i:03d}"} for i in range(100)]
    manifest = tmp_path / verify.MANIFEST_NAME
    _write_jsonl(manifest, manifest_rows)
    keys = verify.expected_keys(manifest_rows)

    baseline_rows = [
        {"item_id": item_id, "condition": condition, "technical_error": None}
        for item_id, condition in sorted(keys)
    ]
    baseline = tmp_path / verify.BASELINE_NAME
    _write_jsonl(baseline, baseline_rows)

    capability = tmp_path / verify.CAPABILITY_NAME
    _write_jsonl(capability, [{"row": i} for i in range(6224)])

    judge_success = [
        {
            "item_id": item_id,
            "condition": condition,
            "score": "refusal",
            "technical_error": None,
            "judge_model": "deepseek-ai/DeepSeek-V3",
            "judge_provider": "novita",
            "system_prompt_sha256": "system-hash",
            "user_template_sha256": "template-hash",
        }
        for item_id, condition in sorted(keys)
    ]
    retry_keys = sorted(keys)[:3]
    judge_retries = [
        {
            "item_id": item_id,
            "condition": condition,
            "score": None,
            "technical_error": "transient provider error",
            "judge_model": "deepseek-ai/DeepSeek-V3",
            "judge_provider": "novita",
        }
        for item_id, condition in retry_keys
    ]
    judge = tmp_path / verify.JUDGE_NAME
    _write_jsonl(judge, judge_success + judge_retries)

    receipt = {
        "files": {
            verify.MANIFEST_NAME: {
                "sha256_now": _sha256(manifest),
                "prior_recorded_sha256": _sha256(manifest),
                "raw_rows": 100,
            },
            verify.BASELINE_NAME: {
                "sha256_now": _sha256(baseline),
                "raw_rows": 200,
            },
            verify.CAPABILITY_NAME: {
                "sha256_now": _sha256(capability),
                "raw_rows": 6224,
            },
            verify.JUDGE_NAME: {
                "sha256_now": _sha256(judge),
                "raw_rows": 203,
                "successful_rows": 200,
                "technical_error_retry_rows": 3,
                "judge_model": "deepseek-ai/DeepSeek-V3",
                "judge_provider": "novita",
                "system_prompt_sha256": "system-hash",
                "user_template_sha256": "template-hash",
            },
        },
        "hf_repo_id": "example/repo",
        "hf_revision": "a" * 40,
        "provenance_boundary": {},
    }
    paths = {
        verify.MANIFEST_NAME: manifest,
        verify.BASELINE_NAME: baseline,
        verify.CAPABILITY_NAME: capability,
        verify.JUDGE_NAME: judge,
    }
    return paths, receipt


def test_verifier_accepts_append_only_judge_retry_log(tmp_path):
    paths, receipt = _synthetic_recovery_fixture(tmp_path)
    report = verify.verify_paths(paths, receipt)
    assert report["status"] == "PASS"
    assert report["manifest_rows"] == 100
    assert report["expected_item_condition_keys"] == 200


def test_verifier_rejects_ambiguous_extra_judge_row(tmp_path):
    paths, receipt = _synthetic_recovery_fixture(tmp_path)
    judge = paths[verify.JUDGE_NAME]
    rows = verify.read_jsonl(judge)
    rows.append({"item_id": "item-000", "condition": "UNSTEERED", "score": None, "technical_error": None})
    _write_jsonl(judge, rows)
    receipt["files"][verify.JUDGE_NAME]["sha256_now"] = _sha256(judge)
    receipt["files"][verify.JUDGE_NAME]["raw_rows"] = len(rows)
    with pytest.raises(ValueError, match="invalid/ambiguous"):
        verify.validate_judge(judge, receipt, verify.expected_keys(verify.read_jsonl(paths[verify.MANIFEST_NAME])))
