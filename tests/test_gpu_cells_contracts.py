from __future__ import annotations

import hashlib

import pytest

from tools.gpu_cells.t11_default_generation import (
    FROZEN_ID_LIST_SHA256,
    FROZEN_QUESTION_IDS,
    MODEL,
    canonical_sha256,
    default_rollout_id,
    generation_row_id,
)
from tools.gpu_cells.t12_c80_activation_extract import (
    select_authoritative_attempts,
    verify_part_manifest_files,
)
from src.ids import generation_row_id as canonical_generation_row_id
from src.ids import rollout_id as canonical_rollout_id


def test_t12_attempt_selection_lowest_valid_else_highest_failed():
    rows = [
        {"rollout_id": "a", "retry": 0, "technical_validity": "truncated"},
        {"rollout_id": "a", "retry": 1, "technical_validity": "valid"},
        {"rollout_id": "a", "retry": 2, "technical_validity": "valid"},
        {"rollout_id": "b", "retry": 0, "technical_validity": "truncated"},
        {"rollout_id": "b", "retry": 4, "technical_validity": "generation_error"},
    ]
    got = {row["rollout_id"]: row for row in select_authoritative_attempts(rows)}
    assert got["a"]["retry"] == 1
    assert got["a"]["technical_validity"] == "valid"
    assert got["b"]["retry"] == 4
    assert got["b"]["technical_validity"] != "valid"


def test_t12_attempt_selection_rejects_duplicate_retry_index():
    rows = [
        {"rollout_id": "x", "retry": 0, "technical_validity": "truncated"},
        {"rollout_id": "x", "retry": 0, "technical_validity": "valid"},
    ]
    with pytest.raises(RuntimeError, match="duplicate retry"):
        select_authoritative_attempts(rows)


def test_t12_resume_manifest_verifies_every_referenced_hash():
    meta = b'{"uid":"x"}\n'
    tensor = b"tensor-bytes"
    doc = {
        "part_index": 7,
        "files": {
            "metadata": {
                "path": "meta_part0007.jsonl",
                "sha256": hashlib.sha256(meta).hexdigest(),
            },
            "all_response": {
                "path": "all_response_part0007.safetensors",
                "sha256": hashlib.sha256(tensor).hexdigest(),
            },
        },
    }
    files = {
        "meta_part0007.jsonl": meta,
        "all_response_part0007.safetensors": tensor,
    }
    assert verify_part_manifest_files(doc, files.get, expected_part_index=7)

    missing = dict(files)
    missing.pop("all_response_part0007.safetensors")
    with pytest.raises(RuntimeError, match="missing file"):
        verify_part_manifest_files(doc, missing.get, expected_part_index=7)

    corrupt = dict(files)
    corrupt["all_response_part0007.safetensors"] = b"corrupt"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_part_manifest_files(doc, corrupt.get, expected_part_index=7)


def test_t11_embedded_block_ids_match_frozen_hashes():
    for block, ids in FROZEN_QUESTION_IDS.items():
        assert len(ids) == 80
        assert len(set(ids)) == 80
        assert canonical_sha256(list(ids)) == FROZEN_ID_LIST_SHA256[block]


def test_t11_default_ids_match_current_canonical_ids():
    for qid in (0, 1, 237):
        for condition_index in range(5):
            observed = default_rollout_id(MODEL, qid, condition_index)
            expected = canonical_rollout_id(
                MODEL, "DEFAULT", "", qid, condition_index
            )
            assert observed == expected
            assert generation_row_id(observed, 0) == canonical_generation_row_id(expected, 0)
