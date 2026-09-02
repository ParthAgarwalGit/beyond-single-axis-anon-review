#!/usr/bin/env python3
"""Validate T02 method_frozen.yaml without model access."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    config_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("configs/method_frozen.yaml")
    )
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    qd = cfg["question_design"]
    blocks = {
        name: spec["question_ids"]
        for name, spec in qd["blocks"].items()
    }

    assert set(blocks) == {"E80", "C80-A", "C80-B"}
    for name, ids in blocks.items():
        assert len(ids) == 80, f"{name}: expected 80 IDs, found {len(ids)}"
        assert len(set(ids)) == 80, f"{name}: duplicate IDs inside block"
        observed_hash = canonical_sha256(ids)
        expected_hash = qd["blocks"][name]["id_list_sha256"]
        assert observed_hash == expected_hash, (
            f"{name}: ID hash mismatch {observed_hash} != {expected_hash}"
        )

    sets = {name: set(ids) for name, ids in blocks.items()}
    assert not sets["E80"] & sets["C80-A"]
    assert not sets["E80"] & sets["C80-B"]
    assert not sets["C80-A"] & sets["C80-B"]
    union = sets["E80"] | sets["C80-A"] | sets["C80-B"]
    assert union == set(range(240)), (
        f"question union must be exactly 0..239; missing={sorted(set(range(240))-union)}, "
        f"extra={sorted(union-set(range(240)))}"
    )

    assert 7 in sets["C80-A"] and 227 in sets["C80-A"], (
        "duplicate text IDs 7 and 227 must both be in C80-A"
    )

    assignment = []
    total_prompt_counts = Counter()
    for block_name, ids in blocks.items():
        counts = Counter()
        for pos, qid in enumerate(ids):
            idx = pos % 5
            counts[idx] += 1
            total_prompt_counts[idx] += 1
            assignment.append(
                {
                    "question_block": block_name,
                    "block_position": pos,
                    "question_id": qid,
                    "prompt_index": idx,
                }
            )
        assert counts == Counter({0: 16, 1: 16, 2: 16, 3: 16, 4: 16}), (
            f"{block_name}: prompt balance wrong: {dict(counts)}"
        )

    assert total_prompt_counts == Counter({0: 48, 1: 48, 2: 48, 3: 48, 4: 48})
    assignment_hash = canonical_sha256(assignment)
    assert assignment_hash == qd["all_blocks_assignment_sha256"], (
        f"assignment hash mismatch {assignment_hash} != "
        f"{qd['all_blocks_assignment_sha256']}"
    )

    assert cfg["activation_extraction"]["primary_pool"] == "ALL_RESPONSE_TOKENS"
    assert cfg["role_vectors_and_axis"]["axis_formula"] == (
        "v = mu_default - mean_r(mu_role_r)"
    )
    assert cfg["activation_extraction"]["middle_layer"]["primary_block_index"] == 16
    assert cfg["role_vectors_and_axis"][
        "project_role_vector_threshold_per_80_id_block"
    ] == 10
    assert cfg["steering"]["unique_conditions"] == [
        "shared_zero",
        "assistant_axis_toward",
        "assistant_axis_away",
        "random_positive",
        "random_negative",
    ]
    assert cfg["causal_role_selection"]["P50_excluded_from_final_causal_axis"] is True
    assert cfg["confirmatory_geometry"]["primary_blocks"] == ["C80-A", "C80-B"]
    assert cfg["confirmatory_geometry"]["primary_pool"] == "ALL_RESPONSE_TOKENS"
    assert cfg["role_judge"]["parse_failure"] == (
        "missing measurement; never convert to 0"
    )

    report = {
        "status": "PASS",
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "question_blocks": {name: len(ids) for name, ids in blocks.items()},
        "question_union": len(union),
        "unique_prompt_counts_across_240": dict(sorted(total_prompt_counts.items())),
        "duplicate_pair_block": "C80-A",
        "assignment_sha256": assignment_hash,
        "review_note": (
            "A PASS validates internal consistency only. Reviewer approval, "
            "runtime revision hashes, T04 hook validation, judge validation, "
            "coefficient freeze, and P50/Axis hashes remain separate gates."
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
