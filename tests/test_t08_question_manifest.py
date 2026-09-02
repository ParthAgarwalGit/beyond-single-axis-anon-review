"""Tests for the frozen T08 question/block manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def load_outputs():
    cfg = yaml.safe_load(
        (ROOT / "configs/method_frozen.yaml").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "design/questions_240_manifest.json").read_text(encoding="utf-8")
    )
    blocks = {
        "E80": json.loads(
            (ROOT / "design/questions_E80.json").read_text(encoding="utf-8")
        ),
        "C80-A": json.loads(
            (ROOT / "design/questions_C80_A.json").read_text(encoding="utf-8")
        ),
        "C80-B": json.loads(
            (ROOT / "design/questions_C80_B.json").read_text(encoding="utf-8")
        ),
    }
    return cfg, manifest, blocks


def test_builder_reproduces_outputs(tmp_path):
    # The builder now fails closed on any --out-dir outside the repo (that is
    # the exact PR #70 bug: a scratch temp dir got committed as a block-file
    # path). Reproducibility must therefore be checked from a repo-relative,
    # git-ignored scratch directory, not pytest's default tmp_path.
    scratch = ROOT / ".cache" / f"test_t08_reproduce_{tmp_path.name}"
    try:
        result = subprocess.run(
            [
                sys.executable,
                "tools/build_t08_question_manifest.py",
                "--config", "configs/method_frozen.yaml",
                "--questions", "data/lu_et_al/questions.json",
                "--out-dir", str(scratch),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(result.stdout)
        assert report["assignment_sha256"] == (
            "6f0213dd56073d3ce87b835e8c7b8a7759ab9b53e78486bced689e3f0c1e6172"
        )
        for filename in (
            "questions_E80.json",
            "questions_C80_A.json",
            "questions_C80_B.json",
        ):
            assert (scratch / filename).read_bytes() == (
                ROOT / "design" / filename
            ).read_bytes()

        generated = json.loads(
            (scratch / "questions_240_manifest.json").read_text(encoding="utf-8")
        )
        committed = json.loads(
            (ROOT / "design/questions_240_manifest.json").read_text(encoding="utf-8")
        )
        # Paths legitimately differ (scratch dir vs. design/) - strip them
        # and compare everything else exactly, then check the paths
        # separately below rather than overwriting one side with the other,
        # which would hide a real path-portability regression instead of
        # catching it.
        def strip_paths(m):
            return {
                **m,
                "block_files": {
                    name: {k: v for k, v in info.items() if k != "path"}
                    for name, info in m["block_files"].items()
                },
                "method_config": {k: v for k, v in m["method_config"].items() if k != "path"},
                "source_questions": {k: v for k, v in m["source_questions"].items() if k != "path"},
            }

        assert strip_paths(generated) == strip_paths(committed)

        block_filenames = {
            "E80": "questions_E80.json",
            "C80-A": "questions_C80_A.json",
            "C80-B": "questions_C80_B.json",
        }
        for block_name, filename in block_filenames.items():
            assert generated["block_files"][block_name]["path"] == (
                f".cache/test_t08_reproduce_{tmp_path.name}/{filename}"
            )
            assert committed["block_files"][block_name]["path"] == f"design/{filename}"
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def test_blocks_partition_all_source_ids():
    _, _, blocks = load_outputs()
    ids = []
    for name, block in blocks.items():
        block_ids = [row["question_id"] for row in block["questions"]]
        assert len(block_ids) == len(set(block_ids)) == 80, name
        ids.extend(block_ids)
    assert len(ids) == len(set(ids)) == 240
    assert set(ids) == set(range(240))


def test_duplicate_pair_stays_in_c80a():
    _, _, blocks = load_outputs()
    c80a = {row["question_id"] for row in blocks["C80-A"]["questions"]}
    assert {7, 227} <= c80a
    text_by_id = {
        row["question_id"]: row["question_text"]
        for row in blocks["C80-A"]["questions"]
    }
    assert text_by_id[7] == text_by_id[227]


def test_prompt_balance_and_assignment_rule():
    cfg, manifest, blocks = load_outputs()
    for block in blocks.values():
        rows = block["questions"]
        counts = Counter(row["prompt_index"] for row in rows)
        assert counts == Counter({0: 16, 1: 16, 2: 16, 3: 16, 4: 16})
        for position, row in enumerate(rows):
            assert row["block_position"] == position
            assert row["prompt_index"] == position % 5
    assert manifest["prompt_assignment"]["counts_across_240"] == {
        "0": 48, "1": 48, "2": 48, "3": 48, "4": 48
    }


def test_source_and_method_hashes_match_files():
    _, manifest, _ = load_outputs()
    method_hash = hashlib.sha256(
        (ROOT / "configs/method_frozen.yaml").read_bytes()
    ).hexdigest()
    source_hash = hashlib.sha256(
        (ROOT / "data/lu_et_al/questions.json").read_bytes()
    ).hexdigest()
    assert manifest["method_config"]["sha256"] == method_hash
    assert manifest["source_questions"]["sha256"] == source_hash


def test_block_file_hashes_match_committed_bytes():
    _, manifest, _ = load_outputs()
    for info in manifest["block_files"].values():
        path = ROOT / info["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]


def test_manifest_paths_are_canonical_repo_relative():
    """Every path recorded in the committed manifest must be portable.

    A machine-local absolute path (e.g. a scratch --out-dir used during
    development) must never end up in committed provenance: it breaks the
    hash test on any other checkout and silently records where a file
    happened to be generated instead of what the frozen artifact actually
    is. See PR #70 review: block_files paths were once committed as
    [local-path]
    """
    _, manifest, _ = load_outputs()
    expected_block_paths = {
        "E80": "design/questions_E80.json",
        "C80-A": "design/questions_C80_A.json",
        "C80-B": "design/questions_C80_B.json",
    }
    for block, info in manifest["block_files"].items():
        assert info["path"] == expected_block_paths[block]

    assert manifest["method_config"]["path"] == "configs/method_frozen.yaml"
    assert manifest["source_questions"]["path"] == "data/lu_et_al/questions.json"

    for path_str in (
        manifest["method_config"]["path"],
        manifest["source_questions"]["path"],
        *(info["path"] for info in manifest["block_files"].values()),
    ):
        assert not Path(path_str).is_absolute(), f"{path_str!r} is not repo-relative"
        assert (ROOT / path_str).exists(), f"{path_str!r} does not resolve under {ROOT}"


def test_question_hashes_match_text():
    _, _, blocks = load_outputs()
    for block in blocks.values():
        for row in block["questions"]:
            observed = hashlib.sha256(
                row["question_text"].encode("utf-8")
            ).hexdigest()[:16]
            assert observed == row["question_content_sha256"]


def test_pre_render_manifest_is_not_generation_ready():
    _, manifest, _ = load_outputs()
    assert manifest["status"] == "DRAFT_PRE_RENDER"
    assert manifest["completion_gate"]["question_manifest_validated"] is True
    assert manifest["completion_gate"]["ready_for_generation"] is False


def test_manifest_path_helper_refuses_a_path_outside_the_repo(tmp_path):
    """Fail-closed check for the bug PR #70 review caught: a scratch/temp
    --out-dir must never silently end up recorded in committed provenance."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_t08_question_manifest_test", ROOT / "tools" / "build_t08_question_manifest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.manifest_path(ROOT / "design" / "questions_E80.json") == "design/questions_E80.json"

    import pytest
    with pytest.raises(ValueError, match="not under"):
        module.manifest_path(tmp_path / "questions_E80.json")
