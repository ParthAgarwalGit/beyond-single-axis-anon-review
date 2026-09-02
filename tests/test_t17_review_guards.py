"""Regression tests for the freeze-critical T17 review guards."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t17 = _load_tool("t17_review_pool_tool", "tools/run_t17_pool_sensitivity.py")
c160 = _load_tool("t17_review_c160_tool", "tools/run_t17_c160_geometry.py")


def _args(**kw):
    base = dict(
        layout="c80", block="C80-A", analysis_status="primary",
        membership="all_valid", retained_roles=None, scores=None,
        membership_authority=None, hf_repo="[Author-A-HF]/persona-artifacts",
        hf_revision="a" * 40,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_default_join_rejects_duplicate_rollout_id(tmp_path):
    p = tmp_path / "C80-A"
    p.mkdir()
    rows = [
        {"rollout_id": "u1", "default_condition_index": 0},
        {"rollout_id": "u1", "default_condition_index": 1},
    ]
    (p / "defaults.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="duplicate rollout_id"):
        t17._c80_default_conditions(tmp_path, "C80-A")


def test_default_join_rejects_missing_uid(tmp_path):
    p = tmp_path / "C80-A"
    p.mkdir()
    (p / "defaults.jsonl").write_text(json.dumps({"default_condition_index": 0}) + "\n")
    with pytest.raises(SystemExit, match="missing rollout_id"):
        t17._c80_default_conditions(tmp_path, "C80-A")


def test_default_join_rejects_t11_row_without_t12_match():
    with pytest.raises(SystemExit, match="have no T12 default row"):
        t17._validate_default_join_complete({"u1": "0", "u2": "1"}, {"u1"})


def test_root_manifest_fallback_is_accepted_when_global(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"middle_block_index": t17.MIDDLE_BLOCK_INDEX}))
    t17._apply_layout("c80", "C80-A")
    got = t17._resolve_c80_manifest(tmp_path, "C80-A")
    assert got["path"] == "manifest.json"
    assert len(got["sha256"]) == 64


def test_manifest_declared_block_mismatch_refuses(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"block": "C80-B"}))
    t17._apply_layout("c80", "C80-A")
    with pytest.raises(SystemExit, match="declares block"):
        t17._resolve_c80_manifest(tmp_path, "C80-A")


def test_e80_primary_refuses_cleanly(tmp_path):
    t17._apply_layout("e80")
    with pytest.raises(SystemExit, match="C80-only"):
        t17._validate_run_request(_args(layout="e80"), {"r"})


def test_score3_requires_retained_roles():
    t17._apply_layout("c80", "C80-A")
    with pytest.raises(SystemExit, match="requires --retained-roles"):
        t17._validate_run_request(_args(analysis_status="sensitivity", membership="score3",
                                        retained_roles=None, scores="scores.jsonl"), None)


def test_score3_cannot_be_primary():
    t17._apply_layout("c80", "C80-A")
    with pytest.raises(SystemExit, match="cannot be stamped primary"):
        t17._validate_run_request(_args(membership="score3", scores="scores.jsonl"), {"r"})


def test_primary_requires_retained_roles():
    t17._apply_layout("c80", "C80-A")
    with pytest.raises(SystemExit, match="requires --retained-roles"):
        t17._validate_run_request(_args(), None)


def test_primary_requires_valid_authority_name(tmp_path):
    wrong = tmp_path / "wrong.md"
    wrong.write_text("authority")
    t17._apply_layout("c80", "C80-A")
    with pytest.raises(SystemExit, match="unexpected membership-authority filename"):
        t17._validate_run_request(_args(membership_authority=str(wrong)), {"r"})


def test_primary_requires_immutable_hf_revision(tmp_path):
    auth = tmp_path / t17.MEMBERSHIP_AUTHORITY_NAME
    auth.write_text("authority")
    t17._apply_layout("c80", "C80-A")
    with pytest.raises(SystemExit, match="immutable 40-hex"):
        t17._validate_run_request(_args(membership_authority=str(auth), hf_revision="main"), {"r"})


def test_valid_primary_guard_returns_authority_hash(tmp_path):
    auth = tmp_path / t17.MEMBERSHIP_AUTHORITY_NAME
    auth.write_text("authority")
    t17._apply_layout("c80", "C80-A")
    ok, sha = t17._validate_run_request(_args(membership_authority=str(auth)), {"r"})
    assert ok is True and len(sha) == 64


def _manifest(block, **overrides):
    m = {
        "layout": "c80", "arms": [block], "analysis_status": "primary",
        "provisional": False, "membership_rule": "all_valid",
        "middle_block_index": c160.MIDDLE_BLOCK_INDEX, "retained_role_count": 275,
        "retained_role_manifest_sha256": "r" * 64,
        "membership_authority_sha256": "a" * 64,
        "hf_dataset_repo": "[Author-A-HF]/persona-artifacts",
        "hf_dataset_revision": "b" * 40,
    }
    m.update(overrides)
    return m


@pytest.mark.parametrize("field,value,match", [
    ("layout", "e80", "expected primary reduction"),
    ("analysis_status", "sensitivity", "not stamped as primary"),
    ("membership_rule", "score3", "must be all_valid"),
    ("middle_block_index", 15, "middle block mismatch"),
    ("retained_role_count", 274, "expected frozen 275-role"),
    ("retained_role_manifest_sha256", None, "manifest hash missing"),
    ("membership_authority_sha256", None, "authority hash missing"),
    ("hf_dataset_revision", "main", "revision missing/invalid"),
])
def test_validate_pair_failure_modes(field, value, match):
    a, b = _manifest("C80-A"), _manifest("C80-B")
    a[field] = value
    with pytest.raises(SystemExit, match=match):
        c160.validate_pair(a, b)


def test_validate_pair_rejects_cross_block_revision_mismatch():
    a, b = _manifest("C80-A"), _manifest("C80-B", hf_dataset_revision="c" * 40)
    with pytest.raises(SystemExit, match="revisions differ"):
        c160.validate_pair(a, b)


def _save_single(path, means):
    names = sorted(means)
    np.savez(path, names=np.array(names, dtype=object), matrix=np.stack([means[n] for n in names]))


def test_combined_default_enforces_each_block_floor_before_pooling(tmp_path):
    ra, rb = tmp_path / "A", tmp_path / "B"
    da, db = ra / "DEFAULT-C80-A", rb / "DEFAULT-C80-B"
    da.mkdir(parents=True); db.mkdir(parents=True)
    means = {str(i): np.array([float(i), 1.0]) for i in range(5)}
    _save_single(da / "all_response.npz", means)
    _save_single(db / "all_response.npz", means)
    ca = {str(i): {"total": 100, "eligible": (90 if i == 0 else 100)} for i in range(5)}
    cb = {str(i): {"total": 100, "eligible": 100} for i in range(5)}
    (da / "condition_counts_by_pool.json").write_text(json.dumps({"all_response": ca}))
    (db / "condition_counts_by_pool.json").write_text(json.dumps({"all_response": cb}))
    with pytest.raises(SystemExit, match="C80-A all_response default floor failed"):
        c160.combined_default(ra, rb, "all_response")
