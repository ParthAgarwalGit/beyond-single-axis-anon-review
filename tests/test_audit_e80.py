"""T09 E80 acceptance auditor: frozen expectations and blocked behaviour."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "audit_e80_acceptance", REPO_ROOT / "tools" / "audit_e80_acceptance.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

LU_PRESENT = (REPO_ROOT / "data" / "lu_et_al" / "roles.json").is_file()


@pytest.mark.skipif(not LU_PRESENT, reason="pinned Lu artifacts not checked out")
def test_expected_cell_counts_match_frozen_design():
    lu = audit.load_lu_inputs(REPO_ROOT / "data" / "lu_et_al")
    assert len(lu["role_ids"]) == 275
    assert len(audit.E80_IDS) == 80
    # 275 roles x 80 questions per role arm; 80 questions x 5 conditions.
    assert len(lu["role_ids"]) * len(audit.E80_IDS) == 22000
    assert len(audit.E80_IDS) * audit.N_DEFAULT_CONDITIONS == 400
    # Every E80 question has a frozen prompt index, balanced 16 per index.
    per_index = {}
    for qid in audit.E80_IDS:
        per_index[audit.FROZEN_PROMPT_INDEX[qid]] = (
            per_index.get(audit.FROZEN_PROMPT_INDEX[qid], 0) + 1
        )
    assert per_index == {i: 16 for i in range(5)}


def test_missing_data_root_produces_blocked_report(tmp_path):
    out = tmp_path / "report.json"
    code = audit.main([
        "--data-root", str(tmp_path / "does-not-exist"),
        "--lu-data", str(REPO_ROOT / "data" / "lu_et_al"),
        "--output", str(out),
        "--allow-dirty",
    ])
    assert code == 1
    report = json.loads(out.read_text())
    assert report["overall"] == "BLOCKED"
    assert len(report["checks"]) == 11
    assert {c["status"] for c in report["checks"]} == {"BLOCKED"}
    assert report["config_sha256"]
    assert report["source_git_sha"]


def test_blocked_without_lu_artifacts(tmp_path):
    out = tmp_path / "report.json"
    code = audit.main([
        "--data-root", str(tmp_path),
        "--lu-data", str(tmp_path / "no-lu"),
        "--output", str(out),
        "--allow-dirty",
    ])
    assert code == 1
    report = json.loads(out.read_text())
    assert report["overall"] == "BLOCKED"
    assert report["inputs"]["lu_artifacts_present"] is False
