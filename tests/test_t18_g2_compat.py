"""Regression tests for the outcome-blind T18/G2 membership compatibility fix."""

import json
from pathlib import Path

import pytest

from tests.test_t18_loader import CFG, N_ROLES, build_run_dir
from tools import run_t18_g2_dimensional_adequacy as compat


def _test_g2():
    geom = CFG["confirmatory_geometry"]
    return {
        "freeze_status": "FROZEN",
        "geometry": {
            "membership": compat.MEMBERSHIP,
            "pool": geom["primary_pool"],
            "block_index": geom["primary_layer"],
            "prompt_arm": geom["primary_prompt_arm"],
            "blocks": list(geom["primary_blocks"]),
            "retained_roles": N_ROLES,
        },
        "authority": {
            "canonical_result_path": "results/t15/confirmatory_reliability.json",
            "membership_authority_pr": 44,
        },
        "t18_compatibility": {
            "judge_filter_for_primary_rows": "FORBIDDEN",
        },
    }


def _patch_g2(monkeypatch):
    monkeypatch.setattr(compat, "_load_g2_record", lambda: (_test_g2(), "a" * 64))


def test_real_g2_record_is_frozen_and_label_independent():
    record = json.loads(Path("docs/G2_GEOMETRY_FREEZE.json").read_text())
    assert record["freeze_status"] == "FROZEN"
    assert record["frozen_before_any_real_t18_outcome_inspected"] is True
    assert record["geometry"]["membership"] == compat.MEMBERSHIP
    assert record["t18_compatibility"]["judge_filter_for_primary_rows"] == "FORBIDDEN"
    assert record["t18_compatibility"]["analysis_parameters_unchanged"] is True


def test_g2_loader_does_not_filter_on_role_judge_scores(tmp_path, monkeypatch):
    _patch_g2(monkeypatch)
    run = build_run_dir(tmp_path, role_score=0, sensitivity_score=0)

    rows, provenance = compat.load_rows_g2(run, CFG)

    assert provenance["membership"] == compat.MEMBERSHIP
    assert provenance["judge_filter_applied"] is False
    assert provenance["n_roles"] == N_ROLES
    assert set(rows.role_roles) == {f"role_{r:02d}" for r in range(N_ROLES)}


def test_g2_loader_does_not_require_judge_jsonl(tmp_path, monkeypatch):
    _patch_g2(monkeypatch)
    run = build_run_dir(tmp_path)
    (run / "judge.jsonl").unlink()

    rows, provenance = compat.load_rows_g2(run, CFG)

    assert len(rows.role_roles) == N_ROLES
    assert provenance["judge_filter_applied"] is False


def test_g2_loader_still_fails_closed_on_eligibility_mismatch(tmp_path, monkeypatch):
    _patch_g2(monkeypatch)
    run = build_run_dir(
        tmp_path,
        eligible_roles=[f"role_{r:02d}" for r in range(N_ROLES - 1)],
    )

    with pytest.raises(SystemExit, match="eligible-role artifact declares"):
        compat.load_rows_g2(run, CFG)
