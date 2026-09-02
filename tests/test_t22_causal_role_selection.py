import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.causal_role_selection import assistant_axis, reconcile_observed_roles, select_p50
from src.t22_method_succession import DEVIATION_ID, prepare_config_successor
from tools.run_t22_p50 import run


def planted(n=70, d=8):
    role_ids = ["assistant"] + [f"role_{i:03d}" for i in range(n - 1)]
    means = np.zeros((n, d), dtype=np.float64)
    means[:, 0] = np.linspace(-2.0, 2.0, n)
    mu_default = np.zeros(d, dtype=np.float64)
    mu_default[0] = 5.0
    return role_ids, means, mu_default


def test_selects_exactly_50_and_excludes_literal_assistant():
    ids, means, default = planted()
    out = select_p50(ids, means, default)
    selected = [r["role_id"] for r in out["selected_roles"]]
    assert len(selected) == 50
    assert len(set(selected)) == 50
    assert "assistant" not in selected
    assert out["p50_size"] == 50


def test_deterministic_and_monotone_scores():
    ids, means, default = planted()
    a = select_p50(ids, means, default)
    b = select_p50(ids, means, default)
    assert a["membership_sha256"] == b["membership_sha256"]
    scores = [r["assistant_side_score"] for r in a["selected_roles"]]
    assert scores == sorted(scores, reverse=True)


def test_lexical_tie_break():
    ids, means, default = planted(n=53)
    for rid in ("role_001", "role_002", "role_003"):
        means[ids.index(rid), 0] = 10.0
    out = select_p50(ids, means, default)
    tied = [r["role_id"] for r in out["selected_roles"][:3]]
    assert tied == ["role_001", "role_002", "role_003"]


def test_missing_literal_assistant_fails_closed():
    ids, means, default = planted()
    ids[0] = "not_assistant"
    with pytest.raises(ValueError):
        select_p50(ids, means, default)


def test_axis_orientation_is_toward_default():
    ids, means, default = planted()
    axis, unit = assistant_axis(default, means)
    assert np.linalg.norm(axis) > 0
    assert np.isclose(np.linalg.norm(unit), 1.0)
    assert default @ unit > means.mean(axis=0) @ unit


def test_reconcile_observed_roles():
    assert reconcile_observed_roles(["a", "b"], ["b", "a", "a"])["match"]
    bad = reconcile_observed_roles(["a", "b"], ["a", "c"])
    assert not bad["match"]
    assert bad["missing"] == ["b"]
    assert bad["unexpected"] == ["c"]


def _write_single(path: Path, names, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, names=np.asarray(names, dtype=object), matrix=np.asarray(matrix))


def _fake_reduced(tmp_path: Path) -> Path:
    root = tmp_path / "e80"
    ids, means, _ = planted(n=70, d=8)
    _write_single(root / "translated" / "available_all_response.npz", ids, means)

    conds = [str(i) for i in range(5)]
    defaults = np.zeros((5, 8), dtype=np.float64)
    defaults[:, 0] = 5.0
    _write_single(root / "default" / "all_response.npz", conds, defaults)
    (root / "default" / "condition_counts_by_pool.json").write_text(
        json.dumps({"all_response": {c: {"eligible": 80, "total": 80} for c in conds}}),
        encoding="utf-8",
    )
    (root / "reduced_manifest.json").write_text(
        json.dumps({
            "layout": "e80",
            "membership_rule": "all_valid",
            "middle_block_index": 16,
            "t12_activation_manifest_sha256": "a" * 64,
        }),
        encoding="utf-8",
    )
    return root


def test_candidate_manifest_records_downstream_exclusion_as_requirement_not_fact(tmp_path):
    reduced = _fake_reduced(tmp_path)
    out_manifest = tmp_path / "P50.json"
    out_report = tmp_path / "report.json"
    rep = run(reduced, out_manifest, out_report, None, finalize_method_succession=False)
    doc = json.loads(out_manifest.read_text(encoding="utf-8"))

    assert rep["status"] == "PASS_CANDIDATE_ONLY"
    assert doc["status"] == "CANDIDATE_NOT_METHOD_BOUND"
    assert "p50_excluded_from_t24_heldout_axis" not in doc
    req = doc["heldout_axis_requirement"]
    assert req["p50_must_be_excluded_from_t24_heldout_axis"] is True
    assert req["verified_by_t22"] is False
    assert req["verification_owner"] == "T24/T25"


def test_successor_registers_deviation_and_fills_manifest_hash_together():
    base = """schema_version: method-freeze.4
causal_role_selection:
  selection_data: E80 score3 historical text
  P50_manifest_sha256: null
deviation_records:
- id: DEV-OLD
  decision_date: '2026-08-01'
  task: OLD
  reason: old
  affected_frozen_fields: [x]
  outcomes_already_inspected: false
  reviewer: reviewer
  reviewer_approval_status: APPROVED
v4_amendment:
  base_config_path: configs/method_frozen.yaml
"""
    manifest_sha = "b" * 64
    successor, body_sha = prepare_config_successor(base, manifest_sha)
    parsed = yaml.safe_load(successor)

    assert parsed["causal_role_selection"]["P50_manifest_sha256"] == manifest_sha
    recs = [r for r in parsed["deviation_records"] if r["id"] == DEVIATION_ID]
    assert len(recs) == 1
    assert "causal_role_selection.selection_data" in recs[0]["affected_frozen_fields"]
    assert "causal_role_selection.P50_manifest_sha256" in recs[0]["affected_frozen_fields"]
    assert len(body_sha) == 64


def test_successor_refuses_split_or_conflicting_p50_fill():
    already_filled = """causal_role_selection:
  P50_manifest_sha256: cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
deviation_records: []
v4_amendment: {}
"""
    with pytest.raises(ValueError, match="already filled"):
        prepare_config_successor(already_filled, "b" * 64)

    split = f"""causal_role_selection:
  P50_manifest_sha256: null
deviation_records:
- id: {DEVIATION_ID}
v4_amendment: {{}}
"""
    with pytest.raises(ValueError, match="already registered"):
        prepare_config_successor(split, "b" * 64)
