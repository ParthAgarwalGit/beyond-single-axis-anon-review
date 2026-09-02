"""Tests for tools/run_t15_t16_resplits.py's per-partition geometry rebuild.

Pins the core statistic (_draw_statistic and its two helpers) against
synthetic data with a known structure, and checks the fail-closed guards
that protect against a resplit silently computing over incomplete data.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "run_t15_t16_resplits_test", ROOT / "tools" / "run_t15_t16_resplits.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()

N_Q, N_R, D = 12, 8, 16


def _synthetic():
    rng = np.random.default_rng(0)
    role_grid = rng.standard_normal((N_Q, N_R, D)).astype(np.float32)
    role_questions = np.arange(1000, 1000 + N_Q)
    role_names = np.array([f"role{i}" for i in range(N_R)], dtype=object)
    # Shift the default grid well away from the role grid so assistant_axis's
    # sign check (default_proj > mean_role_proj) is satisfied by construction.
    default_grid = rng.standard_normal((N_Q, 5, D)).astype(np.float32) + 3.0
    default_questions = role_questions.copy()

    q_index = {int(q): i for i, q in enumerate(role_questions)}
    r_index = {str(r): i for i, r in enumerate(role_names)}
    dq_index = {int(q): i for i, q in enumerate(default_questions)}
    retained_roles = set(role_names.tolist())
    return role_grid, q_index, r_index, default_grid, dq_index, retained_roles, role_questions


def test_draw_statistic_produces_valid_ranges():
    role_grid, q_index, r_index, default_grid, dq_index, retained_roles, role_questions = _synthetic()
    half1, half2 = role_questions[:6].tolist(), role_questions[6:].tolist()

    r, cos, n = TOOL._draw_statistic(role_grid, q_index, r_index, default_grid, dq_index,
                                     retained_roles, half1, half2)

    assert -1.0 <= r <= 1.0
    assert -1.0 <= cos <= 1.0
    assert n == N_R


def test_draw_statistic_is_deterministic():
    role_grid, q_index, r_index, default_grid, dq_index, retained_roles, role_questions = _synthetic()
    half1, half2 = role_questions[:6].tolist(), role_questions[6:].tolist()

    a = TOOL._draw_statistic(role_grid, q_index, r_index, default_grid, dq_index, retained_roles, half1, half2)
    b = TOOL._draw_statistic(role_grid, q_index, r_index, default_grid, dq_index, retained_roles, half1, half2)
    assert a == b


def test_half_role_means_fails_closed_on_missing_question():
    role_grid, q_index, r_index, _, _, retained_roles, _ = _synthetic()
    with pytest.raises(SystemExit, match="no role data at all"):
        TOOL._half_role_means(role_grid, q_index, r_index, retained_roles, [999999])


def test_half_role_means_fails_closed_on_zero_eligible_role():
    role_grid, q_index, r_index, _, _, retained_roles, role_questions = _synthetic()
    role_grid = role_grid.copy()
    role_grid[:6, 0, :] = np.nan  # role0 has no data among the first 6 questions
    with pytest.raises(SystemExit, match="zero eligible cells"):
        TOOL._half_role_means(role_grid, q_index, r_index, retained_roles, role_questions[:6].tolist())


def test_half_default_mean_fails_closed_below_valid_floor():
    _, _, _, default_grid, dq_index, _, role_questions = _synthetic()
    default_grid = default_grid.copy()
    # Blank out all but one row of condition 0 across all 12 questions - well
    # below the frozen 0.95 valid-fraction floor.
    default_grid[1:, 0, :] = np.nan
    with pytest.raises(SystemExit, match="below frozen"):
        TOOL._half_default_mean(default_grid, dq_index, role_questions.tolist())


def test_half_role_means_excludes_nan_cells_from_the_average():
    role_grid, q_index, r_index, _, _, retained_roles, role_questions = _synthetic()
    role_grid = role_grid.copy()
    half = role_questions[:6].tolist()
    # role0's mean over the half should equal the mean of only its non-NaN cells.
    role_grid[0, 0, :] = np.nan
    means = TOOL._half_role_means(role_grid, q_index, r_index, retained_roles, half)
    idx = [q_index[q] for q in half]
    expected = role_grid[idx][1:, 0, :].mean(axis=0)
    assert np.allclose(means["role0"], expected)


def test_verify_manifest_authority_accepts_the_real_committed_manifest():
    import json
    manifest = json.loads((ROOT / "results" / "t15" / "t15_t16_resplit_manifest.json")
                          .read_text(encoding="utf-8"))
    TOOL.verify_manifest_authority(manifest)  # must not raise


def test_verify_manifest_authority_rejects_a_tampered_hash():
    import json
    manifest = json.loads((ROOT / "results" / "t15" / "t15_t16_resplit_manifest.json")
                          .read_text(encoding="utf-8"))
    manifest["draws"][0]["half_1_question_ids"] = manifest["draws"][0]["half_1_question_ids"][::-1]
    with pytest.raises(SystemExit, match="manifest_sha256 does not verify"):
        TOOL.verify_manifest_authority(manifest)


def test_verify_manifest_authority_rejects_a_self_consistent_but_different_manifest():
    """A manifest can be edited and its own manifest_sha256 trivially
    recomputed to match - passing every self-consistency check while no
    longer being the exact outcome-blind manifest PR #71 froze. Only a
    binding to the frozen expected hash catches this."""
    import json
    manifest = json.loads((ROOT / "results" / "t15" / "t15_t16_resplit_manifest.json")
                          .read_text(encoding="utf-8"))
    manifest["draws"][0]["half_1_question_ids"], manifest["draws"][0]["half_2_question_ids"] = (
        manifest["draws"][0]["half_2_question_ids"], manifest["draws"][0]["half_1_question_ids"],
    )
    manifest["manifest_sha256"] = TOOL.canonical_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )
    with pytest.raises(SystemExit, match="not enough"):
        TOOL.verify_manifest_authority(manifest)


def test_verify_manifest_authority_rejects_wrong_authority_or_draw_count():
    import json
    manifest = json.loads((ROOT / "results" / "t15" / "t15_t16_resplit_manifest.json")
                          .read_text(encoding="utf-8"))

    bad_authority = dict(manifest, authority="docs/SOME_OTHER_SPEC.md")
    with pytest.raises(SystemExit, match="manifest authority"):
        TOOL.verify_manifest_authority(bad_authority)

    bad_count = dict(manifest, n_draws=299)
    with pytest.raises(SystemExit, match="n_draws"):
        TOOL.verify_manifest_authority(bad_count)


def test_verify_hf_revision_authority_accepts_the_canonical_revision():
    import json
    canonical = json.loads(TOOL.CANONICAL_RELIABILITY_RESULT.read_text(encoding="utf-8"))
    authorized = canonical["input_provenance"]["bundle_fetched_at_hf_revision"]
    TOOL.verify_hf_revision_authority(authorized)  # must not raise


def test_verify_hf_revision_authority_rejects_a_mismatched_revision():
    with pytest.raises(SystemExit, match="atoms bundle HF revision"):
        TOOL.verify_hf_revision_authority("0" * 40)
