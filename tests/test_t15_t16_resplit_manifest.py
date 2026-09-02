"""Regression tests for the T15/T16 300-resplit manifest generator.

Pins the invariants specified in docs/DEVIATION_2026-08-25_T15_T16_RESPLIT_SPEC.md:
exact partition sizes, prompt-index stratum balance, the forced-together
duplicate pair, determinism, and the provenance-derived master seed.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "build_t15_t16_resplit_manifest_test",
        ROOT / "tools" / "build_t15_t16_resplit_manifest.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def test_master_seed_is_derived_from_the_frozen_assignment_hash():
    assert TOOL.MASTER_SEED == int(TOOL.EXPECTED_ASSIGNMENT_SHA256[:8], 16)
    assert TOOL.MASTER_SEED == 1862407133


def test_manifest_generation_is_deterministic_across_calls():
    a = TOOL.build_manifest(ROOT / "design")
    b = TOOL.build_manifest(ROOT / "design")
    assert a == b
    assert a["manifest_sha256"] == b["manifest_sha256"]


def test_all_300_draws_satisfy_every_frozen_constraint():
    manifest = TOOL.build_manifest(ROOT / "design")
    strata = {int(p): set(ids) for p, ids in manifest["strata"].items()}
    universe = set().union(*strata.values())
    assert len(universe) == 160

    seen = set()
    for draw in manifest["draws"]:
        half_1 = set(draw["half_1_question_ids"])
        half_2 = set(draw["half_2_question_ids"])
        assert len(half_1) == 80 and len(half_2) == 80
        assert half_1.isdisjoint(half_2)
        assert half_1 | half_2 == universe
        assert (7 in half_1) == (227 in half_1), "forced-together pair split"
        for members in strata.values():
            assert len(half_1 & members) == 16
            assert len(half_2 & members) == 16
        key = tuple(sorted(half_1))
        assert key not in seen, "duplicate partition across draws"
        seen.add(key)
        assert draw["spawn_key"] == [draw["draw_index"]]

    assert len(manifest["draws"]) == 300
    assert len(seen) == 300


def test_observed_split_is_recorded_separately_and_not_among_the_300_draws():
    manifest = TOOL.build_manifest(ROOT / "design")
    observed_a = set(manifest["observed_split"]["c80_a_question_ids"])
    assert len(observed_a) == 80
    assert manifest["observed_split"]["assignment_sha256"] == TOOL.EXPECTED_ASSIGNMENT_SHA256

    draw_halves = {
        tuple(sorted(draw["half_1_question_ids"])) for draw in manifest["draws"]
    } | {
        tuple(sorted(draw["half_2_question_ids"])) for draw in manifest["draws"]
    }
    assert tuple(sorted(observed_a)) not in draw_halves, (
        "the historical observed split must not coincide with a generated draw; "
        "if this fires, verify it is a genuine coincidence and not a sampling bug"
    )


def test_forced_together_ids_are_confirmed_in_different_strata():
    strata, strata_of = TOOL.load_question_strata(ROOT / "design")
    a, b = TOOL.FORCED_TOGETHER
    assert strata_of[a] != strata_of[b]


def test_draw_one_partition_rejects_a_stratum_with_both_forced_ids(monkeypatch):
    import numpy as np

    strata = {0: list(range(32))}
    strata_of = {7: 0, 227: 0}
    rng = np.random.default_rng(0)
    with pytest.raises(SystemExit, match="more than one forced ID"):
        TOOL.draw_one_partition(rng, strata, strata_of)


def test_committed_manifest_matches_a_fresh_build_and_is_internally_hashed():
    committed_path = ROOT / "results" / "t15" / "t15_t16_resplit_manifest.json"
    committed = json.loads(committed_path.read_text(encoding="utf-8"))

    committed_stamped_hash = committed["manifest_sha256"]
    recomputed_stamped_hash = TOOL.canonical_sha256(
        {k: v for k, v in committed.items() if k != "manifest_sha256"}
    )
    assert recomputed_stamped_hash == committed_stamped_hash, (
        "the committed manifest's own manifest_sha256 field does not match a hash "
        "recomputed over its own content - the committed file has been hand-edited "
        "or corrupted since it was generated"
    )

    fresh = TOOL.build_manifest(ROOT / "design")
    assert fresh == committed, (
        "results/t15/t15_t16_resplit_manifest.json is stale: it does not match what "
        "build_manifest() produces from the current frozen inputs. Regenerate it via "
        "tools/build_t15_t16_resplit_manifest.py and re-commit."
    )
    assert fresh["manifest_sha256"] == committed_stamped_hash
