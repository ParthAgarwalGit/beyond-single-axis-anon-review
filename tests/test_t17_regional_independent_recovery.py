import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/t17/regional_independent_recovery"
TOOL_PATH = ROOT / "tools/run_t17_regional_independent_recovery.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("t17_regional_recovery_test", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _result():
    return json.loads((OUT / "regional_independent_recovery.json").read_text("utf-8"))


def test_expected_scientific_values_and_fidelity_are_frozen():
    result = _result()

    assert result["status"] == "POST_PRIMARY_SENSITIVITY_COMPLETE"
    assert result["fidelity_gate"]["status"] == "PASS"
    assert np.isclose(
        result["fidelity_gate"]["reconstructed_c160_reasoning_answer_axis_cosine"],
        0.5877841077999876,
    )
    assert np.isclose(
        result["fidelity_gate"]["reconstructed_c160_reasoning_answer_role_r"],
        0.9546374089228963,
    )

    reasoning = result["regional_independent_recovery"]["reasoning"]
    answer = result["regional_independent_recovery"]["answer"]
    within = result["within_block_reasoning_vs_answer"]

    assert reasoning["n_roles"] == 275
    assert answer["n_roles"] == 275
    assert reasoning["bootstrap_valid_replicates"] == 2000
    assert answer["bootstrap_valid_replicates"] == 2000
    assert reasoning["bootstrap_invalid_replicates"] == 0
    assert answer["bootstrap_invalid_replicates"] == 0

    assert np.isclose(reasoning["cross_axis_pearson_r"], 0.9737941803520305)
    assert np.allclose(
        reasoning["cross_axis_pearson_ci95"],
        [0.9674739825165557, 0.9789373791870419],
    )
    assert np.isclose(reasoning["axis_cosine_C80A_C80B"], 0.9246771534321052)
    assert np.isclose(reasoning["project_consistent_spearman"], 0.9721075560428148)

    assert np.isclose(answer["cross_axis_pearson_r"], 0.9600601570530783)
    assert np.allclose(
        answer["cross_axis_pearson_ci95"],
        [0.9512717641280366, 0.9678385903672123],
    )
    assert np.isclose(answer["axis_cosine_C80A_C80B"], 0.8348719274959626)
    assert np.isclose(answer["project_consistent_spearman"], 0.960806093303713)

    assert np.isclose(
        within["C80-A"]["reasoning_answer_axis_cosine"], 0.5841187007983598
    )
    assert np.isclose(
        within["C80-B"]["reasoning_answer_axis_cosine"], 0.5868930507571809
    )
    assert np.isclose(
        within["C80-A"]["reasoning_answer_same_role_own_axis_pearson"],
        0.9567436005010954,
    )
    assert np.isclose(
        within["C80-B"]["reasoning_answer_same_role_own_axis_pearson"],
        0.947056402551915,
    )


def test_committed_analysis_plan_matches_generator_schema_exactly():
    """Prevent the committed aggregate and generator schema from drifting apart."""
    result = _result()
    assert result["analysis_plan"] == TOOL.analysis_plan(2000, 0)


def test_fidelity_targets_are_read_from_canonical_c160_authority():
    result = _result()
    expected_cos, expected_r = TOOL.c160_reasoning_answer_reference()
    gate = result["fidelity_gate"]
    assert gate["expected_c160_reasoning_answer_axis_cosine"] == expected_cos
    assert gate["expected_c160_reasoning_answer_role_r"] == expected_r


def test_default_floor_and_middle_block_come_from_frozen_config():
    cfg = TOOL._CFG
    expected_floor = float(
        cfg["role_vectors_and_axis"]["default_vector"]
        ["minimum_valid_fraction_per_condition"]
    )
    expected_block = int(
        cfg["activation_extraction"]["middle_layer"]["primary_block_index"]
    )
    assert TOOL.DEFAULT_FLOOR == expected_floor == 0.95
    assert TOOL.MIDDLE_BLOCK_INDEX == expected_block == 16


def test_committed_provenance_paths_are_platform_neutral():
    result = _result()
    paths = result["provenance"]["input_sha256"]
    assert paths
    assert all("\\" not in path for path in paths)


def test_claim_boundary_is_not_promoted():
    result = _result()
    boundary = " ".join(result["claim_boundary"]).lower()
    assert "does not alter the frozen all-response t15/t16 primary result" in boundary
    assert "does not establish causal propagation" in boundary
    assert "does not establish a uniquely privileged assistant direction" in boundary
