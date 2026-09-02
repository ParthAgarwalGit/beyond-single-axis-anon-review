"""Explicit code-versus-YAML invariant tests.

The modules DERIVE their constants from the frozen YAML, so these tests pin
both directions at once: they fail if the YAML's frozen values change, and
they fail if a module stops deriving from the YAML (drifting back to local
constants). Literal expected values are intentional.
"""

from src import activation_extraction, generation, statistics, steering
from src.config import block_question_ids
from src.schemas import ARMS


def test_steering_condition_names_are_frozen(cfg):
    assert tuple(steering.CAUSAL_CONDITIONS) == (
        "shared_zero", "assistant_axis_toward", "assistant_axis_away",
        "random_positive", "random_negative",
    )
    assert tuple(steering.CAUSAL_CONDITIONS) == tuple(cfg["steering"]["unique_conditions"])


def test_markers_and_segmentation_cases_are_frozen(cfg):
    assert generation.OPEN_MARKER == "<think>" == cfg["response_segmentation"]["opening_marker"]
    assert generation.CLOSE_MARKER == "</think>" == cfg["response_segmentation"]["closing_marker"]
    assert set(generation.SEGMENTATION_CASES) == {
        "direct_answer", "balanced_reasoning", "malformed_reasoning",
        "empty_answer_after_close",
    } == set(cfg["response_segmentation"]["cases"])


def test_validity_labels_are_frozen(cfg):
    assert set(generation.VALIDITY_LABELS) == {
        "valid", "empty_response", "generation_error", "truncated",
        "serialization_failure", "token_alignment_failure", "degenerate_repetition",
    } == set(cfg["validity"]["labels"])


def test_pools_and_hook_are_frozen(cfg):
    extraction = cfg["activation_extraction"]
    assert extraction["primary_pool"] == "ALL_RESPONSE_TOKENS"
    assert tuple(extraction["sensitivity_pools"]) == ("REASONING_TOKENS", "FINAL_ANSWER_TOKENS")
    assert extraction["hook"]["n_blocks"] == 32
    assert activation_extraction.primary_block(cfg) == 16
    assert activation_extraction.hook_module_path(cfg, 16) == "model.model.layers[16]"
    assert activation_extraction.hidden_states_index(16) == 17


def test_causal_label_sets_are_frozen(cfg):
    assert tuple(statistics.HARMFULNESS_LABELS) == (
        "harmful", "partial", "refused", "degenerate",
    ) == tuple(cfg["causal_outcomes"]["harmfulness_labels"])
    assert tuple(statistics.IDENTITY_LABELS) == (
        "assistant", "human_role", "nonhuman_role", "weird_or_mystical_role",
        "ambiguous", "other", "nonsensical",
    ) == tuple(cfg["causal_outcomes"]["identity_labels"])


def test_arms_and_prompt_count_are_frozen(cfg):
    assert ARMS == ("USER_TRANSLATED_LU", "USER_EXPLICIT", "SYSTEM_LU", "DEFAULT")
    assert len(cfg["prompt_rendering"]["default_conditions"]) == 5


def test_blocks_are_frozen(cfg):
    blocks = block_question_ids(cfg)
    assert set(blocks) == {"E80", "C80-A", "C80-B"}
    assert all(len(ids) == 80 for ids in blocks.values())


def test_config_is_immutable(cfg):
    import pytest
    with pytest.raises(TypeError):
        cfg["steering"]["layer"] = 3
    assert isinstance(cfg["steering"]["unique_conditions"], tuple)
