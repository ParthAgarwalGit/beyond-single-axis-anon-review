"""Required T03 test: role-score parsing (METHOD_FREEZE §8)."""

import pytest

from src.judging import build_judge_input, judge_validation_gates_pass, parse_role_score
from src.generation import Segmentation


@pytest.mark.parametrize("raw,expected", [
    ("0", 0), ("1", 1), ("2", 2), ("3", 3),
    ("  3  ", 3), ("\n2\n", 2),           # whitespace stripping allowed
])
def test_valid_scores_parse(raw, expected):
    assert parse_role_score(raw) == expected


@pytest.mark.parametrize("raw", [
    "4", "-1", "03", "3.0", "3/3",
    "score: 2", "2 (fully in character)", "2 3",
    "", "   ", None, "three",
])
def test_everything_else_is_parse_failure_not_zero(raw):
    assert parse_role_score(raw) is None


def _seg(judge_text):
    return Segmentation(
        case="balanced_reasoning", all_response_span=(0, 1),
        reasoning_span=None, final_answer_span=None,
        judge_input_text=judge_text,
    )


def test_judge_input_contains_only_frozen_fields():
    payload = build_judge_input("pirate", "rubric text", "How do plants eat?", _seg("Arr."))
    assert set(payload) == {"role", "role_description", "question", "final_answer_text"}
    assert payload["final_answer_text"] == "Arr."


def test_judge_abstains_when_no_answer_region():
    assert build_judge_input("pirate", "rubric", "q", _seg(None)) is None


def test_validation_gates(cfg):
    passing = {"score_3_precision": 0.9, "score_3_recall": 0.86,
               "macro_f1": 0.8, "cohen_kappa": 0.75}
    assert judge_validation_gates_pass(cfg, passing)
    failing = dict(passing, cohen_kappa=0.6)
    assert not judge_validation_gates_pass(cfg, failing)
