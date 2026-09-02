"""Required T03 test: token masks (METHOD_FREEZE §6)."""

import pytest

from conftest import char_tokenize
from src.generation import (
    BALANCED_REASONING, DIRECT_ANSWER, EMPTY_ANSWER, MALFORMED_REASONING,
    GenerationRecord, TokenAlignmentError, segment_response,
)

EOS = 0
PROMPT = "USER: hi"


def make_record(output_text, finish_reason="stop", trailing_eos=1):
    return GenerationRecord(
        prompt_token_ids=char_tokenize(PROMPT),
        output_token_ids=char_tokenize(output_text) + [EOS] * trailing_eos,
        output_text=output_text,
        finish_reason=finish_reason,
    )


def test_direct_answer_masks():
    text = "Plants photosynthesise."
    record = make_record(text)
    seg = segment_response(record, char_tokenize, terminal_ids=(EOS,))
    start = len(record.prompt_token_ids)
    assert seg.case == DIRECT_ANSWER
    # Prompt tokens excluded; terminal EOS excluded.
    assert seg.all_response_span == (start, start + len(text))
    assert seg.reasoning_span is None
    assert seg.final_answer_span == seg.all_response_span
    assert seg.judge_input_text == text


def test_balanced_reasoning_masks():
    reasoning = "<think>chart a course</think>"
    answer = "\n\nArr, plants eat sunlight."
    record = make_record(reasoning + answer)
    seg = segment_response(record, char_tokenize, terminal_ids=(EOS,))
    start = len(record.prompt_token_ids)
    assert seg.case == BALANCED_REASONING
    # Reasoning runs from response start through the final </think> inclusive.
    assert seg.reasoning_span == (start, start + len(reasoning))
    # Answer is strictly after the final </think>, EOS excluded.
    assert seg.final_answer_span == (start + len(reasoning), start + len(reasoning + answer))
    assert seg.judge_input_text == answer
    assert seg.flags == []


def test_multiple_pairs_flagged_last_close_wins():
    text = "<think>a</think>mid<think>b</think>final answer"
    seg = segment_response(make_record(text), char_tokenize, terminal_ids=(EOS,))
    assert seg.case == BALANCED_REASONING
    assert "multiple_pairs" in seg.flags
    assert seg.judge_input_text == "final answer"


def test_malformed_markers_leave_only_all_response_pool():
    for text in ("<think>never closes", "orphan</think> close <think>",
                 "</think>close before open<think>x</think>"):
        seg = segment_response(make_record(text), char_tokenize, terminal_ids=(EOS,))
        assert seg.case == MALFORMED_REASONING
        assert seg.reasoning_span is None
        assert seg.final_answer_span is None
        assert seg.judge_input_text is None  # judge abstains
        assert seg.all_response_span is not None


def test_empty_answer_after_close():
    text = "<think>thoughts</think>\n\n  "
    seg = segment_response(make_record(text), char_tokenize, terminal_ids=(EOS,))
    assert seg.case == EMPTY_ANSWER
    assert seg.reasoning_span is not None
    assert seg.final_answer_span is None
    assert seg.judge_input_text is None


def test_boundary_must_align_with_stored_token_ids():
    text = "<think>a</think>answer"
    record = make_record(text)
    # Corrupt the stored IDs: text and tokens can no longer be reconciled.
    record.output_token_ids = record.output_token_ids[2:]
    with pytest.raises(TokenAlignmentError):
        segment_response(record, char_tokenize, terminal_ids=(EOS,))
