"""Required T03 test: response validity (METHOD_FREEZE §7)."""

from conftest import char_tokenize
from src.generation import GenerationRecord, classify_validity, degenerate_repetition

EOS = 0


def make_record(text, finish_reason="stop", output_ids=None):
    return GenerationRecord(
        prompt_token_ids=char_tokenize("USER: hi"),
        output_token_ids=char_tokenize(text) + [EOS] if output_ids is None else output_ids,
        output_text=text,
        finish_reason=finish_reason,
    )


def classify(record):
    return classify_validity(record, char_tokenize, terminal_ids=(EOS,))


def test_valid_response():
    assert classify(make_record("<think>plan</think>A fine answer.")) == "valid"


def test_generation_error():
    assert classify(make_record("partial", finish_reason="error")) == "generation_error"


def test_empty_response():
    assert classify(make_record("", output_ids=[EOS, EOS])) == "empty_response"
    assert classify(make_record("   ")) == "empty_response"


def test_truncated():
    assert classify(make_record("cut off mid-sen", finish_reason="length")) == "truncated"


def test_degenerate_repetition():
    text = "arr " * 40
    assert classify(make_record(text)) == "degenerate_repetition"
    assert degenerate_repetition(char_tokenize(text))
    assert not degenerate_repetition(char_tokenize("a perfectly normal reply about plants"))


def test_token_alignment_failure():
    record = make_record("<think>a</think>answer")
    record.output_token_ids = record.output_token_ids[3:]  # corrupt stored IDs
    assert classify(record) == "token_alignment_failure"


def test_malformed_markers_are_not_a_validity_failure():
    # Malformed reasoning is a segmentation status; the row can still be
    # technically valid and keep its ALL_RESPONSE_TOKENS pool.
    assert classify(make_record("<think>never closes but plenty of text")) == "valid"
