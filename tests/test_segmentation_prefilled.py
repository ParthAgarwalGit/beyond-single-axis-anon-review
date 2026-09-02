"""Segmentation for models that PREFILL the opening <think> into the prompt.

DeepSeek-R1-Distill's chat template ends the prompt with ``<think>`` (via
add_generation_prompt=True), so the model generates INSIDE the reasoning block:
a well-formed output is ``reasoning</think>answer`` with no ``<think>`` and one
``</think>``. Without prefilled-open handling every such output is misclassified
as ``malformed_reasoning`` and becomes unjudgeable.

These CPU-only checks use a char tokenizer where ``</think>`` is a single
sentinel id, mirroring the real tokenizer (``</think>`` -> one token 128014).
"""

from src.generation import (
    GenerationRecord, classify_validity, segment_response,
)

CLOSE_ID = -14   # single-token sentinel standing in for </think> (128014)


def _tok(text):
    """Char tokenizer with </think> and <think> as single sentinel tokens."""
    ids, i = [], 0
    while i < len(text):
        if text.startswith("</think>", i):
            ids.append(CLOSE_ID); i += len("</think>")
        elif text.startswith("<think>", i):
            ids.append(-13); i += len("<think>")
        else:
            ids.append(ord(text[i])); i += 1
    return ids


def _record(answer_reasoning, finish="stop"):
    prompt_ids = _tok("PROMPT<think>")           # prompt prefills <think>
    out_ids = _tok(answer_reasoning)
    return GenerationRecord(prompt_ids, out_ids, answer_reasoning, finish)


def test_prefilled_open_reasoning_is_balanced():
    rec = _record("thinking hard</think>Aye, my final answer.")
    seg = segment_response(rec, _tok, prefilled_open=True, close_token_id=CLOSE_ID)
    assert seg.case == "balanced_reasoning"
    assert seg.judge_input_text == "Aye, my final answer."
    # token spans: reasoning = start..close(incl); answer = after close
    start = len(rec.prompt_token_ids)
    assert seg.reasoning_span[0] == start
    assert seg.final_answer_span[1] == start + len(rec.output_token_ids)
    assert seg.reasoning_span[1] == seg.final_answer_span[0]


def test_same_output_is_malformed_without_prefilled_flag():
    # The exact bug: orphan </think> (open is prefilled) => malformed if the
    # segmenter isn't told the open was prefilled.
    rec = _record("thinking hard</think>Aye, my final answer.")
    seg = segment_response(rec, _tok, prefilled_open=False)
    assert seg.case == "malformed_reasoning"
    assert seg.judge_input_text is None


def test_prefilled_open_classify_validity_valid():
    rec = _record("reason</think>answer")
    v = classify_validity(rec, _tok, prefilled_open=True, close_token_id=CLOSE_ID)
    assert v == "valid"


def test_prefilled_open_unclosed_is_malformed():
    rec = _record("still reasoning, never closed")   # no </think>
    seg = segment_response(rec, _tok, prefilled_open=True, close_token_id=CLOSE_ID)
    assert seg.case == "malformed_reasoning"
    assert "unclosed" in seg.flags[0]


def test_prefilled_open_empty_answer():
    rec = _record("reasoning only</think>   ")       # whitespace answer
    seg = segment_response(rec, _tok, prefilled_open=True, close_token_id=CLOSE_ID)
    assert seg.case == "empty_answer_after_close"
    assert seg.judge_input_text is None


def test_token_boundary_survives_text_retok_mismatch():
    # Simulate the vLLM detok/retok artifact: a tokenizer whose whole-text
    # re-encoding does NOT reproduce the stored ids, but </think> is still a
    # clean single token. Token-based boundary must still segment correctly.
    rec = _record("reason</think>the answer")
    def lossy_tok(text):
        ids = _tok(text)
        return ids + [999] if "</think>" not in text else ids  # perturb non-boundary text
    seg = segment_response(rec, lossy_tok, prefilled_open=True, close_token_id=CLOSE_ID)
    assert seg.case == "balanced_reasoning"
    assert seg.judge_input_text == "the answer"


def test_multiple_closes_uses_final_boundary():
    rec = _record("a</think>b</think>final")
    seg = segment_response(rec, _tok, prefilled_open=True, close_token_id=CLOSE_ID)
    assert seg.case == "balanced_reasoning"
    assert seg.judge_input_text == "final"
    assert "multiple_closes" in seg.flags
