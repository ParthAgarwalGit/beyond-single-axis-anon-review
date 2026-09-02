"""Canonical generation record, response boundaries, segmentation, validity.

Implements METHOD_FREEZE §6–§7. The generation engine's stored
``prompt_token_ids`` and ``output_token_ids`` are authoritative; primary
masks are never reconstructed from separately retokenized strings. Two
alignment checks apply:

1. the WHOLE stored completion text must tokenize back to exactly the
   stored content token IDs (every segmentation case, including direct
   answers and malformed reasoning);
2. the reasoning/answer delimiter is accepted only when the text prefix
   ending at the delimiter tokenizes to an exact prefix of the stored IDs.

All frozen names below are derived from the frozen YAML, never redefined.
"""

from dataclasses import dataclass, field

from .config import load_frozen_config

_CFG, _ = load_frozen_config()

OPEN_MARKER = _CFG["response_segmentation"]["opening_marker"]
CLOSE_MARKER = _CFG["response_segmentation"]["closing_marker"]

# Frozen segmentation case names (YAML ``response_segmentation.cases``).
SEGMENTATION_CASES = tuple(_CFG["response_segmentation"]["cases"])
DIRECT_ANSWER = "direct_answer"
BALANCED_REASONING = "balanced_reasoning"
MALFORMED_REASONING = "malformed_reasoning"
EMPTY_ANSWER = "empty_answer_after_close"
assert set(SEGMENTATION_CASES) == {
    DIRECT_ANSWER, BALANCED_REASONING, MALFORMED_REASONING, EMPTY_ANSWER
}, "code drifted from the frozen segmentation cases"

# Frozen technical-validity labels (YAML ``validity.labels``).
VALIDITY_LABELS = tuple(_CFG["validity"]["labels"])


@dataclass
class GenerationRecord:
    """One generation attempt, holding the engine's authoritative token IDs."""
    prompt_token_ids: list
    output_token_ids: list
    output_text: str
    finish_reason: str  # "stop" | "length" | "error"


@dataclass
class Segmentation:
    case: str
    # Half-open (start, end) spans in ABSOLUTE positions of the concatenated
    # prompt+output sequence; None when the pool is unavailable.
    all_response_span: tuple
    reasoning_span: tuple
    final_answer_span: tuple
    judge_input_text: str  # None => judge abstains (missing measurement)
    flags: list = field(default_factory=list)


class TokenAlignmentError(Exception):
    """Stored token IDs and text segmentation cannot be reconciled."""


def response_start(record):
    return len(record.prompt_token_ids)


def content_token_ids(record, terminal_ids=()):
    """Generated content tokens: output IDs minus trailing padding/EOS/stop."""
    ids = list(record.output_token_ids)
    while ids and ids[-1] in terminal_ids:
        ids.pop()
    return ids


def verify_text_token_alignment(record, tokenize, terminal_ids=()):
    """Whole-completion check: the stored text must tokenize to exactly the
    stored content token IDs. Raises TokenAlignmentError otherwise."""
    content = content_token_ids(record, terminal_ids)
    if tokenize(record.output_text) != content:
        raise TokenAlignmentError(
            "stored output text does not tokenize to the stored content token IDs"
        )
    return content


def _marker_counts_balanced(text):
    opens, closes = text.count(OPEN_MARKER), text.count(CLOSE_MARKER)
    if opens != closes:
        return False
    if opens == 0:
        return True
    return text.rfind(CLOSE_MARKER) > text.rfind(OPEN_MARKER)


def _segment_prefilled_open(record, start, text, content, all_span, tokenize,
                            close_token_id):
    """Segment when the opening ``<think>`` is PREFILLED into the prompt.

    Some reasoning models (e.g. DeepSeek-R1-Distill) end the rendered prompt
    with ``<think>`` via ``add_generation_prompt=True``, so the model begins
    generating INSIDE the reasoning block: a well-formed output contains no
    ``<think>`` and a single ``</think>`` that closes reasoning. The frozen
    ``reasoning_mask`` definition ("response start through the end of the
    final </think>") already assumes exactly this; the case-classification is
    what has to account for the prefilled open.

    When ``close_token_id`` is the single-token id of ``</think>`` the boundary
    is located by TOKEN (authoritative per ``response_tokenization``), which is
    also robust to benign detokenize/retokenize round-trip differences that
    otherwise trip the whole-completion text check.
    """
    opens = text.count(OPEN_MARKER)
    closes = text.count(CLOSE_MARKER)
    if opens != 0:
        return Segmentation(
            case=MALFORMED_REASONING, all_response_span=all_span,
            reasoning_span=None, final_answer_span=None, judge_input_text=None,
            flags=["prefilled_open_but_output_reopened"],
        )
    if closes == 0:
        # The prefilled reasoning block never closed: no final-answer boundary.
        return Segmentation(
            case=MALFORMED_REASONING, all_response_span=all_span,
            reasoning_span=None, final_answer_span=None, judge_input_text=None,
            flags=["prefilled_open_unclosed"],
        )

    if close_token_id is not None:
        if content.count(close_token_id) != closes:
            raise TokenAlignmentError(
                "</think> count differs between output text and stored token IDs"
            )
        tok_idx = len(content) - 1 - content[::-1].index(close_token_id)
        reasoning_len = tok_idx + 1
    else:
        boundary = text.rfind(CLOSE_MARKER) + len(CLOSE_MARKER)
        prefix_ids = tokenize(text[:boundary])
        if prefix_ids != content[: len(prefix_ids)]:
            raise TokenAlignmentError(
                "reasoning boundary does not tokenize to a prefix of the stored output IDs"
            )
        reasoning_len = len(prefix_ids)

    reasoning_span = (start, start + reasoning_len)
    answer_text = text[text.rfind(CLOSE_MARKER) + len(CLOSE_MARKER):]
    flags = ["multiple_closes"] if closes > 1 else []
    if not answer_text.strip():
        return Segmentation(
            case=EMPTY_ANSWER, all_response_span=all_span,
            reasoning_span=reasoning_span, final_answer_span=None,
            judge_input_text=None, flags=flags,
        )
    return Segmentation(
        case=BALANCED_REASONING, all_response_span=all_span,
        reasoning_span=reasoning_span,
        final_answer_span=(start + reasoning_len, start + len(content)),
        judge_input_text=answer_text, flags=flags,
    )


def segment_response(record, tokenize, terminal_ids=(), prefilled_open=False,
                     close_token_id=None):
    """Segment a technically valid completion into the frozen token pools.

    ``tokenize`` must be the same text->token-ID function used for the
    stored output (``add_special_tokens=False``); it verifies alignment,
    it never rebuilds the primary mask.

    ``prefilled_open`` selects the branch for models whose chat template
    prefills the opening ``<think>`` into the prompt (see
    :func:`_segment_prefilled_open`); ``close_token_id`` is the single-token
    id of ``</think>`` for token-authoritative boundary detection.

    Returns a :class:`Segmentation`; raises TokenAlignmentError when text
    and stored token IDs cannot be reconciled (whole-completion check for
    every case, plus the delimiter-boundary prefix check).
    """
    start = response_start(record)

    if prefilled_open:
        content = content_token_ids(record, terminal_ids)
        all_span = (start, start + len(content))
        return _segment_prefilled_open(
            record, start, record.output_text, content, all_span, tokenize,
            close_token_id,
        )

    content = verify_text_token_alignment(record, tokenize, terminal_ids)
    text = record.output_text
    all_span = (start, start + len(content))

    opens, closes = text.count(OPEN_MARKER), text.count(CLOSE_MARKER)

    if opens == 0 and closes == 0:
        return Segmentation(
            case=DIRECT_ANSWER,
            all_response_span=all_span,
            reasoning_span=None,
            final_answer_span=all_span,
            judge_input_text=text,
        )

    if not _marker_counts_balanced(text):
        return Segmentation(
            case=MALFORMED_REASONING,
            all_response_span=all_span,
            reasoning_span=None,
            final_answer_span=None,
            judge_input_text=None,
            flags=["malformed_markers"],
        )

    boundary_char = text.rfind(CLOSE_MARKER) + len(CLOSE_MARKER)
    prefix_ids = tokenize(text[:boundary_char])
    if prefix_ids != content[: len(prefix_ids)]:
        raise TokenAlignmentError(
            "reasoning boundary does not tokenize to a prefix of the stored output IDs"
        )

    flags = ["multiple_pairs"] if opens > 1 else []
    reasoning_span = (start, start + len(prefix_ids))
    answer_text = text[boundary_char:]

    if not answer_text.strip():
        return Segmentation(
            case=EMPTY_ANSWER,
            all_response_span=all_span,
            reasoning_span=reasoning_span,
            final_answer_span=None,
            judge_input_text=None,
            flags=flags,
        )

    return Segmentation(
        case=BALANCED_REASONING,
        all_response_span=all_span,
        reasoning_span=reasoning_span,
        final_answer_span=(start + len(prefix_ids), start + len(content)),
        judge_input_text=answer_text,
        flags=flags,
    )


def degenerate_repetition(token_ids, ngram=8, min_repeats=6):
    """Predeclared repetition detector: the response tail is one n-gram
    repeated at least ``min_repeats`` times consecutively."""
    block = ngram * min_repeats
    if len(token_ids) < block:
        return False
    tail = token_ids[-block:]
    unit = tail[:ngram]
    return all(tail[i : i + ngram] == unit for i in range(0, block, ngram))


def classify_validity(record, tokenize, terminal_ids=(), serialization_ok=True,
                      prefilled_open=False, close_token_id=None):
    """Frozen technical-validity label for one generation attempt.

    ``serialization_ok`` is the result of the record-level reconstruction
    check (``src.schemas.serialization_ok``): False means the persisted
    messages/rendered-prompt cannot be reconstructed exactly, which is the
    frozen ``serialization_failure`` label.

    ``prefilled_open`` / ``close_token_id`` are forwarded to
    :func:`segment_response` for prefilled-``<think>`` models.

    Technical validity is separate from role expression; a role score is
    never substituted for a validity label.
    """
    if record.finish_reason == "error":
        return "generation_error"
    if not serialization_ok:
        return "serialization_failure"
    content = content_token_ids(record, terminal_ids)
    if not content or not record.output_text.strip():
        return "empty_response"
    if record.finish_reason == "length":
        return "truncated"
    if degenerate_repetition(content):
        return "degenerate_repetition"
    try:
        segment_response(record, tokenize, terminal_ids,
                         prefilled_open=prefilled_open, close_token_id=close_token_id)
    except TokenAlignmentError:
        return "token_alignment_failure"
    return "valid"
