
from pathlib import Path
import importlib.util
import sys


# ---------------------------------------------------------------------
# Import the T12 runner as a module.
#
# IMPORTANT:
# register it in sys.modules BEFORE exec_module().
# Python 3.12 dataclasses require this during @dataclass processing.
# ---------------------------------------------------------------------

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "run_t12_activation_recompute.py"
)

MODULE_NAME = "t12_runner"

spec = importlib.util.spec_from_file_location(
    MODULE_NAME,
    SCRIPT,
)

if spec is None or spec.loader is None:
    raise ImportError(
        f"Could not create import spec for {SCRIPT}"
    )

t12 = importlib.util.module_from_spec(spec)

# Required before executing a module containing @dataclass.
sys.modules[MODULE_NAME] = t12

try:
    spec.loader.exec_module(t12)
except Exception:
    # Do not leave a half-imported module behind.
    sys.modules.pop(MODULE_NAME, None)
    raise


class TinyTokenizer:
    """
    Deterministic character-level tokenizer for
    CPU-only unit testing.
    """

    def __call__(
        self,
        text,
        add_special_tokens=False,
    ):
        class Result:
            pass

        result = Result()

        result.input_ids = [
            ord(ch)
            for ch in text
        ]

        return result



class BoundarySensitiveTokenizer:
    """
    Tiny tokenizer that deliberately makes tokenize(prefix) differ from
    the corresponding prefix of tokenize(full_completion).

    This reproduces the exact class of BPE-boundary mismatch that T12 must
    handle because run-003 lacks generation-native token IDs.
    """

    FULL = "<think>abc</think>final"
    PREFIX = "<think>abc</think>"

    def __call__(
        self,
        text,
        add_special_tokens=False,
    ):
        class Result:
            pass

        result = Result()

        if text == "PROMPT":
            result.input_ids = [101, 102]
        elif text == self.FULL:
            result.input_ids = [1, 2, 3, 4, 5, 6]
        elif text == self.PREFIX:
            # Same text prefix, intentionally different tokenization at
            # the boundary compared with FULL[:prefix_tokens].
            result.input_ids = [1, 2, 99, 4]
        else:
            result.input_ids = [
                1000 + ord(ch)
                for ch in text
            ]

        return result

def make_row(
    completion,
    finish_reason="stop",
    rendered_prompt="PROMPT",
):
    return {
        "uid": "test-uid",
        "role": "tester",
        "question_id": 1,
        "prompt_idx": 0,
        "condition": "TEST",
        "rendered_prompt": rendered_prompt,
        "completion": completion,
        "finish_reason": finish_reason,
        "n_out_tokens": len(completion),
    }


def test_direct_answer_pool():

    tok = TinyTokenizer()

    p = t12.prepare_row(
        make_row("hello"),
        tok,
    )

    assert p.validity == "valid"

    assert p.segmentation_case == (
        "direct_answer"
    )

    assert p.reasoning_span is None

    assert (
        p.all_span
        == p.answer_span
    )

    assert (
        p.all_span[0]
        == len("PROMPT")
    )


def test_balanced_reasoning():

    tok = TinyTokenizer()

    completion = (
        "<think>abc</think>final"
    )

    p = t12.prepare_row(
        make_row(completion),
        tok,
    )

    assert p.validity == "valid"

    assert p.reasoning_span is not None
    assert p.answer_span is not None
    assert p.all_span is not None

    assert (
        p.reasoning_span[0]
        == p.all_span[0]
    )

    assert (
        p.reasoning_span[1]
        == p.answer_span[0]
    )

    assert (
        p.answer_span[1]
        == p.all_span[1]
    )


def test_truncated_kept_as_truncated():

    tok = TinyTokenizer()

    p = t12.prepare_row(
        make_row(
            "unfinished reasoning",
            finish_reason="length",
        ),
        tok,
    )

    assert p.validity == "truncated"

    # A truncated output can still have an
    # all-response activation reconstructed.
    # Downstream analyses filter by validity.
    assert p.all_span is not None


def test_token_count_delta_is_audit_only():

    tok = TinyTokenizer()

    row = make_row("abcd")

    row["n_out_tokens"] = 5

    p = t12.prepare_row(
        row,
        tok,
    )

    assert (
        p.token_count_delta
        == -1
    )


def test_metadata_records_reconstructed_token_provenance():

    tok = TinyTokenizer()

    p = t12.prepare_row(
        make_row("hello"),
        tok,
    )

    meta = t12.metadata_for(p)

    assert (
        meta[
            "token_id_provenance"
        ]
        == "reconstructed_from_stored_text"
    )

    assert (
        meta[
            "generation_native_token_ids_available"
        ]
        is False
    )


def test_reconstructed_boundary_fallback_recovers_answer_and_reasoning():
    tok = BoundarySensitiveTokenizer()

    p = t12.prepare_row(
        make_row(BoundarySensitiveTokenizer.FULL),
        tok,
    )

    # The frozen strict prefix check would classify this as alignment
    # failure, but T12's stored-text fallback should recover the same
    # boundary procedure already validated against historical tensors.
    assert p.validity == "valid"
    assert p.segmentation_case == "balanced_reasoning"
    assert (
        p.segmentation_provenance
        == "reconstructed_from_stored_text_boundary"
    )
    assert p.frozen_boundary_prefix_check_passed is False

    prompt_len = 2

    assert p.all_span == (
        prompt_len,
        prompt_len + 6,
    )
    assert p.reasoning_span == (
        prompt_len,
        prompt_len + 4,
    )
    assert p.answer_span == (
        prompt_len + 4,
        prompt_len + 6,
    )

    meta = t12.metadata_for(p)

    assert meta["has_all_response_pool"] is True
    assert meta["has_reasoning_pool"] is True
    assert meta["has_answer_pool"] is True
    assert (
        meta["segmentation_provenance"]
        == "reconstructed_from_stored_text_boundary"
    )


def test_reconstructed_boundary_preserves_truncated_validity():
    tok = BoundarySensitiveTokenizer()

    p = t12.prepare_row(
        make_row(
            BoundarySensitiveTokenizer.FULL,
            finish_reason="length",
        ),
        tok,
    )

    # Segmentation can still be reconstructed physically, while the
    # technical validity label remains the frozen higher-priority label.
    assert p.validity == "truncated"
    assert p.all_span is not None
    assert p.reasoning_span is not None
    assert p.answer_span is not None
    assert (
        p.segmentation_provenance
        == "reconstructed_from_stored_text_boundary"
    )



def test_prompt_seeded_run003_serialization_recovers_reasoning_and_answer():
    tok = TinyTokenizer()

    prompt = "SERIALIZED_ASSISTANT<think>\n"
    completion = "reasoning text</think>final answer"

    p = t12.prepare_row(
        make_row(
            completion,
            rendered_prompt=prompt,
        ),
        tok,
    )

    assert p.validity == "valid"
    assert p.segmentation_case == "balanced_reasoning"
    assert (
        p.segmentation_provenance
        == "prompt_seeded_reconstructed_from_stored_text_boundary"
    )
    assert p.all_span is not None
    assert p.reasoning_span is not None
    assert p.answer_span is not None
    assert p.reasoning_span[0] == p.all_span[0]
    assert p.reasoning_span[1] == p.answer_span[0]
    assert p.answer_span[1] == p.all_span[1]


def test_prompt_seeded_no_close_is_malformed_not_direct_answer():
    tok = TinyTokenizer()

    prompt = "SERIALIZED_ASSISTANT<think>\n"
    completion = "a response with no reasoning close marker"

    p = t12.prepare_row(
        make_row(
            completion,
            rendered_prompt=prompt,
        ),
        tok,
    )

    # The prompt seeded the opening <think>; the completion never closes it, so
    # the reasoning region is open at the end. METHOD_FREEZE 6.2 makes this
    # malformed_reasoning, NOT direct_answer: the all-response pool is retained
    # but the answer/reasoning pools are unavailable and the judge abstains,
    # instead of storing the whole in-progress trace as the answer tensor.
    assert p.segmentation_case == "malformed_reasoning"
    assert p.segmentation_provenance == "prompt_seeded_unclosed_open_marker"
    assert p.all_span is not None
    assert p.reasoning_span is None
    assert p.answer_span is None


def test_prompt_seeded_unclosed_truncated_keeps_all_response_only():
    tok = TinyTokenizer()

    prompt = "SERIALIZED_ASSISTANT<think>\n"
    completion = "reasoning that ran out of budget before closing"

    p = t12.prepare_row(
        make_row(
            completion,
            finish_reason="length",
            rendered_prompt=prompt,
        ),
        tok,
    )

    # The common tail: seeded <think>, budget exhausted mid-trace, no </think>.
    assert p.validity == "truncated"          # technical label preserved
    assert p.segmentation_case == "malformed_reasoning"
    assert p.all_span is not None             # all-response pool retained
    assert p.reasoning_span is None
    assert p.answer_span is None


def test_prompt_seeded_closed_reports_prefix_check():
    tok = TinyTokenizer()

    prompt = "SERIALIZED_ASSISTANT<think>\n"
    completion = "reasoning text</think>final answer"

    p = t12.prepare_row(
        make_row(
            completion,
            rendered_prompt=prompt,
        ),
        tok,
    )

    # The char-level tokenizer is prefix-consistent, so the frozen delimiter
    # prefix-token check passes and is recorded as True (not None) for the
    # prompt-seeded reconstruction path.
    assert p.segmentation_case == "balanced_reasoning"
    assert (
        p.segmentation_provenance
        == "prompt_seeded_reconstructed_from_stored_text_boundary"
    )
    assert p.frozen_boundary_prefix_check_passed is True


def test_prompt_seeded_truncated_row_keeps_truncated_label_and_physical_pools():
    tok = TinyTokenizer()

    prompt = "SERIALIZED_ASSISTANT<think>\n"
    completion = "reasoning text</think>final answer"

    p = t12.prepare_row(
        make_row(
            completion,
            finish_reason="length",
            rendered_prompt=prompt,
        ),
        tok,
    )

    assert p.validity == "truncated"
    assert p.all_span is not None
    assert p.reasoning_span is not None
    assert p.answer_span is not None
    assert (
        p.segmentation_provenance
        == "prompt_seeded_reconstructed_from_stored_text_boundary"
    )


def test_runner_can_start_as_cli():
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "tools/run_t12_activation_recompute.py",
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--dataset-root" in result.stdout
    assert "--output-root" in result.stdout

