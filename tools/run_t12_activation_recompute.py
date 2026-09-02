#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# When invoked as `python tools/run_t12_activation_recompute.py`, Python
# puts tools/ on sys.path. Add the repository root before importing src.*.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers
import yaml
import safetensors
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generation import (
    BALANCED_REASONING,
    EMPTY_ANSWER,
    MALFORMED_REASONING,
    GenerationRecord,
    TokenAlignmentError,
    classify_validity,
    segment_response,
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

STORE_DTYPE = torch.float16
FORMAT_VERSION = "t12-activation-recompute-v3"

MODE_SOURCE_DIR = {
    "translated": "translated",
    "wrapper": "extraction",
    "default": "default",
}


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_metadata() -> dict:
    def run(*args):
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def tokenizer_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(
        text,
        add_special_tokens=False,
    ).input_ids


# ---------------------------------------------------------------------
# Prepared row
# ---------------------------------------------------------------------

@dataclass
class PreparedRow:
    row: dict

    prompt_ids: list[int]
    output_ids: list[int]
    full_ids: list[int]

    validity: str
    segmentation_case: Optional[str]

    all_span: Optional[tuple[int, int]]
    reasoning_span: Optional[tuple[int, int]]
    answer_span: Optional[tuple[int, int]]

    # T12 uses token IDs reconstructed from stored text because run-003
    # did not persist generation-native token IDs. The frozen segmentation
    # routine performs a strict prefix-token check at </think>; with BPE
    # tokenizers, tokenizing a prefix independently can differ from the
    # prefix of tokenizing the whole completion. This provenance records
    # whether the frozen check passed directly or T12 used the validated
    # stored-text boundary reconstruction.
    segmentation_provenance: Optional[str]
    frozen_boundary_prefix_check_passed: Optional[bool]
    segmentation_note: Optional[str]

    token_count_delta: Optional[int]
    preparation_error: Optional[str]


OPEN_MARKER = "<think>"
CLOSE_MARKER = "</think>"



def prompt_seeded_reasoning_boundary(
    prompt_text: str,
    record: GenerationRecord,
    tokenize,
):
    """
    Reconstruct the run-003 reasoning/answer boundary when the serialized
    prompt already opens the reasoning region with ``<think>``.

    Historical run-003 rows are serialized like:

        rendered_prompt = "...<|Assistant|><think>\\n"
        completion      = "reasoning text ... </think>final answer"

    Thus ``record.output_text`` legitimately contains a closing marker
    without an opening marker. The frozen ``segment_response`` function
    only sees output_text, so it treats that row as malformed. For T12,
    we must reconstruct the boundary from the full stored serialization.

    Pooling remains response-only:
      * all-response: prompt end -> completion end
      * reasoning:    prompt end -> through final </think>
      * answer:       tokens after final </think> -> completion end

    Token IDs remain reconstructed from stored text, as required by T12.
    """

    if not prompt_text.rstrip().endswith(OPEN_MARKER):
        raise TokenAlignmentError(
            "prompt-seeded reconstruction requested but rendered_prompt "
            "does not end with <think>"
        )

    text = record.output_text
    start = len(record.prompt_token_ids)
    content = list(record.output_token_ids)
    end = start + len(content)

    # The opening marker that starts the reasoning region is in the prompt.
    # Any additional opening markers in the completion must still balance.
    n_open_in_completion = text.count(OPEN_MARKER)
    n_close_in_completion = text.count(CLOSE_MARKER)

    if n_close_in_completion == 0:
        raise TokenAlignmentError(
            "prompt-seeded reconstruction requested but completion has "
            "no </think>"
        )

    expected_closes = 1 + n_open_in_completion
    if n_close_in_completion != expected_closes:
        raise TokenAlignmentError(
            "unbalanced prompt-seeded reasoning markers: "
            f"prompt_open=1, completion_open={n_open_in_completion}, "
            f"completion_close={n_close_in_completion}"
        )

    if n_open_in_completion:
        if text.rfind(CLOSE_MARKER) < text.rfind(OPEN_MARKER):
            raise TokenAlignmentError(
                "final </think> appears before the final completion <think>"
            )

    close_idx = text.rfind(CLOSE_MARKER)
    boundary_char = close_idx + len(CLOSE_MARKER)

    # This is the exact reconstruction procedure already validated against
    # the historical answer tensors in the five-row GPU gate.
    prefix_ids = tokenize(text[:boundary_char])
    boundary = start + len(prefix_ids)

    # The frozen delimiter prefix-token check (segment_response step 2): does the
    # separately tokenized prefix through the final </think> equal the actual
    # prefix of the stored content token IDs? For prompt-seeded rows this is
    # informative per row rather than None, so a reviewer can read the pass rate
    # instead of trusting the five-row gate to cover 44,400 rows.
    prefix_check_passed = (prefix_ids == content[: len(prefix_ids)])

    if boundary < start or boundary > end:
        raise TokenAlignmentError(
            "prompt-seeded reconstructed boundary lies outside response "
            f"span: start={start}, boundary={boundary}, end={end}"
        )

    all_span = (start, end)
    reasoning_span = (start, boundary)
    answer_text = text[boundary_char:]

    if not answer_text.strip():
        return (
            EMPTY_ANSWER,
            all_span,
            reasoning_span,
            None,
            prefix_check_passed,
        )

    if boundary >= end:
        raise TokenAlignmentError(
            "non-empty answer text produced an empty answer-token span"
        )

    return (
        BALANCED_REASONING,
        all_span,
        reasoning_span,
        (boundary, end),
        prefix_check_passed,
    )


def reconstructed_reasoning_boundary(
    record: GenerationRecord,
    tokenize,
):
    """
    Reconstruct the balanced-reasoning boundary from stored completion text.

    This fallback is used ONLY when frozen ``segment_response`` raises
    ``TokenAlignmentError`` on its delimiter-prefix check.

    Why this is needed:
      * run-003 did not persist generation-native output token IDs;
      * T12 therefore retokenizes the stored completion text;
      * for BPE tokenizers, tokenize(prefix) need not equal a literal prefix
        of tokenize(full_text), even though both are deterministic;
      * the historical activation worker located the answer boundary by
        tokenizing the text through the final </think> separately.

    The five-row historical reconstruction gate in this notebook validated
    that exact stored-text procedure against the historical answer tensors.

    Returns:
        (case, all_span, reasoning_span, answer_span)
    """

    text = record.output_text
    start = len(record.prompt_token_ids)
    content = list(record.output_token_ids)
    end = start + len(content)

    close_idx = text.rfind(CLOSE_MARKER)

    if close_idx < 0:
        raise TokenAlignmentError(
            "reconstruction fallback requested but no final </think> exists"
        )

    boundary_char = close_idx + len(CLOSE_MARKER)
    prefix_ids = tokenize(text[:boundary_char])

    boundary = start + len(prefix_ids)

    if boundary < start or boundary > end:
        raise TokenAlignmentError(
            "reconstructed reasoning boundary lies outside the stored "
            f"response span: start={start}, boundary={boundary}, end={end}"
        )

    all_span = (start, end)
    reasoning_span = (start, boundary)
    answer_text = text[boundary_char:]

    if not answer_text.strip():
        return (
            EMPTY_ANSWER,
            all_span,
            reasoning_span,
            None,
        )

    if boundary >= end:
        raise TokenAlignmentError(
            "non-empty final-answer text produced an empty reconstructed "
            "answer-token span"
        )

    return (
        "balanced_reasoning",
        all_span,
        reasoning_span,
        (boundary, end),
    )


def prepare_row(
    row: dict,
    tokenizer,
) -> PreparedRow:
    """
    Reconstruct token IDs and locate the reasoning/answer boundary for one
    historical run-003 row, trying three boundary procedures in order. The
    frozen strict prefix-token check is the primary method, NOT the only
    one: two validated fallbacks exist, and a reader who only checks
    ``segmentation_provenance`` for ``"frozen_segment_response"`` will
    silently miss rows resolved by either of them.

    1. ``prompt_seeded_reasoning_boundary`` - DeepSeek-R1-Distill's chat
       template prefills the opening ``<think>`` into the prompt. When the
       completion contains the closing ``</think>``, this reconstructs the
       boundary from the full stored serialization.
    2. Frozen ``segment_response`` - the strict delimiter-prefix token
       check, used for non-seeded rows.
    3. ``reconstructed_reasoning_boundary`` - when (2) raises
       ``TokenAlignmentError``, this non-seeded BPE fallback recovers the
       boundary from the separately tokenized stored-text prefix through
       the final ``</think>``. This is the same procedure validated
       against historical answer tensors by the five-row reconstruction
       gate referenced in that function's docstring, not an ad-hoc guess.
       A row resolved this way is NOT dropped or marked invalid solely for
       having failed the strict check; ``frozen_boundary_prefix_check_passed``
       records the distinction, and ``validity`` is corrected back to
       ``"valid"`` when this is the only issue found.

    See ``segmentation_provenance`` on the returned ``PreparedRow`` to tell
    which of the three procedures a given row actually used.
    """

    try:
        prompt_text = row["rendered_prompt"]
        completion = row["completion"]

        prompt_ids = tokenizer_ids(
            tokenizer,
            prompt_text,
        )

        output_ids = tokenizer_ids(
            tokenizer,
            completion,
        )

        record = GenerationRecord(
            prompt_token_ids=prompt_ids,
            output_token_ids=output_ids,
            output_text=completion,
            finish_reason=row.get(
                "finish_reason",
                "stop",
            ),
        )

        tokenize = lambda text: tokenizer_ids(
            tokenizer,
            text,
        )

        # Historical run-003 does not contain generation-native token IDs
        # or original message structures.
        #
        # We therefore:
        #   1. reconstruct token IDs from the stored text;
        #   2. preserve this provenance explicitly;
        #   3. do NOT claim an independent text/token-ID validation.
        validity = classify_validity(
            record,
            tokenize=tokenize,
            terminal_ids=(),
            serialization_ok=True,
        )

        segmentation_provenance = None
        frozen_boundary_prefix_check_passed = None
        segmentation_note = None

        # IMPORTANT run-003 serialization detail:
        # DeepSeek-R1-Distill's chat template prefills the opening <think> into
        # the prompt, so the stored rendered_prompt ends in <think> and the
        # reasoning region is already OPEN when the completion begins. Reading
        # only the completion misreads these rows:
        #   * completion WITH a closing </think>: a normal reasoning response,
        #     but the frozen segmenter sees an orphan close (malformed);
        #   * completion with NO </think>: the seeded reasoning region never
        #     closed, yet the frozen segmenter sees a marker-free completion,
        #     calls it direct_answer, and stores the whole in-progress trace as
        #     the ANSWER tensor. METHOD_FREEZE 6.2 classifies an unclosed open
        #     marker as malformed_reasoning (answer and reasoning pools
        #     unavailable, judge abstains; all-response pool retained).
        # Handle both seeded cases before the frozen segmenter.
        prompt_seeded = prompt_text.rstrip().endswith(OPEN_MARKER)
        prompt_seeded_closed = prompt_seeded and CLOSE_MARKER in completion
        prompt_seeded_unclosed = prompt_seeded and CLOSE_MARKER not in completion

        if prompt_seeded_closed:
            try:
                (
                    segmentation_case,
                    all_span,
                    reasoning_span,
                    answer_span,
                    prefix_check_passed,
                ) = prompt_seeded_reasoning_boundary(
                    prompt_text,
                    record,
                    tokenize,
                )

                segmentation_provenance = (
                    "prompt_seeded_reconstructed_from_stored_text_boundary"
                )
                frozen_boundary_prefix_check_passed = prefix_check_passed
                segmentation_note = (
                    "rendered_prompt contains the opening <think>; "
                    "completion contains the closing </think>. Boundary "
                    "reconstructed from the stored full serialization using "
                    "the historical procedure validated against old tensors. "
                    "frozen_boundary_prefix_check_passed records whether the "
                    "separately tokenized prefix equals the stored ID prefix."
                )

            except Exception as seeded_exc:
                segmentation_case = MALFORMED_REASONING
                all_span = (
                    len(prompt_ids),
                    len(prompt_ids) + len(output_ids),
                ) if output_ids else None
                reasoning_span = None
                answer_span = None
                segmentation_provenance = (
                    "prompt_seeded_boundary_failed"
                )
                frozen_boundary_prefix_check_passed = None
                segmentation_note = (
                    f"{type(seeded_exc).__name__}: {seeded_exc}"
                )

        elif prompt_seeded_unclosed:
            # Unclosed seeded reasoning region -> malformed per METHOD_FREEZE
            # 6.2. All-response pool retained; reasoning/answer pools
            # unavailable, so the row cannot enter the answer/reasoning
            # sensitivities and the judge abstains, which is the contamination
            # the malformed rule exists to quarantine. validity keeps its
            # technical label (e.g. truncated when the budget ran out mid-trace).
            segmentation_case = MALFORMED_REASONING
            all_span = (
                len(prompt_ids),
                len(prompt_ids) + len(output_ids),
            ) if output_ids else None
            reasoning_span = None
            answer_span = None
            segmentation_provenance = "prompt_seeded_unclosed_open_marker"
            frozen_boundary_prefix_check_passed = None
            segmentation_note = (
                "rendered_prompt ends with <think> and the completion has no "
                "</think>: the seeded reasoning region never closed. "
                "METHOD_FREEZE 6.2 classifies an unclosed open marker as "
                "malformed_reasoning (all-response retained; reasoning/answer "
                "unavailable; judge abstains)."
            )

        else:
            try:
                segmentation = segment_response(
                    record,
                    tokenize=tokenize,
                    terminal_ids=(),
                )

                segmentation_case = segmentation.case
                all_span = segmentation.all_response_span
                reasoning_span = segmentation.reasoning_span
                answer_span = segmentation.final_answer_span

                segmentation_provenance = "frozen_segment_response"
                frozen_boundary_prefix_check_passed = (
                    True
                    if segmentation_case in {
                        BALANCED_REASONING,
                        EMPTY_ANSWER,
                    }
                    else None
                )

            except TokenAlignmentError as strict_exc:
                # For rows whose opening/closing markers are both in the
                # completion, retain the reconstruction-aware BPE fallback.
                try:
                    (
                        segmentation_case,
                        all_span,
                        reasoning_span,
                        answer_span,
                    ) = reconstructed_reasoning_boundary(
                        record,
                        tokenize,
                    )

                    segmentation_provenance = (
                        "reconstructed_from_stored_text_boundary"
                    )
                    frozen_boundary_prefix_check_passed = False
                    segmentation_note = (
                        "Frozen delimiter-prefix token check did not pass "
                        "on reconstructed token IDs; used separately "
                        "tokenized stored-text prefix through final "
                        "</think>. "
                        f"Original check: {strict_exc}"
                    )

                    if validity == "token_alignment_failure":
                        validity = "valid"

                except Exception as fallback_exc:
                    segmentation_case = "malformed_reasoning"
                    all_span = (
                        len(prompt_ids),
                        len(prompt_ids) + len(output_ids),
                    ) if output_ids else None
                    reasoning_span = None
                    answer_span = None
                    segmentation_provenance = (
                        "reconstructed_boundary_failed"
                    )
                    frozen_boundary_prefix_check_passed = False
                    segmentation_note = (
                        f"Frozen segmenter: {type(strict_exc).__name__}: "
                        f"{strict_exc}; reconstruction fallback: "
                        f"{type(fallback_exc).__name__}: {fallback_exc}"
                    )

        historical_n = row.get("n_out_tokens")

        if historical_n is None:
            token_count_delta = None
        else:
            token_count_delta = (
                len(output_ids)
                - int(historical_n)
            )

        return PreparedRow(
            row=row,
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            full_ids=prompt_ids + output_ids,
            validity=validity,
            segmentation_case=segmentation_case,
            all_span=all_span,
            reasoning_span=reasoning_span,
            answer_span=answer_span,
            segmentation_provenance=segmentation_provenance,
            frozen_boundary_prefix_check_passed=(
                frozen_boundary_prefix_check_passed
            ),
            segmentation_note=segmentation_note,
            token_count_delta=token_count_delta,
            preparation_error=None,
        )

    except Exception as exc:
        return PreparedRow(
            row=row,
            prompt_ids=[],
            output_ids=[],
            full_ids=[],
            # "preparation_failure" is not in the frozen VALIDITY_LABELS
            # vocabulary (src/generation.py); prepare_row() raised before
            # classify_validity() could run, which is itself a serialization
            # failure - the row could not be turned into a valid
            # GenerationRecord. preparation_error below carries the specific
            # exception for anyone who needs to distinguish this path from
            # classify_validity()'s own serialization_failure branch.
            validity="serialization_failure",
            segmentation_case=None,
            all_span=None,
            reasoning_span=None,
            answer_span=None,
            segmentation_provenance=None,
            frozen_boundary_prefix_check_passed=None,
            segmentation_note=None,
            token_count_delta=None,
            preparation_error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ---------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------

def make_batches(
    rows: list[dict],
    token_budget: int,
    max_batch: int,
):
    """
    Preserve source order.

    Historical n_out_tokens is used only as a rough batching estimate,
    never as authoritative tokenization.
    """

    current = []
    estimated_tokens = 0

    for row in rows:

        estimate = (
            int(row.get("n_out_tokens", 512))
            + 500
        )

        would_overflow = (
            current
            and (
                len(current) >= max_batch
                or estimated_tokens + estimate > token_budget
            )
        )

        if would_overflow:
            yield current
            current = []
            estimated_tokens = 0

        current.append(row)
        estimated_tokens += estimate

    if current:
        yield current


# ---------------------------------------------------------------------
# Activation pooling collector
# ---------------------------------------------------------------------

class PoolCollector:

    def __init__(
        self,
        model,
        middle_layer: int,
        store_dtype=STORE_DTYPE,
    ):
        self.model = model
        self.middle_layer = middle_layer
        self.store_dtype = store_dtype

        self.n_layers = len(
            model.model.layers
        )

        self.hidden_size = (
            model.config.hidden_size
        )

        self.current_all = None
        self.current_answer = None
        self.current_reasoning = None

        self.acc_all = {}
        self.acc_answer = {}
        self.acc_reasoning = {}

        self.handles = [
            model.model.layers[layer]
            .register_forward_hook(
                self._make_hook(layer)
            )
            for layer in range(
                self.n_layers
            )
        ]

    @staticmethod
    def _pool_one(
        h,
        item_index,
        span,
        hidden_size,
    ):
        if span is None:
            return torch.zeros(
                hidden_size,
                dtype=torch.float32,
                device=h.device,
            )

        start, end = span

        if end <= start:
            return torch.zeros(
                hidden_size,
                dtype=torch.float32,
                device=h.device,
            )

        return (
            h[
                item_index,
                start:end,
            ]
            .float()
            .mean(dim=0)
        )

    def _make_hook(self, layer):

        def hook(
            _module,
            _inputs,
            output,
        ):
            h = (
                output[0]
                if isinstance(output, tuple)
                else output
            )

            if self.current_all is None:
                return None

            self.acc_all[layer] = (
                torch.stack(
                    [
                        self._pool_one(
                            h,
                            i,
                            span,
                            self.hidden_size,
                        )
                        for i, span
                        in enumerate(
                            self.current_all
                        )
                    ]
                )
                .to(self.store_dtype)
                .cpu()
            )

            self.acc_answer[layer] = (
                torch.stack(
                    [
                        self._pool_one(
                            h,
                            i,
                            span,
                            self.hidden_size,
                        )
                        for i, span
                        in enumerate(
                            self.current_answer
                        )
                    ]
                )
                .to(self.store_dtype)
                .cpu()
            )

            if layer == self.middle_layer:
                self.acc_reasoning[layer] = (
                    torch.stack(
                        [
                            self._pool_one(
                                h,
                                i,
                                span,
                                self.hidden_size,
                            )
                            for i, span
                            in enumerate(
                                self.current_reasoning
                            )
                        ]
                    )
                    .to(self.store_dtype)
                    .cpu()
                )

            # Read-only hook.
            return None

        return hook

    def set_spans(
        self,
        prepared: list[PreparedRow],
    ):
        self.current_all = [
            p.all_span
            for p in prepared
        ]

        self.current_answer = [
            p.answer_span
            for p in prepared
        ]

        self.current_reasoning = [
            p.reasoning_span
            for p in prepared
        ]

        self.acc_all.clear()
        self.acc_answer.clear()
        self.acc_reasoning.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


# ---------------------------------------------------------------------
# Chunk persistence
# ---------------------------------------------------------------------

def existing_completed_uids(
    mode_dir: Path,
) -> set[str]:

    completed = set()

    for path in sorted(
        mode_dir.glob(
            "meta_part*.jsonl"
        )
    ):
        for row in read_jsonl(path):
            completed.add(
                row["uid"]
            )

    return completed


def next_part_index(
    mode_dir: Path,
) -> int:

    numbers = []

    for path in mode_dir.glob(
        "meta_part*.jsonl"
    ):
        stem = path.stem

        try:
            numbers.append(
                int(
                    stem.replace(
                        "meta_part",
                        "",
                    )
                )
            )
        except ValueError:
            pass

    return (
        max(numbers) + 1
        if numbers
        else 0
    )


def save_chunk(
    mode_dir: Path,
    part_index: int,
    all_store: dict,
    answer_store: dict,
    reasoning_store: dict,
    meta_rows: list[dict],
) -> dict:

    mode_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = f"{part_index:04d}"

    records = {}

    def save_tensor_group(
        label,
        tensors,
    ):
        if not tensors:
            return None

        final_path = (
            mode_dir
            / f"{label}_part{suffix}.safetensors"
        )

        tmp_path = final_path.with_suffix(
            ".safetensors.tmp"
        )

        save_file(
            {
                key: value.contiguous()
                for key, value
                in tensors.items()
            },
            str(tmp_path),
        )

        os.replace(
            tmp_path,
            final_path,
        )

        return {
            "path": final_path.name,
            "bytes": final_path.stat().st_size,
            "sha256": sha256_file(
                final_path
            ),
            "n_tensors": len(tensors),
        }

    records["all_response"] = (
        save_tensor_group(
            "all_response",
            all_store,
        )
    )

    records["answer"] = (
        save_tensor_group(
            "answer",
            answer_store,
        )
    )

    records["reasoning_middle"] = (
        save_tensor_group(
            "reasoning_middle",
            reasoning_store,
        )
    )

    meta_path = (
        mode_dir
        / f"meta_part{suffix}.jsonl"
    )

    tmp_meta = meta_path.with_suffix(
        ".jsonl.tmp"
    )

    with tmp_meta.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in meta_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    os.replace(
        tmp_meta,
        meta_path,
    )

    records["metadata"] = {
        "path": meta_path.name,
        "bytes": meta_path.stat().st_size,
        "sha256": sha256_file(
            meta_path
        ),
        "n_rows": len(meta_rows),
    }

    part_manifest = {
        "format_version": FORMAT_VERSION,
        "part_index": part_index,
        "files": records,
    }

    manifest_path = (
        mode_dir
        / f"part_manifest_{suffix}.json"
    )

    write_json(
        manifest_path,
        part_manifest,
    )

    return part_manifest


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def metadata_for(
    prepared: PreparedRow,
) -> dict:

    row = prepared.row

    def span_length(span):
        if span is None:
            return None

        return (
            span[1]
            - span[0]
        )

    return {
        "uid": row.get("uid"),
        "role": row.get("role"),
        "question_id": row.get(
            "question_id"
        ),
        "prompt_idx": row.get(
            "prompt_idx"
        ),
        "condition": row.get(
            "condition"
        ),
        "source_shard": row.get(
            "shard"
        ),

        "finish_reason": row.get(
            "finish_reason"
        ),

        "historical_n_out_tokens": (
            row.get("n_out_tokens")
        ),

        "reconstructed_n_out_tokens": (
            len(
                prepared.output_ids
            )
        ),

        "token_count_delta": (
            prepared.token_count_delta
        ),

        "validity": (
            prepared.validity
        ),

        "segmentation_case": (
            prepared.segmentation_case
        ),

        "segmentation_provenance": (
            prepared.segmentation_provenance
        ),

        "frozen_boundary_prefix_check_passed": (
            prepared.frozen_boundary_prefix_check_passed
        ),

        "segmentation_note": (
            prepared.segmentation_note
        ),

        "n_all_response_tokens": (
            span_length(
                prepared.all_span
            )
        ),

        "n_reasoning_tokens": (
            span_length(
                prepared.reasoning_span
            )
        ),

        "n_answer_tokens": (
            span_length(
                prepared.answer_span
            )
        ),

        "has_all_response_pool": (
            prepared.all_span
            is not None
        ),

        "has_reasoning_pool": (
            prepared.reasoning_span
            is not None
        ),

        "has_answer_pool": (
            prepared.answer_span
            is not None
        ),

        "preparation_error": (
            prepared.preparation_error
        ),

        # Critical provenance statement.
        "token_id_provenance": (
            "reconstructed_from_stored_text"
        ),

        "generation_native_token_ids_available": (
            False
        ),

        "serialization_check": (
            "stored_rendered_prompt_used_directly;"
            "original_message_structure_not_available"
        ),
    }


# ---------------------------------------------------------------------
# One mode
# ---------------------------------------------------------------------

def source_paths_for_mode(
    dataset_root: Path,
    mode: str,
) -> list[Path]:

    source_dir = (
        dataset_root
        / MODE_SOURCE_DIR[mode]
    )

    paths = sorted(
        source_dir.glob(
            "rollouts_shard*.jsonl"
        )
    )

    if not paths:
        raise FileNotFoundError(
            f"No rollout shards found for "
            f"{mode}: {source_dir}"
        )

    return paths


def run_mode(
    mode: str,
    dataset_root: Path,
    output_root: Path,
    tokenizer,
    model,
    middle_layer: int,
    token_budget: int,
    max_batch: int,
    chunk_size: int,
    limit: Optional[int],
    config_path: Path,
):

    source_paths = (
        source_paths_for_mode(
            dataset_root,
            mode,
        )
    )

    rows = []

    for path in source_paths:
        rows.extend(
            read_jsonl(path)
        )

    original_count = len(rows)

    uid_counts = collections.Counter(
        row["uid"]
        for row in rows
    )

    duplicates = [
        uid
        for uid, count
        in uid_counts.items()
        if count != 1
    ]

    if duplicates:
        raise RuntimeError(
            f"{mode}: duplicate UIDs "
            f"found: {duplicates[:10]}"
        )

    mode_dir = (
        output_root
        / mode
    )

    mode_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = (
        existing_completed_uids(
            mode_dir
        )
    )

    rows = [
        row
        for row in rows
        if row["uid"]
        not in completed
    ]

    if limit is not None:
        rows = rows[:limit]

    print(
        f"\n[{mode}] "
        f"source={original_count:,} "
        f"already_complete={len(completed):,} "
        f"todo={len(rows):,}"
    )

    collector = PoolCollector(
        model=model,
        middle_layer=middle_layer,
        store_dtype=STORE_DTYPE,
    )

    part_index = next_part_index(
        mode_dir
    )

    all_store = {}
    answer_store = {}
    reasoning_store = {}
    meta_rows = []

    processed_since_flush = 0

    t0 = time.time()
    n_processed = 0

    try:

        for batch_rows in make_batches(
            rows,
            token_budget=token_budget,
            max_batch=max_batch,
        ):

            prepared = [
                prepare_row(
                    row,
                    tokenizer,
                )
                for row in batch_rows
            ]

            def _is_runnable(p):
                return bool(p.full_ids) and p.all_span is not None

            # A single predicate partitions both lists in one O(n) pass.
            # The previous `p not in runnable` re-scanned `runnable` for
            # every row (O(n^2)) and relied on PreparedRow's default
            # dataclass value-equality to define membership, which is both
            # slow and the wrong notion of identity for this partition.
            runnable = [
                p
                for p in prepared
                if _is_runnable(p)
            ]

            # Metadata-only records for rows
            # that cannot be teacher-forced.
            not_runnable = [
                p
                for p in prepared
                if not _is_runnable(p)
            ]

            for p in not_runnable:
                meta_rows.append(
                    metadata_for(p)
                )

                processed_since_flush += 1
                n_processed += 1

            if runnable:

                max_length = max(
                    len(p.full_ids)
                    for p in runnable
                )

                pad_id = (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id
                    is not None
                    else tokenizer.eos_token_id
                )

                if pad_id is None:
                    raise RuntimeError(
                        "Tokenizer has neither "
                        "pad_token_id nor eos_token_id"
                    )

                input_ids = torch.full(
                    (
                        len(runnable),
                        max_length,
                    ),
                    pad_id,
                    dtype=torch.long,
                )

                attention_mask = (
                    torch.zeros_like(
                        input_ids
                    )
                )

                for i, p in enumerate(
                    runnable
                ):
                    ids = torch.tensor(
                        p.full_ids,
                        dtype=torch.long,
                    )

                    input_ids[
                        i,
                        : len(ids),
                    ] = ids

                    attention_mask[
                        i,
                        : len(ids),
                    ] = 1

                collector.set_spans(
                    runnable
                )

                try:
                    with torch.inference_mode():
                        model(
                            input_ids=input_ids.to(
                                model.device
                            ),
                            attention_mask=(
                                attention_mask.to(
                                    model.device
                                )
                            ),
                            use_cache=False,
                        )

                except torch.cuda.OutOfMemoryError:

                    torch.cuda.empty_cache()

                    if len(runnable) == 1:
                        raise

                    # Retry one item at a time.
                    for single in runnable:

                        collector.set_spans(
                            [single]
                        )

                        ids = torch.tensor(
                            [single.full_ids],
                            dtype=torch.long,
                            device=model.device,
                        )

                        mask = torch.ones_like(
                            ids
                        )

                        with torch.inference_mode():
                            model(
                                input_ids=ids,
                                attention_mask=mask,
                                use_cache=False,
                            )

                        uid = single.row[
                            "uid"
                        ]

                        all_store[uid] = (
                            torch.stack(
                                [
                                    collector
                                    .acc_all[layer][0]
                                    for layer
                                    in range(
                                        collector.n_layers
                                    )
                                ]
                            )
                        )

                        if (
                            single.answer_span
                            is not None
                        ):
                            answer_store[
                                uid
                            ] = torch.stack(
                                [
                                    collector
                                    .acc_answer[layer][0]
                                    for layer
                                    in range(
                                        collector.n_layers
                                    )
                                ]
                            )

                        if (
                            single.reasoning_span
                            is not None
                        ):
                            reasoning_store[
                                uid
                            ] = (
                                collector
                                .acc_reasoning[
                                    middle_layer
                                ][0]
                            )

                        meta_rows.append(
                            metadata_for(
                                single
                            )
                        )

                        processed_since_flush += 1
                        n_processed += 1

                else:

                    for i, p in enumerate(
                        runnable
                    ):

                        uid = p.row["uid"]

                        all_store[uid] = (
                            torch.stack(
                                [
                                    collector
                                    .acc_all[layer][i]
                                    for layer
                                    in range(
                                        collector.n_layers
                                    )
                                ]
                            )
                        )

                        if (
                            p.answer_span
                            is not None
                        ):
                            answer_store[
                                uid
                            ] = torch.stack(
                                [
                                    collector
                                    .acc_answer[layer][i]
                                    for layer
                                    in range(
                                        collector.n_layers
                                    )
                                ]
                            )

                        if (
                            p.reasoning_span
                            is not None
                        ):
                            reasoning_store[
                                uid
                            ] = (
                                collector
                                .acc_reasoning[
                                    middle_layer
                                ][i]
                            )

                        meta_rows.append(
                            metadata_for(p)
                        )

                        processed_since_flush += 1
                        n_processed += 1

            if (
                processed_since_flush
                >= chunk_size
            ):

                save_chunk(
                    mode_dir=mode_dir,
                    part_index=part_index,
                    all_store=all_store,
                    answer_store=answer_store,
                    reasoning_store=(
                        reasoning_store
                    ),
                    meta_rows=meta_rows,
                )

                part_index += 1

                all_store = {}
                answer_store = {}
                reasoning_store = {}
                meta_rows = []

                processed_since_flush = 0

                elapsed = (
                    time.time()
                    - t0
                )

                rate = (
                    n_processed
                    / max(
                        elapsed,
                        1e-9,
                    )
                    * 3600
                )

                print(
                    f"[{mode}] "
                    f"{n_processed:,}/"
                    f"{len(rows):,} "
                    f"this invocation | "
                    f"{rate:,.0f}/h",
                    flush=True,
                )

        if meta_rows:

            save_chunk(
                mode_dir=mode_dir,
                part_index=part_index,
                all_store=all_store,
                answer_store=answer_store,
                reasoning_store=(
                    reasoning_store
                ),
                meta_rows=meta_rows,
            )

    finally:
        collector.close()

    return build_mode_report(
        mode=mode,
        dataset_root=dataset_root,
        output_root=output_root,
        source_paths=source_paths,
        source_count=original_count,
        config_path=config_path,
    )


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------

def all_mode_metadata(
    mode_dir: Path,
) -> list[dict]:

    rows = []

    for path in sorted(
        mode_dir.glob(
            "meta_part*.jsonl"
        )
    ):
        rows.extend(
            read_jsonl(path)
        )

    return rows


def build_mode_report(
    mode: str,
    dataset_root: Path,
    output_root: Path,
    source_paths: list[Path],
    source_count: int,
    config_path: Path,
) -> dict:

    mode_dir = (
        output_root
        / mode
    )

    meta = all_mode_metadata(
        mode_dir
    )

    validity = collections.Counter(
        row["validity"]
        for row in meta
    )

    segmentation = (
        collections.Counter(
            str(
                row[
                    "segmentation_case"
                ]
            )
            for row in meta
        )
    )

    segmentation_provenance = (
        collections.Counter(
            str(
                row.get(
                    "segmentation_provenance"
                )
            )
            for row in meta
        )
    )

    token_deltas = (
        collections.Counter(
            str(
                row[
                    "token_count_delta"
                ]
            )
            for row in meta
            if row[
                "token_count_delta"
            ]
            is not None
        )
    )

    # Item 1 (review): prompt-seeded rows whose reasoning region was opened in
    # the prompt but never closed are malformed_reasoning, not direct_answer.
    # Report them and their finish_reason split so the reclassified tail is
    # auditable. By construction their rendered_prompt ends with <think>.
    unclosed = [
        row for row in meta
        if row.get("segmentation_provenance")
        == "prompt_seeded_unclosed_open_marker"
    ]
    prompt_seeded_unclosed_report = {
        "count": len(unclosed),
        "rendered_prompt_ends_with_open_marker": True,
        "by_finish_reason": dict(
            sorted(
                collections.Counter(
                    str(row.get("finish_reason")) for row in unclosed
                ).items()
            )
        ),
        "classification": (
            "malformed_reasoning per METHOD_FREEZE 6.2 (all-response retained; "
            "reasoning/answer pools unavailable; judge abstains)"
        ),
    }

    # Item 2 (review): make the frozen delimiter prefix-token check a reported
    # pass rate over the prompt-seeded reconstruction path, rather than a field
    # that is None for ~all production rows.
    seeded_closed = [
        row for row in meta
        if row.get("segmentation_provenance")
        == "prompt_seeded_reconstructed_from_stored_text_boundary"
    ]
    prefix_check = collections.Counter(
        str(row.get("frozen_boundary_prefix_check_passed"))
        for row in seeded_closed
    )
    n_seeded = len(seeded_closed)
    frozen_boundary_prefix_check_report = {
        "population": "prompt_seeded_reconstructed_from_stored_text_boundary",
        "n_rows": n_seeded,
        "counts": dict(sorted(prefix_check.items())),
        "pass_rate": (
            round(prefix_check.get("True", 0) / n_seeded, 6)
            if n_seeded else None
        ),
        "note": (
            "True means the separately tokenized prefix through the final "
            "</think> equals the stored content-ID prefix, i.e. the "
            "reconstruction is exact for that row. Scales the five-row gate to "
            "the whole production set."
        ),
    }

    report = {
        "format_version": FORMAT_VERSION,

        "created_utc": (
            time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            )
        ),

        "git": git_metadata(),

        "config": {
            "path": str(
                config_path
            ),
            "sha256": (
                sha256_file(
                    config_path
                )
            ),
        },

        "mode": mode,

        "source_count": source_count,
        "recomputed_metadata_count": (
            len(meta)
        ),

        "complete": (
            len(meta)
            == source_count
        ),

        "validity_counts": dict(
            sorted(
                validity.items()
            )
        ),

        "segmentation_counts": dict(
            sorted(
                segmentation.items()
            )
        ),

        "segmentation_provenance_counts": dict(
            sorted(
                segmentation_provenance.items()
            )
        ),

        "prompt_seeded_unclosed_report": (
            prompt_seeded_unclosed_report
        ),

        "frozen_boundary_prefix_check_report": (
            frozen_boundary_prefix_check_report
        ),

        "pool_counts": {
            "all_response": sum(
                bool(
                    r[
                        "has_all_response_pool"
                    ]
                )
                for r in meta
            ),
            "answer": sum(
                bool(
                    r[
                        "has_answer_pool"
                    ]
                )
                for r in meta
            ),
            "reasoning": sum(
                bool(
                    r[
                        "has_reasoning_pool"
                    ]
                )
                for r in meta
            ),
        },

        "token_count_delta_counts": dict(
            sorted(
                token_deltas.items()
            )
        ),

        "source_files": [
            {
                "path": str(path),
                "bytes": (
                    path.stat().st_size
                ),
                "sha256": (
                    sha256_file(path)
                ),
            }
            for path in source_paths
        ],

        "token_provenance": {
            "generation_native_token_ids_available": False,
            "method": (
                "token IDs reconstructed "
                "deterministically from stored "
                "rendered_prompt and completion"
            ),
            "historical_n_out_tokens_used_as": (
                "audit-only token-count field"
            ),
            "reasoning_boundary_method": (
                "use frozen segment_response when its strict prefix-token "
                "check passes; otherwise use the separately tokenized "
                "stored-text prefix through the final </think>, matching "
                "the historical worker procedure validated by the five-row "
                "reconstruction gate"
            ),
        },

        "pooling": {
            "primary": (
                "all response tokens"
            ),
            "answer_sensitivity": (
                "final-answer span under frozen text semantics; "
                "stored-text boundary reconstruction is used when "
                "generation-native token IDs are unavailable"
            ),
            "reasoning_sensitivity": (
                "reasoning span under frozen text semantics; "
                "stored-text boundary reconstruction is used when "
                "generation-native token IDs are unavailable; "
                "middle layer only"
            ),
        },
    }

    write_json(
        mode_dir
        / "activation_recompute_report.json",
        report,
    )

    return report


def build_global_manifest(
    output_root: Path,
    config_path: Path,
    model_info: dict,
    middle_layer: int,
    reports: dict,
) -> dict:

    parts = {}

    for mode in reports:

        mode_dir = (
            output_root
            / mode
        )

        parts[mode] = []

        for path in sorted(
            mode_dir.glob(
                "part_manifest_*.json"
            )
        ):
            parts[mode].append(
                json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            )

    manifest = {
        "format_version": FORMAT_VERSION,

        "created_utc": (
            time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            )
        ),

        "git": git_metadata(),

        "config": {
            "path": str(
                config_path
            ),
            "sha256": (
                sha256_file(
                    config_path
                )
            ),
        },

        "model": model_info,

        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
            "pyyaml": yaml.__version__,
        },

        "activation_site": {
            "module": (
                "model.model.layers[L]"
            ),
            "site": (
                "post-MLP residual / "
                "transformer block output"
            ),
            "n_layers": (
                model_info[
                    "n_layers"
                ]
            ),
            "hidden_size": (
                model_info[
                    "hidden_size"
                ]
            ),
            "middle_layer": (
                middle_layer
            ),
            "layer_convention": (
                "block L output; "
                "equivalent to hidden_states[L+1] "
                "except final post-norm ambiguity "
                "avoided by hooks"
            ),
        },

        "storage": {
            "dtype": "float16",
            "all_response_shape": (
                "[32, 4096]"
            ),
            "answer_shape": (
                "[32, 4096]"
            ),
            "reasoning_middle_shape": (
                "[4096]"
            ),
            "chunked_safetensors": True,
        },

        "reports": reports,

        "parts": parts,
    }

    write_json(
        output_root
        / "activation_manifest_v3.json",
        manifest,
    )

    return manifest


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/method_frozen.yaml",
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        help=(
            "Path to "
            "lu-replication/run-003"
        ),
    )

    parser.add_argument(
        "--output-root",
        required=True,
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "translated",
            "wrapper",
            "default",
        ],
        default=[
            "translated",
            "wrapper",
            "default",
        ],
    )

    parser.add_argument(
        "--token-budget",
        type=int,
        default=32768,
    )

    parser.add_argument(
        "--max-batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.token_budget <= 0:
        raise ValueError("--token-budget must be > 0")
    if args.max_batch <= 0:
        raise ValueError("--max-batch must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0 when provided")

    config_path = Path(
        args.config
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Frozen method config not found: {config_path}"
        )

    cfg = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    primary = (
        cfg["models"]["primary"]
    )

    model_id = (
        primary["model_id"]
    )

    model_revision = (
        primary.get("revision")
        or primary.get("model_revision")
        or primary.get("revision_id")
        or primary.get("model_commit")
        or primary.get("commit")
    )

    tokenizer_revision = (
        primary.get("tokenizer_revision")
        or primary.get("tokenizer_commit")
    )

    if model_revision is None:
        raise KeyError(
            "Could not resolve frozen model revision. "
            f"Available primary-model keys: {list(primary.keys())}"
        )

    if tokenizer_revision is None:
        raise KeyError(
            "Could not resolve frozen tokenizer revision. "
            f"Available primary-model keys: {list(primary.keys())}"
        )

    middle_layer = (
        cfg[
            "activation_extraction"
        ][
            "middle_layer"
        ][
            "primary_block_index"
        ]
    )

    print(
        "Loading tokenizer:",
        model_id,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_id,
            revision=(
                tokenizer_revision
            ),
        )
    )

    print(
        "Loading model:",
        model_id,
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_id,
            revision=model_revision,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        .eval()
    )

    n_layers = len(
        model.model.layers
    )

    hidden_size = (
        model.config.hidden_size
    )

    assert n_layers == (
        primary[
            "n_transformer_blocks"
        ]
    )

    assert hidden_size == (
        primary[
            "hidden_size"
        ]
    )

    print(
        f"Model loaded: "
        f"{n_layers} layers, "
        f"hidden={hidden_size}, "
        f"middle={middle_layer}"
    )

    output_root = Path(
        args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_root = Path(
        args.dataset_root
    )

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}"
        )

    reports = {}

    for mode in args.modes:

        reports[mode] = run_mode(
            mode=mode,
            dataset_root=dataset_root,
            output_root=output_root,
            tokenizer=tokenizer,
            model=model,
            middle_layer=middle_layer,
            token_budget=(
                args.token_budget
            ),
            max_batch=args.max_batch,
            chunk_size=args.chunk_size,
            limit=args.limit,
            config_path=config_path,
        )

        build_global_manifest(
            output_root=output_root,
            config_path=config_path,
            model_info={
                "model_id": model_id,
                "model_revision": (
                    model_revision
                ),
                "tokenizer_revision": (
                    tokenizer_revision
                ),
                "n_layers": n_layers,
                "hidden_size": (
                    hidden_size
                ),
                "inference_dtype": (
                    "bfloat16"
                ),
            },
            middle_layer=middle_layer,
            reports=reports,
        )

    print(
        "\nT12 activation recomputation "
        "finished."
    )

    print(
        "Manifest:",
        output_root
        / "activation_manifest_v3.json",
    )


if __name__ == "__main__":
    main()