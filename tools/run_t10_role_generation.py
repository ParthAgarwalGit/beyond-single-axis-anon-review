#!/usr/bin/env python3
"""T10 C80 role-generation worker (prepared; NOT launched here).

Consumes one frozen T10 C80 block manifest and produces one validated
generation row per translated-arm role rollout. Both C80 halves run this same
worker (differing only in the manifest's ``--block``), so their 44,000 IDs
reconcile under identical code and configuration.

Production safeguards (checked before any generation):
  - the actual engine's model revision, tokenizer revision, and chat-template
    SHA are verified against the frozen manifest (``preflight``);
  - decoding settings are loaded from and checked against a frozen T09 runtime
    artifact, never an arbitrary dict (``load_frozen_decoding``);
  - each rollout's rendered-prompt hash is verified against the frozen
    expected hash BEFORE ``engine.generate`` is called;
  - resume validates existing rows against the current manifest/config and
    rejects foreign, duplicate, malformed, or engine-mismatched rows;
  - retryable technical failures are retried under the same rollout ID with a
    retry counter; semantic degeneration is never retried.

Every row is validated against the canonical generation schema before it is
written. The model engine is injected; ``--dry-run`` runs the whole path on CPU
with a deterministic fake engine. Production generation is not auto-wired here:
launch waits on T04, strict-final T08, and the T09 runtime freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ARM = "USER_TRANSLATED_LU"
BLOCK_FILES = {
    "C80-A": "questions_C80_A.json",
    "C80-B": "questions_C80_B.json",
}

# Technical failures worth retrying under the same rollout ID (method
# ``validity.technical_retries``). Semantic outcomes below are terminal and are
# NEVER retried: ``valid``, ``empty_response``, ``degenerate_repetition``.
RETRYABLE_TECHNICAL = frozenset({
    "generation_error", "truncated", "serialization_failure",
    "token_alignment_failure",
})
SEMANTIC_TERMINAL = frozenset({"valid", "empty_response", "degenerate_repetition"})
# Degeneration is called out explicitly so a future edit cannot accidentally
# make it retryable.
NEVER_RETRY = frozenset({"degenerate_repetition"})
DEFAULT_MAX_RETRIES = 3

REQUIRED_DECODING_KEYS = ("temperature", "max_new_tokens", "do_sample", "seed")


def _add_root(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PreflightError(RuntimeError):
    """A production precondition (engine/config/decoding/manifest) is not met."""


class ResumeIntegrityError(RuntimeError):
    """An existing output file is contaminated, foreign, or inconsistent."""


class RenderHashMismatch(RuntimeError):
    """A freshly rendered prompt does not match its frozen expected hash."""


@dataclass
class RoleEngine:
    tokenizer: Any
    generate: Callable[[list[int]], tuple[list[int], str, str]]
    model_revision: str
    tokenizer_revision: str
    chat_template_sha256: str
    terminal_ids: tuple[int, ...] = ()


# --------------------------------------------------------------------------- #
# Production preconditions.
# --------------------------------------------------------------------------- #
def load_frozen_decoding(root: Path, artifact_path: str) -> tuple[dict[str, Any], str]:
    """Load decoding settings from the frozen T09 runtime artifact.

    Production decoding must come from a frozen artifact (the E80 acceptance
    report / T09 runtime freeze), never an arbitrary dict. Returns the decoding
    block and the artifact's SHA-256, which is stamped into every row.
    """
    p = root / artifact_path
    if not p.exists():
        raise PreflightError(f"frozen T09 runtime artifact missing: {artifact_path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    decoding = data.get("frozen_decoding")
    if not isinstance(decoding, dict):
        raise PreflightError(
            f"{artifact_path} has no 'frozen_decoding' object; T09 must freeze it"
        )
    missing = [k for k in REQUIRED_DECODING_KEYS if k not in decoding]
    if missing:
        raise PreflightError(f"frozen decoding missing keys: {missing}")
    return decoding, _sha256_bytes(p.read_bytes())


def preflight(manifest: dict[str, Any], engine: RoleEngine, config_sha: str,
              require_frozen_render_hashes: bool) -> None:
    """Verify engine/config/manifest agree before any generation runs."""
    if manifest.get("arm") != ARM:
        raise PreflightError(f"manifest arm {manifest.get('arm')!r} is not {ARM}")
    if manifest["method_config"]["sha256"] != config_sha:
        raise PreflightError("manifest was built from a different method config")
    model = manifest["model"]
    checks = {
        "model_revision": (engine.model_revision, model.get("model_revision")),
        "tokenizer_revision": (engine.tokenizer_revision, model.get("tokenizer_revision")),
        "chat_template_sha256": (engine.chat_template_sha256, model.get("chat_template_sha256")),
    }
    for field, (actual, expected) in checks.items():
        if not expected:
            raise PreflightError(f"manifest model.{field} is not pinned")
        if actual != expected:
            raise PreflightError(
                f"engine {field} {actual!r} != frozen manifest {expected!r}"
            )
    if require_frozen_render_hashes:
        if manifest.get("status") != "FROZEN":
            raise PreflightError("production run requires a FROZEN manifest")
        freeze = manifest.get("rendered_prompt_hash_freeze", {})
        if freeze.get("status") != "FROZEN" or not freeze.get(
            "every_rollout_carries_expected_hash"
        ):
            raise PreflightError(
                "production run requires frozen rendered-prompt hashes on every rollout"
            )


# --------------------------------------------------------------------------- #
# Cached frozen-text lookups (hash-verified once per key).
# --------------------------------------------------------------------------- #
def _load_role_prompt_cache(root: Path) -> dict[tuple[str, int], str]:
    payload = json.loads(
        (root / "data" / "lu_et_al" / "role_prompts.json").read_text(encoding="utf-8")
    )
    return {(r["role_id"], r["prompt_index"]): r["text"] for r in payload["prompts"]}


def _load_question_cache(root: Path, block: str) -> dict[int, str]:
    payload = json.loads(
        (root / "design" / BLOCK_FILES[block]).read_text(encoding="utf-8")
    )
    return {q["question_id"]: q["question_text"] for q in payload["questions"]}


def build_generation_row(
    root: Path,
    rollout: dict[str, Any],
    engine: RoleEngine,
    cfg: dict[str, Any],
    config_sha: str,
    model_id: str,
    git_sha: str,
    runtime_settings: dict[str, Any],
    prompt_cache: dict[tuple[str, int], str],
    question_cache: dict[str, dict[int, str]],
    retry: int = 0,
    enforce_render_hash: bool = True,
) -> dict[str, Any]:
    _add_root(root)
    from src import ids
    from src.generation import (
        GenerationRecord, classify_validity, content_token_ids, response_start,
        segment_response,
    )
    from src.prompt_rendering import (
        render_prompt, role_arm_messages, tokenize_rendered_prompt,
    )
    from src.schemas import serialization_ok

    role = rollout["role_id"]
    prompt_key = (role, rollout["prompt_index"])
    role_instruction = prompt_cache[prompt_key]
    if hashlib.sha256(role_instruction.encode("utf-8")).hexdigest()[:16] != rollout["role_prompt_content_sha256"]:
        raise ValueError(f"role-prompt hash drift for {prompt_key}")

    block = rollout["block"]
    if block not in question_cache:
        question_cache[block] = _load_question_cache(root, block)
    question_text = question_cache[block][rollout["question_id"]]
    if hashlib.sha256(question_text.encode("utf-8")).hexdigest()[:16] != rollout["question_content_sha256"]:
        raise ValueError(f"question {rollout['question_id']} text hash drift")

    messages = role_arm_messages(cfg, ARM, role_instruction, question_text)
    rendered = render_prompt(engine.tokenizer, messages)
    rendered_sha = ids.text_sha256(rendered)

    # Freeze check BEFORE generation: never spend a forward pass on a prompt
    # that does not match the frozen expected hash.
    expected = rollout.get("expected_rendered_prompt_sha256")
    if enforce_render_hash and expected is not None and rendered_sha != expected:
        raise RenderHashMismatch(
            f"rollout {rollout['rollout_id']}: rendered prompt hash {rendered_sha} "
            f"!= frozen {expected}"
        )

    prompt_ids = tokenize_rendered_prompt(engine.tokenizer, rendered)
    output_ids, output_text, finish_reason = engine.generate(prompt_ids)
    record = GenerationRecord(
        prompt_token_ids=list(prompt_ids),
        output_token_ids=list(output_ids),
        output_text=output_text,
        finish_reason=finish_reason,
    )

    def tokenize(text: str) -> list[int]:
        return tokenize_rendered_prompt(engine.tokenizer, text)

    # Reasoning models (e.g. DeepSeek-R1-Distill) prefill the opening <think>
    # into the prompt, so the output holds only the closing </think>. Detect
    # that from the rendered prompt and locate the boundary by the single
    # </think> token id (authoritative, robust to detok/retok round-trips).
    from src.generation import OPEN_MARKER, CLOSE_MARKER
    prefilled_open = rendered.rstrip().endswith(OPEN_MARKER)
    _close_ids = tokenize(CLOSE_MARKER)
    close_token_id = _close_ids[0] if len(_close_ids) == 1 else None

    rollout_id = rollout["rollout_id"]
    row = {
        "row_id": ids.generation_row_id(rollout_id, retry),
        "rollout_id": rollout_id,
        "model": model_id,
        "model_revision": engine.model_revision,
        "config_sha256": config_sha,
        "git_sha": git_sha,
        "arm": ARM,
        "channel": "user",
        "default_condition_index": None,
        "role_id": role,
        "block": block,
        "question_id": rollout["question_id"],
        "prompt_index": rollout["prompt_index"],
        "retry": retry,
        "messages": messages,
        "rendered_prompt": rendered,
        "messages_sha256": ids.messages_sha256(messages),
        "rendered_prompt_sha256": rendered_sha,
        "tokenizer_revision": engine.tokenizer_revision,
        "chat_template_sha256": engine.chat_template_sha256,
        "runtime_settings": runtime_settings,
        "prompt_token_ids": list(prompt_ids),
        "output_token_ids": list(output_ids),
        "n_prompt_tokens": len(prompt_ids),
        "n_output_tokens": len(output_ids),
        "response_start": response_start(record),
        "output_text": output_text,
        "finish_reason": finish_reason,
    }

    ser_ok = serialization_ok(row)
    validity = classify_validity(
        record, tokenize, engine.terminal_ids, serialization_ok=ser_ok,
        prefilled_open=prefilled_open, close_token_id=close_token_id,
    )
    row["technical_validity"] = validity
    row["technical_failure_code"] = None if validity == "valid" else validity

    content = content_token_ids(record, engine.terminal_ids)
    start = response_start(record)
    row["response_end_exclusive"] = start + len(content)

    all_response = reasoning = final_answer = 0
    if validity == "valid":
        seg = segment_response(
            record, tokenize, engine.terminal_ids,
            prefilled_open=prefilled_open, close_token_id=close_token_id,
        )
        row["segmentation_case"] = seg.case
        all_response = seg.all_response_span[1] - seg.all_response_span[0]
        if seg.reasoning_span:
            reasoning = seg.reasoning_span[1] - seg.reasoning_span[0]
        if seg.final_answer_span:
            final_answer = seg.final_answer_span[1] - seg.final_answer_span[0]
    else:
        row["segmentation_case"] = None
        all_response = len(content)
    row["token_region_counts"] = {
        "all_response": all_response,
        "reasoning": reasoning,
        "final_answer": final_answer,
    }
    return row


# --------------------------------------------------------------------------- #
# Resume: validate existing rows and plan retries.
# --------------------------------------------------------------------------- #
def _is_retryable(validity: str) -> bool:
    return validity in RETRYABLE_TECHNICAL and validity not in NEVER_RETRY


def load_existing_attempts(
    out_path: Path,
    manifest_ids: set[str],
    engine: RoleEngine,
    config_sha: str,
    root: Path | None = None,
) -> dict[str, list[tuple[int, str]]]:
    """Validate an existing output file and return, per rollout, its attempts
    as (retry, technical_validity). Raises on any contamination."""
    if root is not None:
        _add_root(root)
    from src.schemas import SchemaError, validate_row

    attempts: dict[str, list[tuple[int, str]]] = defaultdict(list)
    if not out_path.exists():
        return attempts
    seen_keys: set[tuple[str, int]] = set()
    with out_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ResumeIntegrityError(
                    f"corrupted JSON at line {line_no}: {e}"
                ) from None
            try:
                validate_row(row, "generation")
            except SchemaError as e:
                raise ResumeIntegrityError(f"line {line_no}: malformed row: {e}") from None
            if row["config_sha256"] != config_sha:
                raise ResumeIntegrityError(
                    f"line {line_no}: foreign config_sha256 {row['config_sha256']}"
                )
            if row["rollout_id"] not in manifest_ids:
                raise ResumeIntegrityError(
                    f"line {line_no}: rollout_id {row['rollout_id']} not in this manifest"
                )
            if row["arm"] != ARM:
                raise ResumeIntegrityError(f"line {line_no}: foreign arm {row['arm']}")
            if (row["model_revision"] != engine.model_revision
                    or row["tokenizer_revision"] != engine.tokenizer_revision
                    or row["chat_template_sha256"] != engine.chat_template_sha256):
                raise ResumeIntegrityError(
                    f"line {line_no}: row engine revisions do not match this engine"
                )
            key = (row["rollout_id"], row["retry"])
            if key in seen_keys:
                raise ResumeIntegrityError(f"duplicate (rollout_id, retry) {key}")
            seen_keys.add(key)
            attempts[row["rollout_id"]].append((row["retry"], row["technical_validity"]))
    return attempts


def resume_plan(attempts: dict[str, list[tuple[int, str]]], max_retries: int
                ) -> tuple[set[str], dict[str, int], set[str]]:
    """Return (done, next_retry, exhausted).

    A rollout is done if any attempt reached a terminal outcome, or if all its
    attempts are retryable but the retry budget is spent (exhausted). Otherwise
    it is pending with the next retry index.
    """
    done: set[str] = set()
    next_retry: dict[str, int] = {}
    exhausted: set[str] = set()
    for rid, atts in attempts.items():
        if any(not _is_retryable(v) for _, v in atts):
            done.add(rid)
            continue
        max_r = max(r for r, _ in atts)
        if max_r + 1 > max_retries:
            done.add(rid)
            exhausted.add(rid)
        else:
            next_retry[rid] = max_r + 1
    return done, next_retry, exhausted


def run(
    root: Path,
    manifest: dict[str, Any],
    engine: RoleEngine,
    cfg: dict[str, Any],
    config_sha: str,
    git_sha: str,
    runtime_settings: dict[str, Any],
    out_path: Path,
    limit: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    enforce_render_hash: bool = True,
) -> dict[str, Any]:
    _add_root(root)
    from src.schemas import validate_row

    model_id = cfg["models"]["primary"]["model_id"]
    if manifest["method_config"]["sha256"] != config_sha:
        raise ResumeIntegrityError("manifest was built from a different method config")
    if manifest["arm"] != ARM:
        raise ValueError(f"manifest arm {manifest['arm']} is not {ARM}")

    manifest_ids = {r["rollout_id"] for r in manifest["rollouts"]}
    attempts = load_existing_attempts(
        out_path, manifest_ids, engine, config_sha, root=root
    )
    done, next_retry, exhausted = resume_plan(attempts, max_retries)

    prompt_cache = _load_role_prompt_cache(root)
    question_cache: dict[str, dict[int, str]] = {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    todo = [r for r in manifest["rollouts"] if r["rollout_id"] not in done]
    if limit is not None:
        todo = todo[:limit]

    written = 0
    rollouts_processed = 0
    newly_exhausted = 0
    with out_path.open("a", encoding="utf-8", newline="\n") as fh:
        for rollout in todo:
            rid = rollout["rollout_id"]
            attempt = next_retry.get(rid, 0)
            while attempt <= max_retries:
                row = build_generation_row(
                    root, rollout, engine, cfg, config_sha, model_id,
                    git_sha, runtime_settings, prompt_cache, question_cache,
                    retry=attempt, enforce_render_hash=enforce_render_hash,
                )
                validate_row(row, "generation")
                fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                fh.flush()
                written += 1
                validity = row["technical_validity"]
                if not _is_retryable(validity):
                    break  # terminal: valid or semantic (degeneration never retried)
                attempt += 1
            else:
                newly_exhausted += 1
            rollouts_processed += 1

    return {
        "block": manifest["block"],
        "rollouts_processed": rollouts_processed,
        "rows_written": written,
        "skipped_already_done": len(done),
        "exhausted_before_this_run": len(exhausted),
        "exhausted_this_run": newly_exhausted,
        "remaining": len(manifest_ids) - len(done) - rollouts_processed,
        "output": str(out_path),
    }


# --------------------------------------------------------------------------- #
# Deterministic CPU dry-run engine.
# --------------------------------------------------------------------------- #
class _CharTokenizer:
    chat_template = "char-tokenizer-dryrun-template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [f"<{m['role']}>{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def _dry_run_engine() -> RoleEngine:
    tok = _CharTokenizer()
    answer = "<think>Considering the character.</think>Aye, here is my reply in role."

    def generate(prompt_ids):
        return [ord(c) for c in answer], answer, "stop"

    return RoleEngine(
        tokenizer=tok,
        generate=generate,
        model_revision="dryrun-model-rev",
        tokenizer_revision="dryrun-tok-rev",
        chat_template_sha256=hashlib.sha256(
            tok.chat_template.encode("utf-8")
        ).hexdigest(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/method_frozen.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--decoding-artifact",
        default="results/audit/E80_acceptance_report.json",
        help="Frozen T09 runtime artifact carrying frozen_decoding.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    _add_root(root)
    import yaml

    config_bytes = (root / args.config).read_bytes()
    cfg = yaml.safe_load(config_bytes)
    config_sha = _sha256_bytes(config_bytes)
    manifest = json.loads((root / args.manifest).read_text(encoding="utf-8"))

    if args.dry_run:
        # Dry run: deterministic fake engine, placeholder decoding, no frozen
        # render-hash / T09 requirement (this is a wiring test, not production).
        engine = _dry_run_engine()
        runtime_settings = {
            "temperature": 0.0, "max_new_tokens": 2048, "do_sample": False,
            "seed": 0,
            "note": "PLACEHOLDER dry-run decoding; production loads frozen T09 settings.",
        }
        block = manifest["block"].replace("-", "_")
        out_path = root / "results" / "generation" / f"_dryrun_c80_{block}.jsonl"
        if out_path.exists():
            out_path.unlink()
        summary = run(
            root, manifest, engine, cfg, config_sha, "0" * 40, runtime_settings,
            out_path, limit=args.limit or 10, max_retries=args.max_retries,
            enforce_render_hash=True,
        )
        summary["mode"] = "dry-run"
        print(json.dumps(summary, indent=2))
        return 0

    # Production path: load frozen decoding and run preflight, then refuse to
    # auto-launch. Launch waits on T04, strict-final T08, and the T09 freeze.
    decoding, decoding_sha = load_frozen_decoding(root, args.decoding_artifact)
    raise SystemExit(
        "Production generation is intentionally not wired to auto-launch. "
        "T10 is BLOCKED ON T02-T04-T08 and the T09 runtime freeze. Construct a "
        "real RoleEngine, call preflight(manifest, engine, config_sha, "
        "require_frozen_render_hashes=True), then run() with runtime_settings "
        f"derived from the frozen decoding (artifact sha {decoding_sha})."
    )


if __name__ == "__main__":
    raise SystemExit(main())
