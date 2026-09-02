"""Deterministic ID generation.

IDs are SHA-256 digests of a canonical JSON encoding of the identifying
fields, so the same semantic unit always receives the same ID across
machines, runs, and reorderings. Technical retries reuse the rollout ID and
increment ``retry`` (frozen: retries do not mint new semantic units).
"""

import hashlib
import json

ID_HEX_LEN = 16


def _digest(kind, fields):
    payload = json.dumps(
        {"kind": kind, **fields}, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:ID_HEX_LEN]


def rollout_id(model, arm, role_id, question_id, prompt_index):
    """Semantic rollout ID: one per role/question/arm cell. Retry-stable."""
    return _digest("rollout", {
        "model": model,
        "arm": arm,
        "role_id": role_id,
        "question_id": int(question_id),
        "prompt_index": int(prompt_index),
    })


def generation_row_id(rollout, retry=0):
    """Row ID for one concrete generation attempt of a rollout."""
    return _digest("generation", {"rollout_id": rollout, "retry": int(retry)})


def steering_rollout_id(model, role_id, question_id, prompt_index, condition):
    """Semantic rollout ID for a causal steering cell.

    ``condition`` must be one of the five unique frozen conditions — the
    shared zero is a single condition, never one zero per direction family
    (see steering.build_causal_conditions).
    """
    return _digest("steering_rollout", {
        "model": model,
        "role_id": role_id,
        "question_id": int(question_id),
        "prompt_index": int(prompt_index),
        "condition": condition,
    })


def text_sha256(text):
    """Full SHA-256 of a text field (rendered prompts, message lists)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def messages_sha256(messages):
    """Canonical full SHA-256 of a chat message list.

    Hashes the canonical JSON encoding (sorted keys, no whitespace) so the
    stored hash is reproducible from the stored messages exactly.
    """
    payload = json.dumps(
        messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return text_sha256(payload)
