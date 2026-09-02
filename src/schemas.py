"""Config-aware schema checks and validated JSONL I/O (T03 step 4).

Every persisted JSONL row is validated before writing and after reading,
against both field-level checks and record-level cross-field checks derived
from the active downstream frozen YAML:

- arms, default-condition indices, blocks, validity labels, segmentation
  cases, pools, block bounds, and steering condition names come from the
  downstream frozen config, never from local constants;
- ``question_id`` must belong to the declared block; role-arm
  ``prompt_index`` must equal the frozen positional assignment for that
  question, while DEFAULT rows carry ``prompt_index ==
  default_condition_index`` because the five default conditions are fully
  crossed with every block question (METHOD_FREEZE §5.4) and the rollout
  ID needs the condition to keep each cell distinct;
- stored ``messages_sha256`` / ``rendered_prompt_sha256`` must match the
  hashes recomputed from the stored content (rows labelled
  ``serialization_failure`` are quarantined and exempt, since their content
  is by definition not reconstructible);
- a judge row with ``abstained=True`` cannot carry a role score;
- a steering row's dose columns must be consistent with its condition
  (the shared zero is zero in both columns).

Rows may carry either the immutable historical V3 config SHA or the active V4
SHA. Accepting the historical SHA is a provenance-compatibility rule only: it
does not make V3 the active configuration for new decisions.
"""

import json

from . import ids
from .config import (
    block_question_ids,
    known_frozen_config_shas,
    load_frozen_config,
    prompt_index_for,
)

_CFG, _CONFIG_SHA = load_frozen_config()
_KNOWN_CONFIG_SHAS = known_frozen_config_shas()

_ROLE_ARMS = tuple(_CFG["prompt_rendering"]["conditions"])
DEFAULT_ARM = "DEFAULT"
ARMS = _ROLE_ARMS + (DEFAULT_ARM,)
_DEFAULT_INDICES = tuple(
    c["index"] for c in _CFG["prompt_rendering"]["default_conditions"]
)
_BLOCKS = block_question_ids(_CFG)
_QUESTION_TO_BLOCK_POSITION = {
    qid: (name, pos)
    for name, ids_ in _BLOCKS.items()
    for pos, qid in enumerate(ids_)
}
_VALIDITY_LABELS = tuple(_CFG["validity"]["labels"])
_SEGMENTATION_CASES = tuple(_CFG["response_segmentation"]["cases"])
_POOLS = (_CFG["activation_extraction"]["primary_pool"],
          *_CFG["activation_extraction"]["sensitivity_pools"])
_N_BLOCKS = _CFG["activation_extraction"]["hook"]["n_blocks"]
_STEERING_CONDITIONS = tuple(_CFG["steering"]["unique_conditions"])

_HEX = set("0123456789abcdef")


def _is_int_list(v):
    return isinstance(v, list) and all(isinstance(x, int) for x in v)


def _is_message_list(v):
    return (
        isinstance(v, list) and len(v) > 0
        and all(
            isinstance(m, dict)
            and set(m) == {"role", "content"}
            and m["role"] in ("system", "user", "assistant")
            and isinstance(m["content"], str)
            for m in v
        )
    )


def _is_sha256(v):
    return isinstance(v, str) and len(v) == 64 and set(v) <= _HEX


def _is_git_sha(v):
    return isinstance(v, str) and len(v) == 40 and set(v) <= _HEX


def _is_row_id(v):
    return isinstance(v, str) and len(v) == 16 and set(v) <= _HEX


_COMMON = {
    "row_id": _is_row_id,
    "rollout_id": _is_row_id,
    "model": lambda v: isinstance(v, str) and v,
    "model_revision": lambda v: isinstance(v, str) and v,
    "config_sha256": _is_sha256,
    "git_sha": _is_git_sha,
}

GENERATION_ROW = {
    **_COMMON,
    "arm": lambda v: v in ARMS,
    "channel": lambda v: v in ("user", "system"),
    "default_condition_index": lambda v: v is None or v in _DEFAULT_INDICES,
    "role_id": lambda v: isinstance(v, str),
    "block": lambda v: v in _BLOCKS,
    "question_id": lambda v: isinstance(v, int) and 0 <= v <= 239,
    "prompt_index": lambda v: isinstance(v, int) and 0 <= v < len(_DEFAULT_INDICES),
    "retry": lambda v: isinstance(v, int) and v >= 0,
    "messages": _is_message_list,
    "rendered_prompt": lambda v: isinstance(v, str) and v,
    "messages_sha256": _is_sha256,
    "rendered_prompt_sha256": _is_sha256,
    "tokenizer_revision": lambda v: isinstance(v, str) and v,
    "chat_template_sha256": _is_sha256,
    "runtime_settings": lambda v: isinstance(v, dict) and v,
    "prompt_token_ids": _is_int_list,
    "output_token_ids": _is_int_list,
    "n_prompt_tokens": lambda v: isinstance(v, int) and v >= 0,
    "n_output_tokens": lambda v: isinstance(v, int) and v >= 0,
    "response_start": lambda v: isinstance(v, int) and v >= 0,
    "response_end_exclusive": lambda v: isinstance(v, int) and v >= 0,
    "output_text": lambda v: isinstance(v, str),
    "finish_reason": lambda v: v in ("stop", "length", "error"),
    "technical_validity": lambda v: v in _VALIDITY_LABELS,
    "segmentation_case": lambda v: v is None or v in _SEGMENTATION_CASES,
}

JUDGE_ROW = {
    **_COMMON,
    "generation_row_id": _is_row_id,
    "judge_model": lambda v: isinstance(v, str) and v,
    "judge_raw_response": lambda v: v is None or isinstance(v, str),
    "role_score": lambda v: v is None or v in (0, 1, 2, 3),
    "abstained": lambda v: isinstance(v, bool),
}

ACTIVATION_ROW = {
    **_COMMON,
    "generation_row_id": _is_row_id,
    "pool": lambda v: v in _POOLS,
    "block_index": lambda v: isinstance(v, int) and 0 <= v < _N_BLOCKS,
    "vector_sha256": _is_sha256,
    "n_pooled_tokens": lambda v: isinstance(v, int) and v > 0,
}

# Causal steering rows use the (later-frozen) harmful-question set, not the
# 240 extraction questions, so they carry no extraction block/prompt coupling.
STEERING_ROW = {
    **{k: v for k, v in GENERATION_ROW.items() if k not in (
        "arm", "channel", "default_condition_index", "block",
        "question_id", "prompt_index",
    )},
    "role_id": lambda v: isinstance(v, str) and v,
    "harmful_question_id": lambda v: isinstance(v, str) and v,
    "steering_prompt_id": lambda v: isinstance(v, str) and v,
    "condition": lambda v: v in _STEERING_CONDITIONS,
    "axis_dose": lambda v: isinstance(v, (int, float)),
    "random_dose": lambda v: isinstance(v, (int, float)),
}


class SchemaError(ValueError):
    pass


def serialization_ok(row):
    """True when the stored hashes reproduce from the stored content."""
    return (
        row.get("messages_sha256") == ids.messages_sha256(row.get("messages"))
        and row.get("rendered_prompt_sha256") == ids.text_sha256(row.get("rendered_prompt", ""))
    )


def _check_record_integrity(row, problems):
    """Token-count/boundary, config-hash, and serialization checks shared by
    every row type that stores a full generation record."""
    if row["n_prompt_tokens"] != len(row["prompt_token_ids"]):
        problems.append("n_prompt_tokens does not match prompt_token_ids")
    if row["n_output_tokens"] != len(row["output_token_ids"]):
        problems.append("n_output_tokens does not match output_token_ids")
    if row["response_start"] != len(row["prompt_token_ids"]):
        problems.append("response_start must equal len(prompt_token_ids)")
    if not (row["response_start"] <= row["response_end_exclusive"]
            <= row["response_start"] + len(row["output_token_ids"])):
        problems.append("response_end_exclusive outside the stored output range")
    if row["technical_validity"] != "serialization_failure" and not serialization_ok(row):
        problems.append("stored hashes do not reproduce from stored content")


def _check_generation_cross_fields(row, problems):
    arm = row["arm"]
    if arm == DEFAULT_ARM:
        if row["default_condition_index"] is None:
            problems.append("DEFAULT rows require default_condition_index")
        if row["role_id"] != "":
            problems.append("DEFAULT rows must have empty role_id")
    else:
        if row["default_condition_index"] is not None:
            problems.append("role-arm rows must not set default_condition_index")
        if not row["role_id"]:
            problems.append("role-arm rows require a role_id")
        channel = "system" if arm == "SYSTEM_LU" else "user"
        if row["channel"] != channel:
            problems.append(f"arm {arm} requires channel {channel!r}")

    located = _QUESTION_TO_BLOCK_POSITION.get(row["question_id"])
    if located is None:
        problems.append(f"question_id {row['question_id']} not in any frozen block")
    else:
        block, position = located
        if row["block"] != block:
            problems.append(
                f"question_id {row['question_id']} belongs to {block}, row says {row['block']}"
            )
        if arm == DEFAULT_ARM:
            # The five default conditions are fully crossed with every block
            # question; prompt_index mirrors the condition index so each
            # condition-question cell mints a distinct rollout ID.
            if row["default_condition_index"] != row["prompt_index"]:
                problems.append(
                    f"DEFAULT rows require prompt_index == default_condition_index, "
                    f"got {row['prompt_index']} != {row['default_condition_index']}"
                )
        else:
            frozen_index = prompt_index_for(_CFG, _BLOCKS[block], position)
            if row["prompt_index"] != frozen_index:
                problems.append(
                    f"prompt_index {row['prompt_index']} contradicts frozen assignment "
                    f"{frozen_index} for question {row['question_id']}"
                )

    _check_record_integrity(row, problems)


def _check_judge_cross_fields(row, problems):
    if row["abstained"] and row["role_score"] is not None:
        problems.append("abstained judge row cannot carry a role score")
    if row["abstained"] and row["judge_raw_response"] is not None:
        problems.append("abstained judge row cannot carry a raw response")


def _check_steering_cross_fields(row, problems):
    _check_record_integrity(row, problems)
    condition, axis, random = row["condition"], row["axis_dose"], row["random_dose"]
    expectations = {
        "shared_zero": axis == 0 and random == 0,
        "assistant_axis_toward": axis > 0 and random == 0,
        "assistant_axis_away": axis < 0 and random == 0,
        "random_positive": random > 0 and axis == 0,
        "random_negative": random < 0 and axis == 0,
    }
    if not expectations[condition]:
        problems.append(
            f"dose columns (axis={axis}, random={random}) inconsistent with {condition}"
        )


ROW_TYPES = {
    "generation": (GENERATION_ROW, _check_generation_cross_fields),
    "judge": (JUDGE_ROW, _check_judge_cross_fields),
    "activation": (ACTIVATION_ROW, None),
    "steering": (STEERING_ROW, _check_steering_cross_fields),
}


def validate_row(row, row_type):
    """Raise SchemaError listing every missing/invalid/inconsistent field."""
    if row_type not in ROW_TYPES:
        raise SchemaError(f"unknown row type {row_type!r}")
    if not isinstance(row, dict):
        raise SchemaError("row is not an object")
    schema, cross_check = ROW_TYPES[row_type]
    problems = []
    for field, check in schema.items():
        if field not in row:
            problems.append(f"missing field: {field}")
        elif not check(row[field]):
            problems.append(f"invalid value for {field}: {row[field]!r}")
    if not problems:
        if row["config_sha256"] not in _KNOWN_CONFIG_SHAS:
            problems.append("config_sha256 is not a recognized frozen configuration")
        if cross_check is not None:
            cross_check(row, problems)
    if problems:
        raise SchemaError("; ".join(problems))
    return True


def validate_jsonl_rows(rows, row_type):
    """Validate an iterable of parsed rows; raises on the first bad row
    with its zero-based index."""
    for i, row in enumerate(rows):
        try:
            validate_row(row, row_type)
        except SchemaError as e:
            raise SchemaError(f"row {i}: {e}") from None
    return True


def write_jsonl(path, rows, row_type):
    """Canonical validated JSONL writer: every row is validated before any
    byte is written, so a bad row never truncates or corrupts a file."""
    rows = list(rows)
    validate_jsonl_rows(rows, row_type)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return len(rows)


def read_jsonl(path, row_type):
    """Canonical validated JSONL reader: rows are validated after parsing."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SchemaError(f"row {i}: not valid JSON ({e})") from None
    validate_jsonl_rows(rows, row_type)
    return rows
