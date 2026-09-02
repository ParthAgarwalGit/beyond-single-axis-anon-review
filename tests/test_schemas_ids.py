"""T03 steps 4–5: config-aware JSONL schema checks and deterministic IDs."""

import pytest

from src import ids
from src.config import block_question_ids, load_frozen_config, prompt_index_for
from src.schemas import (
    SchemaError, read_jsonl, serialization_ok, validate_jsonl_rows,
    validate_row, write_jsonl,
)

_, CONFIG_SHA = load_frozen_config()


def frozen_position(cfg, question_id):
    for name, qids in block_question_ids(cfg).items():
        if question_id in qids:
            position = qids.index(question_id)
            return name, prompt_index_for(cfg, qids, position)
    raise AssertionError(f"{question_id} not in any block")


def make_generation_row(cfg, question_id=7):
    block, prompt_index = frozen_position(cfg, question_id)
    messages = [{"role": "user", "content": "hello"}]
    rendered = "rendered"
    rollout = ids.rollout_id("deepseek-r1-distill-llama-8b", "USER_TRANSLATED_LU",
                             "pirate", question_id, prompt_index)
    return {
        "row_id": ids.generation_row_id(rollout),
        "rollout_id": rollout,
        "model": "deepseek-r1-distill-llama-8b",
        "model_revision": "rev-model-1",
        "config_sha256": CONFIG_SHA,
        "git_sha": "a" * 40,
        "arm": "USER_TRANSLATED_LU",
        "channel": "user",
        "default_condition_index": None,
        "role_id": "pirate",
        "block": block,
        "question_id": question_id,
        "prompt_index": prompt_index,
        "retry": 0,
        "messages": messages,
        "rendered_prompt": rendered,
        "messages_sha256": ids.messages_sha256(messages),
        "rendered_prompt_sha256": ids.text_sha256(rendered),
        "tokenizer_revision": "rev-tok-1",
        "chat_template_sha256": "c" * 64,
        "runtime_settings": {"temperature": 0.0, "max_new_tokens": 1500},
        "prompt_token_ids": [1, 2, 3],
        "output_token_ids": [4, 5],
        "n_prompt_tokens": 3,
        "n_output_tokens": 2,
        "response_start": 3,
        "response_end_exclusive": 5,
        "output_text": "hi",
        "finish_reason": "stop",
        "technical_validity": "valid",
        "segmentation_case": "direct_answer",
    }


def make_judge_row():
    return {
        "row_id": "0" * 16, "rollout_id": "1" * 16, "model": "judge",
        "model_revision": "rev-judge-1", "config_sha256": CONFIG_SHA,
        "git_sha": "a" * 40, "generation_row_id": "2" * 16,
        "judge_model": "gpt-x", "judge_raw_response": None,
        "role_score": None, "abstained": True,
    }


def test_valid_generation_row_passes(cfg):
    assert validate_row(make_generation_row(cfg), "generation")


@pytest.mark.parametrize("field,value", [
    ("arm", "HAND_ROLLED_ARM"),                  # unknown arm
    ("git_sha", "g" * 40),                       # nonhex git SHA
    ("git_sha", "abc123"),
    ("question_id", 240),
    ("technical_validity", "role_failure"),
    ("finish_reason", "unknown"),
    ("messages", [{"role": "narrator", "content": "x"}]),
    ("config_sha256", "0" * 64),                 # stale/foreign config hash
    ("runtime_settings", {}),
])
def test_invalid_values_rejected(cfg, field, value):
    row = dict(make_generation_row(cfg), **{field: value})
    with pytest.raises(SchemaError):
        validate_row(row, "generation")


def test_inconsistent_prompt_assignment_rejected(cfg):
    row = make_generation_row(cfg)
    row["prompt_index"] = (row["prompt_index"] + 1) % 5
    with pytest.raises(SchemaError, match="contradicts frozen assignment"):
        validate_row(row, "generation")


def test_wrong_block_rejected(cfg):
    row = make_generation_row(cfg, question_id=7)  # frozen into C80-A
    row["block"] = "E80"
    with pytest.raises(SchemaError, match="belongs to"):
        validate_row(row, "generation")


def test_mismatched_hashes_rejected(cfg):
    row = make_generation_row(cfg)
    row["output_text"] = "hi"
    row["rendered_prompt"] = "tampered"
    assert not serialization_ok(row)
    with pytest.raises(SchemaError, match="hashes do not reproduce"):
        validate_row(row, "generation")


def test_token_count_metadata_must_match_ids(cfg):
    row = dict(make_generation_row(cfg), n_output_tokens=99)
    with pytest.raises(SchemaError, match="n_output_tokens"):
        validate_row(row, "generation")
    row = dict(make_generation_row(cfg), response_start=0)
    with pytest.raises(SchemaError, match="response_start"):
        validate_row(row, "generation")


def test_channel_must_match_arm(cfg):
    row = dict(make_generation_row(cfg), channel="system")
    with pytest.raises(SchemaError, match="requires channel"):
        validate_row(row, "generation")


def test_default_rows_require_condition_and_no_role(cfg):
    row = make_generation_row(cfg)
    row.update(arm="DEFAULT", default_condition_index=row["prompt_index"], role_id="")
    assert validate_row(row, "generation")
    with pytest.raises(SchemaError, match="default_condition_index"):
        validate_row(dict(row, default_condition_index=None), "generation")
    with pytest.raises(SchemaError, match="empty role_id"):
        validate_row(dict(row, role_id="pirate"), "generation")


def test_default_conditions_fully_crossed(cfg):
    # The five default conditions are crossed with every block question:
    # any condition index validates as long as prompt_index mirrors it.
    row = make_generation_row(cfg)
    other = (row["prompt_index"] + 1) % 5
    row.update(
        arm="DEFAULT", role_id="",
        default_condition_index=other, prompt_index=other,
    )
    assert validate_row(row, "generation")
    with pytest.raises(SchemaError, match="prompt_index == default_condition_index"):
        validate_row(dict(row, prompt_index=(other + 1) % 5), "generation")


def test_missing_field_rejected_with_row_index(cfg):
    good = make_generation_row(cfg)
    bad = {k: v for k, v in good.items() if k != "rollout_id"}
    with pytest.raises(SchemaError, match="row 1.*rollout_id"):
        validate_jsonl_rows([good, bad], "generation")


def test_judge_abstention_coupling():
    row = make_judge_row()
    assert validate_row(row, "judge")
    with pytest.raises(SchemaError, match="abstained"):
        validate_row(dict(row, role_score=2), "judge")
    scored = dict(row, abstained=False, judge_raw_response="3", role_score=3)
    assert validate_row(scored, "judge")
    with pytest.raises(SchemaError):
        validate_row(dict(scored, role_score=4), "judge")


def test_steering_row_dose_consistency(cfg):
    base = make_generation_row(cfg)
    for key in ("arm", "channel", "default_condition_index", "block",
                "question_id", "prompt_index"):
        del base[key]
    base.update(
        role_id="pirate", harmful_question_id="hq-01", steering_prompt_id="sp-01",
        condition="shared_zero", axis_dose=0.0, random_dose=0.0,
    )
    assert validate_row(base, "steering")
    # Nonzero dose on the shared zero must be rejected.
    with pytest.raises(SchemaError, match="inconsistent with shared_zero"):
        validate_row(dict(base, axis_dose=8.0), "steering")
    toward = dict(base, condition="assistant_axis_toward", axis_dose=8.0)
    assert validate_row(toward, "steering")
    with pytest.raises(SchemaError, match="inconsistent"):
        validate_row(dict(toward, random_dose=8.0), "steering")


def test_jsonl_writer_reader_roundtrip(cfg, tmp_path):
    rows = [make_generation_row(cfg), make_generation_row(cfg, question_id=1)]
    path = tmp_path / "generation.jsonl"
    assert write_jsonl(path, rows, "generation") == 2
    assert read_jsonl(path, "generation") == rows
    # The writer refuses to write any bytes when one row is invalid.
    bad = dict(rows[0], git_sha="nothex")
    with pytest.raises(SchemaError):
        write_jsonl(tmp_path / "bad.jsonl", [bad], "generation")
    assert not (tmp_path / "bad.jsonl").exists()


def test_ids_are_deterministic_and_distinct():
    a1 = ids.rollout_id("m", "USER_TRANSLATED_LU", "pirate", 7, 2)
    a2 = ids.rollout_id("m", "USER_TRANSLATED_LU", "pirate", 7, 2)
    b = ids.rollout_id("m", "USER_TRANSLATED_LU", "pirate", 8, 2)
    assert a1 == a2
    assert a1 != b
    assert len(a1) == 16


def test_retry_keeps_rollout_id_but_changes_row_id():
    rollout = ids.rollout_id("m", "USER_TRANSLATED_LU", "pirate", 7, 2)
    assert ids.generation_row_id(rollout, retry=0) != ids.generation_row_id(rollout, retry=1)
