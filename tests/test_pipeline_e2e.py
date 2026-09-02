"""End-to-end canonical pipeline smoke test (local CPU, fake model).

Exercises every canonical module in sequence on synthetic data: frozen
config -> prompt rendering -> generation record -> validity/segmentation ->
validated JSONL roundtrip -> activation pooling -> judging -> geometry ->
steering conditions -> outcome rates -> stamped report.
"""

import json

import numpy as np

from conftest import char_tokenize
from src import ids
from src.activation_extraction import pool_response_vector, primary_block
from src.config import block_question_ids, load_frozen_config, prompt_index_for
from src.generation import GenerationRecord, classify_validity, segment_response
from src.geometry import assistant_axis, cross_projections, default_mean, pearson_r, role_mean
from src.judging import build_judge_input, parse_role_score
from src.prompt_rendering import default_messages, role_arm_messages
from src.provenance import stamp_report
from src.schemas import read_jsonl, write_jsonl
from src.statistics import outcome_rates
from src.steering import build_causal_conditions, dose_columns, random_control_direction

EOS = 0
HIDDEN = 4
ROLES = {"pirate": [4.0, 0.0, 1.0, 0.0], "alien": [0.0, 1.0, 0.0, 3.0]}


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(f"[{m['role']}]{m['content']}" for m in messages) + "[assistant]"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return char_tokenize(text)


def fake_generate(tokenizer, messages, role):
    """Deterministic fake engine: renders, 'generates', returns the record."""
    rendered = tokenizer.apply_chat_template(messages)
    prompt_ids = tokenizer.encode(rendered)
    reply = f"<think>plotting as {role}</think>I answer fully in character as {role}."
    return rendered, GenerationRecord(
        prompt_token_ids=prompt_ids,
        output_token_ids=char_tokenize(reply) + [EOS],
        output_text=reply,
        finish_reason="stop",
    )


def test_full_canonical_pipeline(cfg, tmp_path):
    _, config_sha = load_frozen_config()
    tokenizer = FakeTokenizer()
    blocks = block_question_ids(cfg)

    # --- render + generate + validate + persist one row per role ---
    question_id = blocks["C80-A"][0]
    prompt_index = prompt_index_for(cfg, blocks["C80-A"], 0)
    rows, segmentations = [], {}
    for role, _vec in ROLES.items():
        messages = role_arm_messages(cfg, "USER_TRANSLATED_LU", f"You are a {role}.", "Why is the sky blue?")
        rendered, record = fake_generate(tokenizer, messages, role)
        validity = classify_validity(record, tokenizer.encode, terminal_ids=(EOS,))
        assert validity == "valid"
        seg = segment_response(record, tokenizer.encode, terminal_ids=(EOS,))
        segmentations[role] = (record, seg)
        rollout = ids.rollout_id("fake-model", "USER_TRANSLATED_LU", role, question_id, prompt_index)
        rows.append({
            "row_id": ids.generation_row_id(rollout), "rollout_id": rollout,
            "model": "fake-model", "model_revision": "rev-1",
            "config_sha256": config_sha, "git_sha": "a" * 40,
            "arm": "USER_TRANSLATED_LU", "channel": "user",
            "default_condition_index": None, "role_id": role,
            "block": "C80-A", "question_id": question_id, "prompt_index": prompt_index,
            "retry": 0, "messages": messages, "rendered_prompt": rendered,
            "messages_sha256": ids.messages_sha256(messages),
            "rendered_prompt_sha256": ids.text_sha256(rendered),
            "tokenizer_revision": "rev-tok", "chat_template_sha256": "c" * 64,
            "runtime_settings": {"temperature": 0.0},
            "prompt_token_ids": record.prompt_token_ids,
            "output_token_ids": record.output_token_ids,
            "n_prompt_tokens": len(record.prompt_token_ids),
            "n_output_tokens": len(record.output_token_ids),
            "response_start": len(record.prompt_token_ids),
            "response_end_exclusive": len(record.prompt_token_ids) + len(record.output_token_ids) - 1,
            "output_text": record.output_text, "finish_reason": "stop",
            "technical_validity": validity, "segmentation_case": seg.case,
        })
    path = tmp_path / "generation.jsonl"
    write_jsonl(path, rows, "generation")
    assert read_jsonl(path, "generation") == rows

    # --- default rendering never wraps, activation pooling, judging ---
    assert default_messages(cfg, 0, "Why is the sky blue?", "user") == \
        [{"role": "user", "content": "Why is the sky blue?"}]
    assert primary_block(cfg) == 16

    scores = {}
    for role, (record, seg) in segmentations.items():
        n_total = len(record.prompt_token_ids) + len(record.output_token_ids) - 1
        residuals = np.tile(ROLES[role], (n_total, 1)).astype(float)
        pooled = pool_response_vector(residuals, seg.all_response_span,
                                      len(record.prompt_token_ids))
        assert np.allclose(pooled, ROLES[role])
        payload = build_judge_input(role, "rubric", "Why is the sky blue?", seg)
        assert payload is not None and role in payload["final_answer_text"]
        scores[role] = parse_role_score("3")
    assert all(s == 3 for s in scores.values())

    # --- geometry: block-specific means, axes, cross projections ---
    mu_default = default_mean([[np.array([1.0, 1.0, 0.0, 0.0])]] * 5)
    means_a = {r: role_mean([v, v]) for r, v in ROLES.items()}
    means_b = {r: role_mean([np.asarray(v) * 1.1]) for r, v in ROLES.items()}
    axis_a = assistant_axis(mu_default, means_a)
    axis_b = assistant_axis(mu_default, means_b)
    roles, s_a, s_b = cross_projections(means_a, means_b, axis_a, axis_b)
    reliability = pearson_r(s_a, s_b)
    assert -1.0 <= reliability <= 1.0

    # --- steering design + causal outcomes ---
    conditions = build_causal_conditions(alpha=8.0)
    assert len(conditions) == 5
    assert sum(spec["alpha"] == 0.0 for spec in conditions.values()) == 1
    doses = {c: dose_columns(c, 8.0) for c in conditions}
    assert doses["shared_zero"] == (0.0, 0.0)
    random_dir = random_control_direction(cfg, axis_a)
    assert np.isclose(np.linalg.norm(random_dir), 1.0)

    rates = outcome_rates(["refused", "harmful"], ["assistant", "human_role"])
    assert rates["n_completed_nontechnical"] == 2

    # --- stamped report ---
    report = stamp_report({
        "reliability_pearson": reliability,
        "rates": rates,
    }, allow_dirty=True)
    out = tmp_path / "smoke_report.json"
    out.write_text(json.dumps(report, indent=2))
    reloaded = json.loads(out.read_text())
    assert len(reloaded["source_git_sha"]) == 40
    assert reloaded["config_sha256"] == config_sha
