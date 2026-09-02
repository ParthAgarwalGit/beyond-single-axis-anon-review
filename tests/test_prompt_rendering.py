"""Required T03 test: the default prompt never receives a role wrapper."""

import pytest

from src.prompt_rendering import default_messages, role_arm_messages

ROLE = "You are a weathered pirate captain who speaks in nautical slang."
QUESTION = "How do plants make food?"
WRAPPER_FRAGMENTS = (
    ROLE,
    "Stay in character",
    "first person",
    "Never mention being an AI",
)


def test_default_never_receives_role_wrapper(cfg):
    conditions = cfg["prompt_rendering"]["default_conditions"]
    for spec in conditions:
        for channel in ("user", "system"):
            msgs = default_messages(
                cfg, spec["index"], QUESTION, channel, model_name="DeepSeek-R1-Distill-Llama-8B"
            )
            joined = " ".join(m["content"] for m in msgs)
            for fragment in WRAPPER_FRAGMENTS:
                assert fragment not in joined, (
                    f"default condition {spec['index']} ({channel}) leaked a role wrapper"
                )
            assert QUESTION in joined


def test_empty_control_is_question_only(cfg):
    for channel in ("user", "system"):
        msgs = default_messages(cfg, 0, QUESTION, channel)
        assert msgs == [{"role": "user", "content": QUESTION}]


def test_role_arms_match_frozen_templates(cfg):
    user_lu = role_arm_messages(cfg, "USER_TRANSLATED_LU", ROLE, QUESTION)
    assert user_lu == [{"role": "user", "content": f"{ROLE}\n\n{QUESTION}"}]

    system_lu = role_arm_messages(cfg, "SYSTEM_LU", ROLE, QUESTION)
    assert system_lu == [
        {"role": "system", "content": ROLE},
        {"role": "user", "content": QUESTION},
    ]

    explicit = role_arm_messages(cfg, "USER_EXPLICIT", ROLE, QUESTION)
    assert len(explicit) == 1
    assert explicit[0]["content"].startswith(ROLE)
    assert explicit[0]["content"].endswith(QUESTION)
    assert "Stay in character" in explicit[0]["content"]


def test_unknown_arm_rejected(cfg):
    with pytest.raises(ValueError):
        role_arm_messages(cfg, "HAND_ROLLED_ARM", ROLE, QUESTION)


def test_unknown_default_condition_index_rejected(cfg):
    with pytest.raises(ValueError, match="unknown default-condition index"):
        default_messages(cfg, 9, QUESTION, "user")
