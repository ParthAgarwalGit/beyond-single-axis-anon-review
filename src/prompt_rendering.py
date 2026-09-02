"""Canonical prompt rendering (frozen METHOD_FREEZE §5).

Builds chat message lists from the frozen condition templates in the YAML.
Manual model-specific special tokens are forbidden: rendering to text is done
only through ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``
and the rendered text is tokenized with ``add_special_tokens=False``.

The default Assistant conditions never receive a role wrapper of any kind.
"""

from .config import load_frozen_config


def role_arm_messages(cfg, arm, role_instruction, question):
    """Message list for a role arm (USER_TRANSLATED_LU / USER_EXPLICIT / SYSTEM_LU)."""
    conditions = cfg["prompt_rendering"]["conditions"]
    if arm not in conditions:
        raise ValueError(f"unknown arm {arm!r}; frozen arms: {sorted(conditions)}")
    messages = []
    for spec in conditions[arm]["messages"]:
        content = spec["content_template"]
        # The YAML stores literal backslash-n escapes inside the templates.
        content = content.replace("\\n", "\n")
        content = content.replace("{source_role_instruction}", role_instruction)
        content = content.replace("{question}", question)
        messages.append({"role": spec["role"], "content": content})
    return messages


def default_messages(cfg, condition_index, question, channel, model_name=""):
    """Message list for one of the five frozen default-Assistant conditions.

    ``channel`` is ``"user"`` or ``"system"`` and must match the channel of
    the role arm being compared against (frozen: same channel as matched
    role arm). No role wrapper is ever added; the only allowed content is
    the frozen default-condition text.
    """
    if channel not in ("user", "system"):
        raise ValueError(f"channel must be 'user' or 'system', got {channel!r}")
    conditions = cfg["prompt_rendering"]["default_conditions"]
    spec = next((c for c in conditions if c["index"] == condition_index), None)
    if spec is None:
        raise ValueError(
            f"unknown default-condition index {condition_index!r}; "
            f"frozen indices: {sorted(c['index'] for c in conditions)}"
        )
    text = spec["text"].replace("{model_name}", model_name)
    separator = cfg["prompt_rendering"]["separator"]

    if spec["is_empty_control"] or not text:
        # Empty control: the question alone, no empty extra message.
        return [{"role": "user", "content": question}]
    if channel == "system":
        return [
            {"role": "system", "content": text},
            {"role": "user", "content": question},
        ]
    return [{"role": "user", "content": f"{text}{separator}{question}"}]


def render_prompt(tokenizer, messages):
    """Frozen rendering call. Returns the rendered prompt text."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def tokenize_rendered_prompt(tokenizer, rendered_text):
    """Frozen tokenization of rendered text (template supplies special tokens)."""
    return tokenizer.encode(rendered_text, add_special_tokens=False)


def default_condition_count(cfg=None):
    if cfg is None:
        cfg, _ = load_frozen_config()
    return len(cfg["prompt_rendering"]["default_conditions"])
