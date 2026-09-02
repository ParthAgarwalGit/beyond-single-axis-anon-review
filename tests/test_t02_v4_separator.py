"""Regression guard for V4 prompt-rendering separator semantics."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_v4_prompt_rendering_separator_is_two_real_newlines():
    cfg = yaml.safe_load((ROOT / "configs" / "method_frozen_v4.yaml").read_bytes())
    assert cfg["prompt_rendering"]["separator"] == "\n\n"
