"""T04 tests for hook mechanics, token boundaries, and pooled activations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.activation_extraction import hidden_states_index, pool_response_vector
from src.runtime_hooks import GeneratedTokenHook
from tools.run_t04_hook_and_pooling_validation import run_synthetic_pooling_checks


class DummyLayer(torch.nn.Module):
    def forward(self, x):
        return (x, "metadata")


def test_read_only_hook_is_inert():
    layer = DummyLayer()
    hook = GeneratedTokenHook("read_only", capture_tensors=True)
    handle = layer.register_forward_hook(hook)
    try:
        x = torch.randn(1, 3, 8)
        out = layer(x)[0]
    finally:
        handle.remove()
    assert torch.equal(out, x)
    assert hook.diagnostics.prefill_calls == 1
    assert hook.diagnostics.prefill_max_abs_delta == 0.0


def test_zero_hook_is_inert_on_decode():
    layer = DummyLayer()
    hook = GeneratedTokenHook("zero")
    handle = layer.register_forward_hook(hook)

    try:
        # First call is always prompt prefill.
        prefill = torch.randn(1, 1, 8)
        prefill_out = layer(prefill)[0]

        # Second one-token call is decode.
        decode = torch.randn(1, 1, 8)
        decode_out = layer(decode)[0]
    finally:
        handle.remove()

    assert torch.equal(prefill_out, prefill)
    assert torch.equal(decode_out, decode)

    assert hook.diagnostics.prefill_calls == 1
    assert hook.diagnostics.decode_calls == 1
    assert hook.diagnostics.prefill_max_abs_delta == 0.0
    assert hook.diagnostics.decode_max_abs_delta == 0.0


def test_nonzero_hook_changes_decode_but_not_prefill():
    layer = DummyLayer()
    direction = torch.arange(1, 9, dtype=torch.float32)
    direction = direction / torch.linalg.vector_norm(direction)
    hook = GeneratedTokenHook(
        "steer", unit_direction=direction, injection_scale=2.0
    )
    handle = layer.register_forward_hook(hook)
    try:
        prefill = torch.randn(1, 4, 8)
        decode = torch.randn(1, 1, 8)
        prefill_out = layer(prefill)[0]
        decode_out = layer(decode)[0]
    finally:
        handle.remove()

    assert torch.equal(prefill_out, prefill)
    assert not torch.equal(decode_out, decode)
    observed = torch.linalg.vector_norm(
        (decode_out - decode).float(), dim=-1
    ).item()
    assert observed == pytest.approx(2.0, rel=1e-5, abs=1e-5)
    assert hook.diagnostics.prefill_max_abs_delta == 0.0
    assert hook.diagnostics.decode_max_abs_delta > 0.0



def test_one_token_first_call_is_prefill():
    layer = DummyLayer()

    direction = torch.arange(1, 9, dtype=torch.float32)
    direction = direction / torch.linalg.vector_norm(direction)

    hook = GeneratedTokenHook(
        "steer",
        unit_direction=direction,
        injection_scale=2.0,
    )

    handle = layer.register_forward_hook(hook)

    try:
        # Even though the sequence length is one, the first call is prefill.
        one_token_prefill = torch.randn(1, 1, 8)
        prefill_out = layer(one_token_prefill)[0]

        # The next one-token call is decode and must be steered.
        decode = torch.randn(1, 1, 8)
        decode_out = layer(decode)[0]
    finally:
        handle.remove()

    assert torch.equal(prefill_out, one_token_prefill)
    assert not torch.equal(decode_out, decode)

    observed = torch.linalg.vector_norm(
        (decode_out - decode).float(),
        dim=-1,
    ).item()

    assert observed == pytest.approx(
        2.0,
        rel=1e-5,
        abs=1e-5,
    )

    assert hook.diagnostics.prefill_calls == 1
    assert hook.diagnostics.decode_calls == 1
    assert hook.diagnostics.prefill_max_abs_delta == 0.0
    assert hook.diagnostics.decode_max_abs_delta > 0.0

def test_hidden_states_index_mapping():
    assert hidden_states_index(15) == 16
    assert hidden_states_index(16) == 17


def test_hand_written_pooling_suite_passes():
    report = run_synthetic_pooling_checks()
    assert report["status"] == "PASS", report


def test_pool_rejects_prompt_positions():
    residuals = np.arange(24, dtype=np.float64).reshape(6, 4)
    with pytest.raises(ValueError, match="prompt"):
        pool_response_vector(residuals, (0, 4), prompt_len=3)


def test_reports_pass_after_gpu_run():
    hook_path = ROOT / "results/validation/hook_validation_v3.json"
    pool_path = ROOT / "results/validation/pooling_validation_v3.json"
    if not hook_path.exists() or not pool_path.exists():
        pytest.skip(
            "Run tools/run_t04_hook_and_pooling_validation.py on CUDA first"
        )

    hook = json.loads(hook_path.read_text(encoding="utf-8"))
    pool = json.loads(pool_path.read_text(encoding="utf-8"))

    assert hook["status"] == "PASS", hook["checks"]
    assert pool["status"] == "PASS", pool["checks"]

    required_hook_checks = {
        "read_only_logits_match",
        "read_only_tokens_match",
        "zero_logits_match",
        "zero_tokens_match",
        "nonzero_changes_logits_after_first_token",
        "nonzero_prefill_unchanged",
        "nonzero_decode_changed",
        "layer_16_mapping",
        "layer_15_mapping",
        "teacher_forced_all_generated_match",
        "teacher_forced_answer_pool_match",
    }
    assert required_hook_checks <= set(hook["checks"])
    assert all(hook["checks"][name] for name in required_hook_checks)

    required_pool_checks = {
        "synthetic_cases_pass",
        "response_start_equals_prompt_length",
        "all_response_runtime_pool_matches_manual",
        "reasoning_runtime_pool_matches_manual_when_available",
        "answer_runtime_pool_matches_manual_when_available",
        "teacher_forced_answer_pool_match",
    }
    assert required_pool_checks <= set(pool["checks"])
    assert all(pool["checks"][name] for name in required_pool_checks)
