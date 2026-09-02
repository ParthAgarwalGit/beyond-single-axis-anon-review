#!/usr/bin/env python3
"""Run T04 formal hook and pooling validation.

Creates:
- results/validation/hook_validation_v3.json
- results/validation/pooling_validation_v3.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.activation_extraction import hidden_states_index, pool_response_vector
from src.config import load_frozen_config
from src.generation import (
    BALANCED_REASONING,
    DIRECT_ANSWER,
    EMPTY_ANSWER,
    MALFORMED_REASONING,
    GenerationRecord,
    content_token_ids,
    response_start,
    segment_response,
)
from src.runtime_hooks import CaptureLayerOutput, GeneratedTokenHook


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"git_sha": sha, "git_dirty": None if status is None else bool(status)}


def resolve_revision(model_id: str, requested: Optional[str]) -> str:
    """Resolve a Hugging Face revision to an immutable commit SHA."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, revision=requested or "main")
    if not info.sha:
        raise RuntimeError(f"Could not resolve immutable revision for {model_id}")
    return str(info.sha)


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    af = a.detach().float().cpu()
    bf = b.detach().float().cpu()
    delta = (af - bf).abs()
    denom = bf.abs().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        af.reshape(1, -1), bf.reshape(1, -1)
    ).item()
    return {
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
        "max_rel": float((delta / denom).max().item()) if delta.numel() else 0.0,
        "cosine": float(cosine),
    }


def allclose_report(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    report = tensor_diff(a, b)
    report.update(
        {
            "atol": float(atol),
            "rtol": float(rtol),
            "allclose": bool(
                torch.allclose(
                    a.detach().float().cpu(),
                    b.detach().float().cpu(),
                    atol=atol,
                    rtol=rtol,
                )
            ),
        }
    )
    return report


def compare_token_lists(a: list[int], b: list[int]) -> dict[str, Any]:
    first_difference = None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            first_difference = i
            break
    if first_difference is None and len(a) != len(b):
        first_difference = min(len(a), len(b))
    return {
        "equal": a == b,
        "length_a": len(a),
        "length_b": len(b),
        "first_difference_index": first_difference,
    }


def compare_score_lists(
    a: list[torch.Tensor],
    b: list[torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    n = min(len(a), len(b))
    per_step = []
    first_nonclose = None
    global_max_abs = 0.0
    for i in range(n):
        current = allclose_report(a[i], b[i], atol=atol, rtol=rtol)
        current["step"] = i
        per_step.append(current)
        global_max_abs = max(global_max_abs, current["max_abs"])
        if first_nonclose is None and not current["allclose"]:
            first_nonclose = i
    return {
        "n_steps_compared": n,
        "length_a": len(a),
        "length_b": len(b),
        "all_steps_close": len(a) == len(b) and all(row["allclose"] for row in per_step),
        "first_nonclose_step": first_nonclose,
        "max_abs_across_steps": global_max_abs,
        "per_step": per_step,
    }


def terminal_ids(tokenizer) -> tuple[int, ...]:
    values = []
    for value in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        if isinstance(value, int) and value not in values:
            values.append(value)
    return tuple(values)


def render_validation_prompt(tokenizer, prompt_text: str) -> tuple[str, list[int]]:
    messages = [{"role": "user", "content": prompt_text}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if not ids:
        raise ValueError("Rendered validation prompt produced no token IDs")
    return rendered, ids


@torch.inference_mode()
def run_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    hook: Optional[GeneratedTokenHook] = None,
    layer: Optional[int] = None,
) -> dict[str, Any]:
    handle = None
    if hook is not None:
        if layer is None:
            raise ValueError("layer is required when a hook is supplied")
        handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        output = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        if handle is not None:
            handle.remove()

    prompt_len = int(input_ids.shape[1])
    output_ids = output.sequences[0, prompt_len:].detach().cpu().tolist()
    scores = [score.detach().float().cpu() for score in output.scores]
    return {
        "output_ids": output_ids,
        "scores": scores,
        "text": tokenizer.decode(
            output_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "hook_diagnostics": None if hook is None else hook.diagnostics.as_dict(),
    }


@torch.inference_mode()
def teacher_force_full(model, full_ids: torch.Tensor, *, layers: tuple[int, ...]):
    captures = {layer: CaptureLayerOutput() for layer in layers}
    handles = [
        model.model.layers[layer].register_forward_hook(captures[layer])
        for layer in layers
    ]
    try:
        output = model(
            input_ids=full_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    return output, {layer: captures[layer].hidden for layer in layers}


@torch.inference_mode()
def cached_teacher_force_generated(
    model,
    prompt_ids: torch.Tensor,
    generated_ids: list[int],
    *,
    hidden_state_index: int,
) -> torch.Tensor:
    """Process a stored completion one token at a time with KV caching."""
    prefill = model(
        input_ids=prompt_ids,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = prefill.past_key_values
    rows = []
    for token_id in generated_ids:
        token = torch.tensor([[token_id]], dtype=torch.long, device=prompt_ids.device)
        step = model(
            input_ids=token,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = step.past_key_values
        rows.append(step.hidden_states[hidden_state_index][:, -1:, :].detach())
    if not rows:
        raise ValueError("Stored completion contains no token IDs")
    return torch.cat(rows, dim=1)


def make_generation_record(tokenizer, prompt_ids: list[int], output_ids: list[int]) -> GenerationRecord:
    content = list(output_ids)
    terms = set(terminal_ids(tokenizer))
    while content and content[-1] in terms:
        content.pop()
    output_text = tokenizer.decode(
        content,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    finish_reason = "stop" if output_ids and output_ids[-1] in terms else "length"
    return GenerationRecord(
        prompt_token_ids=list(prompt_ids),
        output_token_ids=list(output_ids),
        output_text=output_text,
        finish_reason=finish_reason,
    )


def tokenizer_fn(tokenizer) -> Callable[[str], list[int]]:
    return lambda text: tokenizer.encode(text, add_special_tokens=False)


def validation_direction(hidden_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vector = torch.randn(hidden_size, generator=generator, dtype=torch.float32)
    return vector / torch.linalg.vector_norm(vector)


def run_synthetic_pooling_checks() -> dict[str, Any]:
    """Validate masks and arithmetic means on hand-written token sequences."""
    eos = 999_999

    def tok(text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def record(prompt_len: int, text: str) -> GenerationRecord:
        return GenerationRecord(
            prompt_token_ids=list(range(prompt_len)),
            output_token_ids=tok(text) + [eos],
            output_text=text,
            finish_reason="stop",
        )

    cases: dict[str, dict[str, Any]] = {}

    direct = record(3, "abc")
    seg = segment_response(direct, tok, terminal_ids=(eos,))
    cases["direct_answer"] = {
        "pass": (
            seg.case == DIRECT_ANSWER
            and seg.all_response_span == (3, 6)
            and seg.reasoning_span is None
            and seg.final_answer_span == (3, 6)
            and response_start(direct) == 3
        ),
        "segmentation": asdict(seg),
    }

    text = "<think>r</think>answer"
    balanced = record(2, text)
    seg = segment_response(balanced, tok, terminal_ids=(eos,))
    boundary = len("<think>r</think>")
    cases["balanced_reasoning"] = {
        "pass": (
            seg.case == BALANCED_REASONING
            and seg.all_response_span == (2, 2 + len(text))
            and seg.reasoning_span == (2, 2 + boundary)
            and seg.final_answer_span == (2 + boundary, 2 + len(text))
            and seg.judge_input_text == "answer"
        ),
        "segmentation": asdict(seg),
    }

    text_multi = "<think>a</think><think>b</think>z"
    multi = record(1, text_multi)
    seg_multi = segment_response(multi, tok, terminal_ids=(eos,))
    last_boundary = text_multi.rfind("</think>") + len("</think>")
    cases["multiple_pairs"] = {
        "pass": (
            seg_multi.case == BALANCED_REASONING
            and seg_multi.final_answer_span == (1 + last_boundary, 1 + len(text_multi))
            and "multiple_pairs" in seg_multi.flags
        ),
        "segmentation": asdict(seg_multi),
    }

    malformed = record(4, "<think>unfinished")
    seg_bad = segment_response(malformed, tok, terminal_ids=(eos,))
    cases["malformed_reasoning"] = {
        "pass": (
            seg_bad.case == MALFORMED_REASONING
            and seg_bad.all_response_span is not None
            and seg_bad.reasoning_span is None
            and seg_bad.final_answer_span is None
            and seg_bad.judge_input_text is None
        ),
        "segmentation": asdict(seg_bad),
    }

    empty = record(2, "<think>x</think>   ")
    seg_empty = segment_response(empty, tok, terminal_ids=(eos,))
    cases["empty_answer_after_close"] = {
        "pass": (
            seg_empty.case == EMPTY_ANSWER
            and seg_empty.final_answer_span is None
            and seg_empty.judge_input_text is None
        ),
        "segmentation": asdict(seg_empty),
    }

    prompt_len = 2
    text_pool = "<think>x</think>yz"
    pooled_record = record(prompt_len, text_pool)
    pooled_seg = segment_response(pooled_record, tok, terminal_ids=(eos,))
    total_len = prompt_len + len(text_pool)
    residuals = np.arange(total_len * 4, dtype=np.float64).reshape(total_len, 4)

    all_vec = pool_response_vector(residuals, pooled_seg.all_response_span, prompt_len)
    reason_vec = pool_response_vector(residuals, pooled_seg.reasoning_span, prompt_len)
    answer_vec = pool_response_vector(residuals, pooled_seg.final_answer_span, prompt_len)

    a0, a1 = pooled_seg.all_response_span
    r0, r1 = pooled_seg.reasoning_span
    f0, f1 = pooled_seg.final_answer_span
    cases["arithmetic_pooling"] = {
        "pass": bool(
            np.array_equal(all_vec, residuals[a0:a1].mean(axis=0))
            and np.array_equal(reason_vec, residuals[r0:r1].mean(axis=0))
            and np.array_equal(answer_vec, residuals[f0:f1].mean(axis=0))
        ),
        "all_response_vector": all_vec.tolist(),
        "reasoning_vector": reason_vec.tolist(),
        "final_answer_vector": answer_vec.tolist(),
    }

    rejected = False
    try:
        pool_response_vector(residuals, (0, prompt_len + 1), prompt_len)
    except ValueError:
        rejected = True
    cases["prompt_leakage_rejected"] = {"pass": rejected}

    return {
        "status": "PASS" if all(v["pass"] for v in cases.values()) else "FAIL",
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/method_frozen.yaml")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--validation-alpha",
        type=float,
        default=0.10,
        help="Validation-only; never reuse as the causal coefficient.",
    )
    parser.add_argument("--equivalence-atol", type=float, default=1e-5)
    parser.add_argument("--equivalence-rtol", type=float, default=1e-5)
    parser.add_argument("--teacher-force-atol", type=float, default=5e-2)
    parser.add_argument("--teacher-force-rtol", type=float, default=5e-2)
    parser.add_argument(
        "--prompt",
        default=(
            "Answer with exactly one short sentence: Why should a scientific "
            "measurement be checked twice?"
        ),
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="eager",
    )
    parser.add_argument("--output-dir", default="results/validation")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config_path = ROOT / args.config
    loaded_config = load_frozen_config()

    if (
        isinstance(loaded_config, tuple)
        and len(loaded_config) == 2
    ):
        cfg, config_sha = loaded_config
    else:
        cfg = loaded_config
        config_sha = hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest()
    primary = cfg["models"]["primary"]
    model_id = args.model_id or primary["model_id"]
    requested_model_revision = args.model_revision or primary.get("model_revision")
    requested_tokenizer_revision = args.tokenizer_revision or primary.get("tokenizer_revision")

    model_revision = resolve_revision(model_id, requested_model_revision)
    tokenizer_revision = resolve_revision(model_id, requested_tokenizer_revision)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=tokenizer_revision, trust_remote_code=False
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The 8B integration validation requires CUDA. CPU-only tests can still run via pytest."
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=dtype,
        device_map={"": 0},
        trust_remote_code=False,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    rendered_prompt, prompt_token_ids = render_validation_prompt(tokenizer, args.prompt)
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=model.device)

    primary_layer = int(cfg["activation_extraction"]["middle_layer"]["primary_block_index"])
    sensitivity_layer = int(
        cfg["activation_extraction"]["middle_layer"]["off_by_one_sensitivity_block_index"]
    )
    if primary_layer != 16 or sensitivity_layer != 15:
        raise ValueError("Frozen method must specify primary block 16 and sensitivity block 15")

    baseline = run_generate(
        model, tokenizer, input_ids, max_new_tokens=args.max_new_tokens
    )
    if not baseline["output_ids"]:
        raise RuntimeError("Baseline generation returned no output tokens")

    record = make_generation_record(tokenizer, prompt_token_ids, baseline["output_ids"])
    seg = segment_response(
        record,
        tokenizer_fn(tokenizer),
        terminal_ids=terminal_ids(tokenizer),
    )

    # T04 must always be able to validate an answer-only pool. If the short
    # greedy completion ends with malformed/unclosed reasoning, preserve it
    # for the hook comparisons but use a clearly recorded direct-answer
    # completion for the teacher-forced pooling check.
    teacher_force_fallback_used = False
    stored_output_ids = list(baseline["output_ids"])
    if seg.final_answer_span is None:
        teacher_force_fallback_used = True
        fallback_text = "Careful rechecking catches measurement errors."
        stored_output_ids = tokenizer.encode(
            fallback_text, add_special_tokens=False
        )
        record = GenerationRecord(
            prompt_token_ids=list(prompt_token_ids),
            output_token_ids=list(stored_output_ids),
            output_text=fallback_text,
            finish_reason="stop",
        )
        seg = segment_response(
            record,
            tokenizer_fn(tokenizer),
            terminal_ids=terminal_ids(tokenizer),
        )

    full_ids_list = prompt_token_ids + stored_output_ids
    full_ids = torch.tensor([full_ids_list], dtype=torch.long, device=model.device)

    full_output, captures = teacher_force_full(
        model, full_ids, layers=(sensitivity_layer, primary_layer)
    )

    layer_mapping = {}
    for layer in (sensitivity_layer, primary_layer):
        hs_index = hidden_states_index(layer)
        mapping = allclose_report(
            captures[layer],
            full_output.hidden_states[hs_index],
            atol=args.equivalence_atol,
            rtol=args.equivalence_rtol,
        )
        mapping.update(
            {
                "block_index": layer,
                "hidden_states_index": hs_index,
                "hook_path": f"model.model.layers[{layer}]",
            }
        )
        layer_mapping[str(layer)] = mapping

    primary_hidden = full_output.hidden_states[hidden_states_index(primary_layer)][0].detach()
    prompt_len = len(prompt_token_ids)
    runtime_pools = {}
    for name, span in (
        ("all_response", seg.all_response_span),
        ("reasoning", seg.reasoning_span),
        ("final_answer", seg.final_answer_span),
    ):
        if span is None:
            runtime_pools[name] = {"available": False, "span": None}
            continue
        pooled = pool_response_vector(
            primary_hidden.float().cpu().numpy(), span, prompt_len
        )
        start, end = span
        manual = primary_hidden[start:end].float().mean(dim=0).cpu().numpy()
        runtime_pools[name] = {
            "available": True,
            "span": list(span),
            "n_tokens": end - start,
            "manual_pool_max_abs_diff": float(np.max(np.abs(pooled - manual))),
            "manual_pool_allclose": bool(
                np.allclose(
                    pooled,
                    manual,
                    atol=args.equivalence_atol,
                    rtol=args.equivalence_rtol,
                )
            ),
        }

    all_start, all_end = seg.all_response_span
    mean_residual_norm = float(
        torch.linalg.vector_norm(
            primary_hidden[all_start:all_end].float(), dim=-1
        ).mean().cpu()
    )
    hidden_size = int(primary_hidden.shape[-1])
    direction = validation_direction(hidden_size, args.seed)
    injection_scale = float(args.validation_alpha * mean_residual_norm)

    read_hook = GeneratedTokenHook("read_only")
    read_only = run_generate(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        hook=read_hook,
        layer=primary_layer,
    )

    zero_hook = GeneratedTokenHook("zero")
    zero = run_generate(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        hook=zero_hook,
        layer=primary_layer,
    )

    steer_hook = GeneratedTokenHook(
        "steer", unit_direction=direction, injection_scale=injection_scale
    )
    nonzero = run_generate(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        hook=steer_hook,
        layer=primary_layer,
    )

    read_logits = compare_score_lists(
        baseline["scores"],
        read_only["scores"],
        atol=args.equivalence_atol,
        rtol=args.equivalence_rtol,
    )
    zero_logits = compare_score_lists(
        baseline["scores"],
        zero["scores"],
        atol=args.equivalence_atol,
        rtol=args.equivalence_rtol,
    )
    nonzero_logits = compare_score_lists(
        baseline["scores"],
        nonzero["scores"],
        atol=args.equivalence_atol,
        rtol=args.equivalence_rtol,
    )

    read_tokens = compare_token_lists(baseline["output_ids"], read_only["output_ids"])
    zero_tokens = compare_token_lists(baseline["output_ids"], zero["output_ids"])
    nonzero_tokens = compare_token_lists(baseline["output_ids"], nonzero["output_ids"])

    cached_generated = cached_teacher_force_generated(
        model,
        input_ids,
        stored_output_ids,
        hidden_state_index=hidden_states_index(primary_layer),
    )
    one_shot_generated = full_output.hidden_states[hidden_states_index(primary_layer)][
        :, prompt_len : prompt_len + len(stored_output_ids), :
    ]
    teacher_force_all = allclose_report(
        cached_generated,
        one_shot_generated,
        atol=args.teacher_force_atol,
        rtol=args.teacher_force_rtol,
    )

    teacher_answer = {"available": seg.final_answer_span is not None}
    if seg.final_answer_span is not None:
        answer_start, answer_end = seg.final_answer_span
        rel_start = answer_start - prompt_len
        rel_end = answer_end - prompt_len
        full_answer_pool = one_shot_generated[:, rel_start:rel_end, :].float().mean(dim=1)
        cached_answer_pool = cached_generated[:, rel_start:rel_end, :].float().mean(dim=1)
        teacher_answer.update(
            allclose_report(
                cached_answer_pool,
                full_answer_pool,
                atol=args.teacher_force_atol,
                rtol=args.teacher_force_rtol,
            )
        )

    synthetic = run_synthetic_pooling_checks()

    hook_checks = {
        "read_only_logits_match": read_logits["all_steps_close"],
        "read_only_tokens_match": read_tokens["equal"],
        "zero_logits_match": zero_logits["all_steps_close"],
        "zero_tokens_match": zero_tokens["equal"],
        "nonzero_changes_logits_after_first_token": (
            nonzero_logits["first_nonclose_step"] is not None
            and nonzero_logits["first_nonclose_step"] >= 1
        ),
        "nonzero_prefill_unchanged": nonzero["hook_diagnostics"]["prefill_max_abs_delta"] == 0.0,
        "nonzero_decode_changed": nonzero["hook_diagnostics"]["decode_max_abs_delta"] > 0.0,
        "layer_16_mapping": layer_mapping[str(primary_layer)]["allclose"],
        "layer_15_mapping": layer_mapping[str(sensitivity_layer)]["allclose"],
        "teacher_forced_all_generated_match": teacher_force_all["allclose"],
        "teacher_forced_answer_pool_match": (
            teacher_answer.get("allclose", False) if teacher_answer["available"] else False
        ),
    }
    hook_status = "PASS" if all(hook_checks.values()) else "FAIL"

    pooling_checks = {
        "synthetic_cases_pass": synthetic["status"] == "PASS",
        "response_start_equals_prompt_length": response_start(record) == len(prompt_token_ids),
        "all_response_runtime_pool_matches_manual": runtime_pools["all_response"].get(
            "manual_pool_allclose", False
        ),
        "reasoning_runtime_pool_matches_manual_when_available": (
            not runtime_pools["reasoning"]["available"]
            or runtime_pools["reasoning"].get("manual_pool_allclose", False)
        ),
        "answer_runtime_pool_matches_manual_when_available": (
            not runtime_pools["final_answer"]["available"]
            or runtime_pools["final_answer"].get("manual_pool_allclose", False)
        ),
        "teacher_forced_answer_pool_match": (
            teacher_answer.get("allclose", False) if teacher_answer["available"] else False
        ),
    }
    pooling_status = "PASS" if all(pooling_checks.values()) else "FAIL"

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "attention_implementation": args.attn_implementation,
        "seed": args.seed,
        **git_metadata(ROOT),
    }
    provenance = {
        "model_id": model_id,
        "requested_model_revision": requested_model_revision,
        "resolved_model_revision": model_revision,
        "requested_tokenizer_revision": requested_tokenizer_revision,
        "resolved_tokenizer_revision": tokenizer_revision,
        "chat_template_sha256": sha256_text(tokenizer.chat_template or ""),
        "method_config_path": str(config_path.relative_to(ROOT)),
        "method_config_sha256": config_sha,
        "rendered_prompt_sha256": sha256_text(rendered_prompt),
        "prompt_token_count": len(prompt_token_ids),
        "generation_start_index": len(prompt_token_ids),
    }

    hook_report = {
        "schema_version": "t04-hook-validation/3",
        "task": "T04",
        "created_at_utc": utc_now(),
        "status": hook_status,
        "provenance": provenance,
        "environment": env,
        "frozen_hook_definition": {
            "module_path": f"model.model.layers[{primary_layer}]",
            "tensor": "first tensor in block output before final model norm",
            "primary_block_index": primary_layer,
            "primary_hidden_states_index": hidden_states_index(primary_layer),
            "sensitivity_block_index": sensitivity_layer,
            "sensitivity_hidden_states_index": hidden_states_index(sensitivity_layer),
            "prompt_prefill_positions_modified": False,
            "generated_response_positions_modified": True,
            "first_sampled_response_token_unsteered": True,
        },
        "tolerances": {
            "same_path_atol": args.equivalence_atol,
            "same_path_rtol": args.equivalence_rtol,
            "teacher_force_atol": args.teacher_force_atol,
            "teacher_force_rtol": args.teacher_force_rtol,
        },
        "validation_only_nonzero_scale": {
            "alpha": args.validation_alpha,
            "observed_mean_residual_norm": mean_residual_norm,
            "injection_scale": injection_scale,
            "note": "Validation-only; do not reuse as the causal coefficient.",
        },
        "baseline": {
            "output_token_count": len(baseline["output_ids"]),
            "output_token_ids": baseline["output_ids"],
            "decoded_text": baseline["text"],
        },
        "read_only": {
            "logit_comparison": read_logits,
            "token_comparison": read_tokens,
            "hook_diagnostics": read_only["hook_diagnostics"],
        },
        "zero_vector": {
            "logit_comparison": zero_logits,
            "token_comparison": zero_tokens,
            "hook_diagnostics": zero["hook_diagnostics"],
        },
        "nonzero_vector": {
            "logit_comparison": nonzero_logits,
            "token_comparison": nonzero_tokens,
            "hook_diagnostics": nonzero["hook_diagnostics"],
            "decoded_text": nonzero["text"],
            "tokens_changed": not nonzero_tokens["equal"],
            "note": (
                "The residual and downstream logits must change. Greedy tokens are "
                "reported but may stay unchanged if the same token remains the argmax."
            ),
        },
        "layer_indexing": layer_mapping,
        "teacher_forcing": {
            "stored_completion_output_token_count": len(stored_output_ids),
            "fallback_direct_answer_used": teacher_force_fallback_used,
            "one_shot_vs_cached_all_generated": teacher_force_all,
            "one_shot_vs_cached_answer_pool": teacher_answer,
        },
        "timing": {
            "read_only": read_only["hook_diagnostics"],
            "zero_vector": zero["hook_diagnostics"],
            "nonzero_vector": nonzero["hook_diagnostics"],
        },
        "checks": hook_checks,
    }

    pooling_report = {
        "schema_version": "t04-pooling-validation/3",
        "task": "T04",
        "created_at_utc": utc_now(),
        "status": pooling_status,
        "provenance": provenance,
        "segmentation": {
            "case": seg.case,
            "all_response_span": list(seg.all_response_span),
            "reasoning_span": None if seg.reasoning_span is None else list(seg.reasoning_span),
            "final_answer_span": None if seg.final_answer_span is None else list(seg.final_answer_span),
            "flags": list(seg.flags),
            "prompt_length": prompt_len,
            "content_output_token_count": len(
                content_token_ids(record, terminal_ids(tokenizer))
            ),
            "teacher_force_fallback_direct_answer_used": teacher_force_fallback_used,
        },
        "runtime_pools": runtime_pools,
        "teacher_forcing": {"one_shot_vs_cached_answer_pool": teacher_answer},
        "synthetic_hand_written_sequences": synthetic,
        "checks": pooling_checks,
    }

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hook_path = out_dir / "hook_validation_v3.json"
    pool_path = out_dir / "pooling_validation_v3.json"
    hook_path.write_text(json.dumps(hook_report, indent=2, ensure_ascii=False) + "\n")
    pool_path.write_text(json.dumps(pooling_report, indent=2, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "hook_status": hook_status,
                "pooling_status": pooling_status,
                "hook_report": str(hook_path.relative_to(ROOT)),
                "pooling_report": str(pool_path.relative_to(ROOT)),
                "nonzero_tokens_changed": not nonzero_tokens["equal"],
                "nonzero_first_changed_logit_step": nonzero_logits["first_nonclose_step"],
            },
            indent=2,
        )
    )
    return 0 if hook_status == "PASS" and pooling_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
