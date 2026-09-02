#!/usr/bin/env python3
"""Capture the real outcome-blind Qwen3-32B calibration artifact for T24.

This fills the only large-data gap left by the reviewed T24 protocol. It reads
only the frozen 100-item T21 prompt manifest, runs the pinned Qwen checkpoint
UNSTEERED with thinking disabled, captures block-output residual activations at
layers 46..53, embeds the released Assistant Axis, and then invokes the already
reviewed numerical-freeze runner.

The activation-row convention is fixed here before T29 outcomes are inspected:
* natural_L: every hook-visible row during HF generate (prefill + decode calls);
* default_L: one equal-item reference row per manifest item, obtained by averaging
  that item's hook-visible rows. This makes the sphere centre an equal-item
  default-Assistant centroid instead of allowing long prompts/responses to dominate.

This capture convention is a project calibration choice, not a Lu et al. method.
It must be independently reviewed before the expensive production run.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
ARTIFACT_REPO = "lu-christina/assistant-axis-vectors"
ARTIFACT_REVISION = "3b3b788432ad33e3a28d9ff08e88a530c0740814"
AXIS_FILE = "qwen-3-32b/assistant_axis.pt"
AXIS_SHA256 = "a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211"
CAP_FILE = "qwen-3-32b/capping_config.pt"
CAP_SHA256 = "6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24"
SOURCE_ID = "T21_JBB_BEHAVIOR_ONLY_ADAPTED_NOT_LU_PERSONA_JAILBREAK"
MANIFEST_SHA256 = "d611e904f3b5d47c71e5ab1f3fa1a84ead6cfd5d94dba337f5c3b71aafef7793"
LAYERS = tuple(range(46, 54))
N_ITEMS = 100
MAX_NEW_TOKENS = 256
CAPTURE_SCHEMA = "t24-qwen-calibration/1.0"


def die(msg):
    raise SystemExit(f"T24 CAPTURE ABORTED (fail-closed): {msg}")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_sha():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              check=True, text=True, capture_output=True).stdout.strip()
    except Exception as exc:
        die(f"cannot resolve source git SHA: {exc}")


def load_manifest(path):
    path = Path(path)
    if not path.exists():
        die(f"frozen manifest missing: {path}")
    if sha256_file(path) != MANIFEST_SHA256:
        die("frozen 100-item T21 manifest hash mismatch")
    rows = [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]
    if len(rows) != N_ITEMS or len({r.get("item_id") for r in rows}) != N_ITEMS:
        die(f"expected {N_ITEMS} unique manifest rows")
    for r in rows:
        if not isinstance(r.get("messages"), list) or not r["messages"]:
            die(f"manifest item {r.get('item_id')} has no messages")
    return rows


def download_artifacts(work, token):
    from huggingface_hub import hf_hub_download
    out = Path(work) / "source_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    axis = Path(hf_hub_download(ARTIFACT_REPO, AXIS_FILE, repo_type="dataset",
                                revision=ARTIFACT_REVISION, token=token, local_dir=out))
    cap = Path(hf_hub_download(ARTIFACT_REPO, CAP_FILE, repo_type="dataset",
                               revision=ARTIFACT_REVISION, token=token, local_dir=out))
    if sha256_file(axis) != AXIS_SHA256:
        die("released Assistant Axis SHA mismatch")
    if sha256_file(cap) != CAP_SHA256:
        die("released capping config SHA mismatch")
    return axis, cap


def load_axis(path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        obj = obj.get("axis")
    if obj is None:
        die("released Axis artifact has no axis")
    arr = obj.detach().cpu().float().numpy() if torch.is_tensor(obj) else np.asarray(obj, np.float32)
    if arr.ndim != 2 or arr.shape[0] <= max(LAYERS) or not np.isfinite(arr).all():
        die(f"released Axis malformed: {arr.shape}")
    return arr.astype(np.float32, copy=False)


def load_model(token):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        die("CUDA GPU required for genuine Qwen3-32B BF16 capture")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=token,
                                        use_fast=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, token=token, dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True, use_safetensors=True,
        trust_remote_code=False).eval()
    devices = {str(p.device) for p in model.parameters()} | {str(x) for x in (getattr(model, "hf_device_map", {}) or {}).values()}
    if any(any(k in d.lower() for k in ("cpu", "disk", "meta")) for d in devices):
        die(f"model offloaded outside CUDA: {sorted(devices)}")
    if not hasattr(model, "model") or not hasattr(model.model, "layers") or len(model.model.layers) <= max(LAYERS):
        die("model.model.layers does not expose layers 46..53")
    return model, tok


@contextmanager
def hooks(model):
    import torch
    state = {L: [] for L in LAYERS}
    handles = []
    def make_hook(L):
        def hook(_m, _i, out):
            t = out if torch.is_tensor(out) else out[0] if isinstance(out, (tuple, list)) else None
            if t is None or t.ndim != 3 or t.shape[0] != 1:
                die(f"layer {L}: unexpected hook output")
            state[L].append(t[0].detach().to(device="cpu", dtype=torch.float16).numpy().copy())
            return out
        return hook
    for L in LAYERS:
        handles.append(model.model.layers[L].register_forward_hook(make_hook(L)))
    try:
        yield state
    finally:
        for h in handles:
            h.remove()


def shard_path(root, idx, item_id):
    d = hashlib.sha256(str(item_id).encode()).hexdigest()[:16]
    return Path(root) / "shards" / f"{idx:03d}_{d}.npz"


def render(tok, messages):
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError as exc:
        die(f"chat template does not support enable_thinking=False: {exc}")
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    return text, enc


def run_item(model, tok, row, state, out_path):
    import torch
    for L in LAYERS:
        state[L].clear()
    text, enc = render(tok, row["messages"])
    prompt_sha = hashlib.sha256(text.encode()).hexdigest()
    if out_path.exists():
        with np.load(out_path, allow_pickle=False) as z:
            if str(z["item_id"]) != str(row["item_id"]) or str(z["prompt_sha256"]) != prompt_sha:
                die(f"resume shard provenance mismatch: {out_path.name}")
        return
    input_device = model.get_input_embeddings().weight.device
    ids = enc["input_ids"].to(input_device)
    mask = enc.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(ids)
    mask = mask.to(input_device)
    with torch.inference_mode():
        generated = model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                                   max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
                                   pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    arrays = {
        "item_id": np.asarray(str(row["item_id"])),
        "prompt_sha256": np.asarray(prompt_sha),
        "generated_tokens": np.asarray(int(generated.shape[1] - ids.shape[1]), dtype=np.int64),
    }
    n_rows = None
    for L in LAYERS:
        if not state[L]:
            die(f"{row['item_id']}: layer {L} captured no activations")
        x = np.concatenate(state[L], axis=0).astype(np.float16, copy=False)
        if not np.isfinite(x).all():
            die(f"{row['item_id']}: non-finite layer {L} activations")
        n_rows = x.shape[0] if n_rows is None else n_rows
        if x.shape[0] != n_rows:
            die(f"{row['item_id']}: hook row counts differ across layers")
        arrays[f"natural_{L}"] = x
        arrays[f"default_{L}"] = x.astype(np.float32).mean(axis=0).astype(np.float32)
    arrays["natural_rows"] = np.asarray(int(n_rows), dtype=np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, out_path)


def assemble(rows, work, axis):
    shards = [shard_path(work, i, r["item_id"]) for i, r in enumerate(rows)]
    if not all(p.exists() for p in shards):
        die("cannot assemble: one or more per-item shards are missing")
    output = {
        "calibration_source_id": np.asarray(SOURCE_ID),
        "calibration_manifest_sha256": np.asarray(MANIFEST_SHA256),
        "capture_schema": np.asarray(CAPTURE_SCHEMA),
        "capture_source_git_sha": np.asarray(git_sha()),
        "capture_row_unit_natural": np.asarray("all hook-visible rows during unsteered generate"),
        "capture_row_unit_default": np.asarray("equal-item mean of that item's hook-visible rows"),
    }
    for L in LAYERS:
        ns, ds = [], []
        for p in shards:
            with np.load(p, allow_pickle=False) as z:
                ns.append(np.asarray(z[f"natural_{L}"], dtype=np.float16))
                ds.append(np.asarray(z[f"default_{L}"], dtype=np.float32))
        output[f"natural_{L}"] = np.concatenate(ns, axis=0)
        output[f"default_{L}"] = np.stack(ds, axis=0)
        output[f"axis_{L}"] = np.asarray(axis[L], dtype=np.float32)
    out = Path(work) / "t24_qwen_calibration.npz"
    np.savez(out, **output)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = ap.parse_args(argv)
    work = Path(args.work_dir).expanduser().resolve(); work.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest)
    token = args.hf_token.strip() or None
    axis_pt, cap_pt = download_artifacts(work, token)
    axis = load_axis(axis_pt)
    model, tok = load_model(token)
    with hooks(model) as state:
        for i, row in enumerate(rows):
            p = shard_path(work, i, row["item_id"])
            run_item(model, tok, row, state, p)
            print(f"[{i+1:03d}/{N_ITEMS}] {row['item_id']} -> {p.name}")
    del model; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    calibration = assemble(rows, work, axis)
    results = ROOT / "results" / "t24"; results.mkdir(parents=True, exist_ok=True)
    params = work / "source_control_parameters.npz"
    freeze_json = results / "source_control_freeze.json"
    subprocess.run([
        sys.executable, str(ROOT / "tools" / "run_t24_source_control_freeze.py"),
        "--calibration-npz", str(calibration),
        "--assistant-axis-pt", str(axis_pt),
        "--capping-config-pt", str(cap_pt),
        "--out-npz", str(params), "--out-json", str(freeze_json),
    ], cwd=str(ROOT), check=True)
    receipt = {
        "schema_version": CAPTURE_SCHEMA,
        "task": "T24",
        "status": "REAL_QWEN_CALIBRATION_CAPTURED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_sha": git_sha(),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16", "thinking": False},
        "manifest_sha256": MANIFEST_SHA256,
        "n_items": N_ITEMS,
        "layers": list(LAYERS),
        "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "use_cache": True, "calibration_only": True},
        "natural_L": "all hook-visible rows during unsteered HF generate",
        "default_L": "one equal-item mean row per manifest item over that item's hook-visible rows",
        "calibration_npz_external_path": str(calibration),
        "calibration_npz_sha256": sha256_file(calibration),
        "numerical_controls_npz_external_path": str(params),
        "numerical_controls_npz_sha256": sha256_file(params),
        "assistant_axis_sha256": sha256_file(axis_pt),
        "capping_config_sha256": sha256_file(cap_pt),
        "outcome_labels_used": False,
        "claim_boundary": "Capture/freeze provenance only; causal specificity requires T29 outcomes.",
    }
    receipt_path = results / "t24_calibration_capture_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("T24 numerical freeze complete")
    print(" calibration:", calibration, receipt["calibration_npz_sha256"])
    print(" controls:   ", params, receipt["numerical_controls_npz_sha256"])
    print(" freeze:     ", freeze_json)
    print(" receipt:    ", receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
