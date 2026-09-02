# ================================================================================
# T12 · C80 activation extraction → Hugging Face
# --------------------------------------------------------------------------------
# Teacher-forces stored C80 completions and pools block-output residuals over the
# frozen response regions. This reproducibility helper is import-safe for CPU tests
# and fails closed on both frozen attempt selection and resumable part integrity.
#
# Production artifacts already exist; do not rerun merely to clean this helper.
# ================================================================================

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

HF_TOKEN = ""
BLOCK = "C80-B"
HF_DATA_REPO = "[Author-A-HF]/persona-artifacts"
C80_ROLLOUTS = "results/generation/c80_role_C80_B.final.jsonl"
HF_OUT_REPO = "[Author-A-HF]/persona-artifacts"
HF_OUT_PREFIX = "t12-c80/C80-B"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_REV = "6a6f4aa4197940add57724a7707d069478df56b1"
MIDDLE_BLOCK = 16
PART_SIZE = 256
SMOKE_N = 8
DTYPE_STORE = "float16"
FLUSH_EVERY = 20
EXTRACTOR_VERSION = "t12-c80-extract-cell-v2"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select_authoritative_attempts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the exact frozen T09/T10 attempt-selection rule.

    Per rollout: select the LOWEST retry with technical_validity == ``valid``.
    If no valid attempt exists, retain the HIGHEST retry failed row for technical
    failure provenance only. Duplicate retry indices for one rollout fail closed.
    """
    by_uid: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        uid = row.get("rollout_id")
        if not isinstance(uid, str) or not uid:
            raise RuntimeError("attempt row missing rollout_id")
        try:
            retry = int(row.get("retry", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"rollout {uid}: invalid retry index {row.get('retry')!r}") from exc
        if retry < 0:
            raise RuntimeError(f"rollout {uid}: negative retry index {retry}")
        row = dict(row)
        row["retry"] = retry
        by_uid.setdefault(uid, []).append(row)

    selected: list[dict[str, Any]] = []
    for uid, attempts in by_uid.items():
        retries = [int(r["retry"]) for r in attempts]
        if len(retries) != len(set(retries)):
            raise RuntimeError(f"rollout {uid}: duplicate retry index in raw attempts")
        valid = [r for r in attempts if r.get("technical_validity") == "valid"]
        if valid:
            chosen = min(valid, key=lambda r: int(r["retry"]))
        else:
            chosen = max(attempts, key=lambda r: int(r["retry"]))
        selected.append(chosen)
    return sorted(selected, key=lambda r: r["rollout_id"])


def _self_test_attempt_selection() -> None:
    rows = [
        {"rollout_id": "a", "retry": 0, "technical_validity": "truncated"},
        {"rollout_id": "a", "retry": 1, "technical_validity": "valid"},
        {"rollout_id": "a", "retry": 2, "technical_validity": "valid"},
        {"rollout_id": "b", "retry": 0, "technical_validity": "truncated"},
        {"rollout_id": "b", "retry": 3, "technical_validity": "generation_error"},
    ]
    got = {r["rollout_id"]: r for r in select_authoritative_attempts(rows)}
    assert got["a"]["retry"] == 1, "must choose lowest-retry valid attempt"
    assert got["b"]["retry"] == 3, "must choose highest-retry failed attempt when none valid"


def verify_part_manifest_files(
    manifest_doc: dict[str, Any],
    load_bytes: Callable[[str], bytes | None],
    *,
    expected_part_index: int | None = None,
) -> bool:
    """Verify one part manifest and every referenced file/hash.

    ``load_bytes`` returns bytes for a relative part file or ``None`` when absent.
    If a manifest exists, any missing/malformed/hash-mismatched referenced file is
    an integrity error rather than a reason to silently skip or trust the part.
    """
    if not isinstance(manifest_doc, dict):
        raise RuntimeError("part manifest is not a JSON object")
    if expected_part_index is not None and manifest_doc.get("part_index") != expected_part_index:
        raise RuntimeError(
            f"part manifest index {manifest_doc.get('part_index')!r} != expected {expected_part_index}"
        )
    files = manifest_doc.get("files")
    if not isinstance(files, dict) or "metadata" not in files:
        raise RuntimeError("part manifest must contain a files mapping with metadata")
    if not files:
        raise RuntimeError("part manifest contains no file records")

    seen_paths: set[str] = set()
    for label, spec in files.items():
        if not isinstance(spec, dict):
            raise RuntimeError(f"part manifest entry {label!r} is not an object")
        rel = spec.get("path")
        expected_sha = spec.get("sha256")
        if not isinstance(rel, str) or not rel or Path(rel).name != rel:
            raise RuntimeError(f"part manifest entry {label!r} has unsafe path {rel!r}")
        if rel in seen_paths:
            raise RuntimeError(f"part manifest references {rel!r} more than once")
        seen_paths.add(rel)
        if not isinstance(expected_sha, str) or not _HEX64.fullmatch(expected_sha):
            raise RuntimeError(f"part manifest entry {label!r} has invalid SHA-256")
        payload = load_bytes(rel)
        if payload is None:
            raise RuntimeError(f"part manifest references missing file {rel}")
        observed = sha256_bytes(payload)
        if observed != expected_sha:
            raise RuntimeError(
                f"part file {rel} hash mismatch: {observed[:12]} != {expected_sha[:12]}"
            )
    return True


def main() -> None:
    # Exercise the exact attempt rule before touching GPU/HF. A dedicated CPU test
    # imports the same function, so this is defense in depth rather than the only test.
    _self_test_attempt_selection()

    import numpy as np
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors.torch import save as st_save
    from transformers import AutoModelForCausalLM, AutoTokenizer

    api = HfApi(token=HF_TOKEN)
    store_dtype = getattr(torch, DTYPE_STORE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("no GPU visible")

    if os.path.exists(C80_ROLLOUTS):
        roll_path = C80_ROLLOUTS
        print(f"using local rollouts: {roll_path}")
    else:
        roll_path = hf_hub_download(
            HF_DATA_REPO, C80_ROLLOUTS, repo_type="dataset", token=HF_TOKEN
        )

    raw_rows: list[dict[str, Any]] = []
    with open(roll_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_rows.append(json.loads(line))
    print(f"{BLOCK}: {len(raw_rows)} attempt rows")

    rows = select_authoritative_attempts(raw_rows)
    n_valid = sum(r.get("technical_validity") == "valid" for r in rows)
    n_failed_provenance = len(rows) - n_valid
    print(
        f"  selected {len(rows)} authoritative rollouts: {n_valid} valid; "
        f"{n_failed_provenance} highest-retry failed provenance rows"
    )

    config_shas = {r.get("config_sha256") for r in rows}
    if len(config_shas) != 1 or None in config_shas:
        raise RuntimeError(f"selected rollouts carry invalid/multiple config hashes: {config_shas}")
    config_sha256 = config_shas.pop()
    blocks_seen = {r.get("block") for r in rows}
    if blocks_seen != {BLOCK}:
        raise RuntimeError(f"expected only block {BLOCK}, found {blocks_seen}")

    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REV, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    layers = model.model.layers
    n_layers = len(layers)
    hidden = model.config.hidden_size
    if n_layers != 32 or not 0 <= MIDDLE_BLOCK < n_layers:
        raise RuntimeError("model block layout does not match frozen 32-block contract")

    captured: list[Any] = [None] * n_layers

    def make_hook(i: int):
        def hook(_m, _in, out):
            captured[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    for i, layer in enumerate(layers):
        layer.register_forward_hook(make_hook(i))

    seg_has_reasoning = {"balanced_reasoning", "empty_answer_after_close"}
    seg_has_answer = {"balanced_reasoning", "direct_answer"}

    def spans(row: dict[str, Any]):
        rs = int(row["response_start"])
        re_ = int(row["response_end_exclusive"])
        trc = row.get("token_region_counts") or {}
        rc = int(trc.get("reasoning", 0))
        ac = int(trc.get("final_answer", 0))
        seg = row.get("segmentation_case")
        valid = row.get("technical_validity") == "valid"
        all_span = (rs, re_)
        has_all = valid and re_ > rs
        reasoning = (rs, rs + rc) if rc > 0 and seg in seg_has_reasoning else None
        if seg == "direct_answer":
            answer = (rs, re_)
        elif ac > 0 and seg in seg_has_answer:
            answer = (re_ - ac, re_)
        else:
            answer = None
        return all_span, reasoning, answer, has_all

    @torch.no_grad()
    def extract(row: dict[str, Any]):
        ids = list(row["prompt_token_ids"]) + list(row["output_token_ids"])
        all_span, reasoning_span, answer_span, has_all = spans(row)
        if not has_all:
            return None, {
                "has_all_response_pool": False,
                "has_reasoning_pool": False,
                "has_answer_pool": False,
            }
        a0, a1 = all_span
        if not 0 <= a0 < a1 <= len(ids):
            raise RuntimeError(f"span {all_span} outside sequence length {len(ids)}")
        inp = torch.tensor([ids], device=device)
        model(inp, use_cache=False)
        blocks = torch.stack([captured[i][0] for i in range(n_layers)])

        def pool(span, layer=None):
            s, e = span
            sl = blocks[:, s:e, :] if layer is None else blocks[layer, s:e, :]
            return sl.mean(dim=-2).to(store_dtype).cpu()

        out = {"all_response": pool(all_span)}
        if answer_span:
            out["answer"] = pool(answer_span)
        if reasoning_span:
            out["reasoning_middle"] = pool(reasoning_span, layer=MIDDLE_BLOCK)
        return out, {
            "has_all_response_pool": True,
            "has_answer_pool": answer_span is not None,
            "has_reasoning_pool": reasoning_span is not None,
        }

    def meta_for(row: dict[str, Any], avail: dict[str, bool]) -> dict[str, Any]:
        return {
            "uid": row["rollout_id"],
            "role": row.get("role_id"),
            "question_id": row.get("question_id"),
            "prompt_index": row.get("prompt_index"),
            "default_condition_index": row.get("default_condition_index"),
            "condition": row.get("arm"),
            "block": row.get("block"),
            "validity": row.get("technical_validity"),
            "segmentation_case": row.get("segmentation_case"),
            "finish_reason": row.get("finish_reason"),
            "n_output_tokens": row.get("n_output_tokens"),
            "token_region_counts": row.get("token_region_counts"),
            "selected_retry": row.get("retry"),
            **avail,
        }

    local_out = Path("./t12_c80_" + BLOCK.replace("-", "_"))
    local_out.mkdir(parents=True, exist_ok=True)
    existing = set(api.list_repo_files(HF_OUT_REPO, repo_type="dataset"))
    parts = [rows[i : i + PART_SIZE] for i in range(0, len(rows), PART_SIZE)]
    print(f"{len(parts)} parts of up to {PART_SIZE}; staging to {local_out}")

    def load_part_bytes(rel: str) -> bytes | None:
        local_path = local_out / rel
        if local_path.exists():
            return local_path.read_bytes()
        remote = f"{HF_OUT_PREFIX}/{rel}"
        if remote not in existing:
            return None
        downloaded = hf_hub_download(
            HF_OUT_REPO, remote, repo_type="dataset", token=HF_TOKEN
        )
        return Path(downloaded).read_bytes()

    def already_done(idx: int) -> bool:
        suffix = f"{idx:04d}"
        manifest_name = f"part_manifest_{suffix}.json"
        payload = load_part_bytes(manifest_name)
        if payload is None:
            # Existence of metadata alone is deliberately NOT enough to skip.
            return False
        try:
            doc = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"cannot parse existing {manifest_name}: {exc}") from exc
        verify_part_manifest_files(doc, load_part_bytes, expected_part_index=idx)
        return True

    def flush_to_hf(msg: str) -> None:
        for attempt in range(10):
            try:
                api.upload_folder(
                    folder_path=str(local_out),
                    path_in_repo=HF_OUT_PREFIX,
                    repo_id=HF_OUT_REPO,
                    repo_type="dataset",
                    commit_message=msg,
                )
                return
            except Exception as exc:
                if "429" in str(exc) and attempt < 9:
                    print("  HF commit rate-limited; sleeping 120s ...")
                    time.sleep(120)
                    continue
                raise

    def run_part(idx: int, part: list[dict[str, Any]], smoke: bool = False) -> None:
        suffix = f"{idx:04d}"
        if not smoke and already_done(idx):
            print(f"  part {suffix} verified complete — skip")
            return
        all_t: dict[str, Any] = {}
        ans_t: dict[str, Any] = {}
        rea_t: dict[str, Any] = {}
        metas: list[dict[str, Any]] = []
        for row in part:
            pools, avail = extract(row)
            uid = row["rollout_id"]
            if pools:
                all_t[uid] = pools["all_response"]
                if "answer" in pools:
                    ans_t[uid] = pools["answer"]
                if "reasoning_middle" in pools:
                    rea_t[uid] = pools["reasoning_middle"]
            metas.append(meta_for(row, avail))

        if smoke:
            print(
                f"  SMOKE: {len(all_t)} all / {len(ans_t)} answer / "
                f"{len(rea_t)} reasoning tensors from {len(part)} rows"
            )
            if not all_t:
                raise RuntimeError("smoke produced no all-response tensors")
            if not all(t.shape == (n_layers, hidden) for t in all_t.values()):
                raise RuntimeError("bad all-response tensor shape")
            if not all(t.shape == (n_layers, hidden) for t in ans_t.values()):
                raise RuntimeError("bad answer tensor shape")
            if not all(t.shape == (hidden,) for t in rea_t.values()):
                raise RuntimeError("bad reasoning tensor shape")
            if not all(torch.isfinite(t).all() for t in all_t.values()):
                raise RuntimeError("non-finite activation in smoke")
            return

        file_manifest: dict[str, dict[str, Any]] = {}
        for label, store in (
            ("all_response", all_t),
            ("answer", ans_t),
            ("reasoning_middle", rea_t),
        ):
            if not store:
                continue
            data = st_save({key: value.contiguous() for key, value in store.items()})
            name = f"{label}_part{suffix}.safetensors"
            (local_out / name).write_bytes(data)
            file_manifest[label] = {
                "path": name,
                "sha256": sha256_bytes(data),
                "n_tensors": len(store),
            }

        meta_bytes = (
            "\n".join(json.dumps(m, ensure_ascii=False) for m in metas) + "\n"
        ).encode("utf-8")
        meta_name = f"meta_part{suffix}.jsonl"
        (local_out / meta_name).write_bytes(meta_bytes)
        file_manifest["metadata"] = {
            "path": meta_name,
            "sha256": sha256_bytes(meta_bytes),
            "n_rows": len(metas),
        }
        part_doc = {
            "schema_version": "t12-c80-part-manifest/2.0",
            "part_index": idx,
            "source_rollout_ids_sha256": hashlib.sha256(
                json.dumps(
                    [r["rollout_id"] for r in part],
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
            "files": file_manifest,
        }
        manifest_path = local_out / f"part_manifest_{suffix}.json"
        manifest_path.write_text(
            json.dumps(part_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Verify the just-written part through the same path used by resume.
        verify_part_manifest_files(
            part_doc,
            lambda rel: (local_out / rel).read_bytes() if (local_out / rel).exists() else None,
            expected_part_index=idx,
        )
        print(
            f"  part {suffix}: {len(all_t)} all / {len(ans_t)} answer / "
            f"{len(rea_t)} reasoning"
        )

    print(f"SMOKE gate on {SMOKE_N} selected rows ...")
    run_part(0, rows[:SMOKE_N], smoke=True)
    print("smoke OK — starting full run\n")

    t0 = time.time()
    for idx, part in enumerate(parts):
        run_part(idx, part)
        if (idx + 1) % FLUSH_EVERY == 0 or (idx + 1) == len(parts):
            flush_to_hf(f"T12 {BLOCK} activations through part {idx:04d}")
            print(
                f"  flushed -> HF [{min((idx + 1) * PART_SIZE, len(rows))}/{len(rows)}] "
                f"{time.time() - t0:.0f}s"
            )

    top = {
        "format_version": "t12-c80-activation-v2",
        "block": BLOCK,
        "model_id": MODEL,
        "model_revision": MODEL_REV,
        "hidden_size": hidden,
        "n_layers": n_layers,
        "middle_block_index": MIDDLE_BLOCK,
        "config_sha256": config_sha256,
        "extractor": EXTRACTOR_VERSION,
        "activation_site": "model.model.layers[L] output (forward hook, before final norm)",
        "source": {
            "path": C80_ROLLOUTS,
            "sha256": sha256_file(roll_path),
            "n_attempt_rows": len(raw_rows),
            "n_selected_rollouts": len(rows),
            "n_selected_valid": n_valid,
            "n_selected_failed_provenance_only": n_failed_provenance,
        },
        "attempt_selection": {
            "rule": "lowest-retry valid; if none valid, highest-retry failed provenance row",
            "failed_selected_rows_are_excluded_from_pooling": True,
        },
        "resume": {
            "skip_requires_part_manifest": True,
            "skip_requires_every_referenced_file_hash": True,
        },
        "pools": {
            "all_response": "all 32 layers",
            "answer": "all 32 layers",
            "reasoning_middle": f"block {MIDDLE_BLOCK} only",
        },
        "note": (
            "Pooled over stored native token spans; no retokenization. Parts are "
            "resumed only after manifest + all referenced hashes verify."
        ),
    }
    (local_out / "manifest.json").write_text(
        json.dumps(top, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    flush_to_hf(f"T12 {BLOCK} activations complete")
    print(
        "\nDONE ->",
        f"https://huggingface.co/datasets/{HF_OUT_REPO}/tree/main/{HF_OUT_PREFIX}",
    )


if __name__ == "__main__":
    main()
