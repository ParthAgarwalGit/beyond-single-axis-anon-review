# ================================================================================
# T11 · canonical default (Assistant-baseline) generation → Hugging Face
# --------------------------------------------------------------------------------
# Reproducibility helper for the five frozen default conditions x one C80 block.
# The submission-critical production defaults already exist; this helper is for a
# clean rerun only. It fails closed against the current canonical T11 contract and
# refuses to treat legacy/non-canonical HF rows as completed work.
# ================================================================================

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

HF_TOKEN = ""
BLOCK = "C80-B"  # C80-A or C80-B
C80_ROLLOUTS = "results/generation/c80_role_C80_B.final.jsonl"
HF_DATA_REPO = "[Author-A-HF]/persona-artifacts"
HF_OUT_REPO = "[Author-A-HF]/persona-artifacts"
# Keep reproducibility reruns separate from the already-analysed production rows.
HF_OUT_PREFIX = "t11-defaults-repro/C80-B"

# Fill with the reviewed repository commit whose contract you are executing.
# A real rerun refuses to proceed without a 40-hex source commit.
EXECUTION_GIT_SHA = ""

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_REV = "6a6f4aa4197940add57724a7707d069478df56b1"
TOKENIZER_REV = MODEL_REV
CHAT_TEMPLATE_SHA256 = "56a1447ad31926fdc21fb07e56e5642bd9c850c4f52d8c8af7bbe5f079a84f5f"
ACTIVE_V4_CONFIG_SHA256 = "f7e7dd9df8c7c72d5da08bf8201295dfc91f3208837d8cd0867054c393984e59"
MATCHED_C80_V3_CONFIG_SHA256 = "73be73df4640c2d32bfbc8b6009741c0fadd8c466acb703d1f99df35d5c8cd79"
MODEL_DISPLAY = "DeepSeek-R1"
MAX_NEW = 8192
SMOKE_N = 5
SEP = "\n\n"
OPEN = "<think>"
CLOSE = "</think>"
CONTRACT_VERSION = "t11-gpu-cell-canonical-v2"

DEFAULT_CONDITIONS = (
    (0, ""),
    (1, "You are an AI assistant."),
    (2, "You are a large language model."),
    (3, f"You are {MODEL_DISPLAY}."),
    (4, "Respond as yourself."),
)

FROZEN_QUESTION_IDS = {
    "C80-A": (
        1, 3, 4, 5, 6, 7, 9, 13, 16, 22, 26, 28, 30, 33, 37, 39, 42, 44,
        47, 49, 53, 59, 60, 62, 64, 67, 68, 69, 72, 76, 78, 80, 86, 89, 94,
        100, 105, 116, 119, 121, 123, 125, 128, 129, 130, 131, 138, 141, 144,
        145, 154, 155, 157, 160, 164, 165, 167, 171, 172, 174, 176, 178, 185,
        195, 200, 201, 205, 206, 208, 209, 214, 215, 216, 217, 222, 224, 226,
        227, 231, 237,
    ),
    "C80-B": (
        0, 10, 12, 14, 21, 24, 31, 32, 35, 40, 45, 46, 52, 54, 56, 58, 63,
        65, 66, 70, 73, 79, 83, 84, 88, 90, 96, 98, 101, 102, 104, 107, 108,
        109, 110, 112, 114, 115, 118, 124, 132, 133, 139, 143, 146, 148, 151,
        152, 156, 159, 161, 163, 168, 169, 170, 173, 177, 179, 181, 184, 187,
        192, 194, 196, 197, 198, 203, 204, 207, 210, 213, 220, 221, 223, 225,
        228, 229, 232, 233, 235,
    ),
}
FROZEN_ID_LIST_SHA256 = {
    "C80-A": "2be301226b19beb92ae2ca9a03b40c38d9ac05af9f3c5382e7ea4d5b19fddba7",
    "C80-B": "9dd34b57ad67c678b33a55ce8c8d348208d504775665d31bd4edc6f30c219a8d",
}
FROZEN_C80_SOURCE = {
    "C80-A": {
        "sha256": "c7229c926066271eb99bc6b647ee8a7446e67071d0b60d5c6592d42566f1238c",
        "n_rows": 22286,
    },
    "C80-B": {
        "sha256": "2d82dbe1ef2ab49068d8f2ad5f544db2da4bd3e1d7d44ec89415d0b8bb84c1b8",
        "n_rows": 22735,
    },
}

_HEX16 = re.compile(r"^[0-9a-f]{16}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def messages_sha256(messages: list[dict[str, str]]) -> str:
    return text_sha256(
        json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


def _digest(kind: str, fields: dict[str, Any]) -> str:
    payload = json.dumps(
        {"kind": kind, **fields},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def default_rollout_id(model: str, question_id: int, condition_index: int) -> str:
    """Exact parity with current src.ids.rollout_id for a DEFAULT row."""
    return _digest(
        "rollout",
        {
            "model": model,
            "arm": "DEFAULT",
            "role_id": "",
            "question_id": int(question_id),
            "prompt_index": int(condition_index),
        },
    )


def generation_row_id(rollout_id: str, retry: int = 0) -> str:
    return _digest("generation", {"rollout_id": rollout_id, "retry": int(retry)})


def _validate_static_contract(block: str) -> None:
    if block not in FROZEN_QUESTION_IDS:
        raise RuntimeError(f"BLOCK must be C80-A or C80-B, got {block!r}")
    qids = FROZEN_QUESTION_IDS[block]
    if len(qids) != 80 or len(set(qids)) != 80:
        raise RuntimeError(f"embedded {block} question inventory is not 80 unique IDs")
    if canonical_sha256(list(qids)) != FROZEN_ID_LIST_SHA256[block]:
        raise RuntimeError(f"embedded {block} question-ID contract hash drifted")
    if not _HEX40.fullmatch(EXECUTION_GIT_SHA):
        raise RuntimeError(
            "EXECUTION_GIT_SHA must be filled with the reviewed 40-hex source commit "
            "before a real T11 rerun; refusing unbound execution provenance"
        )
    if not _HEX64.fullmatch(ACTIVE_V4_CONFIG_SHA256):
        raise RuntimeError("invalid active V4 config SHA")


def _common_suffix(strings: list[str]) -> str:
    if not strings:
        raise ValueError("cannot recover a question from an empty prompt set")
    suffix = strings[0]
    for other in strings[1:]:
        k = 0
        lim = min(len(suffix), len(other))
        while k < lim and suffix[-1 - k] == other[-1 - k]:
            k += 1
        suffix = suffix[len(suffix) - k :] if k else ""
        if not suffix:
            break
    return suffix


def recover_questions_from_frozen_c80(rows: list[dict[str, Any]], block: str) -> dict[int, str]:
    """Recover question text only after the whole C80 artifact is hash-bound."""
    expected = set(FROZEN_QUESTION_IDS[block])
    by_qid: dict[int, list[str]] = collections.defaultdict(list)
    for row in rows:
        if row.get("block") != block or row.get("arm") != "USER_TRANSLATED_LU":
            raise RuntimeError("C80 source contains a foreign block/arm row")
        if row.get("config_sha256") != MATCHED_C80_V3_CONFIG_SHA256:
            raise RuntimeError("C80 source row is not stamped with the frozen historical V3 hash")
        if row.get("model") != MODEL or row.get("model_revision") != MODEL_REV:
            raise RuntimeError("C80 source row model/revision drift")
        qid = row.get("question_id")
        if qid not in expected:
            raise RuntimeError(f"C80 source contains question {qid!r} outside frozen {block}")
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 1:
            raise RuntimeError("translated C80 source row must carry exactly one user message")
        msg = messages[0]
        if msg.get("role") != "user" or not isinstance(msg.get("content"), str):
            raise RuntimeError("translated C80 source message does not match the frozen user arm")
        by_qid[int(qid)].append(msg["content"])

    if set(by_qid) != expected:
        missing = sorted(expected - set(by_qid))
        raise RuntimeError(f"C80 source does not cover the exact frozen question set; missing={missing}")

    questions: dict[int, str] = {}
    for qid in FROZEN_QUESTION_IDS[block]:
        suffix = _common_suffix(by_qid[qid]).strip()
        if not suffix:
            raise RuntimeError(f"question recovery failed for {qid}")
        if not all(content.endswith(suffix) for content in by_qid[qid]):
            raise RuntimeError(f"question recovery is not a common suffix for {qid}")
        questions[qid] = suffix
    return questions


def default_messages(condition_index: int, question: str) -> list[dict[str, str]]:
    spec = dict(DEFAULT_CONDITIONS)
    if condition_index not in spec:
        raise ValueError(f"unknown default condition {condition_index}")
    text = spec[condition_index]
    if not text:
        return [{"role": "user", "content": question}]
    return [{"role": "user", "content": f"{text}{SEP}{question}"}]


def _content_token_ids(out_ids: list[int], terminal_ids: set[int]) -> list[int]:
    content = list(out_ids)
    while content and content[-1] in terminal_ids:
        content.pop()
    return content


def _degenerate_repetition(token_ids: list[int], ngram: int = 8, min_repeats: int = 6) -> bool:
    block = ngram * min_repeats
    if len(token_ids) < block:
        return False
    tail = token_ids[-block:]
    unit = tail[:ngram]
    return all(tail[i : i + ngram] == unit for i in range(0, block, ngram))


def segment_prefilled_open(
    tok: Any,
    prompt_ids: list[int],
    out_ids: list[int],
    output_text: str,
    finish_reason: str,
) -> dict[str, Any]:
    """Self-contained parity with current src.generation prefilled-open semantics."""
    terminal_ids = {x for x in (tok.eos_token_id, tok.pad_token_id) if x is not None}
    content = _content_token_ids(out_ids, terminal_ids)
    rs = len(prompt_ids)
    re_ = rs + len(content)

    if finish_reason == "error":
        return {
            "technical_validity": "generation_error",
            "segmentation_case": None,
            "response_start": rs,
            "response_end_exclusive": re_,
            "token_region_counts": {"all_response": len(content)},
            "role_judgeable": False,
        }
    if not content or not output_text.strip():
        return {
            "technical_validity": "empty_response",
            "segmentation_case": None,
            "response_start": rs,
            "response_end_exclusive": re_,
            "token_region_counts": {"all_response": len(content)},
            "role_judgeable": False,
        }
    if finish_reason == "length":
        return {
            "technical_validity": "truncated",
            "segmentation_case": None,
            "response_start": rs,
            "response_end_exclusive": re_,
            "token_region_counts": {"all_response": len(content)},
            "role_judgeable": False,
        }
    if _degenerate_repetition(content):
        return {
            "technical_validity": "degenerate_repetition",
            "segmentation_case": None,
            "response_start": rs,
            "response_end_exclusive": re_,
            "token_region_counts": {"all_response": len(content)},
            "role_judgeable": False,
        }

    opens = output_text.count(OPEN)
    closes = output_text.count(CLOSE)
    if opens != 0 or closes == 0:
        return {
            "technical_validity": "valid",
            "segmentation_case": "malformed_reasoning",
            "response_start": rs,
            "response_end_exclusive": re_,
            "token_region_counts": {"all_response": len(content)},
            "role_judgeable": False,
        }

    close_ids = tok.encode(CLOSE, add_special_tokens=False)
    if len(close_ids) == 1:
        close_id = close_ids[0]
        if content.count(close_id) != closes:
            return {
                "technical_validity": "token_alignment_failure",
                "segmentation_case": None,
                "response_start": rs,
                "response_end_exclusive": re_,
                "token_region_counts": {"all_response": len(content)},
                "role_judgeable": False,
            }
        idx = len(content) - 1 - content[::-1].index(close_id)
        reasoning_len = idx + 1
    else:
        boundary = output_text.rfind(CLOSE) + len(CLOSE)
        prefix_ids = tok.encode(output_text[:boundary], add_special_tokens=False)
        if prefix_ids != content[: len(prefix_ids)]:
            return {
                "technical_validity": "token_alignment_failure",
                "segmentation_case": None,
                "response_start": rs,
                "response_end_exclusive": re_,
                "token_region_counts": {"all_response": len(content)},
                "role_judgeable": False,
            }
        reasoning_len = len(prefix_ids)

    answer_text = output_text[output_text.rfind(CLOSE) + len(CLOSE) :]
    if not answer_text.strip():
        case = "empty_answer_after_close"
        answer_len = 0
    else:
        case = "balanced_reasoning"
        answer_len = len(content) - reasoning_len
    return {
        "technical_validity": "valid",
        "segmentation_case": case,
        "response_start": rs,
        "response_end_exclusive": re_,
        "token_region_counts": {
            "all_response": len(content),
            "reasoning": reasoning_len,
            "final_answer": answer_len,
        },
        "role_judgeable": case == "balanced_reasoning",
    }


def validate_canonical_default_row(
    row: dict[str, Any],
    question: str,
    condition_index: int,
    tok: Any,
    block: str,
) -> None:
    """Fail-closed subset of current main's GENERATION_ROW + cross-field contract."""
    required = {
        "row_id", "rollout_id", "model", "model_revision", "config_sha256", "git_sha",
        "arm", "channel", "default_condition_index", "role_id", "block", "question_id",
        "prompt_index", "retry", "messages", "rendered_prompt", "messages_sha256",
        "rendered_prompt_sha256", "tokenizer_revision", "chat_template_sha256",
        "runtime_settings", "prompt_token_ids", "output_token_ids", "n_prompt_tokens",
        "n_output_tokens", "response_start", "response_end_exclusive", "output_text",
        "finish_reason", "technical_validity", "segmentation_case",
    }
    missing = sorted(required - set(row))
    if missing:
        raise RuntimeError(f"default row missing canonical fields: {missing}")
    if not _HEX16.fullmatch(str(row["rollout_id"])) or not _HEX16.fullmatch(str(row["row_id"])):
        raise RuntimeError("default row carries a non-canonical 16-hex ID")
    expected_rollout = default_rollout_id(MODEL, int(row["question_id"]), condition_index)
    if row["rollout_id"] != expected_rollout:
        raise RuntimeError("default rollout_id does not match canonical src.ids semantics")
    if row["row_id"] != generation_row_id(expected_rollout, int(row["retry"])):
        raise RuntimeError("generation row_id does not match canonical retry-stable semantics")
    if row["model"] != MODEL or row["model_revision"] != MODEL_REV:
        raise RuntimeError("default row model/revision drift")
    if row["tokenizer_revision"] != TOKENIZER_REV or row["chat_template_sha256"] != CHAT_TEMPLATE_SHA256:
        raise RuntimeError("default row tokenizer/chat-template provenance drift")
    if row["config_sha256"] != ACTIVE_V4_CONFIG_SHA256:
        raise RuntimeError("new T11 rows must stamp the active V4 method config")
    if row["git_sha"] != EXECUTION_GIT_SHA:
        raise RuntimeError("default row is not bound to EXECUTION_GIT_SHA")
    if row["arm"] != "DEFAULT" or row["channel"] != "user" or row["role_id"] != "":
        raise RuntimeError("default row arm/channel/role_id violates the canonical contract")
    if row["block"] != block or int(row["question_id"]) not in FROZEN_QUESTION_IDS[block]:
        raise RuntimeError("default row block/question drift")
    if row["default_condition_index"] != condition_index or row["prompt_index"] != condition_index:
        raise RuntimeError("DEFAULT rows require prompt_index == default_condition_index")
    if int(row["retry"]) != 0:
        raise RuntimeError("this helper emits only retry 0; retries need an explicit rerun policy")

    expected_messages = default_messages(condition_index, question)
    if row["messages"] != expected_messages:
        raise RuntimeError("stored default messages do not reproduce from the frozen renderer")
    rendered = tok.apply_chat_template(expected_messages, tokenize=False, add_generation_prompt=True)
    if row["rendered_prompt"] != rendered:
        raise RuntimeError("stored rendered prompt differs from tokenizer.apply_chat_template")
    if row["messages_sha256"] != messages_sha256(expected_messages):
        raise RuntimeError("messages_sha256 mismatch")
    if row["rendered_prompt_sha256"] != text_sha256(rendered):
        raise RuntimeError("rendered_prompt_sha256 mismatch")
    expected_prompt_ids = tok.encode(rendered, add_special_tokens=False)
    if list(row["prompt_token_ids"]) != list(expected_prompt_ids):
        raise RuntimeError("stored prompt token IDs differ from canonical rendered-prompt tokenization")
    if row["n_prompt_tokens"] != len(row["prompt_token_ids"]):
        raise RuntimeError("n_prompt_tokens mismatch")
    if row["n_output_tokens"] != len(row["output_token_ids"]):
        raise RuntimeError("n_output_tokens mismatch")
    if row["response_start"] != len(row["prompt_token_ids"]):
        raise RuntimeError("response_start mismatch")
    if not row["response_start"] <= row["response_end_exclusive"] <= row["response_start"] + len(row["output_token_ids"]):
        raise RuntimeError("response_end_exclusive outside output range")
    if row["finish_reason"] not in {"stop", "length", "error"}:
        raise RuntimeError(f"unexpected finish_reason {row['finish_reason']!r}")
    validities = {
        "valid", "empty_response", "generation_error", "truncated",
        "serialization_failure", "token_alignment_failure", "degenerate_repetition",
    }
    if row["technical_validity"] not in validities:
        raise RuntimeError("technical_validity outside frozen set")
    cases = {None, "direct_answer", "balanced_reasoning", "malformed_reasoning", "empty_answer_after_close"}
    if row["segmentation_case"] not in cases:
        raise RuntimeError("segmentation_case outside frozen set")


def build_row(
    qid: int,
    condition_index: int,
    question: str,
    out: Any,
    tok: Any,
    block: str,
    source_sha256: str,
) -> dict[str, Any]:
    oo = out.outputs[0]
    messages = default_messages(condition_index, question)
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(rendered, add_special_tokens=False)
    if list(out.prompt_token_ids) != list(prompt_ids):
        raise RuntimeError(
            "vLLM chat prompt IDs differ from the frozen tokenizer rendering; refusing to persist"
        )
    output_ids = list(oo.token_ids)
    finish = oo.finish_reason if oo.finish_reason in {"stop", "length", "error"} else "error"
    seg = segment_prefilled_open(tok, list(prompt_ids), output_ids, oo.text, finish)
    rid = default_rollout_id(MODEL, qid, condition_index)
    runtime_settings = {
        "engine": "vLLM",
        "temperature": 0.0,
        "do_sample": False,
        "seed": 0,
        "max_new_tokens": MAX_NEW,
        "contract_version": CONTRACT_VERSION,
        "matched_c80_generation_config_sha256": MATCHED_C80_V3_CONFIG_SHA256,
    }
    row = {
        "row_id": generation_row_id(rid, 0),
        "rollout_id": rid,
        "model": MODEL,
        "model_revision": MODEL_REV,
        "config_sha256": ACTIVE_V4_CONFIG_SHA256,
        "git_sha": EXECUTION_GIT_SHA,
        "arm": "DEFAULT",
        "channel": "user",
        "default_condition_index": condition_index,
        "role_id": "",
        "block": block,
        "question_id": int(qid),
        "prompt_index": int(condition_index),
        "retry": 0,
        "messages": messages,
        "rendered_prompt": rendered,
        "messages_sha256": messages_sha256(messages),
        "rendered_prompt_sha256": text_sha256(rendered),
        "tokenizer_revision": TOKENIZER_REV,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "runtime_settings": runtime_settings,
        "prompt_token_ids": list(prompt_ids),
        "output_token_ids": output_ids,
        "n_prompt_tokens": len(prompt_ids),
        "n_output_tokens": len(output_ids),
        "output_text": oo.text,
        "finish_reason": finish,
        "question_content_sha256": text_sha256(question)[:16],
        "source_c80_artifact_sha256": source_sha256,
        "matched_c80_generation_config_sha256": MATCHED_C80_V3_CONFIG_SHA256,
        **seg,
    }
    validate_canonical_default_row(row, question, condition_index, tok, block)
    return row


def main() -> None:
    _validate_static_contract(BLOCK)

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=HF_TOKEN)
    roll_path = (
        C80_ROLLOUTS
        if os.path.exists(C80_ROLLOUTS)
        else hf_hub_download(HF_DATA_REPO, C80_ROLLOUTS, repo_type="dataset", token=HF_TOKEN)
    )
    observed_source_sha = sha256_file(roll_path)
    expected_source = FROZEN_C80_SOURCE[BLOCK]
    if observed_source_sha != expected_source["sha256"]:
        raise RuntimeError(
            f"{BLOCK} C80 source SHA mismatch: {observed_source_sha} != {expected_source['sha256']}"
        )

    rows = []
    with open(roll_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != expected_source["n_rows"]:
        raise RuntimeError(
            f"{BLOCK} C80 source row count {len(rows)} != committed {expected_source['n_rows']}"
        )
    questions = recover_questions_from_frozen_c80(rows, BLOCK)
    qids = list(FROZEN_QUESTION_IDS[BLOCK])
    print(f"{BLOCK}: exact frozen source verified; recovered {len(qids)} questions")

    import subprocess as _sp
    import sys as _sys

    _sp.run([_sys.executable, "-m", "pip", "install", "-q", "ninja"], check=False)
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        revision=MODEL_REV,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=MAX_NEW + 1024,
    )
    tok = llm.get_tokenizer()
    template = getattr(tok, "chat_template", None)
    if not isinstance(template, str):
        raise RuntimeError("tokenizer has no string chat_template")
    observed_template_sha = text_sha256(template)
    if observed_template_sha != CHAT_TEMPLATE_SHA256:
        raise RuntimeError(
            f"chat template SHA drift: {observed_template_sha} != {CHAT_TEMPLATE_SHA256}"
        )
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, seed=0)

    existing_files = set(api.list_repo_files(HF_OUT_REPO, repo_type="dataset"))
    out_name = f"{HF_OUT_PREFIX}/defaults.jsonl"
    existing_rows: list[dict[str, Any]] = []
    done: set[tuple[int, int]] = set()
    if out_name in existing_files:
        p = hf_hub_download(HF_OUT_REPO, out_name, repo_type="dataset", token=HF_TOKEN)
        with open(p, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = int(row.get("question_id", -1))
                ci = int(row.get("default_condition_index", -1))
                if qid not in questions or ci not in range(5):
                    raise RuntimeError(f"existing T11 row {line_no} outside frozen cells")
                validate_canonical_default_row(row, questions[qid], ci, tok, BLOCK)
                key = (qid, ci)
                if key in done:
                    raise RuntimeError(f"duplicate existing T11 cell {key}")
                done.add(key)
                existing_rows.append(row)

    all_cells = [(qid, ci, txt) for qid in qids for ci, txt in DEFAULT_CONDITIONS]
    if len(all_cells) != 400:
        raise RuntimeError("frozen block must contain exactly 80 x 5 = 400 default cells")
    work = [cell for cell in all_cells if (cell[0], cell[1]) not in done]
    print(f"{len(work)} canonical default rows to generate ({len(done)} validated existing)")

    def generate(batch: list[tuple[int, int, str]]) -> list[dict[str, Any]]:
        messages = [default_messages(ci, questions[qid]) for qid, ci, _ in batch]
        outs = llm.chat(messages, sp, use_tqdm=False)
        if len(outs) != len(batch):
            raise RuntimeError("vLLM returned a different number of outputs than requested")
        return [
            build_row(qid, ci, questions[qid], out, tok, BLOCK, observed_source_sha)
            for (qid, ci, _), out in zip(batch, outs)
        ]

    new_rows: list[dict[str, Any]] = []
    if work:
        smoke_batch = work[: min(SMOKE_N, len(work))]
        print(f"SMOKE: {len(smoke_batch)} rows")
        smoke_rows = generate(smoke_batch)
        for row in smoke_rows:
            print(
                "  ",
                {
                    k: row[k]
                    for k in (
                        "question_id",
                        "default_condition_index",
                        "rollout_id",
                        "segmentation_case",
                        "technical_validity",
                        "finish_reason",
                    )
                },
            )
        new_rows.extend(smoke_rows)
        if len(work) > len(smoke_batch):
            new_rows.extend(generate(work[len(smoke_batch) :]))
        print("smoke + full generation complete")

    combined = existing_rows + new_rows
    keys = {(int(r["question_id"]), int(r["default_condition_index"])) for r in combined}
    if len(combined) != len(keys):
        raise RuntimeError("combined defaults contain duplicate question/condition cells")
    expected_keys = {(qid, ci) for qid in qids for ci in range(5)}
    if keys != expected_keys:
        missing = sorted(expected_keys - keys)
        raise RuntimeError(f"combined defaults do not close the frozen 400-cell grid; missing={missing[:10]}")

    local = Path("./t11_defaults_" + BLOCK.replace("-", "_"))
    local.mkdir(parents=True, exist_ok=True)
    defaults_path = local / "defaults.jsonl"
    with defaults_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in sorted(combined, key=lambda r: (int(r["question_id"]), int(r["default_condition_index"]))):
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": "t11-gpu-cell-output/2.0",
        "task": "T11",
        "status": "REPRODUCIBILITY_RERUN_CANONICAL",
        "block": BLOCK,
        "model": MODEL,
        "model_revision": MODEL_REV,
        "tokenizer_revision": TOKENIZER_REV,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "execution_git_sha": EXECUTION_GIT_SHA,
        "execution_method_config_sha256": ACTIVE_V4_CONFIG_SHA256,
        "matched_c80_generation_config_sha256": MATCHED_C80_V3_CONFIG_SHA256,
        "source_c80_artifact_sha256": observed_source_sha,
        "source_c80_authoritative_n_rows": expected_source["n_rows"],
        "question_id_list_sha256": FROZEN_ID_LIST_SHA256[BLOCK],
        "n_questions": 80,
        "n_conditions": 5,
        "n_records": 400,
        "n_new_records": len(new_rows),
        "n_validated_existing_records": len(existing_rows),
        "decoding": {
            "engine": "vLLM",
            "temperature": 0.0,
            "do_sample": False,
            "seed": 0,
            "max_new_tokens": MAX_NEW,
        },
        "defaults_jsonl_sha256": sha256_file(defaults_path),
        "contract_version": CONTRACT_VERSION,
        "note": (
            "Canonical DEFAULT rows use prompt_index == default_condition_index, "
            "current V4 execution provenance, and a separately named historical "
            "C80 comparability hash. Existing rows are validated before skip."
        ),
    }
    with (local / "manifest.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    api.upload_folder(
        folder_path=str(local),
        path_in_repo=HF_OUT_PREFIX,
        repo_id=HF_OUT_REPO,
        repo_type="dataset",
        commit_message=f"T11 canonical reproducibility defaults {BLOCK}: {len(new_rows)} new rows",
    )
    print(
        f"DONE: {len(new_rows)} new + {len(existing_rows)} validated existing -> "
        f"https://huggingface.co/datasets/{HF_OUT_REPO}/tree/main/{HF_OUT_PREFIX}"
    )


if __name__ == "__main__":
    main()
