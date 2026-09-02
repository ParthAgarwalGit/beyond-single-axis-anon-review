#!/usr/bin/env python3
"""Freeze exact USER_TRANSLATED_LU rendered-prompt hashes before generation.

Produces one JSONL row for every role x question cell:
  275 roles x 240 questions = 66,000 rows.

This script imports the canonical T03 prompt renderer and deterministic-ID
functions. It refuses to run when model/tokenizer/chat-template provenance is
not pinned in configs/method_frozen.yaml.

The output stores hashes and lengths, not full prompt text, to keep the design
artifact compact. Generation must reproduce the stored hash before model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/method_frozen.yaml")
    parser.add_argument(
        "--questions-manifest", default="design/questions_240_manifest.json"
    )
    parser.add_argument(
        "--role-prompts", default="data/lu_et_al/role_prompts.json"
    )
    parser.add_argument(
        "--output",
        default="design/rendered_prompt_hashes_USER_TRANSLATED_LU.jsonl",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))

    from src.config import load_frozen_config
    from src.ids import rollout_id
    from src.prompt_rendering import (
        render_prompt,
        role_arm_messages,
        tokenize_rendered_prompt,
    )

    config_path = root / args.config
    raw_config = config_path.read_bytes()
    loaded_config = load_frozen_config()

    if (
        isinstance(loaded_config, tuple)
        and len(loaded_config) == 2
    ):
        cfg, loaded_sha = loaded_config
    else:
        cfg = loaded_config
        loaded_sha = hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest()
    config_sha = sha256_bytes(raw_config)
    if loaded_sha != config_sha:
        raise ValueError("canonical config loader returned a different hash")
    if cfg.get("freeze_status") != "FROZEN":
        raise ValueError("render freeze requires method freeze_status: FROZEN")

    model = cfg["models"]["primary"]
    required_runtime = ("model_revision", "tokenizer_revision", "chat_template_sha256")
    missing = [key for key in required_runtime if not model.get(key)]
    if missing:
        raise ValueError(
            "pin these primary-model fields before render freeze: "
            + ", ".join(missing)
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model["model_id"],
        revision=model["tokenizer_revision"],
        trust_remote_code=False,
    )
    actual_template = tokenizer.chat_template
    if not isinstance(actual_template, str) or not actual_template:
        raise ValueError("tokenizer has no chat template")
    actual_template_sha = hashlib.sha256(
        actual_template.encode("utf-8")
    ).hexdigest()
    if actual_template_sha != model["chat_template_sha256"]:
        raise ValueError(
            "chat-template hash mismatch: "
            f"{actual_template_sha} != {model['chat_template_sha256']}"
        )

    qmanifest = json.loads(
        (root / args.questions_manifest).read_text(encoding="utf-8")
    )
    if qmanifest["method_config"]["sha256"] != config_sha:
        raise ValueError("question manifest was built from another method config")

    questions = []
    for block_name, info in qmanifest["block_files"].items():
        block_path = root / info["path"]
        if hashlib.sha256(block_path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"block file hash mismatch: {block_path}")
        block = json.loads(block_path.read_text(encoding="utf-8"))
        questions.extend(block["questions"])
    questions.sort(
        key=lambda row: (
            ["E80", "C80-A", "C80-B"].index(row["question_block"]),
            row["block_position"],
        )
    )
    if len(questions) != 240:
        raise ValueError("expected 240 questions")

    rp_payload = json.loads(
        (root / args.role_prompts).read_text(encoding="utf-8")
    )
    prompts = rp_payload["prompts"]
    expected_prompt_count = cfg["source_provenance"]["source_counts"]["role_prompts"]
    if len(prompts) != expected_prompt_count:
        raise ValueError(
            f"expected {expected_prompt_count} role prompts, found {len(prompts)}"
        )

    prompt_by_role_index = {}
    roles = set()
    for row in prompts:
        key = (row["role_id"], row["prompt_index"])
        if key in prompt_by_role_index:
            raise ValueError(f"duplicate role prompt {key}")
        observed_text_sha = hashlib.sha256(
            row["text"].encode("utf-8")
        ).hexdigest()[:16]
        if observed_text_sha != row["content_sha256"]:
            raise ValueError(f"role prompt hash mismatch: {key}")
        prompt_by_role_index[key] = row
        roles.add(row["role_id"])

    expected_roles = cfg["source_provenance"]["source_counts"]["roles"]
    if len(roles) != expected_roles:
        raise ValueError(f"expected {expected_roles} roles, found {len(roles)}")

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    seen_rollouts = set()

    with output.open("w", encoding="utf-8", newline="\n") as fh:
        for role_id in sorted(roles):
            for q in questions:
                prompt_index = q["prompt_index"]
                prompt = prompt_by_role_index[(role_id, prompt_index)]
                messages = role_arm_messages(
                    cfg,
                    "USER_TRANSLATED_LU",
                    prompt["text"],
                    q["question_text"],
                )
                rendered = render_prompt(tokenizer, messages)
                prompt_ids = tokenize_rendered_prompt(tokenizer, rendered)
                rid = rollout_id(
                    model["model_id"],
                    "USER_TRANSLATED_LU",
                    role_id,
                    q["question_id"],
                    prompt_index,
                )
                if rid in seen_rollouts:
                    raise ValueError(f"duplicate rollout ID {rid}")
                seen_rollouts.add(rid)

                row = {
                    "rollout_id": rid,
                    "arm": "USER_TRANSLATED_LU",
                    "channel": "user",
                    "role_id": role_id,
                    "prompt_uid": prompt["prompt_uid"],
                    "role_prompt_content_sha256": prompt["content_sha256"],
                    "question_id": q["question_id"],
                    "question_uid": q["question_uid"],
                    "question_content_sha256": q["question_content_sha256"],
                    "question_block": q["question_block"],
                    "block_position": q["block_position"],
                    "prompt_index": prompt_index,
                    "messages_sha256": hashlib.sha256(
                        canonical_json_text(messages).encode("utf-8")
                    ).hexdigest(),
                    "rendered_prompt_sha256": hashlib.sha256(
                        rendered.encode("utf-8")
                    ).hexdigest(),
                    "rendered_prompt_character_count": len(rendered),
                    "rendered_prompt_token_count": len(prompt_ids),
                    "model_id": model["model_id"],
                    "model_revision": model["model_revision"],
                    "tokenizer_revision": model["tokenizer_revision"],
                    "chat_template_sha256": actual_template_sha,
                    "method_config_sha256": config_sha,
                }
                fh.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                n += 1

    expected_rows = expected_roles * 240
    if n != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, wrote {n}")

    print(json.dumps({
        "status": "PASS",
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "n_rows": n,
        "n_roles": expected_roles,
        "n_questions": 240,
        "arm": "USER_TRANSLATED_LU",
        "model_revision": model["model_revision"],
        "tokenizer_revision": model["tokenizer_revision"],
        "chat_template_sha256": actual_template_sha,
        "method_config_sha256": config_sha,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
