#!/usr/bin/env python3
"""Freeze one T10 C80 role-generation block manifest (prepare; do not launch).

T10 generates the translated-arm role-conditioned outputs for the two untouched
confirmatory blocks:

    275 roles x 160 question IDs (C80-A + C80-B) = 44,000 new outputs

split into two 80-question halves (22,000 each). This builder freezes ONE block
so the two halves use identical code and configuration and differ only in
``--block``. [Author A]'s half and [Author B]'s half are the two blocks.

Each role-question pair uses exactly one role prompt, at the question's frozen
positional prompt index (T08 prompt_assignment). So a block contributes:

    275 roles x 80 questions = 22,000 rollouts, balanced 16 per prompt index.

Deterministic rollout IDs come from src.ids.rollout_id; the builder renders no
model input and needs no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BLOCK_FILES = {
    "C80-A": "questions_C80_A.json",
    "C80-B": "questions_C80_B.json",
}
CONFIRMATORY_BLOCKS = ("C80-A", "C80-B")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def short_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


EXPECTED_BLOCK_SCHEMA = "t08-question-block/1.0"


def _load_render_hash_lookup(path: Path, arm: str) -> dict[tuple[str, int, int], str]:
    """Expected rendered-prompt hashes, keyed by (role_id, question_id,
    prompt_index), from a frozen T08 render-hash JSONL. Used to freeze the
    prompt hash the worker must reproduce before generation."""
    lookup: dict[tuple[str, int, int], str] = {}
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("arm") != arm:
                raise ValueError(f"render-hash line {line_no}: arm {row.get('arm')!r}")
            key = (row["role_id"], row["question_id"], row["prompt_index"])
            if key in lookup:
                raise ValueError(f"duplicate render-hash key {key}")
            lookup[key] = row["rendered_prompt_sha256"]
    return lookup


def build_manifest(root: Path, config_path: Path, block: str, out_path: Path,
                   strict_final: bool, design_dir: Path | None = None,
                   render_hashes: Path | None = None) -> dict[str, Any]:
    if block not in CONFIRMATORY_BLOCKS:
        raise ValueError(f"--block must be one of {CONFIRMATORY_BLOCKS}")
    sys.path.insert(0, str(root))
    import yaml

    from src.ids import rollout_id

    design_dir = design_dir or (root / "design")

    config_bytes = config_path.read_bytes()
    cfg = yaml.safe_load(config_bytes)
    config_sha = sha256_bytes(config_bytes)

    model = cfg["models"]["primary"]
    arm = "USER_TRANSLATED_LU"
    if cfg["prompt_rendering"]["conditions"][arm]["status"] != "PRIMARY_DEEPSEEK_EXTRACTION_ARM":
        raise ValueError("T10 uses the primary translated arm")

    if strict_final:
        if cfg.get("freeze_status") != "FROZEN":
            raise ValueError("strict-final requires freeze_status: FROZEN")
        for key in ("model_revision", "tokenizer_revision", "chat_template_sha256"):
            if not model.get(key):
                raise ValueError(f"strict-final requires pinned {key}")
        if render_hashes is None:
            raise ValueError(
                "strict-final requires --render-hashes (frozen T08 rendered-prompt "
                "hashes) so the worker can verify each prompt before generation"
            )

    # Frozen block questions.
    block_path = design_dir / BLOCK_FILES[block]
    block_payload = json.loads(block_path.read_text(encoding="utf-8"))
    # Schema compatibility of the T08 block artifact.
    if block_payload.get("schema_version") != EXPECTED_BLOCK_SCHEMA:
        raise ValueError(
            f"{block}: block schema {block_payload.get('schema_version')!r} "
            f"!= {EXPECTED_BLOCK_SCHEMA!r}"
        )
    if block_payload.get("block") != block:
        raise ValueError(f"{block}: block file declares {block_payload.get('block')!r}")
    if block_payload["method_config"]["sha256"] != config_sha:
        raise ValueError(
            f"{block}: built from a different method config "
            f"({block_payload['method_config']['sha256']} != {config_sha})"
        )
    if strict_final and block_payload.get("status") != "FROZEN":
        raise ValueError(
            f"strict-final requires the T08 {block} artifact to be FROZEN, "
            f"found {block_payload.get('status')!r}"
        )
    questions = block_payload["questions"]
    if len(questions) != 80:
        raise ValueError(f"{block}: expected 80 questions, found {len(questions)}")
    if len({q["question_id"] for q in questions}) != 80:
        raise ValueError(f"{block}: duplicate question_id in block file")

    # Role prompts (hash-verified), indexed by (role_id, prompt_index).
    rp_payload = json.loads(
        (root / "data" / "lu_et_al" / "role_prompts.json").read_text(encoding="utf-8")
    )
    prompts = rp_payload["prompts"]
    expected_roles = cfg["source_provenance"]["source_counts"]["roles"]
    prompts_per_role = cfg["source_provenance"]["source_counts"]["prompts_per_role"]
    expected_prompt_total = cfg["source_provenance"]["source_counts"]["role_prompts"]
    if len(prompts) != expected_prompt_total:
        raise ValueError(
            f"expected {expected_prompt_total} role prompts, found {len(prompts)}"
        )
    by_role_index: dict[tuple[str, int], dict[str, Any]] = {}
    per_role_indices: dict[str, set[int]] = {}
    for row in prompts:
        if short_sha256(row["text"]) != row["content_sha256"]:
            raise ValueError(f"role-prompt hash mismatch {row['prompt_uid']}")
        key = (row["role_id"], row["prompt_index"])
        if key in by_role_index:
            raise ValueError(f"duplicate role prompt {key}")
        by_role_index[key] = row
        per_role_indices.setdefault(row["role_id"], set()).add(row["prompt_index"])
    roles = set(per_role_indices)
    if len(roles) != expected_roles:
        raise ValueError(f"expected {expected_roles} roles, found {len(roles)}")
    expected_index_set = set(range(prompts_per_role))
    bad_roles = {
        r: sorted(idx) for r, idx in per_role_indices.items()
        if idx != expected_index_set
    }
    if bad_roles:
        raise ValueError(
            f"roles without exactly indices {sorted(expected_index_set)}: "
            f"{dict(list(bad_roles.items())[:3])}"
        )

    render_lookup = None
    if render_hashes is not None:
        render_lookup = _load_render_hash_lookup(render_hashes, arm)

    rollouts = []
    per_prompt_index = Counter()
    seen: set[str] = set()
    for role_id in sorted(roles):
        for q in questions:
            prompt_index = q["prompt_index"]
            key = (role_id, prompt_index)
            if key not in by_role_index:
                raise ValueError(f"role {role_id!r} has no prompt index {prompt_index}")
            prompt = by_role_index[key]
            rid = rollout_id(
                model["model_id"], arm, role_id, q["question_id"], prompt_index
            )
            if rid in seen:
                raise ValueError(f"duplicate rollout id {rid}")
            seen.add(rid)
            rollout = {
                "rollout_id": rid,
                "arm": arm,
                "channel": "user",
                "role_id": role_id,
                "block": block,
                "question_id": q["question_id"],
                "question_uid": q["question_uid"],
                "question_content_sha256": q["question_content_sha256"],
                "block_position": q["block_position"],
                "prompt_index": prompt_index,
                "role_prompt_uid": prompt["prompt_uid"],
                "role_prompt_content_sha256": prompt["content_sha256"],
            }
            if render_lookup is not None:
                rk = (role_id, q["question_id"], prompt_index)
                if rk not in render_lookup:
                    raise ValueError(f"render-hash manifest missing {rk}")
                rollout["expected_rendered_prompt_sha256"] = render_lookup[rk]
            rollouts.append(rollout)
            per_prompt_index[prompt_index] += 1

    expected = expected_roles * 80
    if len(rollouts) != expected:
        raise ValueError(f"expected {expected} rollouts, built {len(rollouts)}")
    if any(v != expected_roles * 16 for v in per_prompt_index.values()):
        raise ValueError(f"prompt-index imbalance: {dict(per_prompt_index)}")

    manifest = {
        "schema_version": "t10-c80-manifest/1.0",
        "task": "T10",
        "title": f"Translated-arm role-generation manifest for {block}",
        "status": "FROZEN" if strict_final else "DRAFT_PRE_RENDER",
        "block": block,
        "other_half_block": "C80-B" if block == "C80-A" else "C80-A",
        "arm": arm,
        "method_config": {"sha256": config_sha, "freeze_status": cfg.get("freeze_status")},
        "model": {
            "model_id": model["model_id"],
            "model_revision": model.get("model_revision"),
            "tokenizer_revision": model.get("tokenizer_revision"),
            "chat_template_sha256": model.get("chat_template_sha256"),
        },
        "identical_code_note": (
            "Both C80 halves are produced by this same builder and worker, "
            "differing only in --block, so the 44,000 IDs reconcile."
        ),
        "workload": {
            "n_roles": expected_roles,
            "n_questions": 80,
            "n_outputs": expected,
            "both_blocks_total": expected_roles * 160,
        },
        "counts": {
            "per_prompt_index": {str(k): v for k, v in sorted(per_prompt_index.items())},
        },
        "rendered_prompt_hash_freeze": {
            "status": "FROZEN" if render_lookup is not None else "PENDING",
            "source": str(render_hashes) if render_hashes is not None else None,
            "every_rollout_carries_expected_hash": render_lookup is not None,
        },
        "rollouts_sha256": canonical_sha256([r["rollout_id"] for r in rollouts]),
        "rollouts": rollouts,
    }
    write_json(out_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/method_frozen.yaml")
    parser.add_argument("--block", required=True, choices=list(CONFIRMATORY_BLOCKS))
    parser.add_argument("--out", default=None)
    parser.add_argument("--strict-final", action="store_true")
    parser.add_argument(
        "--render-hashes", default=None,
        help="Frozen T08 rendered-prompt hash JSONL for the translated arm; "
             "required for --strict-final so the worker can verify each prompt.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = args.out or f"design/c80_role_manifest_{args.block.replace('-', '_')}.json"
    manifest = build_manifest(
        root, root / args.config, args.block, root / out, args.strict_final,
        render_hashes=(root / args.render_hashes) if args.render_hashes else None,
    )
    print(json.dumps({
        "status": manifest["status"],
        "block": manifest["block"],
        "manifest": str(root / out),
        "manifest_sha256": sha256_file(root / out),
        "n_outputs": manifest["workload"]["n_outputs"],
        "both_blocks_total": manifest["workload"]["both_blocks_total"],
        "per_prompt_index": manifest["counts"]["per_prompt_index"],
        "rollouts_sha256": manifest["rollouts_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
