#!/usr/bin/env python3
"""Build and validate the T08 240-ID, three-block question manifest.

Required outputs:
  design/questions_E80.json
  design/questions_C80_A.json
  design/questions_C80_B.json
  design/questions_240_manifest.json

The block IDs and prompt-index rule are read from configs/method_frozen.yaml.
The script never invents or rebalances IDs. It fails if T01/T02 provenance,
duplicate handling, prompt balance, or hashes drift.

A question-only manifest may be created before tokenizer revisions are pinned.
Use --strict-final before generation. Strict-final requires a validated rendered
prompt hash manifest for all 275 x 240 translated-arm role/question cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

BLOCK_TO_FILENAME = {
    "E80": "questions_E80.json",
    "C80-A": "questions_C80_A.json",
    "C80-B": "questions_C80_B.json",
}


def manifest_path(path: Path) -> str:
    """Repo-relative POSIX path for anything recorded in a committed manifest.

    Committed provenance must be portable across machines and operating
    systems. Recording ``path.as_posix()`` alone only fixes the separator
    style - it silently commits an absolute, machine-local path (e.g. a
    scratch --out-dir) if the caller ever passes one. Resolve against ROOT
    and fail closed rather than commit something unusable elsewhere.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        raise ValueError(
            f"refusing to record a non-repo-relative path in committed provenance: "
            f"{resolved} is not under {ROOT}"
        )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def short_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_questions(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("questions")
    if not isinstance(rows, list):
        raise ValueError("questions artifact must contain a 'questions' list")

    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        qid = row.get("question_id")
        if not isinstance(qid, int):
            raise ValueError(f"invalid question_id: {qid!r}")
        if qid in by_id:
            raise ValueError(f"duplicate question_id in source artifact: {qid}")
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"question {qid}: missing text")
        observed = short_text_sha256(text)
        if row.get("content_sha256") != observed:
            raise ValueError(
                f"question {qid}: content hash mismatch "
                f"{row.get('content_sha256')} != {observed}"
            )
        by_id[qid] = row
    return payload, by_id


def validate_source(cfg: dict[str, Any], questions_path: Path,
                    source_payload: dict[str, Any],
                    questions: dict[int, dict[str, Any]]) -> None:
    provenance = cfg["source_provenance"]
    expected_hash = provenance["questions_artifact_sha256_repository"]
    observed_hash = sha256_file(questions_path)
    if observed_hash != expected_hash:
        raise ValueError(
            "T01 questions artifact hash mismatch: "
            f"expected {expected_hash}, observed {observed_hash}"
        )

    expected_n = provenance["source_counts"]["question_ids"]
    expected_unique = provenance["source_counts"]["unique_question_texts"]
    if len(questions) != expected_n:
        raise ValueError(f"expected {expected_n} questions, found {len(questions)}")
    if set(questions) != set(range(expected_n)):
        raise ValueError("question IDs must be exactly 0..239")

    unique_texts = len({row["text"] for row in questions.values()})
    if unique_texts != expected_unique:
        raise ValueError(
            f"expected {expected_unique} unique texts, found {unique_texts}"
        )

    source_commit = source_payload.get("source", {}).get("pinned_commit")
    if source_commit != provenance["lu_pinned_commit"]:
        raise ValueError(
            f"source commit mismatch: {source_commit} != "
            f"{provenance['lu_pinned_commit']}"
        )


def load_and_validate_render_manifest(
    path: Path,
    cfg: dict[str, Any],
    assignments: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_by_question = {
        row["question_id"]: (
            row["question_block"],
            row["block_position"],
            row["prompt_index"],
        )
        for row in assignments
    }
    expected_roles = cfg["source_provenance"]["source_counts"]["roles"]
    expected_rows = expected_roles * len(assignments)

    n = 0
    rollout_ids: set[str] = set()
    role_ids: set[str] = set()
    per_question = Counter()
    per_block = Counter()
    method_hashes = set()
    model_revisions = set()
    tokenizer_revisions = set()
    chat_template_hashes = set()

    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            required = {
                "rollout_id", "arm", "role_id", "question_id",
                "question_block", "block_position", "prompt_index",
                "messages_sha256", "rendered_prompt_sha256",
                "model_id", "model_revision", "tokenizer_revision",
                "chat_template_sha256", "method_config_sha256",
            }
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"render manifest line {line_no}: missing {sorted(missing)}"
                )
            if row["arm"] != "USER_TRANSLATED_LU":
                raise ValueError(
                    f"render manifest line {line_no}: unexpected arm {row['arm']}"
                )
            qid = row["question_id"]
            if qid not in expected_by_question:
                raise ValueError(
                    f"render manifest line {line_no}: unknown question {qid}"
                )
            expected = expected_by_question[qid]
            observed = (
                row["question_block"],
                row["block_position"],
                row["prompt_index"],
            )
            if observed != expected:
                raise ValueError(
                    f"render manifest line {line_no}: assignment drift "
                    f"{observed} != {expected}"
                )
            rid = row["rollout_id"]
            if rid in rollout_ids:
                raise ValueError(f"duplicate rollout_id: {rid}")
            rollout_ids.add(rid)
            role_ids.add(row["role_id"])
            per_question[qid] += 1
            per_block[row["question_block"]] += 1
            method_hashes.add(row["method_config_sha256"])
            model_revisions.add(row["model_revision"])
            tokenizer_revisions.add(row["tokenizer_revision"])
            chat_template_hashes.add(row["chat_template_sha256"])

    if n != expected_rows:
        raise ValueError(f"expected {expected_rows} render rows, found {n}")
    if len(role_ids) != expected_roles:
        raise ValueError(f"expected {expected_roles} roles, found {len(role_ids)}")
    bad_q = {qid: count for qid, count in per_question.items()
             if count != expected_roles}
    if bad_q:
        raise ValueError(
            f"each question must have {expected_roles} role rows; bad={bad_q}"
        )
    if len(method_hashes) != 1:
        raise ValueError("render manifest contains multiple method-config hashes")
    if len(model_revisions) != 1 or None in model_revisions:
        raise ValueError("render manifest must have one non-null model revision")
    if len(tokenizer_revisions) != 1 or None in tokenizer_revisions:
        raise ValueError("render manifest must have one non-null tokenizer revision")
    if len(chat_template_hashes) != 1 or None in chat_template_hashes:
        raise ValueError("render manifest must have one non-null chat-template hash")

    return {
        "status": "FROZEN",
        "path": manifest_path(path),
        "sha256": sha256_file(path),
        "n_rows": n,
        "n_roles": len(role_ids),
        "n_questions": len(per_question),
        "rows_by_block": dict(sorted(per_block.items())),
        "method_config_sha256": next(iter(method_hashes)),
        "model_revision": next(iter(model_revisions)),
        "tokenizer_revision": next(iter(tokenizer_revisions)),
        "chat_template_sha256": next(iter(chat_template_hashes)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/method_frozen.yaml")
    parser.add_argument("--questions", default="data/lu_et_al/questions.json")
    parser.add_argument("--out-dir", default="design")
    parser.add_argument(
        "--render-manifest",
        default=None,
        help="Optional translated-arm rendered-prompt hash JSONL.",
    )
    parser.add_argument(
        "--strict-final",
        action="store_true",
        help="Fail unless T02 is FROZEN and the complete render manifest is supplied.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    questions_path = Path(args.questions)
    out_dir = Path(args.out_dir)

    config_bytes = config_path.read_bytes()
    cfg = yaml.safe_load(config_bytes)
    config_sha = sha256_bytes(config_bytes)

    if args.strict_final and cfg.get("freeze_status") != "FROZEN":
        raise ValueError(
            "strict-final requires freeze_status: FROZEN in method_frozen.yaml"
        )

    source_payload, questions = load_questions(questions_path)
    validate_source(cfg, questions_path, source_payload, questions)

    design = cfg["question_design"]
    blocks_cfg = design["blocks"]
    expected_names = {"E80", "C80-A", "C80-B"}
    if set(blocks_cfg) != expected_names:
        raise ValueError(f"expected blocks {sorted(expected_names)}")

    prompt_count = cfg["source_provenance"]["source_counts"]["prompts_per_role"]
    all_ids: list[int] = []
    assignments: list[dict[str, Any]] = []
    block_payloads: dict[str, dict[str, Any]] = {}

    for block_name in ("E80", "C80-A", "C80-B"):
        ids = list(blocks_cfg[block_name]["question_ids"])
        if len(ids) != 80 or len(set(ids)) != 80:
            raise ValueError(f"{block_name}: expected 80 unique IDs")

        expected_id_hash = blocks_cfg[block_name]["id_list_sha256"]
        observed_id_hash = canonical_sha256(ids)
        if observed_id_hash != expected_id_hash:
            raise ValueError(
                f"{block_name}: ID-list hash drift "
                f"{observed_id_hash} != {expected_id_hash}"
            )

        rows = []
        prompt_counts = Counter()
        for position, qid in enumerate(ids):
            if qid not in questions:
                raise ValueError(f"{block_name}: missing source question {qid}")
            prompt_index = position % prompt_count
            prompt_counts[prompt_index] += 1
            source = questions[qid]
            row = {
                "question_block": block_name,
                "block_position": position,
                "question_id": qid,
                "question_uid": source["question_uid"],
                "question_text": source["text"],
                "question_content_sha256": source["content_sha256"],
                "question_character_count": len(source["text"]),
                "question_word_count": len(source["text"].split()),
                "prompt_index": prompt_index,
            }
            rows.append(row)
            assignments.append({
                "question_block": block_name,
                "block_position": position,
                "question_id": qid,
                "prompt_index": prompt_index,
            })

        expected_per_prompt = len(ids) // prompt_count
        if prompt_counts != Counter(
            {i: expected_per_prompt for i in range(prompt_count)}
        ):
            raise ValueError(
                f"{block_name}: prompt-index imbalance {dict(prompt_counts)}"
            )

        block_payloads[block_name] = {
            "schema_version": "t08-question-block/1.0",
            "task": "T08",
            "status": (
                "FROZEN" if cfg.get("freeze_status") == "FROZEN"
                else "REVIEW_CANDIDATE"
            ),
            "block": block_name,
            "block_role": blocks_cfg[block_name]["role"],
            "method_config": {
                "path": manifest_path(config_path),
                "sha256": config_sha,
                "freeze_status": cfg.get("freeze_status"),
                "schema_version": cfg.get("schema_version"),
            },
            "source_questions": {
                "path": manifest_path(questions_path),
                "sha256": sha256_file(questions_path),
                "lu_repository": cfg["source_provenance"]["lu_repository"],
                "lu_pinned_commit": cfg["source_provenance"]["lu_pinned_commit"],
            },
            "prompt_assignment": {
                "rule": cfg["prompt_assignment"]["rule"],
                "prompt_count": prompt_count,
                "counts": {
                    str(k): int(v) for k, v in sorted(prompt_counts.items())
                },
            },
            "summary": {
                "n_question_ids": len(rows),
                "n_unique_question_texts": len(
                    {row["question_content_sha256"] for row in rows}
                ),
                "character_total": sum(
                    row["question_character_count"] for row in rows
                ),
                "character_mean": round(
                    sum(row["question_character_count"] for row in rows) / len(rows),
                    4,
                ),
                "word_mean": round(
                    sum(row["question_word_count"] for row in rows) / len(rows),
                    4,
                ),
                "id_list_sha256": observed_id_hash,
                "assignment_sha256": canonical_sha256(assignments[-len(rows):]),
            },
            "questions": rows,
        }
        all_ids.extend(ids)

    if len(all_ids) != 240 or len(set(all_ids)) != 240:
        raise ValueError("three blocks must contain 240 disjoint question IDs")
    if set(all_ids) != set(range(240)):
        raise ValueError("three blocks must partition IDs 0..239")

    duplicate_groups = design["duplicate_text_groups"]
    for group in duplicate_groups:
        pair = group["question_ids"]
        required_block = group["required_common_block"]
        required_ids = set(blocks_cfg[required_block]["question_ids"])
        if not set(pair) <= required_ids:
            raise ValueError(
                f"duplicate pair {pair} must share {required_block}"
            )
        texts = {questions[qid]["text"] for qid in pair}
        if len(texts) != 1:
            raise ValueError(f"declared duplicate pair {pair} no longer matches")

    assignment_hash = canonical_sha256(assignments)
    expected_assignment_hash = design["all_blocks_assignment_sha256"]
    if assignment_hash != expected_assignment_hash:
        raise ValueError(
            "full assignment hash drift: "
            f"{assignment_hash} != {expected_assignment_hash}"
        )

    prompt_counts_240 = Counter(row["prompt_index"] for row in assignments)
    expected_240 = 240 // prompt_count
    if prompt_counts_240 != Counter(
        {i: expected_240 for i in range(prompt_count)}
    ):
        raise ValueError(
            f"240-ID prompt-index imbalance {dict(prompt_counts_240)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    block_file_info = {}
    for block_name, payload in block_payloads.items():
        path = out_dir / BLOCK_TO_FILENAME[block_name]
        write_json(path, payload)
        block_file_info[block_name] = {
            "path": manifest_path(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    render_info = {
        "status": "PENDING",
        "required_before_generation": True,
        "reason": (
            "Exact rendered-prompt hashes require a pinned model revision, "
            "tokenizer revision, chat-template hash, and the canonical prompt renderer."
        ),
    }
    if args.render_manifest:
        render_info = load_and_validate_render_manifest(
            Path(args.render_manifest), cfg, assignments
        )

    if args.strict_final and render_info["status"] != "FROZEN":
        raise ValueError(
            "strict-final requires a complete validated rendered-prompt manifest"
        )

    manifest = {
        "schema_version": "t08-question-manifest/1.0",
        "task": "T08",
        "title": "Frozen 240-ID, three-block extraction manifest",
        "status": (
            "FROZEN"
            if cfg.get("freeze_status") == "FROZEN"
            and render_info["status"] == "FROZEN"
            else "DRAFT_PRE_RENDER"
        ),
        "method_config": {
            "path": manifest_path(config_path),
            "sha256": config_sha,
            "freeze_status": cfg.get("freeze_status"),
            "schema_version": cfg.get("schema_version"),
        },
        "source_questions": {
            "path": manifest_path(questions_path),
            "sha256": sha256_file(questions_path),
            "n_question_ids": len(questions),
            "n_unique_question_texts": len(
                {row["text"] for row in questions.values()}
            ),
            "lu_repository": cfg["source_provenance"]["lu_repository"],
            "lu_pinned_commit": cfg["source_provenance"]["lu_pinned_commit"],
        },
        "partition": {
            "blocks": ["E80", "C80-A", "C80-B"],
            "n_ids_per_block": 80,
            "n_ids_total": 240,
            "union_is_0_to_239": True,
            "blocks_are_disjoint": True,
            "duplicate_text_groups": duplicate_groups,
            "duplicate_removed_sensitivity": {
                "drop_question_id": 227,
                "keep_question_id": 7,
            },
        },
        "prompt_assignment": {
            "rule": cfg["prompt_assignment"]["rule"],
            "same_mapping_for_every_role_model_and_arm": cfg[
                "prompt_assignment"
            ]["same_mapping_for_every_role_model_and_arm"],
            "prompt_count": prompt_count,
            "counts_per_block": {
                block: block_payloads[block]["prompt_assignment"]["counts"]
                for block in ("E80", "C80-A", "C80-B")
            },
            "counts_across_240": {
                str(k): int(v) for k, v in sorted(prompt_counts_240.items())
            },
            "assignment_sha256": assignment_hash,
        },
        "block_files": block_file_info,
        "rendered_prompt_hash_freeze": render_info,
        "analysis_blinding": {
            "C80_geometry_must_not_be_inspected_before_both_blocks_complete": True,
            "confirmatory_blocks": ["C80-A", "C80-B"],
        },
        "completion_gate": {
            "question_manifest_validated": True,
            "method_freeze_is_frozen": cfg.get("freeze_status") == "FROZEN",
            "rendered_prompt_hashes_frozen": render_info["status"] == "FROZEN",
            "ready_for_generation": (
                cfg.get("freeze_status") == "FROZEN"
                and render_info["status"] == "FROZEN"
            ),
        },
    }

    combined_manifest_path = out_dir / "questions_240_manifest.json"
    write_json(combined_manifest_path, manifest)

    print(json.dumps({
        "status": manifest["status"],
        "manifest": str(combined_manifest_path),
        "manifest_sha256": sha256_file(combined_manifest_path),
        "method_config_sha256": config_sha,
        "source_questions_sha256": sha256_file(questions_path),
        "assignment_sha256": assignment_hash,
        "block_files": block_file_info,
        "rendered_prompt_hash_freeze": render_info,
        "ready_for_generation": manifest["completion_gate"][
            "ready_for_generation"
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
