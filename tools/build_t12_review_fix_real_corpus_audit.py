#!/usr/bin/env python3
"""Reproduce the T12 E80 review-fix real-corpus metadata audit.

CPU-only except for tokenizer loading. This script does not load the model,
run a forward pass, or modify activation tensors. It re-runs the exact T12
row-preparation/segmentation logic over the stored E80 rollout JSONL files and
writes the machine-readable audit used by PR #28.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from run_t12_activation_recompute import (  # noqa: E402
    MALFORMED_REASONING,
    OPEN_MARKER,
    prepare_row,
    read_jsonl,
    source_paths_for_mode,
)

MODES = ("translated", "wrapper", "default")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_rows(dataset_root: Path, mode: str) -> list[dict]:
    rows = []
    for path in source_paths_for_mode(dataset_root, mode):
        rows.extend(read_jsonl(path))
    uids = [row["uid"] for row in rows]
    if len(uids) != len(set(uids)):
        dupes = [uid for uid, n in collections.Counter(uids).items() if n != 1]
        raise RuntimeError(f"{mode}: duplicate/non-unique uids: {dupes[:10]}")
    return rows


def audit_mode(dataset_root: Path, mode: str, tokenizer) -> dict:
    rows = load_rows(dataset_root, mode)
    prepared = [prepare_row(row, tokenizer) for row in rows]

    failures = [p for p in prepared if p.preparation_error is not None]
    if failures:
        raise RuntimeError(
            f"{mode}: {len(failures)} preparation failures; "
            f"first={failures[0].preparation_error}"
        )

    prefix_rows = [
        p for p in prepared
        if p.segmentation_provenance
        == "prompt_seeded_reconstructed_from_stored_text_boundary"
    ]
    prefix_counts = collections.Counter(
        str(bool(p.frozen_boundary_prefix_check_passed)) for p in prefix_rows
    )
    if any(p.frozen_boundary_prefix_check_passed is not True for p in prefix_rows):
        raise RuntimeError(
            f"{mode}: frozen boundary prefix check did not pass for all seeded-closed rows"
        )

    unclosed = [
        p for p in prepared
        if p.segmentation_provenance == "prompt_seeded_unclosed_open_marker"
    ]
    by_finish = collections.Counter(
        str(p.row.get("finish_reason", "stop")) for p in unclosed
    )

    segmentation_counts = collections.Counter(
        str(p.segmentation_case) for p in prepared
    )
    pool_counts = {
        "all_response": sum(p.all_span is not None for p in prepared),
        "answer": sum(p.answer_span is not None for p in prepared),
        "reasoning": sum(p.reasoning_span is not None for p in prepared),
    }

    unclosed_report = {
        "all_classified_malformed_reasoning": all(
            p.segmentation_case == MALFORMED_REASONING for p in unclosed
        ),
        "all_rendered_prompts_end_with_think": all(
            str(p.row.get("rendered_prompt", "")).rstrip().endswith(OPEN_MARKER)
            for p in unclosed
        ),
        "all_response_retained": all(p.all_span is not None for p in unclosed),
        "answer_pool_unavailable": all(p.answer_span is None for p in unclosed),
        "by_finish_reason": dict(sorted(by_finish.items())),
        "count": len(unclosed),
        "reasoning_pool_unavailable": all(
            p.reasoning_span is None for p in unclosed
        ),
        "uids": [p.row["uid"] for p in unclosed],
    }

    if not all(
        [
            unclosed_report["all_classified_malformed_reasoning"],
            unclosed_report["all_rendered_prompts_end_with_think"],
            unclosed_report["all_response_retained"],
            unclosed_report["answer_pool_unavailable"],
            unclosed_report["reasoning_pool_unavailable"],
        ]
    ):
        raise RuntimeError(f"{mode}: seeded-unclosed invariant failed")

    return {
        "frozen_boundary_prefix_check": {
            "counts": dict(sorted(prefix_counts.items())),
            "n_rows": len(prefix_rows),
            "pass_rate": 1.0 if not prefix_rows else (
                prefix_counts.get("True", 0) / len(prefix_rows)
            ),
            "population": (
                "prompt_seeded_reconstructed_from_stored_text_boundary"
            ),
        },
        "pool_counts": pool_counts,
        "prompt_seeded_unclosed": unclosed_report,
        "segmentation_counts": dict(sorted(segmentation_counts.items())),
        "source_rows": len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        required=True,
        help=(
            "Local root containing run-003 translated/extraction/default "
            "rollout shards."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/t12_runtime_resolved.yaml",
        help="Runtime-resolved T12 config used to pin the tokenizer revision.",
    )
    parser.add_argument(
        "--out",
        default="results/t12/provenance/review_fix_real_corpus_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    config_path = Path(args.config)
    out_path = Path(args.out)

    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    primary = cfg["models"]["primary"]
    tokenizer = AutoTokenizer.from_pretrained(
        primary["model_id"],
        revision=primary["tokenizer_revision"],
    )

    modes = {
        mode: audit_mode(dataset_root, mode, tokenizer)
        for mode in MODES
    }

    status = "PASS"
    for report in modes.values():
        if report["frozen_boundary_prefix_check"]["pass_rate"] != 1.0:
            status = "FAIL"
        u = report["prompt_seeded_unclosed"]
        if not all(
            [
                u["all_classified_malformed_reasoning"],
                u["all_rendered_prompts_end_with_think"],
                u["all_response_retained"],
                u["answer_pool_unavailable"],
                u["reasoning_pool_unavailable"],
            ]
        ):
            status = "FAIL"

    payload = {
        "activation_tensors_modified": False,
        "audit_type": "T12 review-fix real-corpus metadata audit",
        "git_sha": git_sha(),
        "model_forward_pass_rerun": False,
        "modes": modes,
        "runtime_config_sha256": sha256_file(config_path),
        "status": status,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    for mode in MODES:
        u = payload["modes"][mode]["prompt_seeded_unclosed"]
        p = payload["modes"][mode]["frozen_boundary_prefix_check"]
        print(
            f"{mode}: rows={payload['modes'][mode]['source_rows']} "
            f"seeded_unclosed={u['count']} prefix={p['n_rows']} "
            f"pass={p['pass_rate']:.3f}"
        )
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
