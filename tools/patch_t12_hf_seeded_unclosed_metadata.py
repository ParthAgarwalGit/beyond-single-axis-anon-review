#!/usr/bin/env python3
"""Patch the 78 T12 E80 seeded-unclosed metadata rows, reproducibly.

This is a metadata-only operation. It never reads or rewrites activation tensor
files. Affected UIDs are read from the committed T12 audit JSON rather than
copied into this script.

Typical local-only verification:

  python tools/patch_t12_hf_seeded_unclosed_metadata.py \
      --repo-root /path/to/persona-artifacts \
      --dry-run

Apply locally and write a receipt:

  python tools/patch_t12_hf_seeded_unclosed_metadata.py \
      --repo-root /path/to/persona-artifacts \
      --apply

Apply and atomically push only changed metadata files to Hugging Face:

  python tools/patch_t12_hf_seeded_unclosed_metadata.py \
      --repo-root /path/to/persona-artifacts \
      --apply --push --repo-id [Author-A-HF]/persona-artifacts

`HF_TOKEN` (or an existing huggingface_hub login) is required only for --push.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

EXPECTED_BY_MODE = {"translated": 39, "wrapper": 39}
EXPECTED_TOTAL = sum(EXPECTED_BY_MODE.values())
SEEDED_UNCLOSED_NOTE = (
    "rendered_prompt ends with <think> and the completion has no </think>: "
    "the seeded reasoning region never closed. METHOD_FREEZE 6.2 classifies "
    "an unclosed open marker as malformed_reasoning (all-response retained; "
    "reasoning/answer unavailable; judge abstains)."
)
PATCH_VALUES = {
    "segmentation_case": "malformed_reasoning",
    "segmentation_provenance": "prompt_seeded_unclosed_open_marker",
    "frozen_boundary_prefix_check_passed": None,
    "segmentation_note": SEEDED_UNCLOSED_NOTE,
    "n_reasoning_tokens": None,
    "n_answer_tokens": None,
    "has_reasoning_pool": False,
    "has_answer_pool": False,
}


@dataclass(frozen=True)
class Target:
    uid: str
    mode: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_targets(audit_path: Path) -> dict[str, Target]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    targets: dict[str, Target] = {}

    for mode, expected in EXPECTED_BY_MODE.items():
        report = audit["modes"][mode]["prompt_seeded_unclosed"]
        uids = report["uids"]
        if report["count"] != expected or len(uids) != expected:
            raise RuntimeError(
                f"{mode}: expected {expected} affected rows, "
                f"audit says count={report['count']} len(uids)={len(uids)}"
            )
        if len(set(uids)) != len(uids):
            raise RuntimeError(f"{mode}: duplicate UID in audit target list")
        for uid in uids:
            if uid in targets:
                raise RuntimeError(f"UID appears in multiple target modes: {uid}")
            targets[uid] = Target(uid=uid, mode=mode)

    if len(targets) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"expected {EXPECTED_TOTAL} unique targets, found {len(targets)}"
        )
    return targets


def iter_meta_files(repo_root: Path):
    for path in sorted(repo_root.rglob("meta_part*.jsonl")):
        if path.is_file():
            yield path


def encode_rows(rows: list[dict]) -> bytes:
    return (
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    ).encode("utf-8")


def patch_file(path: Path, targets: dict[str, Target]) -> tuple[bytes, list[str]]:
    raw = path.read_bytes()
    rows = []
    target_uids = []

    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

        uid = row.get("uid")
        if uid in targets:
            # Fail closed on unexpected pre-patch states. Fully patched rows are
            # accepted, making repeated verification idempotent. A partially
            # patched row is rejected rather than silently normalized.
            already_patched = all(row.get(k) == v for k, v in PATCH_VALUES.items())
            if not already_patched:
                if row.get("has_answer_pool") is not True:
                    raise RuntimeError(
                        f"{path}:{line_no} target {uid}: expected stale "
                        "has_answer_pool=true before patch"
                    )
                if row.get("segmentation_case") != "direct_answer":
                    raise RuntimeError(
                        f"{path}:{line_no} target {uid}: expected stale "
                        "segmentation_case=direct_answer before patch"
                    )
                for key, value in PATCH_VALUES.items():
                    row[key] = value
            target_uids.append(uid)

        rows.append(row)

    if not target_uids:
        return raw, []
    return encode_rows(rows), target_uids


def write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def push_to_hub(repo_root: Path, changed_files: list[Path], repo_id: str) -> str:
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise RuntimeError(
            "--push requires huggingface_hub; install with `pip install huggingface_hub`"
        ) from exc

    operations = [
        CommitOperationAdd(
            path_in_repo=path.relative_to(repo_root).as_posix(),
            path_or_fileobj=str(path),
        )
        for path in changed_files
    ]
    if not operations:
        raise RuntimeError("nothing to push")

    info = HfApi().create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=(
            "T12: reclassify 78 seeded-unclosed E80 metadata rows as malformed reasoning"
        ),
    )
    revision = getattr(info, "oid", None)
    if not revision:
        raise RuntimeError(f"HF create_commit returned no immutable oid: {info}")
    return str(revision)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        required=True,
        help="Local checkout/snapshot root of the HF dataset repository.",
    )
    parser.add_argument(
        "--audit",
        default="results/t12/provenance/review_fix_real_corpus_audit.json",
        help="Committed audit containing the canonical 39+39 UID lists.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Verify only; write nothing.")
    mode.add_argument("--apply", action="store_true", help="Patch metadata files locally.")
    parser.add_argument("--push", action="store_true", help="Push changed metadata files to HF.")
    parser.add_argument(
        "--repo-id",
        default="[Author-A-HF]/persona-artifacts",
        help="HF dataset repo for --push.",
    )
    parser.add_argument(
        "--receipt",
        default="results/t12/provenance/hf_metadata_patch_receipt.json",
        help="Receipt path; written only with --apply.",
    )
    return parser.parse_args()


def collections_counter(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def main() -> None:
    args = parse_args()
    if args.push and not args.apply:
        raise SystemExit("--push requires --apply")

    repo_root = Path(args.repo_root).resolve()
    audit_path = Path(args.audit)
    receipt_path = Path(args.receipt)
    if not repo_root.is_dir():
        raise FileNotFoundError(repo_root)
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)

    targets = load_targets(audit_path)
    found: dict[str, Path] = {}
    staged: list[tuple[Path, bytes, bytes, list[str]]] = []

    meta_files = list(iter_meta_files(repo_root))
    if not meta_files:
        raise RuntimeError(f"no meta_part*.jsonl found under {repo_root}")

    for path in meta_files:
        before = path.read_bytes()
        after, uids = patch_file(path, targets)
        if not uids:
            continue
        for uid in uids:
            if uid in found:
                raise RuntimeError(
                    f"target UID {uid} appears in multiple metadata files: "
                    f"{found[uid]} and {path}"
                )
            found[uid] = path
        staged.append((path, before, after, uids))

    missing = sorted(set(targets) - set(found))
    extra = sorted(set(found) - set(targets))
    if missing or extra or len(found) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"target coverage failed: found={len(found)} missing={missing[:10]} "
            f"extra={extra[:10]}"
        )

    by_mode = collections_counter(targets[uid].mode for uid in found)
    if by_mode != EXPECTED_BY_MODE:
        raise RuntimeError(f"mode counts do not match freeze: {by_mode}")

    print(
        f"verified {len(found)} target UIDs across {len(staged)} metadata files: "
        f"{by_mode}"
    )

    if not args.apply:
        print("dry-run only; no files changed")
        return

    file_receipts = []
    changed_paths = []
    for path, before, after, uids in staged:
        path.write_bytes(after)
        landed = path.read_bytes()
        if landed != after:
            raise RuntimeError(f"write verification failed: {path}")
        changed_paths.append(path)
        file_receipts.append(
            {
                "path_in_repo": path.relative_to(repo_root).as_posix(),
                "n_target_rows": len(uids),
                "target_uids": sorted(uids),
                "sha256_before": sha256_bytes(before),
                "sha256_after": sha256_bytes(after),
            }
        )

    receipt = {
        "format_version": "t12-hf-metadata-patch-v1",
        "activation_tensors_modified": False,
        "repo_id": args.repo_id,
        "target_count": len(found),
        "target_counts_by_mode": by_mode,
        "patch_values": PATCH_VALUES,
        "files": file_receipts,
        "hf_revision": None,
    }

    if args.push:
        revision = push_to_hub(repo_root, changed_paths, args.repo_id)
        receipt["hf_revision"] = revision
        print(f"pushed HF dataset revision: {revision}")

    write_receipt(receipt_path, receipt)
    print(f"wrote receipt: {receipt_path}")


if __name__ == "__main__":
    main()
