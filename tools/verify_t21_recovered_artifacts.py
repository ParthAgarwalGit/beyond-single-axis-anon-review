#!/usr/bin/env python3
"""Verify the recovered T21 artifacts against the committed recovery record.

By default this downloads the exact files from the immutable Hugging Face revision
recorded in ``results/t21/T21_ARTIFACT_RECOVERY.json``. Pass ``--local-dir`` to
verify an already-downloaded directory instead. The generation/judgment hashes
are recovery-time attestations; only the frozen evaluation manifest is checked
against a historical pre-existing SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

ROOT = Path(__file__).resolve().parents[1]
RECOVERY_PATH = ROOT / "results" / "t21" / "T21_ARTIFACT_RECOVERY.json"

MANIFEST_NAME = "t21_eval_manifest_frozen.jsonl"
BASELINE_NAME = "t21_source_baseline_generations.jsonl"
CAPABILITY_NAME = "t21_full_capability_generations.jsonl"
JUDGE_NAME = (
    "t21_source_harm_judgments_deepseek_v3_appendix_d2_2_"
    "source_max_v5_novita.jsonl"
)
FILE_NAMES = (MANIFEST_NAME, BASELINE_NAME, CAPABILITY_NAME, JUDGE_NAME)

CONDITIONS = {
    "UNSTEERED",
    "ASSISTANT_AXIS_CAP_SOURCE_SETTING",
}
SCORES = {
    "refusal",
    "refusal_and_justification",
    "nonsensical",
    "out_of_context",
    "related_but_no_info",
    "almost_enough_info",
    "enough_info",
    "enough_info_and_follow_perfectly",
    "other",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {line_no}: {exc}") from exc
    return rows


def expected_keys(manifest_rows: Iterable[Mapping[str, object]]) -> set[Tuple[str, str]]:
    ids = [str(row["item_id"]) for row in manifest_rows]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("frozen manifest must contain exactly 100 unique item_id values")
    return {(item_id, condition) for item_id in ids for condition in CONDITIONS}


def validate_manifest(path: Path, receipt: Mapping[str, object]) -> Tuple[List[dict], set[Tuple[str, str]]]:
    rows = read_jsonl(path)
    rec = receipt["files"][MANIFEST_NAME]
    observed = sha256_file(path)
    if observed != rec["sha256_now"]:
        raise ValueError(f"manifest recovery SHA mismatch: {observed}")
    if observed != rec["prior_recorded_sha256"]:
        raise ValueError("manifest does not match the historical frozen SHA-256")
    if len(rows) != rec["raw_rows"] or len(rows) != 100:
        raise ValueError(f"manifest row count mismatch: {len(rows)}")
    return rows, expected_keys(rows)


def validate_baseline(path: Path, receipt: Mapping[str, object], keys: set[Tuple[str, str]]) -> None:
    rows = read_jsonl(path)
    rec = receipt["files"][BASELINE_NAME]
    if sha256_file(path) != rec["sha256_now"]:
        raise ValueError("baseline recovery SHA mismatch")
    if len(rows) != rec["raw_rows"] or len(rows) != 200:
        raise ValueError(f"baseline row count mismatch: {len(rows)}")
    row_keys = [(str(r.get("item_id")), str(r.get("condition"))) for r in rows]
    counts = Counter(row_keys)
    if set(row_keys) != keys or any(n != 1 for n in counts.values()):
        raise ValueError("baseline does not contain exactly one row for every frozen item-condition key")
    if any(r.get("technical_error") for r in rows):
        raise ValueError("baseline contains a technical-error generation")


def validate_capability(path: Path, receipt: Mapping[str, object]) -> None:
    rows = read_jsonl(path)
    rec = receipt["files"][CAPABILITY_NAME]
    if sha256_file(path) != rec["sha256_now"]:
        raise ValueError("capability recovery SHA mismatch")
    if len(rows) != rec["raw_rows"] or len(rows) != 6224:
        raise ValueError(f"capability row count mismatch: {len(rows)}")


def validate_judge(path: Path, receipt: Mapping[str, object], keys: set[Tuple[str, str]]) -> None:
    rows = read_jsonl(path)
    rec = receipt["files"][JUDGE_NAME]
    if sha256_file(path) != rec["sha256_now"]:
        raise ValueError("judge recovery SHA mismatch")
    if len(rows) != rec["raw_rows"]:
        raise ValueError(f"judge raw row count mismatch: {len(rows)}")

    successful: List[dict] = []
    retry_errors: List[dict] = []
    invalid: List[dict] = []
    for row in rows:
        score = row.get("score")
        technical_error = row.get("technical_error")
        if score in SCORES and not technical_error:
            successful.append(row)
        elif score is None and technical_error:
            retry_errors.append(row)
        else:
            invalid.append(row)

    if invalid:
        raise ValueError(f"judge contains {len(invalid)} invalid/ambiguous rows")
    if len(successful) != rec["successful_rows"]:
        raise ValueError(f"judge successful row count mismatch: {len(successful)}")
    if len(retry_errors) != rec["technical_error_retry_rows"]:
        raise ValueError(f"judge retry-error count mismatch: {len(retry_errors)}")

    success_keys = [(str(r.get("item_id")), str(r.get("condition"))) for r in successful]
    counts = Counter(success_keys)
    if set(success_keys) != keys or any(n != 1 for n in counts.values()):
        raise ValueError("judge does not contain exactly one successful judgment per frozen key")

    retry_keys = {(str(r.get("item_id")), str(r.get("condition"))) for r in retry_errors}
    if not retry_keys.issubset(keys):
        raise ValueError("judge retry-error rows include keys outside the frozen population")

    if {r.get("judge_model") for r in successful} != {rec["judge_model"]}:
        raise ValueError("judge model provenance mismatch")
    if {r.get("judge_provider") for r in successful} != {rec["judge_provider"]}:
        raise ValueError("judge provider provenance mismatch")
    if {r.get("system_prompt_sha256") for r in successful} != {rec["system_prompt_sha256"]}:
        raise ValueError("judge system-prompt hash mismatch")
    if {r.get("user_template_sha256") for r in successful} != {rec["user_template_sha256"]}:
        raise ValueError("judge user-template hash mismatch")


def verify_paths(paths: Mapping[str, Path], receipt: Mapping[str, object]) -> Dict[str, object]:
    missing = [name for name in FILE_NAMES if name not in paths or not paths[name].is_file()]
    if missing:
        raise ValueError(f"missing recovered artifacts: {missing}")

    manifest_rows, keys = validate_manifest(paths[MANIFEST_NAME], receipt)
    validate_baseline(paths[BASELINE_NAME], receipt, keys)
    validate_capability(paths[CAPABILITY_NAME], receipt)
    validate_judge(paths[JUDGE_NAME], receipt, keys)

    return {
        "status": "PASS",
        "manifest_rows": len(manifest_rows),
        "expected_item_condition_keys": len(keys),
        "hf_repo_id": receipt["hf_repo_id"],
        "hf_revision": receipt["hf_revision"],
        "provenance_boundary": receipt["provenance_boundary"],
    }


def download_from_receipt(receipt: Mapping[str, object], out_dir: Path) -> Dict[str, Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for remote verification") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    repo_id = str(receipt["hf_repo_id"])
    revision = str(receipt["hf_revision"])
    prefix = str(receipt["hf_prefix"]).rstrip("/")
    paths: Dict[str, Path] = {}
    for name in FILE_NAMES:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=f"{prefix}/{name}",
            token=token,
            local_dir=str(out_dir),
        )
        paths[name] = Path(downloaded)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir",
        type=Path,
        help="directory containing the four recovered JSONL files; omit to download the immutable HF pin",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path(".cache/t21-recovered-verification"),
        help="download location when --local-dir is omitted",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = json.loads(RECOVERY_PATH.read_text(encoding="utf-8"))
    if args.local_dir:
        paths = {name: args.local_dir / name for name in FILE_NAMES}
    else:
        paths = download_from_receipt(receipt, args.download_dir)
    report = verify_paths(paths, receipt)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
