#!/usr/bin/env python
"""Materialize the frozen 300-partition manifest for the T15/T16 repeated
frozen-question resplit sensitivity.

Specification: docs/DEVIATION_2026-08-25_T15_T16_RESPLIT_SPEC.md. This script
produces ONLY question-ID partition membership. It does not touch
activations, role means, or Axis geometry - that is a separate, later step
that must consume this manifest as a frozen input. No resplit outcome is
inspected anywhere in this file.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
METHOD_FROZEN_CONFIG = ROOT / "configs" / "method_frozen.yaml"
N_DRAWS = 300
N_STRATA = 5
STRATUM_SIZE = 32
HALF_SIZE = 80
FORCED_TOGETHER = (7, 227)
ALL_BLOCK_NAMES = ("E80", "C80-A", "C80-B")
BLOCK_DESIGN_FILENAMES = {
    "E80": "questions_E80.json",
    "C80-A": "questions_C80_A.json",
    "C80-B": "questions_C80_B.json",
}
EXPECTED_ASSIGNMENT_SHA256 = (
    "6f0213dd56073d3ce87b835e8c7b8a7759ab9b53e78486bced689e3f0c1e6172"
)
MASTER_SEED = int(EXPECTED_ASSIGNMENT_SHA256[:8], 16)


def die(msg: str) -> None:
    raise SystemExit(f"T15/T16 resplit manifest ABORTED (fail-closed): {msg}")


def canonical_sha256(value) -> str:
    # Must match tools/build_t08_question_manifest.py::canonical_json_bytes exactly -
    # this function is used to cross-verify hashes that tool independently computes
    # over the same frozen inputs.
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_frozen_block_authority(design_dir: Path, config_path: Path = METHOD_FROZEN_CONFIG) -> str:
    """Recompute every hash this tool depends on from raw frozen inputs, fail closed on drift.

    Independently reproduces, from configs/method_frozen.yaml and the three
    design/questions_{E80,C80_A,C80_B}.json files, exactly what
    tools/build_t08_question_manifest.py computes: each block's id_list_sha256
    and per-block assignment_sha256, and the combined 240-item
    all_blocks_assignment_sha256 that the master seed is derived from. This
    tool must not trust a hardcoded constant for that hash without recomputing
    it from the actual frozen inputs on every run. Also independently verifies
    16-per-prompt-index balance within each 80-question block (not just the
    pooled 160-question C80 universe, which load_question_strata checks
    separately).
    """
    if not config_path.exists():
        die(f"missing frozen method config: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    qd = (cfg or {}).get("question_design")
    if qd is None:
        die(f"{config_path} is missing question_design")
    blocks_cfg = qd.get("blocks") or {}

    all_assignments = []
    for name in ALL_BLOCK_NAMES:
        block_cfg = blocks_cfg.get(name)
        if block_cfg is None:
            die(f"{config_path} is missing question_design.blocks.{name!r}")
        frozen_ids = block_cfg.get("question_ids")
        if not isinstance(frozen_ids, list) or len(frozen_ids) != 80:
            die(f"{name}: frozen config question_ids is not a list of 80 IDs")
        if canonical_sha256(frozen_ids) != block_cfg.get("id_list_sha256"):
            die(f"{name}: id_list_sha256 in the frozen config does not match its own question_ids")

        design_path = design_dir / BLOCK_DESIGN_FILENAMES[name]
        if not design_path.exists():
            die(f"missing frozen design file: {design_path}")
        doc = json.loads(design_path.read_text(encoding="utf-8"))
        rows = sorted(doc.get("questions") or [], key=lambda r: r["block_position"])
        if len(rows) != 80:
            die(f"{design_path}: expected 80 questions, found {len(rows)}")
        if [r["question_id"] for r in rows] != frozen_ids:
            die(
                f"{name}: {design_path} question order/membership does not match "
                "the frozen config's question_ids"
            )

        block_assignments = [
            {
                "question_block": r["question_block"],
                "block_position": r["block_position"],
                "question_id": r["question_id"],
                "prompt_index": r["prompt_index"],
            }
            for r in rows
        ]
        recomputed_block_hash = canonical_sha256(block_assignments)
        committed_block_hash = (doc.get("summary") or {}).get("assignment_sha256")
        if recomputed_block_hash != committed_block_hash:
            die(
                f"{name}: recomputed per-block assignment hash does not match "
                f"{design_path}'s own summary.assignment_sha256"
            )

        pidx_counts = collections.Counter(r["prompt_index"] for r in rows)
        expected_per_prompt = 80 // N_STRATA
        if pidx_counts != collections.Counter({i: expected_per_prompt for i in range(N_STRATA)}):
            die(f"{name}: prompt-index imbalance within this block: {dict(pidx_counts)}")

        all_assignments.extend(block_assignments)

    recomputed_combined_hash = canonical_sha256(all_assignments)
    if recomputed_combined_hash != qd.get("all_blocks_assignment_sha256"):
        die(
            "recomputed 240-item all_blocks_assignment_sha256 does not match "
            f"{config_path}'s question_design.all_blocks_assignment_sha256"
        )
    if recomputed_combined_hash != EXPECTED_ASSIGNMENT_SHA256:
        die(
            f"the frozen config's all_blocks_assignment_sha256 ({recomputed_combined_hash}) "
            f"does not match this tool's EXPECTED_ASSIGNMENT_SHA256 "
            f"({EXPECTED_ASSIGNMENT_SHA256}); the master seed would silently change - "
            "refusing to proceed"
        )
    return recomputed_combined_hash


def load_question_strata(design_dir: Path):
    """Return {prompt_index: sorted [question_id, ...]} over the 160 C80 questions."""
    rows = []
    for name in ("questions_C80_A.json", "questions_C80_B.json"):
        path = design_dir / name
        if not path.exists():
            die(f"missing frozen design file: {path}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(doc["questions"])

    if len(rows) != 160:
        die(f"expected 160 C80 questions, found {len(rows)}")
    qids = [r["question_id"] for r in rows]
    if len(set(qids)) != 160:
        die("duplicate question_id across C80-A/C80-B question files")

    strata: dict[int, list[int]] = {p: [] for p in range(N_STRATA)}
    for r in rows:
        p = r["prompt_index"]
        if p not in strata:
            die(f"question_id {r['question_id']} has out-of-range prompt_index {p!r}")
        strata[p].append(r["question_id"])

    for p, members in strata.items():
        if len(members) != STRATUM_SIZE:
            die(
                f"prompt_index {p} stratum has {len(members)} questions; "
                f"expected {STRATUM_SIZE}"
            )
        members.sort()

    a, b = FORCED_TOGETHER
    strata_of = {qid: p for p, members in strata.items() for qid in members}
    if strata_of.get(a) is None or strata_of.get(b) is None:
        die(f"forced-together IDs {FORCED_TOGETHER} not found in the question universe")
    if strata_of[a] == strata_of[b]:
        die(
            f"forced-together IDs {FORCED_TOGETHER} unexpectedly share a stratum "
            "(spec assumes they do not); the sampling procedure must be revised, "
            "not silently applied"
        )
    return strata, strata_of


def load_observed_split(design_dir: Path):
    a_doc = json.loads((design_dir / "questions_C80_A.json").read_text(encoding="utf-8"))
    b_doc = json.loads((design_dir / "questions_C80_B.json").read_text(encoding="utf-8"))
    a_ids = sorted(r["question_id"] for r in a_doc["questions"])
    b_ids = sorted(r["question_id"] for r in b_doc["questions"])
    if len(a_ids) != HALF_SIZE or len(b_ids) != HALF_SIZE:
        die("observed C80-A/C80-B split is not 80/80")
    if set(a_ids) & set(b_ids):
        die("observed C80-A/C80-B split is not disjoint")
    return {
        "source": "design/questions_C80_A.json + design/questions_C80_B.json",
        "c80_a_question_ids": a_ids,
        "c80_b_question_ids": b_ids,
        "assignment_sha256": EXPECTED_ASSIGNMENT_SHA256,
    }


def draw_one_partition(rng: np.random.Generator, strata: dict[int, list[int]],
                       strata_of: dict[int, int]) -> tuple[list[int], list[int]]:
    half_1: set[int] = set()
    half_2: set[int] = set()
    for p in range(N_STRATA):
        members = strata[p]
        forced = [q for q in FORCED_TOGETHER if strata_of[q] == p]
        if forced:
            if len(forced) != 1:
                die(f"stratum {p} unexpectedly contains more than one forced ID: {forced}")
            forced_id = forced[0]
            rest = [q for q in members if q != forced_id]
            perm = rng.permutation(rest)
            chosen = {forced_id} | {int(x) for x in perm[: STRATUM_SIZE // 2 - 1]}
        else:
            perm = rng.permutation(members)
            chosen = {int(x) for x in perm[: STRATUM_SIZE // 2]}
        if len(chosen) != STRATUM_SIZE // 2:
            die(f"stratum {p} produced {len(chosen)} chosen IDs; expected {STRATUM_SIZE // 2}")
        half_1 |= chosen
        half_2 |= set(members) - chosen

    if len(half_1) != HALF_SIZE or len(half_2) != HALF_SIZE:
        die(f"draw produced half sizes {len(half_1)}/{len(half_2)}; expected {HALF_SIZE}/{HALF_SIZE}")
    if half_1 & half_2:
        die("draw produced overlapping halves")
    a, b = FORCED_TOGETHER
    if (a in half_1) != (b in half_1):
        die(f"draw separated forced-together IDs {FORCED_TOGETHER}")
    return sorted(half_1), sorted(half_2)


def build_manifest(design_dir: Path, config_path: Path = METHOD_FROZEN_CONFIG) -> dict:
    verified_assignment_sha256 = verify_frozen_block_authority(design_dir, config_path)
    strata, strata_of = load_question_strata(design_dir)
    observed = load_observed_split(design_dir)

    master = np.random.SeedSequence(MASTER_SEED)
    children = master.spawn(N_DRAWS)

    draws = []
    seen_partitions = set()
    for i, child in enumerate(children):
        rng = np.random.default_rng(child)
        half_1, half_2 = draw_one_partition(rng, strata, strata_of)
        key = tuple(half_1)
        if key in seen_partitions:
            die(f"draw {i} duplicates an earlier partition; sampling is not producing distinct draws")
        seen_partitions.add(key)
        draws.append({
            "draw_index": i,
            "spawn_key": list(child.spawn_key),
            "half_1_question_ids": half_1,
            "half_2_question_ids": half_2,
        })

    manifest = {
        "schema_version": "t15-t16-resplit-manifest/1.0",
        "task": "T15_T16_REPEATED_FROZEN_QUESTION_RESPLITS",
        "authority": "docs/DEVIATION_2026-08-25_T15_T16_RESPLIT_SPEC.md",
        "n_draws": N_DRAWS,
        "master_seed": MASTER_SEED,
        "master_seed_derivation": (
            f"int(all_blocks_assignment_sha256[:8], 16) = "
            f"int({EXPECTED_ASSIGNMENT_SHA256[:8]!r}, 16)"
        ),
        "all_blocks_assignment_sha256": verified_assignment_sha256,
        "all_blocks_assignment_sha256_independently_recomputed": True,
        "strata": {str(p): strata[p] for p in range(N_STRATA)},
        "forced_together_question_ids": list(FORCED_TOGETHER),
        "observed_split": observed,
        "draws": draws,
        "outcome_blind": True,
        "outcome_note": (
            "This manifest contains only question-ID partition membership. "
            "No geometry, activation, or statistic was computed to produce it."
        ),
    }
    manifest["manifest_sha256"] = canonical_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    )
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design-dir", default=str(ROOT / "design"))
    ap.add_argument("--out", default=str(ROOT / "results" / "t15" / "t15_t16_resplit_manifest.json"))
    args = ap.parse_args()

    manifest_a = build_manifest(Path(args.design_dir))
    manifest_b = build_manifest(Path(args.design_dir))
    if manifest_a["manifest_sha256"] != manifest_b["manifest_sha256"]:
        die("manifest generation is not deterministic across two runs in this process")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(manifest_a, indent=2) + "\n")
    print(f"[t15-t16-resplit] {manifest_a['n_draws']} draws, manifest_sha256="
          f"{manifest_a['manifest_sha256']}")
    print(f"[t15-t16-resplit] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
