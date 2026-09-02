#!/usr/bin/env python3
"""T17 post-primary regional independent-recovery sensitivity.

Rebuild reasoning-only and final-answer-only Assistant axes independently from
C80-A and C80-B using the committed T17 matched-triple reductions and
block-specific regional default representations.

This analysis is a response-region sensitivity. It does not alter the frozen
all-response T15/T16 result, does not establish causal reasoning-to-answer
propagation, and is not Lu-comparable.

Reported for each region:
- C80-A/B Axis cosine;
- cross-built same-role Pearson r (A role means on B Axis; B on A Axis);
- project-consistent Spearman;
- same-Axis projection Pearson r;
- 2,000 role-bootstrap percentile 95% CI on the fixed cross-projection pairs.

The runner first reconstructs the committed pooled C160 reasoning-vs-answer
comparison and fails closed unless it matches the canonical T17 C160 report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import provenance
from src.config import load_frozen_config
from src.geometry import assistant_axis, cross_projections, pearson_r
from src.t17_geometry import (
    REGIONS,
    compare_regions,
    default_vector,
    weighted_merge_matched,
    weighted_merge_single,
)

FORMAT_VERSION = "t17-regional-independent-recovery-v1"
FIDELITY_ATOL = 1e-9
N_ROLES = 275
HIDDEN_SIZE = 4096
C160_REPORT_PATH = REPO_ROOT / "results/t17/c160/t17_c160_role_space_report.json"

_CFG, _CFG_SHA = load_frozen_config()
DEFAULT_FLOOR = float(
    _CFG["role_vectors_and_axis"]["default_vector"]
    ["minimum_valid_fraction_per_condition"]
)
MIDDLE_BLOCK_INDEX = int(
    _CFG["activation_extraction"]["middle_layer"]["primary_block_index"]
)

CLAIM_BOUNDARY = [
    "Does not alter the frozen all-response T15/T16 primary result.",
    "Reasoning/answer analyses remain response-region sensitivities and are not Lu-comparable.",
    "Does not establish causal propagation from reasoning to final answer.",
    "Does not establish a uniquely privileged assistant direction.",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text("utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def load_single(path: Path):
    with np.load(path, allow_pickle=True) as z:
        names = [str(x) for x in z["names"]]
        mat = z["matrix"]
    if len(names) != len(mat):
        raise ValueError(f"{path}: names/matrix length mismatch")
    return {n: np.asarray(mat[i], dtype=np.float64) for i, n in enumerate(names)}


def load_matched(path: Path, pools=REGIONS):
    with np.load(path, allow_pickle=True) as z:
        names = [str(x) for x in z["names"]]
        cols = {p: z[p] for p in pools}
    return {
        name: {p: np.asarray(cols[p][i], dtype=np.float64) for p in pools}
        for i, name in enumerate(names)
    }


def validate_manifest(manifest, expected_block):
    expected = {
        "layout": "c80",
        "analysis_status": "primary",
        "provisional": False,
        "membership_rule": "all_valid",
        "middle_block_index": MIDDLE_BLOCK_INDEX,
        "retained_role_count": N_ROLES,
    }
    for key, want in expected.items():
        if manifest.get(key) != want:
            raise ValueError(
                f"{expected_block}: manifest {key}={manifest.get(key)!r}, expected {want!r}"
            )
    if manifest.get("arms") != [expected_block]:
        raise ValueError(f"{expected_block}: manifest arms mismatch")
    for key in (
        "retained_role_manifest_sha256",
        "membership_authority_sha256",
        "hf_dataset_repo",
        "hf_dataset_revision",
        "config_sha256",
    ):
        if not manifest.get(key):
            raise ValueError(f"{expected_block}: missing {key}")


def verify_npz_inventory(root: Path, manifest):
    inventory = manifest.get("npz_sha256", {})
    if not inventory:
        raise ValueError(f"{root}: manifest has no npz_sha256 inventory")
    for rel, expected in inventory.items():
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(path)
        got = sha256_file(path)
        if got != expected:
            raise ValueError(f"{path}: SHA-256 {got} != {expected}")


def project_rank(x):
    # Deliberately matches src.confirmatory_geometry._rank.
    return np.argsort(np.argsort(np.asarray(x, dtype=np.float64))).astype(float)


def project_spearman(x, y):
    return pearson_r(project_rank(x), project_rank(y))


def block_default(root: Path, block: str, region: str):
    droot = root / f"DEFAULT-{block}"
    means = load_single(droot / f"{region}.npz")
    counts = load_json(droot / "condition_counts_by_pool.json")[region]
    vec = default_vector(means, counts, floor=DEFAULT_FLOOR)
    if vec.shape != (HIDDEN_SIZE,) or not np.isfinite(vec).all():
        raise ValueError(f"{block}/{region}: invalid default vector")
    return vec


def combined_default(root_a: Path, root_b: Path, region: str):
    da = root_a / "DEFAULT-C80-A"
    db = root_b / "DEFAULT-C80-B"
    ma = load_single(da / f"{region}.npz")
    mb = load_single(db / f"{region}.npz")
    ca = load_json(da / "condition_counts_by_pool.json")[region]
    cb = load_json(db / "condition_counts_by_pool.json")[region]

    # Each block must clear the frozen pool-specific availability floor first.
    default_vector(ma, ca, floor=DEFAULT_FLOOR)
    default_vector(mb, cb, floor=DEFAULT_FLOOR)

    ea = {k: int(v["eligible"]) for k, v in ca.items()}
    eb = {k: int(v["eligible"]) for k, v in cb.items()}
    merged_means, merged_eligible = weighted_merge_single(
        ma, ea, mb, eb, require_same_keys=True
    )
    merged_counts = {}
    for k in sorted(set(ca) | set(cb)):
        merged_counts[k] = {
            "total": int(ca.get(k, {}).get("total", 0))
            + int(cb.get(k, {}).get("total", 0)),
            "eligible": int(ca.get(k, {}).get("eligible", 0))
            + int(cb.get(k, {}).get("eligible", 0)),
        }
        if int(merged_eligible[k]) != merged_counts[k]["eligible"]:
            raise ValueError(f"{region}: merged default count mismatch for condition {k}")
    return default_vector(merged_means, merged_counts, floor=DEFAULT_FLOOR)


def c160_reasoning_answer_reference(path: Path = C160_REPORT_PATH):
    """Read the fidelity target from the canonical committed T17 C160 report."""
    report = load_json(path)
    matches = [
        row for row in report.get("region_comparisons", [])
        if {row.get("region_a"), row.get("region_b")} == {"reasoning", "answer"}
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{path}: expected exactly one reasoning/answer region comparison, got {len(matches)}"
        )
    row = matches[0]
    if int(row.get("n_roles", -1)) != N_ROLES:
        raise ValueError(f"{path}: reasoning/answer comparison does not use {N_ROLES} roles")
    return (
        float(row["axis_cosine"]),
        float(row["same_role_own_axis_projection_pearson"]),
    )


def analysis_plan(n_boot: int, seed: int):
    """Canonical schema for the committed aggregate artifact."""
    return {
        "task": "T17_REGIONAL_INDEPENDENT_RECOVERY",
        "analysis_status": "POST_PRIMARY_SENSITIVITY",
        "motivation": (
            "Test whether the reasoning-only and final-answer-only assistant directions "
            "are each reproducible across the disjoint C80-A/C80-B question sets, rather "
            "than the pooled response-region difference being attributable to unstable "
            "estimation."
        ),
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "block_index": MIDDLE_BLOCK_INDEX,
        "prompt_arm": "USER_TRANSLATED_LU",
        "membership": "label_independent_technical_validity",
        "role_cohort": "matched_triple within each C80 block",
        "regions": ["reasoning", "answer"],
        "n_roles_expected": N_ROLES,
        "axis_formula": "default_mean - equal_weighted_mean(role_means), unit-normalized",
        "default_rule": "five region-specific condition means, equal 0.2 weighting, block-specific",
        "cross_projection": "A role means on B axis; B role means on A axis",
        "primary_descriptive_statistics": [
            "cross_axis_role_projection_pearson_r",
            "axis_cosine_C80A_C80B",
        ],
        "secondary_statistics": [
            "project_consistent_spearman",
            "same_axis_projection_pearson",
            "within_block_reasoning_answer_axis_cosine",
            "within_block_reasoning_answer_same_role_own_axis_pearson",
        ],
        "bootstrap": {
            "replicates": n_boot,
            "seed": seed,
            "unit": "role",
            "rule": "resample fixed aligned cross-projection pairs; axes are not rebuilt",
            "ci": "percentile_95",
        },
        "claim_boundary": list(CLAIM_BOUNDARY),
    }


def regional_recovery(region, tri_a, tri_b, defaults_a, defaults_b, boot_idx):
    roles = sorted(set(tri_a) & set(tri_b))
    if len(roles) != N_ROLES:
        raise ValueError(f"{region}: expected {N_ROLES} common roles, got {len(roles)}")

    rm_a = {r: np.asarray(tri_a[r][region], dtype=np.float64) for r in roles}
    rm_b = {r: np.asarray(tri_b[r][region], dtype=np.float64) for r in roles}
    axis_a = assistant_axis(defaults_a[region], rm_a)
    axis_b = assistant_axis(defaults_b[region], rm_b)

    common, s_a, s_b = cross_projections(rm_a, rm_b, axis_a, axis_b)
    if common != roles:
        raise ValueError(f"{region}: cross-projection role ordering changed")

    own_a = np.array([np.dot(rm_a[r], axis_a) for r in roles], dtype=np.float64)
    own_b = np.array([np.dot(rm_b[r], axis_b) for r in roles], dtype=np.float64)

    boot = np.array([
        pearson_r(s_a[idx], s_b[idx])
        for idx in boot_idx
        if np.std(s_a[idx]) and np.std(s_b[idx])
    ])
    if len(boot) != len(boot_idx):
        raise ValueError(f"{region}: one or more bootstrap replicates were degenerate")

    rows = []
    for i, role in enumerate(roles):
        rows.append({
            "role_id": role,
            f"{region}_c80a_on_c80b_axis": float(s_a[i]),
            f"{region}_c80b_on_c80a_axis": float(s_b[i]),
            f"{region}_c80a_own_axis": float(own_a[i]),
            f"{region}_c80b_own_axis": float(own_b[i]),
        })

    return {
        "region": region,
        "n_roles": len(roles),
        "cross_axis_pearson_r": float(pearson_r(s_a, s_b)),
        "cross_axis_pearson_ci95": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
        "project_consistent_spearman": float(project_spearman(s_a, s_b)),
        "axis_cosine_C80A_C80B": float(np.dot(axis_a, axis_b)),
        "same_axis_projection_pearson": float(pearson_r(own_a, own_b)),
        "bootstrap_replicates": int(len(boot_idx)),
        "bootstrap_valid_replicates": int(len(boot)),
        "bootstrap_invalid_replicates": 0,
        "_rows": rows,
    }


def within_block(block_name, tri, defaults):
    roles = sorted(tri)
    rr = {r: np.asarray(tri[r]["reasoning"], dtype=np.float64) for r in roles}
    aa = {r: np.asarray(tri[r]["answer"], dtype=np.float64) for r in roles}
    axis_r = assistant_axis(defaults["reasoning"], rr)
    axis_a = assistant_axis(defaults["answer"], aa)
    sr = np.array([np.dot(rr[r], axis_r) for r in roles])
    sa = np.array([np.dot(aa[r], axis_a) for r in roles])
    return {
        "block": block_name,
        "n_roles": len(roles),
        "reasoning_answer_axis_cosine": float(np.dot(axis_r, axis_a)),
        "reasoning_answer_same_role_own_axis_pearson": float(pearson_r(sr, sa)),
        "reasoning_answer_project_consistent_spearman": float(project_spearman(sr, sa)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--c80-a", default=str(REPO_ROOT / "results/t17/C80-A"))
    ap.add_argument("--c80-b", default=str(REPO_ROOT / "results/t17/C80-B"))
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "results/t17/regional_independent_recovery"),
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument(
        "--write-role-details",
        action="store_true",
        help=(
            "also write regional_cross_projections.csv (derived 275-role detail; "
            "not a canonical aggregate authority)"
        ),
    )
    args = ap.parse_args(argv)

    pre_write_dirty = provenance.git_dirty()
    if pre_write_dirty and not args.allow_dirty:
        raise RuntimeError(
            "refusing canonical regional sensitivity run from a dirty tree; "
            "commit/stash first or use --allow-dirty for development only"
        )

    root_a, root_b, out = Path(args.c80_a), Path(args.c80_b), Path(args.out)
    manifest_a_path = root_a / "reduced_manifest.json"
    manifest_b_path = root_b / "reduced_manifest.json"
    ma, mb = load_json(manifest_a_path), load_json(manifest_b_path)
    validate_manifest(ma, "C80-A")
    validate_manifest(mb, "C80-B")
    for key in (
        "retained_role_manifest_sha256",
        "membership_authority_sha256",
        "hf_dataset_repo",
        "hf_dataset_revision",
        "config_sha256",
    ):
        if ma[key] != mb[key]:
            raise ValueError(f"C80-A/B {key} differ")
    verify_npz_inventory(root_a, ma)
    verify_npz_inventory(root_b, mb)

    counts_a_path = root_a / "C80-A/counts.json"
    counts_b_path = root_b / "C80-B/counts.json"
    counts_a, counts_b = load_json(counts_a_path), load_json(counts_b_path)
    tri_a_path = root_a / "C80-A/matched_triple.npz"
    tri_b_path = root_b / "C80-B/matched_triple.npz"
    tri_a, tri_b = load_matched(tri_a_path), load_matched(tri_b_path)
    if set(tri_a) != set(tri_b) or len(tri_a) != N_ROLES:
        raise ValueError("C80-A/B matched-triple role cohorts differ or are incomplete")

    defaults_a = {r: block_default(root_a, "C80-A", r) for r in REGIONS}
    defaults_b = {r: block_default(root_b, "C80-B", r) for r in REGIONS}

    # Fidelity gate against the already committed pooled C160 regional result.
    expected_cos, expected_r = c160_reasoning_answer_reference(C160_REPORT_PATH)
    tri_c160, _ = weighted_merge_matched(
        tri_a, counts_a["matched_triple"],
        tri_b, counts_b["matched_triple"],
        REGIONS,
    )
    c160_means = {r: {role: tri_c160[role][r] for role in tri_c160} for r in REGIONS}
    c160_defaults = {r: combined_default(root_a, root_b, r) for r in REGIONS}
    pairs, _, _ = compare_regions(c160_means, c160_defaults)
    ra = next(
        x for x in pairs if {x["region_a"], x["region_b"]} == {"reasoning", "answer"}
    )
    got_cos = float(ra["axis_cosine"])
    got_r = float(ra["same_role_own_axis_projection_pearson"])
    if not np.isclose(got_cos, expected_cos, atol=FIDELITY_ATOL, rtol=0):
        raise RuntimeError("C160 fidelity gate failed for reasoning/answer Axis cosine")
    if not np.isclose(got_r, expected_r, atol=FIDELITY_ATOL, rtol=0):
        raise RuntimeError("C160 fidelity gate failed for reasoning/answer role correlation")

    roles = sorted(tri_a)
    rng = np.random.default_rng(args.seed)
    boot_idx = rng.integers(0, len(roles), size=(args.n_boot, len(roles)))
    reg = {
        "reasoning": regional_recovery(
            "reasoning", tri_a, tri_b, defaults_a, defaults_b, boot_idx
        ),
        "answer": regional_recovery(
            "answer", tri_a, tri_b, defaults_a, defaults_b, boot_idx
        ),
    }
    within = {
        "C80-A": within_block("C80-A", tri_a, defaults_a),
        "C80-B": within_block("C80-B", tri_b, defaults_b),
    }

    used = [
        manifest_a_path,
        manifest_b_path,
        tri_a_path,
        tri_b_path,
        counts_a_path,
        counts_b_path,
        root_a / "DEFAULT-C80-A/answer.npz",
        root_a / "DEFAULT-C80-A/reasoning.npz",
        root_a / "DEFAULT-C80-A/condition_counts_by_pool.json",
        root_b / "DEFAULT-C80-B/answer.npz",
        root_b / "DEFAULT-C80-B/reasoning.npz",
        root_b / "DEFAULT-C80-B/condition_counts_by_pool.json",
        C160_REPORT_PATH,
    ]
    input_hashes = {
        p.relative_to(REPO_ROOT).as_posix(): sha256_file(p)
        for p in used
    }

    public_reg = {
        region: {k: v for k, v in reg[region].items() if not k.startswith("_")}
        for region in ("reasoning", "answer")
    }
    result = {
        "format_version": FORMAT_VERSION,
        "task": "T17_REGIONAL_INDEPENDENT_RECOVERY",
        "status": "POST_PRIMARY_SENSITIVITY_COMPLETE",
        "analysis_plan": analysis_plan(args.n_boot, args.seed),
        "fidelity_gate": {
            "status": "PASS",
            "expected_c160_reasoning_answer_axis_cosine": expected_cos,
            "reconstructed_c160_reasoning_answer_axis_cosine": got_cos,
            "expected_c160_reasoning_answer_role_r": expected_r,
            "reconstructed_c160_reasoning_answer_role_r": got_r,
            "absolute_tolerance": FIDELITY_ATOL,
        },
        "regional_independent_recovery": public_reg,
        "within_block_reasoning_vs_answer": within,
        "provenance": {
            "source_git_sha": provenance.source_git_sha(),
            "source_git_dirty_before_outputs": bool(pre_write_dirty),
            "numpy_version": np.__version__,
            "retained_role_manifest_sha256": ma["retained_role_manifest_sha256"],
            "membership_authority_sha256": ma["membership_authority_sha256"],
            "hf_dataset_repo": ma["hf_dataset_repo"],
            "hf_dataset_revision": ma["hf_dataset_revision"],
            "config_sha256": ma["config_sha256"],
            "input_sha256": input_hashes,
        },
        "claim_boundary": list(CLAIM_BOUNDARY),
    }

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "regional_independent_recovery.json", result)

    rows = []
    for region in ("reasoning", "answer"):
        x = public_reg[region]
        rows.append({
            "region": region,
            "n_roles": x["n_roles"],
            "axis_cosine_C80A_C80B": x["axis_cosine_C80A_C80B"],
            "cross_axis_pearson_r": x["cross_axis_pearson_r"],
            "ci95_low": x["cross_axis_pearson_ci95"][0],
            "ci95_high": x["cross_axis_pearson_ci95"][1],
            "project_consistent_spearman": x["project_consistent_spearman"],
            "same_axis_projection_pearson": x["same_axis_projection_pearson"],
        })
    write_csv(
        out / "regional_independent_recovery_summary.csv",
        list(rows[0]),
        rows,
    )

    if args.write_role_details:
        by_role = {r: {"role_id": r} for r in roles}
        for region in ("reasoning", "answer"):
            for row in reg[region]["_rows"]:
                by_role[row["role_id"]].update(row)
        projection_rows = [by_role[r] for r in roles]
        write_csv(
            out / "regional_cross_projections.csv",
            list(projection_rows[0]),
            projection_rows,
        )

    print(json.dumps({
        region: public_reg[region] for region in ("reasoning", "answer")
    }, indent=2))


if __name__ == "__main__":
    main()
