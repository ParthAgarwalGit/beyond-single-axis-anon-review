#!/usr/bin/env python3
"""T17 C160 role-space PCA and response-region geometry.

Consumes two PRIMARY C80 reductions produced by run_t17_pool_sensitivity.py,
forms C160 by exact count-weighted union of the C80-A and C80-B means, and reports:
- PRIMARY all-response C160 role-space PCA on all eligible all-response outputs;
- SENSITIVITY matched all-response / answer / reasoning role-space PCA;
- SENSITIVITY regional Axis cosine and same-role projection correlations.

PCA is descriptive only. It never defines or reorients the Assistant Axis.
Answer-only and reasoning-only geometry are not Lu-comparable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import provenance
from src.config import load_frozen_config
from src.t17_geometry import (
    REGIONS,
    compare_regions,
    default_vector,
    role_space_pca,
    weighted_merge_matched,
    weighted_merge_single,
)

FORMAT_VERSION = "t17-c160-role-space-v2"
_CFG, _CFG_SHA = load_frozen_config()
DEFAULT_FLOOR = float(_CFG["role_vectors_and_axis"]["default_vector"]["minimum_valid_fraction_per_condition"])
MIDDLE_BLOCK_INDEX = int(_CFG["activation_extraction"]["middle_layer"]["primary_block_index"])
_SHA40 = re.compile(r"^[0-9a-fA-F]{40}$")


def die(msg):
    sys.exit(f"\nT17 C160 ABORTED (fail-closed): {msg}\n")


def _dirty_snapshot(allow_dirty=False):
    """Capture tree state before any result artifact is written."""
    dirty = provenance.git_dirty()
    if dirty and not allow_dirty:
        raise RuntimeError(
            "refusing to run canonical T17 C160 analysis from a dirty working tree; "
            "commit first or pass --allow-dirty (development only)"
        )
    return dirty


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_csv(path, fieldnames, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    p.write_text(buf.getvalue(), encoding="utf-8", newline="\n")


def load_single(path):
    with np.load(path, allow_pickle=True) as z:
        names = [str(x) for x in z["names"]]
        mat = z["matrix"]
    return {n: mat[i].astype(np.float64) for i, n in enumerate(names)}


def load_matched(path, pools):
    with np.load(path, allow_pickle=True) as z:
        names = [str(x) for x in z["names"]]
        cols = {p: z[p] for p in pools}
    return {n: {p: cols[p][i].astype(np.float64) for p in pools} for i, n in enumerate(names)}


def load_manifest(root):
    p = Path(root) / "reduced_manifest.json"
    if not p.exists():
        die(f"missing {p}")
    return json.loads(p.read_text("utf-8")), sha256(p)


def verify_reduction(root, manifest):
    stated = manifest.get("npz_sha256", {})
    if not stated:
        die("reduced manifest has no npz_sha256 inventory")
    for rel, want in stated.items():
        p = Path(root) / rel
        if not p.exists():
            die(f"reduced manifest lists missing artifact {p}")
        got = sha256(p)
        if got != want:
            die(f"reduced artifact hash mismatch {rel}: {got[:12]} != {want[:12]}")


def validate_pair(a, b):
    for label, m, block in (("A", a, "C80-A"), ("B", b, "C80-B")):
        if m.get("layout") != "c80" or m.get("arms") != [block]:
            die(f"{label}: expected primary reduction for {block}, got {m.get('layout')}/{m.get('arms')}")
        if m.get("analysis_status") != "primary" or m.get("provisional") is not False:
            die(f"{label}: reduction is not stamped as primary/non-provisional")
        if m.get("membership_rule") != "all_valid":
            die(f"{label}: primary membership must be all_valid")
        if int(m.get("middle_block_index", -1)) != MIDDLE_BLOCK_INDEX:
            die(f"{label}: middle block mismatch")
        if int(m.get("retained_role_count") or -1) != 275:
            die(f"{label}: expected frozen 275-role manifest, got {m.get('retained_role_count')}")
        if not m.get("retained_role_manifest_sha256"):
            die(f"{label}: retained-role manifest hash missing")
        if not m.get("membership_authority_sha256"):
            die(f"{label}: membership-authority hash missing")
        if not m.get("hf_dataset_repo"):
            die(f"{label}: HF dataset repo missing")
        rev = m.get("hf_dataset_revision")
        if not isinstance(rev, str) or not _SHA40.fullmatch(rev):
            die(f"{label}: immutable 40-hex HF dataset revision missing/invalid")
    if a.get("retained_role_manifest_sha256") != b.get("retained_role_manifest_sha256"):
        die("C80-A/B retained-role manifest hashes differ")
    if a.get("membership_authority_sha256") != b.get("membership_authority_sha256"):
        die("C80-A/B membership-authority hashes differ")
    if a.get("hf_dataset_repo") != b.get("hf_dataset_repo"):
        die("C80-A/B HF dataset repos differ")
    if a.get("hf_dataset_revision") != b.get("hf_dataset_revision"):
        die("C80-A/B HF dataset revisions differ")


def _check_block_default_floor(means, counts, pool, label):
    try:
        default_vector(means, counts, floor=DEFAULT_FLOOR)
    except ValueError as exc:
        die(f"{label} {pool} default floor failed: {exc}")


def combined_default(root_a, root_b, pool):
    """Construct a T17-specific pooled C160 default after independent block checks.

    Each C80 half must first clear the frozen >=95% pool-availability floor on all
    five default conditions. Only then are the per-condition means count-weighted
    into a pooled C160 default. This pooled vector is a T17 descriptive construction;
    it does not replace the independent C80-A/B defaults used by T15/T16.
    """
    da = Path(root_a) / "DEFAULT-C80-A"
    db = Path(root_b) / "DEFAULT-C80-B"
    ma, mb = load_single(da / f"{pool}.npz"), load_single(db / f"{pool}.npz")
    ca = json.loads((da / "condition_counts_by_pool.json").read_text("utf-8"))[pool]
    cb = json.loads((db / "condition_counts_by_pool.json").read_text("utf-8"))[pool]

    _check_block_default_floor(ma, ca, pool, "C80-A")
    _check_block_default_floor(mb, cb, pool, "C80-B")

    ea = {k: int(v["eligible"]) for k, v in ca.items()}
    eb = {k: int(v["eligible"]) for k, v in cb.items()}
    merged, merged_eligible = weighted_merge_single(ma, ea, mb, eb, require_same_keys=True)
    counts = {}
    for k in sorted(set(ca) | set(cb)):
        counts[k] = {
            "total": int(ca.get(k, {}).get("total", 0)) + int(cb.get(k, {}).get("total", 0)),
            "eligible": int(ca.get(k, {}).get("eligible", 0)) + int(cb.get(k, {}).get("eligible", 0)),
        }
        if merged_eligible.get(k) != counts[k]["eligible"]:
            die(f"{pool} default count mismatch for condition {k}")
    pooled = default_vector(merged, counts, floor=DEFAULT_FLOOR)
    return pooled, counts


def _analysis_meta(name):
    registry = {
        "all_response_primary_all_eligible": {
            "analysis_status": "PRIMARY",
            "lu_comparable": True,
            "scope_note": "Primary T17 all-response role-space PCA under label-independent membership.",
        },
        "matched_all_response": {
            "analysis_status": "SENSITIVITY",
            "lu_comparable": True,
            "scope_note": "Matched-cohort all-response companion to regional sensitivities.",
        },
        "matched_answer": {
            "analysis_status": "SENSITIVITY",
            "lu_comparable": False,
            "scope_note": "Answer-only geometry is a response-region sensitivity and is not Lu-comparable.",
        },
        "matched_reasoning": {
            "analysis_status": "SENSITIVITY",
            "lu_comparable": False,
            "scope_note": "Reasoning-only geometry is a response-region sensitivity and is not Lu-comparable.",
        },
        "region_comparisons": {
            "analysis_status": "SENSITIVITY",
            "lu_comparable": False,
            "scope_note": "Cross-region Axis/projection comparisons are sensitivity analyses, not the primary Lu-comparable estimand.",
        },
    }
    return registry[name]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--c80-a", required=True, help="primary C80-A reduced directory")
    ap.add_argument("--c80-b", required=True, help="primary C80-B reduced directory")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "t17" / "c160"))
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="DEVELOPMENT ONLY: permit provenance stamping from a dirty working tree",
    )
    args = ap.parse_args(argv)
    pre_write_dirty = _dirty_snapshot(args.allow_dirty)

    ra, rb = Path(args.c80_a), Path(args.c80_b)
    ma, sha_ma = load_manifest(ra); mb, sha_mb = load_manifest(rb)
    validate_pair(ma, mb)
    verify_reduction(ra, ma); verify_reduction(rb, mb)

    counts_a = json.loads((ra / "C80-A" / "counts.json").read_text("utf-8"))
    counts_b = json.loads((rb / "C80-B" / "counts.json").read_text("utf-8"))

    ar_a = load_single(ra / "C80-A" / "available_all_response.npz")
    ar_b = load_single(rb / "C80-B" / "available_all_response.npz")
    ar_c160, ar_counts = weighted_merge_single(
        ar_a, counts_a["available"]["all_response"],
        ar_b, counts_b["available"]["all_response"], require_same_keys=True)
    if len(ar_c160) != 275:
        die(f"primary all-response C160 has {len(ar_c160)} roles, expected 275")

    tri_a = load_matched(ra / "C80-A" / "matched_triple.npz", REGIONS)
    tri_b = load_matched(rb / "C80-B" / "matched_triple.npz", REGIONS)
    tri_c160, tri_counts = weighted_merge_matched(
        tri_a, counts_a["matched_triple"], tri_b, counts_b["matched_triple"], REGIONS)
    if len(tri_c160) != 275:
        die(f"matched three-region C160 has {len(tri_c160)} roles, expected 275")

    defaults, default_counts = {}, {}
    for pool in REGIONS:
        defaults[pool], default_counts[pool] = combined_default(ra, rb, pool)

    pca = {}
    curves = []
    primary_name = "all_response_primary_all_eligible"
    primary_summary, primary_curve, _, _ = role_space_pca(ar_c160, defaults["all_response"])
    primary_summary.update(_analysis_meta(primary_name))
    pca[primary_name] = primary_summary
    for row in primary_curve:
        curves.append({"analysis": primary_name, "analysis_status": "PRIMARY", **row})

    region_role_means = {p: {r: tri_c160[r][p] for r in tri_c160} for p in REGIONS}
    for pool in REGIONS:
        name = f"matched_{pool}"
        summary, curve, _, _ = role_space_pca(region_role_means[pool], defaults[pool])
        summary.update(_analysis_meta(name))
        pca[name] = summary
        for row in curve:
            curves.append({"analysis": name, "analysis_status": "SENSITIVITY", **row})

    pairwise, projection_rows, axes = compare_regions(region_role_means, defaults)
    region_meta = _analysis_meta("region_comparisons")
    for row in pairwise:
        row.update(region_meta)

    report = {
        "format_version": FORMAT_VERSION,
        "task": "T17_C160_role_space_PCA_and_region_geometry",
        "status": "MIXED_PRIMARY_AND_SENSITIVITY",
        "membership": "label_independent_technical_validity",
        "claim_scope": "role-space variance structure; not intrinsic dimensionality of a single persona",
        "middle_block_index": MIDDLE_BLOCK_INDEX,
        "retained_roles": 275,
        "software": {"numpy": np.__version__},
        "c160_construction": (
            "C80-A and C80-B per-role means merged by exact contributing-output counts; "
            "equivalent to concatenating the contributing C160 rows before taking role means"
        ),
        "c160_default_construction": (
            "T17-specific pooled default: each response-region default first clears the >=95% floor "
            "independently in C80-A and C80-B; only then are per-condition means count-weighted across "
            "blocks and equally averaged over five conditions. This pooled vector does not replace the "
            "independent defaults used for T15/T16 confirmatory axes."
        ),
        "analysis_registry": {
            name: _analysis_meta(name)
            for name in ("all_response_primary_all_eligible", "matched_all_response",
                         "matched_answer", "matched_reasoning", "region_comparisons")
        },
        "pca_preprocessing": {"centering": "across role means", "standardization": "none"},
        "pca": pca,
        "region_comparisons": pairwise,
        "matched_triple_count_summary": {
            "min_per_role": int(min(tri_counts.values())),
            "max_per_role": int(max(tri_counts.values())),
        },
        "all_response_count_summary": {
            "min_per_role": int(min(ar_counts.values())),
            "max_per_role": int(max(ar_counts.values())),
        },
        "default_counts_by_region": default_counts,
        "input_provenance": {
            "c80_a_manifest_sha256": sha_ma,
            "c80_b_manifest_sha256": sha_mb,
            "retained_role_manifest_sha256": ma.get("retained_role_manifest_sha256"),
            "membership_authority_sha256": ma.get("membership_authority_sha256"),
            "hf_dataset_repo": ma.get("hf_dataset_repo"),
            "hf_dataset_revision": ma.get("hf_dataset_revision"),
        },
        "interpretation_guardrails": [
            "Only all_response_primary_all_eligible is the PRIMARY T17 PCA analysis.",
            "Answer-only, reasoning-only, matched regional PCA, and cross-region comparisons are SENSITIVITY analyses.",
            "Answer-only and reasoning-only geometry are not Lu-comparable.",
            "PCA is descriptive and does not define or reorient the Assistant Axis.",
            "PC1 sign is aligned to the Axis for display only; report absolute Axis-PC1 alignment.",
            "Differences across reasoning/answer/all-response are role-space differences and do not establish that reasoning training caused them.",
        ],
    }
    report = provenance.stamp_report(
        report,
        allow_dirty=args.allow_dirty,
        dirty=pre_write_dirty,
    )

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "t17_c160_role_space_report.json", report)
    np.savez(out / "t17_c160_defaults.npz",
             **{p: defaults[p].astype(np.float32) for p in REGIONS})
    write_csv(out / "t17_variance_curves.csv",
              ["analysis", "analysis_status", "component", "explained_variance_ratio",
               "cumulative_explained_variance"], curves)
    write_csv(out / "t17_region_comparison.csv",
              ["region_a", "region_b", "axis_cosine", "same_role_own_axis_projection_pearson",
               "same_role_all_response_reference_projection_pearson", "n_roles",
               "analysis_status", "lu_comparable", "scope_note"], pairwise)
    proj_fields = ["role_id"]
    for p in REGIONS:
        proj_fields += [f"proj_{p}_own_axis", f"proj_{p}_all_response_axis"]
    write_csv(out / "t17_region_role_projections.csv", proj_fields, projection_rows)
    np.savez(out / "t17_region_axes.npz", **{p: axes[p].astype(np.float32) for p in REGIONS})
    print(json.dumps({
        "n_roles": 275,
        "primary_all_response_pc1": pca[primary_name]["pc1_variance_explained"],
        "primary_all_response_components_to_70": pca[primary_name]["components_to_70pct"],
        "region_comparisons_status": "SENSITIVITY",
    }, indent=2))


if __name__ == "__main__":
    main()
