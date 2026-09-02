#!/usr/bin/env python
"""T15/T16-S — run the 300 repeated frozen-question resplit sensitivity.

Two phases:

1. ``build-atoms``: stream the raw T12 C80/DEFAULT-C80 shards from the pinned
   HF revision and reduce them to per-question resolution (not the
   already-block-averaged bundle tools/gpu_cells/t15_confirmatory_reduce.py
   produces, which collapses all 80 questions per block into a single role
   mean and therefore cannot support an arbitrary question resplit). Saves a
   private intermediate NPZ ("atoms": one vector per question x role, and
   one vector per question x default-condition) spanning the pooled
   160-question C80 universe. This intermediate is NOT committed - it is
   ~1GB+ and reconstructible from the pinned HF bytes on demand.

2. ``analyze``: for each of the 300 draws in the frozen manifest plus the
   observed C80-A/C80-B split, restrict the atoms to that draw's 80/80
   question halves, rebuild role means and the default mean per half exactly
   as tools/gpu_cells/t15_confirmatory_reduce.py does (mean per condition,
   then equal 0.2 weighting), build each half's Axis via the same
   src.geometry.assistant_axis used everywhere else in this project, and
   recompute the cross-axis same-role Pearson r and Axis cosine.

Per the frozen spec (docs/DEVIATION_2026-08-25_T15_T16_RESPLIT_SPEC.md),
this recomputes only the primary statistic and cosine per draw - not the
full null-permutation/bootstrap suite, which was run once for the observed
split (results/t15/confirmatory_reliability.json) and is not re-run 300
times here; that was never part of the frozen spec for this sensitivity.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_frozen_config  # noqa: E402
from src.geometry import assistant_axis, cross_projections, pearson_r  # noqa: E402
from src.provenance import git_dirty, stamp_report  # noqa: E402
from tools.build_t15_t16_resplit_manifest import canonical_sha256  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = "[Author-A-HF]/persona-artifacts"
BLOCK_INDEX = 16
DEFAULT_VALID_FLOOR = 0.95
N_STRATA = 5
BLOCK_TO_DESIGN_FILE = {
    "C80-A": ROOT / "design" / "questions_C80_A.json",
    "C80-B": ROOT / "design" / "questions_C80_B.json",
}
EXPECTED_MANIFEST_AUTHORITY = "docs/DEVIATION_2026-08-25_T15_T16_RESPLIT_SPEC.md"
EXPECTED_N_DRAWS = 300
CANONICAL_RELIABILITY_RESULT = ROOT / "results" / "t15" / "confirmatory_reliability.json"

# The exact manifest_sha256 committed and reviewed in PR #71
# (results/t15/t15_t16_resplit_manifest.json). Self-consistency alone (the
# stamped hash recomputes from the file's own content) is not sufficient:
# a manifest could be edited - a half_1/half_2 question set swapped, say -
# and its own manifest_sha256 trivially recomputed to match, passing every
# self-consistency check while no longer being the outcome-blind manifest
# PR #71 actually froze. Binding to this literal, reviewable in git history
# rather than derivable from editing the manifest JSON itself, is what
# makes that attack visible.
EXPECTED_MANIFEST_SHA256 = (
    "f24520be2e72bf03a37bd368990b5fa1392a2835f44cb0594ddf3410a92b038f"
)


def _provenance_dirty_snapshot(allow_dirty: bool) -> bool:
    """Capture tree state before outputs are written; production is clean-only."""
    dirty = git_dirty()
    if dirty and not allow_dirty:
        die("refusing to run resplit analysis from a dirty working tree; "
            "commit first or pass --allow-dirty (development only)")
    return dirty


def verify_manifest_authority(manifest: dict) -> None:
    """Fail closed unless the manifest is exactly the frozen, hash-verified
    300-draw manifest, rather than trusting manifest_sha256 as an unverified
    label copied from the file."""
    if manifest.get("authority") != EXPECTED_MANIFEST_AUTHORITY:
        die(f"manifest authority {manifest.get('authority')!r} != "
            f"{EXPECTED_MANIFEST_AUTHORITY!r}")
    if manifest.get("n_draws") != EXPECTED_N_DRAWS:
        die(f"manifest n_draws {manifest.get('n_draws')!r} != {EXPECTED_N_DRAWS}")
    stamped = manifest.get("manifest_sha256")
    recomputed = canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if recomputed != stamped:
        die(f"manifest_sha256 does not verify: recomputed {recomputed} != stamped {stamped} "
            "- the manifest content does not match its own hash")
    if recomputed != EXPECTED_MANIFEST_SHA256:
        die(f"manifest_sha256 {recomputed} != the exact frozen hash PR #71 committed "
            f"({EXPECTED_MANIFEST_SHA256}) - self-consistency alone is not enough; this "
            "must be the exact outcome-blind manifest, not merely a validly-formed one")


def verify_hf_revision_authority(atoms_revision: str) -> None:
    """Fail closed unless the atoms bundle was built from exactly the HF
    revision authorized by the canonical T15/T16 confirmatory result,
    rather than trusting whatever revision happens to be embedded in the
    atoms file."""
    if not CANONICAL_RELIABILITY_RESULT.exists():
        die(f"missing canonical authority for the pinned HF revision: {CANONICAL_RELIABILITY_RESULT}")
    canonical = json.loads(CANONICAL_RELIABILITY_RESULT.read_text(encoding="utf-8"))
    authorized = canonical.get("input_provenance", {}).get("bundle_fetched_at_hf_revision")
    if not authorized:
        die(f"{CANONICAL_RELIABILITY_RESULT} has no input_provenance.bundle_fetched_at_hf_revision")
    if atoms_revision != authorized:
        die(f"atoms bundle HF revision {atoms_revision!r} != the revision authorized by "
            f"{CANONICAL_RELIABILITY_RESULT} ({authorized!r})")


def die(msg: str) -> None:
    raise SystemExit(f"T15/T16-S resplit analysis ABORTED (fail-closed): {msg}")


def _eligible(row) -> bool:
    return row.get("validity") == "valid" and bool(row.get("has_all_response_pool"))


def _cosine(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class _Mean:
    __slots__ = ("s", "n")

    def __init__(self):
        self.s = None
        self.n = 0

    def add(self, v):
        self.s = v.copy() if self.s is None else self.s + v
        self.n += 1

    def mean(self):
        if self.n == 0:
            return None
        return self.s / self.n


def _stream(repo, prefix, revision, token=None):
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.numpy import load_file

    files = list_repo_files(repo, repo_type="dataset", revision=revision, token=token)
    metas = sorted(f for f in files if f.startswith(f"{prefix}/meta_part") and f.endswith(".jsonl"))
    if not metas:
        die(f"no meta_part files found under {prefix} at revision {revision}")
    for meta_path in metas:
        tensor_path = meta_path.replace("meta_part", "all_response_part").replace(".jsonl", ".safetensors")
        mp = hf_hub_download(repo, meta_path, repo_type="dataset", revision=revision, token=token)
        tp = hf_hub_download(repo, tensor_path, repo_type="dataset", revision=revision, token=token)
        tensors = load_file(tp)
        with open(mp, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                t = tensors.get(row["uid"])
                if t is None:
                    continue
                yield row, np.asarray(t[BLOCK_INDEX], dtype=np.float64)
        del tensors


def build_atoms(revision: str, out_path: Path, token: str | None = None) -> None:
    print(f"[t15-t16-s] streaming raw shards at pinned revision {revision} ...")
    role_atoms: dict[tuple[int, str], _Mean] = collections.defaultdict(_Mean)
    default_atoms: dict[tuple[int, int], _Mean] = collections.defaultdict(_Mean)
    default_uid_to_condition: dict[str, int] = {}

    for block in ("C80-A", "C80-B"):
        from huggingface_hub import hf_hub_download
        d = hf_hub_download(REPO, f"t11-defaults/{block}/defaults.jsonl",
                            repo_type="dataset", revision=revision, token=token)
        for line in open(d, encoding="utf-8"):
            r = json.loads(line)
            default_uid_to_condition[r["rollout_id"]] = int(r["default_condition_index"])

    for block in ("C80-A", "C80-B"):
        n_seen = n_kept = 0
        for row, vec in _stream(REPO, f"t12-c80/{block}", revision, token):
            n_seen += 1
            if not _eligible(row):
                continue
            role_atoms[(int(row["question_id"]), row["role"])].add(vec)
            n_kept += 1
        print(f"  {block} roles: {n_seen} rollouts, {n_kept} eligible")

    for block in ("C80-A", "C80-B"):
        n_seen = n_kept = 0
        for row, vec in _stream(REPO, f"t12-c80/DEFAULT-{block}", revision, token):
            n_seen += 1
            if not _eligible(row):
                continue
            cond = default_uid_to_condition.get(row["uid"])
            if cond is None:
                die(f"{block}: default uid {row['uid']} missing from T11 records")
            default_atoms[(int(row["question_id"]), cond)].add(vec)
            n_kept += 1
        print(f"  DEFAULT-{block}: {n_seen} rollouts, {n_kept} eligible")

    role_q = sorted({q for q, _ in role_atoms})
    role_names = sorted({r for _, r in role_atoms})
    role_grid = np.full((len(role_q), len(role_names), len(next(iter(role_atoms.values())).mean())),
                        np.nan, dtype=np.float32)
    q_index = {q: i for i, q in enumerate(role_q)}
    r_index = {r: i for i, r in enumerate(role_names)}
    for (q, r), m in role_atoms.items():
        role_grid[q_index[q], r_index[r]] = m.mean().astype(np.float32)

    default_q = sorted({q for q, _ in default_atoms})
    dq_index = {q: i for i, q in enumerate(default_q)}
    default_grid = np.full((len(default_q), N_STRATA, role_grid.shape[-1]), np.nan, dtype=np.float32)
    for (q, c), m in default_atoms.items():
        default_grid[dq_index[q], c] = m.mean().astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        role_grid=role_grid, role_questions=np.array(role_q), role_names=np.array(role_names, dtype=object),
        default_grid=default_grid, default_questions=np.array(default_q),
        hf_revision=np.array(revision),
    )
    print(f"[t15-t16-s] wrote atoms bundle -> {out_path} "
          f"(role_grid {role_grid.shape}, default_grid {default_grid.shape})")


def _load_retained_roles() -> set[str]:
    path = ROOT / "results" / "t14" / "retained_roles.json"
    if not path.exists():
        die(f"missing frozen retained-role set: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    roles = doc.get("retained_roles") or doc.get("roles")
    if not roles:
        die(f"{path} has no retained_roles/roles field")
    return set(roles)


def _half_role_means(role_grid, q_index, r_index, retained_roles, question_ids):
    idx = [q_index[q] for q in question_ids if q in q_index]
    if len(idx) != len(question_ids):
        die(f"{len(question_ids) - len(idx)} question IDs in this half have no role data at all")
    sub = role_grid[idx]  # [n_q, n_roles, d]
    means = {}
    for r in sorted(retained_roles):
        ri = r_index.get(r)
        if ri is None:
            continue
        col = sub[:, ri, :]
        valid = ~np.isnan(col).any(axis=1)
        if not valid.any():
            die(f"role {r!r} has zero eligible cells among this half's questions")
        means[r] = col[valid].mean(axis=0).astype(np.float64)
    return means


def _half_default_mean(default_grid, dq_index, question_ids):
    idx = [dq_index[q] for q in question_ids if q in dq_index]
    if len(idx) != len(question_ids):
        die(f"{len(question_ids) - len(idx)} question IDs in this half have no default data at all")
    sub = default_grid[idx]  # [n_q, 5, d]
    cond_means = []
    for c in range(N_STRATA):
        col = sub[:, c, :]
        valid = ~np.isnan(col).any(axis=1)
        frac = float(valid.mean())
        if frac < DEFAULT_VALID_FLOOR:
            die(f"default condition {c} valid fraction {frac:.4f} below frozen {DEFAULT_VALID_FLOOR} floor")
        cond_means.append(col[valid].mean(axis=0).astype(np.float64))
    return np.mean(cond_means, axis=0)


def _draw_statistic(role_grid, q_index, r_index, default_grid, dq_index, retained_roles,
                    half_1, half_2):
    rm1 = _half_role_means(role_grid, q_index, r_index, retained_roles, half_1)
    rm2 = _half_role_means(role_grid, q_index, r_index, retained_roles, half_2)
    mu1 = _half_default_mean(default_grid, dq_index, half_1)
    mu2 = _half_default_mean(default_grid, dq_index, half_2)

    common_roles = sorted(set(rm1) & set(rm2))
    if len(common_roles) < 3:
        die(f"only {len(common_roles)} roles present in both halves")
    rm1 = {r: rm1[r] for r in common_roles}
    rm2 = {r: rm2[r] for r in common_roles}

    unit1 = assistant_axis(mu1, rm1)
    unit2 = assistant_axis(mu2, rm2)
    roles, s1, s2 = cross_projections(rm1, rm2, unit1, unit2)
    r_primary = pearson_r(s1, s2)
    cosine = _cosine(unit1, unit2)
    return r_primary, cosine, len(roles)


def analyze(atoms_path: Path, manifest_path: Path, out_path: Path, allow_dirty: bool = False) -> None:
    pre_write_dirty = _provenance_dirty_snapshot(allow_dirty)

    z = np.load(atoms_path, allow_pickle=True)
    role_grid = z["role_grid"]
    role_questions = z["role_questions"]
    role_names = z["role_names"]
    default_grid = z["default_grid"]
    default_questions = z["default_questions"]
    hf_revision = str(z["hf_revision"])
    verify_hf_revision_authority(hf_revision)
    q_index = {int(q): i for i, q in enumerate(role_questions)}
    r_index = {str(r): i for i, r in enumerate(role_names)}
    dq_index = {int(q): i for i, q in enumerate(default_questions)}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_manifest_authority(manifest)
    retained_roles = _load_retained_roles()
    print(f"[t15-t16-s] {len(retained_roles)} retained roles; "
          f"atoms cover {len(role_questions)} questions x {len(role_names)} roles")

    observed = manifest["observed_split"]
    obs_r, obs_cos, obs_n = _draw_statistic(
        role_grid, q_index, r_index, default_grid, dq_index, retained_roles,
        observed["c80_a_question_ids"], observed["c80_b_question_ids"],
    )
    print(f"[t15-t16-s] observed split: r={obs_r:.6f} cosine={obs_cos:.6f} n_roles={obs_n}")

    draws_out = []
    for draw in manifest["draws"]:
        r, cos, n = _draw_statistic(
            role_grid, q_index, r_index, default_grid, dq_index, retained_roles,
            draw["half_1_question_ids"], draw["half_2_question_ids"],
        )
        draws_out.append({
            "draw_index": draw["draw_index"],
            "cross_axis_pearson_r": round(r, 6),
            "axis_cosine": round(cos, 6),
            "n_common_roles": n,
        })
        if draw["draw_index"] % 25 == 0:
            print(f"  draw {draw['draw_index']}/300: r={r:.6f} cosine={cos:.6f}")

    rs = np.array([d["cross_axis_pearson_r"] for d in draws_out])
    coss = np.array([d["axis_cosine"] for d in draws_out])

    result = {
        "task": "T15_T16_REPEATED_FROZEN_QUESTION_RESPLIT_SENSITIVITY",
        "authority": "docs/DEVIATION_2026-08-25_T15_T16_RESPLIT_SPEC.md",
        "manifest_sha256": manifest.get("manifest_sha256"),
        "hf_revision": hf_revision,
        "n_draws": len(draws_out),
        "observed_split": {
            "cross_axis_pearson_r": round(obs_r, 6),
            "axis_cosine": round(obs_cos, 6),
            "n_common_roles": obs_n,
        },
        "draws": draws_out,
        "summary": {
            "cross_axis_pearson_r": {
                "min": round(float(rs.min()), 6),
                "median": round(float(np.median(rs)), 6),
                "max": round(float(rs.max()), 6),
                "mean": round(float(rs.mean()), 6),
                "std": round(float(rs.std()), 6),
            },
            "axis_cosine": {
                "min": round(float(coss.min()), 6),
                "median": round(float(np.median(coss)), 6),
                "max": round(float(coss.max()), 6),
                "mean": round(float(coss.mean()), 6),
                "std": round(float(coss.std()), 6),
            },
            "observed_split_percentile_within_draws": {
                "cross_axis_pearson_r": round(float((rs < obs_r).mean() * 100), 2),
                "axis_cosine": round(float((coss < obs_cos).mean() * 100), 2),
            },
        },
    }
    result = stamp_report(result, dirty=pre_write_dirty, allow_dirty=allow_dirty)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(f"[t15-t16-s] wrote {out_path}")
    print(f"[t15-t16-s] r: observed={obs_r:.6f}  300-draw median={result['summary']['cross_axis_pearson_r']['median']:.6f}"
          f"  [{result['summary']['cross_axis_pearson_r']['min']:.6f}, {result['summary']['cross_axis_pearson_r']['max']:.6f}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("build-atoms")
    p1.add_argument("--revision", required=True, help="40-hex pinned HF dataset revision")
    p1.add_argument("--out", default=str(ROOT / ".cache" / "t15_t16_resplit_atoms.npz"))
    p1.add_argument("--token", default=None)

    p2 = sub.add_parser("analyze")
    p2.add_argument("--atoms", default=str(ROOT / ".cache" / "t15_t16_resplit_atoms.npz"))
    p2.add_argument("--manifest", default=str(ROOT / "results" / "t15" / "t15_t16_resplit_manifest.json"))
    p2.add_argument("--out", default=str(ROOT / "results" / "t15" / "t15_t16_resplit_results.json"))
    p2.add_argument("--allow-dirty", action="store_true",
                    help="development only; production requires a clean tree")

    args = ap.parse_args()
    if args.cmd == "build-atoms":
        build_atoms(args.revision, Path(args.out), args.token)
    elif args.cmd == "analyze":
        analyze(Path(args.atoms), Path(args.manifest), Path(args.out), allow_dirty=args.allow_dirty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
