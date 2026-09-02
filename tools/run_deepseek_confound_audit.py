"""DeepSeek confound/audit bundle — post-primary sensitivities on frozen artifacts.

Three CPU-only checks against the frozen C80 recovery result. No generation,
no model inference, no judging, no re-extraction; every input is a committed
frozen artifact (or the pinned HF metadata the reduced manifests already
bind to), and nothing here modifies or retunes the T15/T16 primary.

  1. response-length confound   — is the cross-built role ordering explained
                                  by mean generated-response length?
  2. default-condition stability — is the equal-weight five-condition default
                                  reference materially dependent on any one
                                  default prompt?
  3. held-out-role reconstruction — does the recovered ordering hold when
                                  each role is excluded from BOTH axes that
                                  score it (full leave-one-role-out rebuild)?

Fidelity gate: before any sensitivity runs, the script reconstructs the
primary axes from the frozen inputs and must reproduce the frozen headline
cross-axis Pearson r to within 1e-6, proving input/pipeline fidelity.
Everything is fail-closed; a missing frozen input aborts that sub-analysis
with the exact missing path rather than recreating production data.

Usage:
    python tools/run_deepseek_confound_audit.py \
        --meta-root <dir with t12-c80/{C80-A,C80-B}/meta_part*.jsonl at the
                     pinned revision> \
        [--out results/audit/deepseek_confound]
"""

import argparse
import csv
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.geometry import assistant_axis, cross_projections, pearson_r  # noqa: E402
from src.provenance import git_dirty, stamp_report  # noqa: E402
from src.t17_geometry import default_vector  # noqa: E402

BLOCKS = ("C80-A", "C80-B")
HEADLINE_TOL = 1e-6


# ---------------------------------------------------------------------------
# frozen-input loading (fail closed, hashes recorded)
# ---------------------------------------------------------------------------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\x00" % len(data))
    h.update(data)
    return h.hexdigest()


def expected_meta_manifest(revision):
    """Fetch the pinned revision's t12-c80 metadata tree listing from HF:
    {relpath: (size, git_blob_sha1)} for every meta_part file."""
    from huggingface_hub import HfApi
    info = HfApi().repo_info("[Author-A-HF]/persona-artifacts", repo_type="dataset",
                             revision=revision, files_metadata=True)
    expected = {}
    for s in info.siblings:
        if s.rfilename.startswith("t12-c80/") and "meta_part" in s.rfilename:
            expected[s.rfilename] = (int(s.size), s.blob_id)
    if not expected:
        raise SystemExit("AUDIT BLOCKED: pinned revision lists no t12-c80 metadata")
    return expected


def verify_meta_tree(meta_root, expected):
    """Fail-closed byte verification of the local metadata tree against the
    pinned revision's git blob ids. Returns {relpath: sha256} of verified
    files. Rejects missing, extra, resized, or modified files."""
    meta_root = Path(meta_root)
    local = {str(p.relative_to(meta_root)).replace("\\", "/")
             for p in meta_root.rglob("meta_part*.jsonl")}
    missing = sorted(set(expected) - local)
    extra = sorted(local - set(expected))
    if missing or extra:
        raise SystemExit(
            f"AUDIT BLOCKED: metadata tree does not match pinned revision "
            f"(missing {len(missing)}: {missing[:3]}; extra {len(extra)}: {extra[:3]})")
    hashes = {}
    for rel, (size, blob_id) in sorted(expected.items()):
        data = (meta_root / rel).read_bytes()
        if len(data) != size or _git_blob_sha1(data) != blob_id:
            raise SystemExit(
                f"AUDIT BLOCKED: {rel} does not match the pinned revision "
                f"(size {len(data)} vs {size}; blob {_git_blob_sha1(data)[:12]} "
                f"vs {blob_id[:12]}) — supply the exact bytes from that revision")
        hashes[rel] = hashlib.sha256(data).hexdigest()
    return hashes


def _require(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"AUDIT BLOCKED: required frozen input missing: {p}")
    return p


def _load_named_npz(path):
    """names/matrix npz written by the T17 reduce; returns {name: vector}."""
    z = np.load(_require(path), allow_pickle=True)
    names = [str(n) for n in z["names"]]
    matrix = np.asarray(z["matrix"], dtype=np.float64)
    if len(names) != matrix.shape[0]:
        raise SystemExit(f"AUDIT BLOCKED: names/matrix mismatch in {path}")
    return dict(zip(names, matrix))


def load_frozen_inputs():
    inputs, hashes = {}, {}
    for block in BLOCKS:
        base = REPO_ROOT / "results" / "t17" / block
        role_npz = base / block / "available_all_response.npz"
        def_npz = base / f"DEFAULT-{block}" / "all_response.npz"
        cc_path = base / f"DEFAULT-{block}" / "condition_counts.json"
        man_path = base / "reduced_manifest.json"
        inputs[block] = {
            "role_means": _load_named_npz(role_npz),
            "default_means": _load_named_npz(def_npz),
            "condition_counts": json.loads(_require(cc_path).read_text()),
            "manifest": json.loads(_require(man_path).read_text()),
        }
        for p in (role_npz, def_npz, cc_path, man_path):
            hashes[str(p.relative_to(REPO_ROOT))] = _sha256(p)

    t15_path = _require(REPO_ROOT / "results" / "t15" / "confirmatory_reliability.json")
    inputs["t15"] = json.loads(t15_path.read_text())
    hashes[str(t15_path.relative_to(REPO_ROOT))] = _sha256(t15_path)

    t14_path = _require(REPO_ROOT / "results" / "t14" / "retained_roles.json")
    t14 = json.loads(t14_path.read_text())
    inputs["retained_roles"] = sorted(
        r["role_id"] if isinstance(r, dict) else str(r)
        for r in (t14.get("eligible_roles_primary") or t14.get("retained_roles")
                  or t14.get("roles"))
    )
    hashes[str(t14_path.relative_to(REPO_ROOT))] = _sha256(t14_path)
    return inputs, hashes


def frozen_default(block_inputs):
    """Equal-weight five-condition default via the frozen T17 constructor."""
    return default_vector(block_inputs["default_means"],
                          block_inputs["condition_counts"])


def frozen_headline_r(t15):
    for key in ("cross_axis_pearson_r", "pearson_r"):
        if key in t15:
            return float(t15[key])
        if isinstance(t15.get("primary"), dict) and key in t15["primary"]:
            return float(t15["primary"][key])
    raise SystemExit("AUDIT BLOCKED: frozen headline r not found in T15 record")


# ---------------------------------------------------------------------------
# small stats helpers (numpy only; average-rank Spearman)
# ---------------------------------------------------------------------------

def _ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_r(x, y):
    return pearson_r(_ranks(x), _ranks(y))


def ols_slope_r2(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(slope), 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def residualize(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return np.asarray(y, float) - (slope * np.asarray(x, float) + intercept)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# analysis 1: response-length confound
# ---------------------------------------------------------------------------

def role_length_means(meta_root, block):
    files = sorted(glob.glob(str(Path(meta_root) / "t12-c80" / block / "meta_part*.jsonl")))
    if not files:
        raise SystemExit(
            f"AUDIT BLOCKED: no meta_part files for {block} under {meta_root} "
            "(expected the pinned-revision t12-c80 metadata)")
    sums, sums_reason, sums_answer, counts = {}, {}, {}, {}
    n_rows = 0
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("validity") != "valid" or not row.get("has_all_response_pool"):
                    continue
                role = row["role"]
                trc = row.get("token_region_counts") or {}
                n_all = trc.get("all_response")
                if n_all is None:
                    raise SystemExit(
                        f"AUDIT BLOCKED: token_region_counts.all_response missing in {fp}")
                sums[role] = sums.get(role, 0) + int(n_all)
                counts[role] = counts.get(role, 0) + 1
                if trc.get("reasoning") is not None:
                    sums_reason[role] = sums_reason.get(role, 0) + int(trc["reasoning"])
                if trc.get("final_answer") is not None:
                    sums_answer[role] = sums_answer.get(role, 0) + int(trc["final_answer"])
                n_rows += 1
    means = {r: sums[r] / counts[r] for r in sums}
    means_reason = {r: sums_reason[r] / counts[r] for r in sums_reason if counts.get(r)}
    means_answer = {r: sums_answer[r] / counts[r] for r in sums_answer if counts.get(r)}
    return means, means_reason, means_answer, n_rows, files


def analysis_length(inputs, meta_root, axes, scores):
    roles, s_a, s_b = scores
    out = {"question": "is role position along the cross-built Assistant "
                       "direction associated with mean generated-response length, "
                       "and does the recovery survive removing that association?"}
    per_block_rows = []
    lengths = {}
    meta_files_used = {}
    for block, s in (("C80-A", s_a), ("C80-B", s_b)):
        means, means_reason, means_answer, n_rows, files = role_length_means(meta_root, block)
        missing = [r for r in roles if r not in means]
        if missing:
            raise SystemExit(f"AUDIT BLOCKED: {block} lacks length data for roles {missing[:5]}")
        length_vec = np.array([means[r] for r in roles])
        lengths[block] = length_vec
        meta_files_used[block] = {"n_files": len(files), "n_valid_rows": n_rows}
        slope, r2 = ols_slope_r2(length_vec, s)
        entry = {
            "block": block,
            "n_roles": len(roles),
            "pearson_length_vs_score": pearson_r(length_vec, s),
            "spearman_length_vs_score": spearman_r(length_vec, s),
            "ols_slope_score_per_token": slope,
            "ols_r2": r2,
            "mean_length_min": float(length_vec.min()),
            "mean_length_max": float(length_vec.max()),
        }
        for name, mm in (("reasoning", means_reason), ("final_answer", means_answer)):
            if len(mm) == len(means):
                vec = np.array([mm[r] for r in roles])
                entry[f"pearson_{name}_tokens_vs_score"] = pearson_r(vec, s)
                entry[f"spearman_{name}_tokens_vs_score"] = spearman_r(vec, s)
        per_block_rows.append(entry)

    resid_a = residualize(lengths["C80-A"], s_a)
    resid_b = residualize(lengths["C80-B"], s_b)
    out["per_block"] = per_block_rows
    out["residualized_recovery"] = {
        "pearson_r": pearson_r(resid_a, resid_b),
        "spearman_r": spearman_r(resid_a, resid_b),
        "frozen_headline_r": axes["headline_r"],
        "delta_vs_headline": pearson_r(resid_a, resid_b) - axes["headline_r"],
    }
    out["meta_rows_used"] = meta_files_used
    out["length_measure"] = ("token_region_counts.all_response per row "
                             "(generated content tokens, identical token set "
                             "to the pooled representation), averaged per role "
                             "over valid rows with the all-response pool")
    rows_csv = [{"role": r,
                 "mean_len_A": float(lengths["C80-A"][i]),
                 "mean_len_B": float(lengths["C80-B"][i]),
                 "score_A_on_axisB": float(s_a[i]),
                 "score_B_on_axisA": float(s_b[i])}
                for i, r in enumerate(roles)]
    return out, rows_csv


# ---------------------------------------------------------------------------
# analysis 2: default-condition stability
# ---------------------------------------------------------------------------

def analysis_default_stability(inputs, axes, scores):
    roles, _, _ = scores
    out = {"question": "is the recovery materially dependent on any single "
                       "default prompt, or is the equal-weight five-condition "
                       "default reference stable?"}
    conds = sorted(inputs["C80-A"]["default_means"],
                   key=lambda c: int(c) if str(c).isdigit() else str(c))
    per_block = {}
    for block in BLOCKS:
        dm = inputs[block]["default_means"]
        eq = axes[block]["mu_default"]
        other = "C80-B" if block == "C80-A" else "C80-A"
        opp_axis = axes[other]["unit"]
        pairwise = {f"{a}|{b}": cosine(dm[a], dm[b])
                    for i, a in enumerate(conds) for b in conds[i + 1:]}
        projections = {c: float(np.dot(dm[c], opp_axis)) for c in conds}
        pvals = np.array(list(projections.values()))
        per_block[block] = {
            "pairwise_cosines": pairwise,
            "pairwise_cosine_min": min(pairwise.values()),
            "proj_on_opposite_frozen_axis": projections,
            "proj_range": float(pvals.max() - pvals.min()),
            "proj_sd": float(pvals.std(ddof=1)),
            "dist_from_equal_weight_centroid": {
                c: float(np.linalg.norm(dm[c] - eq)) for c in conds},
        }

    single_rows = []
    role_means = {b: {r: inputs[b]["role_means"][r] for r in roles} for b in BLOCKS}

    def build(block, mu_d):
        return assistant_axis(mu_d, role_means[block])

    def recovery(unit_a, unit_b):
        _, sa, sb = cross_projections(role_means["C80-A"], role_means["C80-B"],
                                      unit_a, unit_b)
        return pearson_r(sa, sb)

    for label, mu_of in (
        [(f"single_cond_{c}",
          lambda b, c=c: inputs[b]["default_means"][c]) for c in conds]
        + [(f"loo_cond_{c}",
            lambda b, c=c: np.mean([inputs[b]["default_means"][x]
                                    for x in conds if x != c], axis=0))
           for c in conds]
    ):
        entry = {"variant": label}
        try:
            ua = build("C80-A", mu_of("C80-A"))
            ub = build("C80-B", mu_of("C80-B"))
            entry["axis_cos_vs_primary_A"] = cosine(ua, axes["C80-A"]["unit"])
            entry["axis_cos_vs_primary_B"] = cosine(ub, axes["C80-B"]["unit"])
            entry["recovery_r"] = recovery(ua, ub)
            entry["sign_check"] = "PASS"
        except ValueError as exc:
            entry["sign_check"] = f"FAIL: {exc}"
        single_rows.append(entry)

    rec = [e["recovery_r"] for e in single_rows if "recovery_r" in e]
    cosines = [min(e["axis_cos_vs_primary_A"], e["axis_cos_vs_primary_B"])
               for e in single_rows if "axis_cos_vs_primary_A" in e]
    out["per_block"] = per_block
    out["axis_variants"] = single_rows
    out["summary"] = {
        "recovery_r_min": min(rec), "recovery_r_max": max(rec),
        "recovery_r_range": max(rec) - min(rec),
        "axis_cos_vs_primary_min": min(cosines),
        "axis_cos_vs_primary_median": float(np.median(cosines)),
        "frozen_headline_r": axes["headline_r"],
        "n_sign_check_failures": sum(1 for e in single_rows
                                     if e["sign_check"] != "PASS"),
    }
    return out, single_rows


# ---------------------------------------------------------------------------
# analysis 3: proper held-out-role reconstruction
# ---------------------------------------------------------------------------

def analysis_heldout_roles(inputs, axes, scores):
    roles, _, _ = scores
    out = {"question": "does the recovered ordering hold when each role is "
                       "excluded from BOTH axes used to score it?"}
    role_means = {b: {r: np.asarray(inputs[b]["role_means"][r]) for r in roles}
                  for b in BLOCKS}
    mats = {b: np.stack([role_means[b][r] for r in roles]) for b in BLOCKS}
    sums = {b: mats[b].sum(axis=0) for b in BLOCKS}
    n = len(roles)
    held_rows = []
    for i, r in enumerate(roles):
        entry = {"role": r}
        units = {}
        for b in BLOCKS:
            grand = (sums[b] - mats[b][i]) / (n - 1)
            v = axes[b]["mu_default"] - grand
            unit = v / np.linalg.norm(v)
            # frozen sign-check semantics on the reduced role set
            default_proj = float(np.dot(axes[b]["mu_default"], unit))
            others = np.delete(mats[b], i, axis=0)
            if default_proj <= float(np.mean(others @ unit)):
                entry["sign_check"] = f"FAIL({b})"
            units[b] = unit
            entry[f"loro_axis_cos_vs_full_{b[-1]}"] = cosine(unit, axes[b]["unit"])
        entry["score_A_on_loroB"] = float(np.dot(mats["C80-A"][i], units["C80-B"]))
        entry["score_B_on_loroA"] = float(np.dot(mats["C80-B"][i], units["C80-A"]))
        held_rows.append(entry)

    sa = np.array([e["score_A_on_loroB"] for e in held_rows])
    sb = np.array([e["score_B_on_loroA"] for e in held_rows])
    cos_a = np.array([e["loro_axis_cos_vs_full_A"] for e in held_rows])
    cos_b = np.array([e["loro_axis_cos_vs_full_B"] for e in held_rows])
    most_influential = sorted(
        held_rows, key=lambda e: min(e["loro_axis_cos_vs_full_A"],
                                     e["loro_axis_cos_vs_full_B"]))[:5]
    out["heldout_recovery"] = {
        "pearson_r": pearson_r(sa, sb),
        "spearman_r": spearman_r(sa, sb),
        "n_roles": n,
        "frozen_headline_r": axes["headline_r"],
        "n_sign_check_failures": sum(1 for e in held_rows if "sign_check" in e),
    }
    out["loro_axis_cosines"] = {
        "min_A": float(cos_a.min()), "median_A": float(np.median(cos_a)),
        "min_B": float(cos_b.min()), "median_B": float(np.median(cos_b)),
        "most_influential_roles": [
            {"role": e["role"],
             "cos_A": e["loro_axis_cos_vs_full_A"],
             "cos_B": e["loro_axis_cos_vs_full_B"]} for e in most_influential],
    }
    return out, held_rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--meta-root", required=True,
                    help="directory containing t12-c80/{C80-A,C80-B}/meta_part*.jsonl "
                         "downloaded at the reduced manifests' pinned HF revision")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "audit" / "deepseek_confound"))
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)

    pre_write_dirty = git_dirty()
    if pre_write_dirty and not args.allow_dirty:
        raise SystemExit("refusing to run the audit from a dirty tree; commit "
                         "first or pass --allow-dirty (development only)")

    inputs, input_hashes = load_frozen_inputs()
    retained = inputs["retained_roles"]

    # frozen primary axes + fidelity gate
    axes = {"headline_r": frozen_headline_r(inputs["t15"])}
    for block in BLOCKS:
        mu_d = frozen_default(inputs[block])
        rm = {r: inputs[block]["role_means"][r] for r in retained}
        if len(rm) != len(retained):
            raise SystemExit(f"AUDIT BLOCKED: {block} role means missing retained roles")
        axes[block] = {"mu_default": mu_d, "unit": assistant_axis(mu_d, rm)}
    roles, s_a, s_b = cross_projections(
        {r: inputs["C80-A"]["role_means"][r] for r in retained},
        {r: inputs["C80-B"]["role_means"][r] for r in retained},
        axes["C80-A"]["unit"], axes["C80-B"]["unit"])
    reproduced_r = pearson_r(s_a, s_b)
    if abs(reproduced_r - axes["headline_r"]) > HEADLINE_TOL:
        raise SystemExit(
            f"AUDIT BLOCKED: fidelity gate failed — reconstructed r={reproduced_r!r} "
            f"vs frozen {axes['headline_r']!r}; inputs do not reproduce the primary")
    scores = (roles, s_a, s_b)

    hf_revs = {b: inputs[b]["manifest"].get("hf_dataset_revision") for b in BLOCKS}
    if len(set(hf_revs.values())) != 1 or not all(hf_revs.values()):
        raise SystemExit(f"AUDIT BLOCKED: reduced manifests disagree on the "
                         f"pinned metadata revision: {hf_revs}")
    pinned_rev = next(iter(hf_revs.values()))
    meta_hashes = verify_meta_tree(args.meta_root, expected_meta_manifest(pinned_rev))

    a1, a1_csv = analysis_length(inputs, args.meta_root, axes, scores)
    a2, a2_csv = analysis_default_stability(inputs, axes, scores)
    a3, a3_csv = analysis_heldout_roles(inputs, axes, scores)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, payload, rows in (
            ("response_length_sensitivity", a1, a1_csv),
            ("default_condition_sensitivity", a2, a2_csv),
            ("heldout_role_recovery", a3, a3_csv)):
        (out / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
        write_csv(out / f"{name}.csv", rows)
        outputs[f"{name}.json"] = _sha256(out / f"{name}.json")
        outputs[f"{name}.csv"] = _sha256(out / f"{name}.csv")

    (out / "metadata_file_hashes.json").write_text(
        json.dumps({"revision": pinned_rev, "sha256": meta_hashes}, indent=2) + "\n")
    outputs["metadata_file_hashes.json"] = _sha256(out / "metadata_file_hashes.json")
    meta_aggregate = hashlib.sha256("\n".join(
        f"{k} {v}" for k, v in sorted(meta_hashes.items())).encode()).hexdigest()

    provenance = stamp_report({
        "task": "DEEPSEEK_CONFOUND_AUDIT",
        "fidelity_gate": {"reconstructed_r": reproduced_r,
                          "frozen_headline_r": axes["headline_r"],
                          "tolerance": HEADLINE_TOL, "status": "PASS"},
        "frozen_inputs_sha256": input_hashes,
        "t12_c80_meta_hf_revision": pinned_rev,
        "t12_c80_meta_binding": {
            "method": "per-file size + git blob sha1 verified against the "
                      "pinned revision tree via HF API; fail-closed on any "
                      "missing/extra/modified file",
            "n_files_verified": len(meta_hashes),
            "aggregate_sha256": meta_aggregate,
            "per_file_sha256": "metadata_file_hashes.json",
        },
        "meta_rows_used": a1["meta_rows_used"],
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "command": "python tools/run_deepseek_confound_audit.py --meta-root <pinned>",
        "output_sha256": outputs,
        "status_per_analysis": {"response_length": "PASS",
                                "default_stability": "PASS",
                                "heldout_roles": "PASS"},
    }, allow_dirty=args.allow_dirty, dirty=pre_write_dirty)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(json.dumps({
        "fidelity_r": reproduced_r,
        "length": {b["block"]: {"pearson": b["pearson_length_vs_score"],
                                "r2": b["ols_r2"]} for b in a1["per_block"]},
        "residualized_r": a1["residualized_recovery"]["pearson_r"],
        "default_recovery_r_range": [a2["summary"]["recovery_r_min"],
                                     a2["summary"]["recovery_r_max"]],
        "default_axis_cos_min": a2["summary"]["axis_cos_vs_primary_min"],
        "heldout_r": a3["heldout_recovery"]["pearson_r"],
        "loro_cos_min": min(a3["loro_axis_cosines"]["min_A"],
                            a3["loro_axis_cosines"]["min_B"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
