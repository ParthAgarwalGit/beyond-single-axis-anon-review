#!/usr/bin/env python3
"""T17 response-region reduction and per-block geometry.

Primary C80 output membership is label-independent technical validity under the
2026-08-17 confirmatory-membership deviation. The production primary run also
requires an explicit frozen retained-role manifest. A score-3 path remains only
as an explicitly unvalidated sensitivity.

C80-A and C80-B are reduced separately because their default means are block-specific.
The separate ``run_t17_c160_geometry.py`` command combines those two reduced blocks
by count-weighted row-union logic and performs the C160 role-space PCA and
cross-region sensitivity geometry.
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
from src.pool_sensitivity import (
    F5_POOLS,
    POOLS,
    TRIPLE_POOLS,
    available_case_sensitivity,
    default_mean_enforced,
    matched_f5,
    matched_triple,
    reduce_records,
)

FORMAT_VERSION = "t17-pool-sensitivity-v4"
PRIMARY_MEMBERSHIP = "all_valid"
MEMBERSHIP_AUTHORITY_NAME = "DEVIATION_2026-08-17_CONFIRMATORY_MEMBERSHIP.md"
DEFAULT_HF_REPO = "[Author-A-HF]/persona-artifacts"
_SHA40 = re.compile(r"^[0-9a-fA-F]{40}$")

_CFG, _CFG_SHA = load_frozen_config()
MIDDLE_BLOCK_INDEX = int(_CFG["activation_extraction"]["middle_layer"]["primary_block_index"])
DEFAULT_FLOOR = float(_CFG["role_vectors_and_axis"]["default_vector"]["minimum_valid_fraction_per_condition"])
_PRIMARY_ARM_CONFIG = _CFG["confirmatory_geometry"]["primary_prompt_arm"]

LAYOUTS = ("e80", "c80")
C80_BLOCKS = ("C80-A", "C80-B")
ARMS = ("translated", "wrapper")
DEFAULT_MODE = "default"
PRIMARY_ARM = "translated"
LAYOUT = "e80"
_ARM_CONFIG_NAME = {
    "translated": "USER_TRANSLATED_LU",
    "C80-A": "USER_TRANSLATED_LU",
    "C80-B": "USER_TRANSLATED_LU",
}


def die(msg):
    sys.exit(f"\nT17 ABORTED (fail-closed): {msg}\n")


def _dirty_snapshot(allow_dirty=False, stage="T17"):
    """Capture tree state before writes; production runs are clean-only."""
    dirty = provenance.git_dirty()
    if dirty and not allow_dirty:
        raise RuntimeError(
            f"refusing to run {stage} from a dirty working tree; commit first or "
            "pass --allow-dirty (development only)"
        )
    return dirty


def _assert_primary_arm():
    if _ARM_CONFIG_NAME[PRIMARY_ARM] != _PRIMARY_ARM_CONFIG:
        die(f"frozen primary_prompt_arm {_PRIMARY_ARM_CONFIG!r} != expected "
            f"{_ARM_CONFIG_NAME[PRIMARY_ARM]!r}; T17 primary-arm mapping is stale")


def _apply_layout(layout, block=None):
    global ARMS, DEFAULT_MODE, PRIMARY_ARM, LAYOUT
    if layout not in LAYOUTS:
        die(f"unknown layout {layout!r}")
    if layout == "e80":
        ARMS, DEFAULT_MODE, PRIMARY_ARM = ("translated", "wrapper"), "default", "translated"
    else:
        if block not in C80_BLOCKS:
            die(f"--layout c80 requires --block from {C80_BLOCKS}; got {block!r}")
        ARMS, DEFAULT_MODE, PRIMARY_ARM = (block,), f"DEFAULT-{block}", block
    LAYOUT = layout
    _assert_primary_arm()


_assert_primary_arm()


def _write_bytes_lf(path: Path, text: str) -> str:
    if not text.endswith("\n"):
        text += "\n"
    data = text.replace("\r\n", "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _write_json_lf(path, doc):
    return _write_bytes_lf(Path(path), json.dumps(doc, indent=2, ensure_ascii=False))


def _write_csv_lf(path, fieldnames, rows):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    return _write_bytes_lf(Path(path), buf.getvalue())


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _save_matched(path, means, pools):
    names = sorted(means)
    arrays = {"names": np.array(names, dtype=object)}
    for p in pools:
        arrays[p] = np.stack([means[r][p] for r in names]) if names else np.zeros((0, 0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _load_matched(path, pools):
    with np.load(path, allow_pickle=True) as z:
        names = [str(n) for n in z["names"]]
        cols = {p: z[p] for p in pools}
    return {n: {p: cols[p][i] for p in pools} for i, n in enumerate(names)}


def _save_single(path, means):
    names = sorted(means)
    matrix = np.stack([means[k] for k in names]) if names else np.zeros((0, 0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, names=np.array(names, dtype=object), matrix=matrix)


def _load_single(path):
    with np.load(path, allow_pickle=True) as z:
        names = [str(n) for n in z["names"]]
        matrix = z["matrix"]
    return {n: matrix[i] for i, n in enumerate(names)}


def _validate_manifest_content(path: Path, expected_mode: str):
    """Validate fields that are present without inventing requirements for old manifests."""
    try:
        doc = json.loads(path.read_text("utf-8"))
    except Exception as exc:
        die(f"cannot parse T12 manifest {path}: {exc}")
    if not isinstance(doc, dict):
        die(f"T12 manifest {path} is not a JSON object")
    declared = doc.get("block")
    if declared is not None and str(declared) != expected_mode:
        die(f"T12 manifest {path} declares block {declared!r}; expected {expected_mode!r}")
    if "middle_block_index" in doc and int(doc["middle_block_index"]) != MIDDLE_BLOCK_INDEX:
        die(f"T12 manifest {path} middle_block_index={doc['middle_block_index']} != {MIDDLE_BLOCK_INDEX}")
    return doc


def _resolve_c80_manifest(t12_root: Path, mode: str):
    """Prefer the production per-mode manifest, but accept a legacy/root manifest.

    PR #39 writes ``manifest.json`` inside its local output directory and uploads
    that directory under a mode-specific HF prefix. Mounted copies therefore
    normally expose ``<root>/<mode>/manifest.json``. Some exports instead place a
    single manifest at the supplied root; accepting that layout prevents a false
    first-contact failure while content checks still fail closed on a declared
    block mismatch.
    """
    candidates = [t12_root / mode / "manifest.json", t12_root / "manifest.json"]
    for p in candidates:
        if p.exists():
            _validate_manifest_content(p, mode)
            try:
                rel = str(p.relative_to(t12_root))
            except ValueError:
                rel = str(p)
            return {"path": rel, "sha256": _sha256_file(p)}
    die(f"T12 manifest missing for {mode}; checked {candidates[0]} and {candidates[1]}")


def _t12_manifest_provenance(t12_root: Path):
    if LAYOUT == "c80":
        return {mode: _resolve_c80_manifest(t12_root, mode)
                for mode in list(ARMS) + [DEFAULT_MODE]}
    m = t12_root / "activation_manifest_v3.json"
    if not m.exists():
        die(f"T12 activation manifest not found at {m}")
    return {"e80": {"path": "activation_manifest_v3.json", "sha256": _sha256_file(m)}}


def _c80_default_conditions(t11_root: Path, block: str):
    f = t11_root / block / "defaults.jsonl"
    if not f.exists():
        die(f"C80 default-condition source not found at {f}; pass --t11-defaults")
    cond = {}
    with f.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                die(f"{f}: malformed JSON: {exc}")
            idx = r.get("default_condition_index")
            uid = r.get("rollout_id")
            if idx is None or uid is None:
                die(f"{f}: record missing rollout_id/default_condition_index")
            if uid in cond:
                die(f"{f}: duplicate rollout_id {uid!r}")
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                die(f"{f}: non-integer default_condition_index for {uid!r}: {idx!r}")
            if idx not in range(5):
                die(f"{f}: default_condition_index for {uid!r} outside [0,4]: {idx}")
            cond[uid] = str(idx)
    if not cond:
        die(f"{f} contained no default records")
    return cond


def _validate_default_join_complete(default_conditions, seen_default_uids):
    extra = sorted(set(default_conditions) - set(seen_default_uids))
    if extra:
        die(f"{len(extra)} T11 default rollout_id(s) have no T12 default row; examples: {extra[:5]}")


def _t12_record_stream(t12_root: Path, scores_by_uid, default_conditions=None):
    from safetensors import safe_open

    seen_default_uids = set()
    for mode in list(ARMS) + [DEFAULT_MODE]:
        mdir = t12_root / mode
        if not mdir.exists():
            die(f"missing T12 mode dir {mdir}")
        metas = sorted(mdir.glob("meta_part*.jsonl"))
        if not metas:
            die(f"no meta_part*.jsonl in {mdir}")
        for meta_path in metas:
            suffix = meta_path.name[len("meta_part"):-len(".jsonl")]
            handles = {}
            for pool, fname, mid_only in (
                ("all_response", f"all_response_part{suffix}.safetensors", False),
                ("answer", f"answer_part{suffix}.safetensors", False),
                ("reasoning", f"reasoning_middle_part{suffix}.safetensors", True),
            ):
                fp = mdir / fname
                handles[pool] = (safe_open(str(fp), framework="numpy"), mid_only) if fp.exists() else (None, mid_only)
            with meta_path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    uid = row.get("uid")
                    if not uid:
                        die(f"{meta_path}: metadata row missing uid")
                    pools, has = {}, {}
                    for pool in POOLS:
                        has[pool] = bool(row.get(f"has_{pool}_pool"))
                        handle, mid_only = handles[pool]
                        if handle is not None and uid in handle.keys():
                            t = handle.get_tensor(uid)
                            pools[pool] = np.asarray(t if mid_only else t[MIDDLE_BLOCK_INDEX], dtype=np.float64)
                        else:
                            pools[pool] = None
                    condition = row.get("condition")
                    if default_conditions is not None and mode == DEFAULT_MODE:
                        if uid in seen_default_uids:
                            die(f"duplicate T12 default uid {uid!r}")
                        seen_default_uids.add(uid)
                        if uid not in default_conditions:
                            die(f"default uid {uid!r} missing from T11 records")
                        condition = default_conditions[uid]
                    rec = {
                        "uid": uid,
                        "mode": mode,
                        "role": row.get("role"),
                        "condition": condition,
                        "validity": row.get("validity"),
                        "has": has,
                        "pools": pools,
                    }
                    if scores_by_uid is not None:
                        rec["lu_score"] = scores_by_uid.get(uid)
                    yield rec
    if default_conditions is not None:
        _validate_default_join_complete(default_conditions, seen_default_uids)


def _membership(rule):
    if rule == "all_valid":
        return (lambda r: r.get("validity") == "valid",
                "PRIMARY C80 rule: label-independent technical validity; pool availability is enforced per analysis")
    if rule == "score3":
        return (lambda r: r.get("validity") == "valid" and r.get("lu_score") == 3,
                "UNVALIDATED SENSITIVITY ONLY: valid output and automatic judge lu_score == 3")
    raise ValueError(rule)


def _load_retained_roles(path):
    p = Path(path)
    if not p.exists():
        die(f"retained-role manifest does not exist: {p}")
    doc = json.loads(p.read_text("utf-8"))
    for key in ("eligible_roles_primary", "retained_roles", "roles"):
        if isinstance(doc, dict) and isinstance(doc.get(key), list):
            roles = set(doc[key])
            if not roles:
                die(f"retained-role manifest {path} has an empty role list")
            return roles, _sha256_file(p)
    if isinstance(doc, list) and doc:
        return set(doc), _sha256_file(p)
    die(f"retained-role manifest {path} has no recognised role list")


def _load_scores(path):
    out = {}
    sp = Path(path)
    if not sp.exists():
        die(f"score source does not exist: {sp}")
    files = sorted(sp.glob("**/scores.jsonl")) if sp.is_dir() else [sp]
    if not files:
        die(f"no scores.jsonl found under {sp}")
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[r["uid"]] = r.get("lu_score")
    return out


def _primary_authority(path):
    p = Path(path)
    if not p.exists():
        die(f"primary C80 T17 requires merged membership authority; missing {p}")
    if p.name != MEMBERSHIP_AUTHORITY_NAME:
        die(f"unexpected membership-authority filename {p.name!r}")
    if p.stat().st_size == 0:
        die(f"membership authority {p} is empty")
    return _sha256_file(p)


def _validate_run_request(a, retained_roles):
    """Fail closed on analysis/membership provenance before touching activation data."""
    if a.analysis_status == "primary" and LAYOUT != "c80":
        die("primary T17 is C80-only; E80 runs are sensitivity/exploratory")

    if a.membership == "score3":
        if a.analysis_status != "sensitivity":
            die("score3 is UNVALIDATED SENSITIVITY ONLY and cannot be stamped primary")
        if retained_roles is None:
            die("score3 requires --retained-roles; output membership and role eligibility must stay paired")
        if not a.scores:
            die("score3 sensitivity requires --scores")

    is_primary = (LAYOUT == "c80" and a.analysis_status == "primary")
    authority_sha = None
    if is_primary:
        if a.membership != PRIMARY_MEMBERSHIP:
            die("primary C80 T17 must use --membership all_valid")
        if retained_roles is None:
            die("primary C80 T17 requires --retained-roles <frozen T14 manifest>")
        if not a.hf_repo:
            die("primary C80 T17 requires --hf-repo")
        if not a.hf_revision or not _SHA40.fullmatch(a.hf_revision):
            die("primary C80 T17 requires --hf-revision as an immutable 40-hex HF commit SHA")
        authority_sha = _primary_authority(a.membership_authority)
    return is_primary, authority_sha


def cmd_reduce(a):
    _apply_layout(a.layout, a.block)
    pre_write_dirty = _dirty_snapshot(a.allow_dirty, "T17 reduction")

    retained_roles = retained_sha = None
    if a.retained_roles:
        retained_roles, retained_sha = _load_retained_roles(a.retained_roles)
    is_primary, authority_sha = _validate_run_request(a, retained_roles)

    t12_root = Path(a.t12_root)
    out = Path(a.out)
    ok, rule_desc = _membership(a.membership)
    t12_manifest_provenance = _t12_manifest_provenance(t12_root)

    scores_by_uid = _load_scores(a.scores) if a.membership == "score3" else None

    default_conditions = None
    if LAYOUT == "c80":
        if not a.t11_defaults:
            die("--layout c80 requires --t11-defaults")
        default_conditions = _c80_default_conditions(Path(a.t11_defaults), PRIMARY_ARM)

    reduced = reduce_records(
        _t12_record_stream(t12_root, scores_by_uid, default_conditions),
        ok,
        retained_roles=retained_roles,
        arms=ARMS,
        default_mode=DEFAULT_MODE,
    )

    npz_hashes = {}
    for arm in ARMS:
        ad = out / arm
        m = reduced["arms"][arm]
        _save_matched(ad / "matched_f5.npz", m["matched_f5"]["means"], F5_POOLS)
        _save_matched(ad / "matched_triple.npz", m["matched_triple"]["means"], TRIPLE_POOLS)
        for pool in POOLS:
            _save_single(ad / f"available_{pool}.npz", m["available"]["means"][pool])
        _write_json_lf(ad / "counts.json", {
            "matched_f5": m["matched_f5"]["counts"],
            "matched_triple": m["matched_triple"]["counts"],
            "available": m["available"]["counts"],
        })
        for f in ("matched_f5.npz", "matched_triple.npz",
                  "available_all_response.npz", "available_answer.npz", "available_reasoning.npz"):
            npz_hashes[f"{arm}/{f}"] = _sha256_file(ad / f)

    dd = out / DEFAULT_MODE
    for pool in POOLS:
        means = reduced["default"]["condition_means_by_pool"][pool]
        _save_single(dd / f"{pool}.npz", means)
        npz_hashes[f"{DEFAULT_MODE}/{pool}.npz"] = _sha256_file(dd / f"{pool}.npz")
    _write_json_lf(dd / "condition_counts.json", reduced["default"]["condition_counts"])
    _write_json_lf(dd / "condition_counts_by_pool.json", reduced["default"]["condition_counts_by_pool"])

    manifest = provenance.stamp_report({
        "format_version": FORMAT_VERSION,
        "stage": "reduce",
        "analysis_status": a.analysis_status,
        "provisional": not is_primary,
        "middle_block_index": MIDDLE_BLOCK_INDEX,
        "default_valid_floor": DEFAULT_FLOOR,
        "layout": LAYOUT,
        "arms": list(ARMS),
        "default_mode": DEFAULT_MODE,
        "membership_rule": a.membership,
        "membership_description": rule_desc,
        "membership_authority": a.membership_authority if is_primary else None,
        "membership_authority_sha256": authority_sha,
        "retained_role_manifest": a.retained_roles,
        "retained_role_manifest_sha256": retained_sha,
        "retained_role_count": len(retained_roles) if retained_roles is not None else None,
        "hf_dataset_repo": a.hf_repo,
        "hf_dataset_revision": a.hf_revision,
        "t12_root": str(t12_root),
        "t12_activation_manifests": t12_manifest_provenance,
        "npz_sha256": npz_hashes,
        "default_condition_counts": reduced["default"]["condition_counts"],
        "default_condition_counts_by_pool": reduced["default"]["condition_counts_by_pool"],
        "claim_scope": (
            "primary prompt-conditioned role geometry under label-independent technical-validity membership"
            if is_primary else "sensitivity/exploratory only"
        ),
    }, allow_dirty=a.allow_dirty, dirty=pre_write_dirty)
    _write_json_lf(out / "reduced_manifest.json", manifest)
    print(f"reduced -> {out}")
    for arm in ARMS:
        c = reduced["arms"][arm]
        print(f"  {arm}: f5 roles={len(c['matched_f5']['counts'])}; "
              f"triple roles={len(c['matched_triple']['counts'])}")


def _verify_npz(reduced_dir, manifest):
    for rel, want in manifest.get("npz_sha256", {}).items():
        p = reduced_dir / rel
        if not p.exists() or _sha256_file(p) != want:
            die(f"reduced artifact hash mismatch: {rel}")


def _analyze(reduced_dir: Path):
    mpath = reduced_dir / "reduced_manifest.json"
    if not mpath.exists():
        die("reduced_manifest.json missing")
    manifest = json.loads(mpath.read_text("utf-8"))
    _verify_npz(reduced_dir, manifest)

    dnpz = reduced_dir / DEFAULT_MODE / "all_response.npz"
    dcounts = reduced_dir / DEFAULT_MODE / "condition_counts_by_pool.json"
    if not dnpz.exists() or not dcounts.exists():
        die("default all-response means / per-pool counts missing")
    cond_means = _load_single(dnpz)
    pool_counts = json.loads(dcounts.read_text("utf-8"))["all_response"]
    default_vec = default_mean_enforced(cond_means, pool_counts, floor=DEFAULT_FLOOR,
                                        eligible_key="eligible")

    arms_out = {}
    for arm in ARMS:
        ad = reduced_dir / arm
        f5 = _load_matched(ad / "matched_f5.npz", F5_POOLS)
        tri = _load_matched(ad / "matched_triple.npz", TRIPLE_POOLS)
        avail = {p: _load_single(ad / f"available_{p}.npz") for p in POOLS}
        arms_out[arm] = {
            "matched_f5": matched_f5(default_vec, f5),
            "matched_triple": matched_triple(default_vec, tri),
            "available_case_sensitivity": available_case_sensitivity(default_vec, avail),
        }

    from src.pool_sensitivity import INTERPRETATION
    return {
        "format_version": FORMAT_VERSION,
        "stage": "analyze",
        "analysis_status": manifest.get("analysis_status"),
        "provisional": manifest.get("provisional", True),
        "membership_rule": manifest.get("membership_rule"),
        "membership_description": manifest.get("membership_description"),
        "membership_authority_sha256": manifest.get("membership_authority_sha256"),
        "retained_role_manifest_sha256": manifest.get("retained_role_manifest_sha256"),
        "retained_role_count": manifest.get("retained_role_count"),
        "hf_dataset_repo": manifest.get("hf_dataset_repo"),
        "hf_dataset_revision": manifest.get("hf_dataset_revision"),
        "primary_arm": PRIMARY_ARM,
        "middle_block_index": MIDDLE_BLOCK_INDEX,
        "default_valid_floor": DEFAULT_FLOOR,
        "arms": arms_out,
        "interpretation": INTERPRETATION,
    }


def cmd_analyze(a):
    pre_write_dirty = _dirty_snapshot(a.allow_dirty, "T17 analysis")
    reduced = Path(a.reduced)
    mpath = reduced / "reduced_manifest.json"
    if not mpath.exists():
        die("reduced_manifest.json missing")
    man = json.loads(mpath.read_text("utf-8"))
    layout = man.get("layout", "e80")
    block = man.get("arms", [None])[0] if layout == "c80" else None
    _apply_layout(layout, block)
    report = _analyze(reduced)
    report["layout"] = LAYOUT
    report = provenance.stamp_report(
        report,
        allow_dirty=a.allow_dirty,
        dirty=pre_write_dirty,
    )
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    _write_json_lf(out / "pool_sensitivity_report.json", report)
    prim = report["arms"][PRIMARY_ARM]
    suffix = "" if report.get("analysis_status") == "primary" and not report.get("provisional") else ".SENSITIVITY"
    _write_csv_lf(out / f"F5_all_response_vs_answer_only{suffix}.csv",
                  ["role_id", "proj_all_response", "proj_answer_only"],
                  prim["matched_f5"]["rows"])
    if prim["matched_triple"]["rows"]:
        _write_csv_lf(out / f"reasoning_vs_answer{suffix}.csv",
                      ["role_id", "proj_all_response", "proj_answer_only", "proj_reasoning_only"],
                      prim["matched_triple"]["rows"])
    print(f"analyze -> {out}")


def _synthetic_records(seed=0, d=48, n_roles=30):
    rng = np.random.default_rng(seed)
    e0 = np.zeros(d); e0[0] = 1.0
    A = 6.0
    roles = [f"role_{i:02d}" for i in range(n_roles)]
    spread = np.linspace(-4.0, 4.0, n_roles)
    recs, uid = [], 0

    def rec(mode, role, cond, validity, has, vecs):
        nonlocal uid
        uid += 1
        return {"uid": f"u{uid}", "mode": mode, "role": role, "condition": cond,
                "validity": validity, "has": has,
                "pools": {p: (vecs.get(p) if has.get(p) else None) for p in POOLS}}

    for i, r in enumerate(roles):
        for _ in range(6):
            ans = spread[i] * e0 + rng.normal(0, 0.05, d)
            rea = A * e0 + rng.normal(0, 0.1, d)
            recs.append(rec(PRIMARY_ARM, r, "c", "valid",
                            {"all_response": True, "answer": True, "reasoning": True},
                            {"all_response": 0.5 * (ans + rea), "answer": ans, "reasoning": rea}))
        for _ in range(3):
            allr = (spread[i] + rng.normal(0, 2.0)) * e0 + rng.normal(0, 0.05, d)
            recs.append(rec(PRIMARY_ARM, r, "c", "valid",
                            {"all_response": True, "answer": False, "reasoning": False},
                            {"all_response": allr}))
    for c in range(5):
        for _ in range(20):
            base = A * e0 + rng.normal(0, 0.05, d)
            recs.append(rec(DEFAULT_MODE, None, f"cond_{c}", "valid",
                            {"all_response": True, "answer": True, "reasoning": True},
                            {"all_response": base, "answer": base, "reasoning": base}))
    return recs


def cmd_self_test(_a):
    _apply_layout("e80")
    reduced = reduce_records(iter(_synthetic_records()), lambda r: r["validity"] == "valid",
                             arms=("translated",), default_mode="default")
    d = reduced["default"]
    dv = default_mean_enforced(d["condition_means_by_pool"]["all_response"],
                               d["condition_counts_by_pool"]["all_response"],
                               floor=DEFAULT_FLOOR, eligible_key="eligible")
    m = reduced["arms"]["translated"]
    f5 = matched_f5(dv, m["matched_f5"]["means"])
    tri = matched_triple(dv, m["matched_triple"]["means"])
    checks = {
        "f5_roles": f5["n_roles"] == 30,
        "answer_ordering": f5["pearson_all_vs_answer"] > 0.9,
        "reasoning_differs": abs(tri["pearson_all_vs_reasoning"]) < 0.5,
        "five_default_conditions": len(d["condition_means_by_pool"]["reasoning"]) == 5,
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if not all(checks.values()):
        die("self-test failed")
    print("self-test OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reduce")
    r.add_argument("--t12-root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--membership", choices=("all_valid", "score3"), default="all_valid")
    r.add_argument("--analysis-status", choices=("primary", "sensitivity"), default="sensitivity")
    r.add_argument("--retained-roles", default=None)
    r.add_argument("--scores", default=None)
    r.add_argument("--membership-authority", default=str(REPO_ROOT / "docs" / MEMBERSHIP_AUTHORITY_NAME))
    r.add_argument("--layout", choices=LAYOUTS, default="e80")
    r.add_argument("--block", choices=C80_BLOCKS, default=None)
    r.add_argument("--t11-defaults", default=None)
    r.add_argument("--hf-repo", default=DEFAULT_HF_REPO,
                   help="HF dataset repo containing the mounted T12/T11 inputs")
    r.add_argument("--hf-revision", default=None,
                   help="immutable 40-hex HF dataset commit; required for primary C80")
    r.add_argument(
        "--allow-dirty",
        action="store_true",
        help="DEVELOPMENT ONLY: permit provenance stamping from a dirty working tree",
    )
    r.set_defaults(func=cmd_reduce)

    an = sub.add_parser("analyze")
    an.add_argument("--reduced", required=True)
    an.add_argument("--out", default=str(REPO_ROOT / "results" / "t17"))
    an.add_argument(
        "--allow-dirty",
        action="store_true",
        help="DEVELOPMENT ONLY: permit provenance stamping from a dirty working tree",
    )
    an.set_defaults(func=cmd_analyze)

    st = sub.add_parser("self-test")
    st.set_defaults(func=cmd_self_test)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
