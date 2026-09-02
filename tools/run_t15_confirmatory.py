#!/usr/bin/env python
"""T15/T16 - confirmatory split-half Axis reliability runner.

Consumes the reduced C80-A/C80-B block-16 all-response bundle produced by
``tools/gpu_cells/t15_confirmatory_reduce.py``. Primary C80 membership is the
label-independent technical-validity rule authorized by
``docs/DEVIATION_2026-08-17_CONFIRMATORY_MEMBERSHIP.md``.

The primary statistic is cross-axis same-role Pearson r: C80-A role means are
projected on the independently constructed C80-B Axis and vice versa. Reliability
is interpreted against all three frozen nulls, the bootstrap CI, and LOO stability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.confirmatory_geometry import confirmatory_reliability  # noqa: E402
from src.config import load_frozen_config  # noqa: E402
from src.geometry import confirmatory_role_set  # noqa: E402
from src.provenance import git_dirty, stamp_report  # noqa: E402

_ROLE = "role__"
_SUFFIX_A, _SUFFIX_B = "_a", "_b"
_EXPECTED_MEMBERSHIP = "label_independent_technical_validity"
_EXPECTED_AUTHORITY = "docs/DEVIATION_2026-08-17_CONFIRMATORY_MEMBERSHIP.md"
_HEX = set("0123456789abcdef")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_immutable_hf_revision(value, field="bundle_revision"):
    """Return a normalized 40-hex HF commit SHA or fail closed.

    Branch names such as ``main`` are intentionally rejected: the reviewer-visible
    provenance contract is that the analysed bundle is fetched at immutable bytes.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an immutable 40-hex HF commit SHA")
    value = value.strip().lower()
    if len(value) != 40 or any(ch not in _HEX for ch in value):
        raise ValueError(f"{field} must be an immutable 40-hex HF commit SHA")
    return value


def _frozen_params():
    cfg, _ = load_frozen_config()
    return {
        "threshold": int(cfg["role_vectors_and_axis"]
                         ["project_role_vector_threshold_per_80_id_block"]),
        "block_index": int(cfg["confirmatory_geometry"]["primary_layer"]),
        "arm": cfg["confirmatory_geometry"]["primary_prompt_arm"],
        "pool": cfg["confirmatory_geometry"]["primary_pool"],
        "hidden_size": int(cfg["models"]["primary"]["hidden_size"]),
        "default_valid_floor": float(cfg["role_vectors_and_axis"]["default_vector"]
                                     ["minimum_valid_fraction_per_condition"]),
    }


def _provenance_dirty_snapshot(allow_dirty=False):
    """Capture tree state before outputs are written; production is clean-only."""
    dirty = git_dirty()
    if dirty and not allow_dirty:
        raise RuntimeError(
            "refusing to run canonical T15 analysis from a dirty working tree; "
            "commit first or pass --allow-dirty (development only)"
        )
    return dirty


def _load_bundle(bundle_path, counts_path):
    npz = np.load(bundle_path)
    counts = json.loads(Path(counts_path).read_text(encoding="utf-8"))
    role_means_a, role_means_b = {}, {}
    for key in npz.files:
        if not key.startswith(_ROLE):
            continue
        if key.endswith(_SUFFIX_A):
            role_means_a[key[len(_ROLE):-len(_SUFFIX_A)]] = npz[key]
        elif key.endswith(_SUFFIX_B):
            role_means_b[key[len(_ROLE):-len(_SUFFIX_B)]] = npz[key]
    return npz["mu_default_a"], role_means_a, npz["mu_default_b"], role_means_b, counts


def _validate_bundle(mu_a, rm_a, mu_b, rm_b, counts, params):
    prov = counts.get("provenance", {})
    expected = {
        "block_index": params["block_index"],
        "arm": params["arm"],
        "pool": params["pool"],
        "membership": _EXPECTED_MEMBERSHIP,
        "membership_authority": _EXPECTED_AUTHORITY,
    }
    for key, value in expected.items():
        if prov.get(key) != value:
            raise ValueError(f"bundle {key} {prov.get(key)!r} != expected {value!r}")

    # New reductions must carry the exact immutable revision they consumed. The
    # historical canonical bundle predates this field and is represented honestly
    # by null in the committed report rather than by backfilling a guessed revision.
    source_revision = prov.get("source_hf_revision")
    if source_revision is not None:
        _require_immutable_hf_revision(source_revision, field="source_hf_revision")

    shape = (params["hidden_size"],)
    if np.asarray(mu_a).shape != shape or np.asarray(mu_b).shape != shape:
        raise ValueError(f"default-vector shape mismatch; expected {shape}")
    for block, means in (("C80-A", rm_a), ("C80-B", rm_b)):
        bad = [r for r, v in means.items() if np.asarray(v).shape != shape]
        if bad:
            raise ValueError(f"{block}: non-frozen role-vector shape for {bad[0]!r}")

    fracs = prov.get("default_valid_fractions")
    if not isinstance(fracs, dict):
        raise ValueError("bundle missing default_valid_fractions provenance")
    for block in ("C80-A", "C80-B"):
        vals = fracs.get(block)
        if not isinstance(vals, dict) or len(vals) != 5:
            raise ValueError(f"{block}: expected five default valid fractions")
        if min(float(x) for x in vals.values()) < params["default_valid_floor"]:
            raise ValueError(f"{block}: default condition below frozen validity floor")
    return prov


def analyze(bundle_path, counts_path, out_path, n_perm=5000, n_boot=2000, seed=0,
            bundle_revision=None, allow_dirty=False):
    pre_write_dirty = _provenance_dirty_snapshot(allow_dirty)
    bundle_revision = _require_immutable_hf_revision(bundle_revision)
    params = _frozen_params()
    mu_a, rm_a, mu_b, rm_b, counts = _load_bundle(bundle_path, counts_path)
    prov = _validate_bundle(mu_a, rm_a, mu_b, rm_b, counts, params)

    retained = confirmatory_role_set(counts["a"], counts["b"], params["threshold"])
    if len(retained) < 3:
        raise ValueError(f"only {len(retained)} roles clear the split-half threshold")

    result = confirmatory_reliability(
        mu_a, rm_a, mu_b, rm_b, retained,
        n_perm=n_perm, n_boot=n_boot, seed=seed,
    )

    # Flatten the numerical result so the standalone analyzer uses the same schema
    # as the canonical run record committed under results/t15/.
    report = {
        "task": "T15_T16_confirmatory_axis_reliability",
        "membership": _EXPECTED_MEMBERSHIP,
        "frozen_params": {
            "block_index": params["block_index"],
            "arm": params["arm"],
            "pool": params["pool"],
            "threshold": params["threshold"],
        },
        "n_retained_roles": result["n_retained_roles"],
        "cross_axis_pearson_r": result["cross_axis_pearson_r"],
        "secondary": result["secondary"],
        "nulls": result["nulls"],
        "bootstrap": result["bootstrap"],
        "leave_one_role_out": result["leave_one_role_out"],
        "input_provenance": {
            "bundle_sha256": _sha256_file(bundle_path),
            "counts_sha256": _sha256_file(counts_path),
            "t12_manifest_sha256": prov.get("t12_manifest_sha256"),
            "arm": prov.get("arm"),
            "block_index": prov.get("block_index"),
            "pool": prov.get("pool"),
            "membership": prov.get("membership"),
            "reduce_side_hf_revision": prov.get("source_hf_revision"),
            "reduce_side_hf_revision_note": (
                "immutable dataset revision the REDUCTION consumed, as recorded by the "
                "reduce cell. Null for bundles produced before the cell pinned a "
                "revision: those reductions read a mutable branch tip, so the exact "
                "input bytes are attested only by the T12 manifest SHAs above."),
            "bundle_fetched_at_hf_revision": bundle_revision,
            "bundle_fetched_at_hf_revision_note": (
                "immutable revision THIS analysis downloaded the bundle from; together "
                "with the bundle/counts SHA-256 it pins the analysed bytes exactly"),
            "membership_authority": prov.get("membership_authority"),
            "eligible_outputs": {
                "C80-A": int(sum(counts["a"].values())),
                "C80-B": int(sum(counts["b"].values())),
            },
            "default_valid_fractions": prov.get("default_valid_fractions"),
            "source_repo": prov.get("source_repo"),
        },
        "seed": seed,
    }
    report = stamp_report(
        report,
        allow_dirty=allow_dirty,
        dirty=pre_write_dirty,
    )
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"[T15/T16] retained={report['n_retained_roles']} "
        f"r={report['cross_axis_pearson_r']:.4f} "
        f"perm_p={report['nulls']['role_correspondence_permutation']['p_ge_observed']:.4f} "
        f"flip_p={report['nulls']['common_orientation_sign_flip']['p_ge_observed']:.4f} "
        f"iso_p={report['nulls']['isotropic_random_direction']['p_ge_observed']:.4f} "
        f"ci95={report['bootstrap']['ci95']} -> {out_path}"
    )
    return report


def self_test():
    rng = np.random.default_rng(7)
    d, n = 256, 20
    axis = rng.standard_normal(d); axis /= np.linalg.norm(axis)
    coord = rng.uniform(-3.0, 3.0, n)
    a, b, ca, cb = {}, {}, {}, {}
    for i in range(n):
        name = f"r{i:02d}"
        base = coord[i] * axis
        a[name] = base + 0.3 * rng.standard_normal(d)
        b[name] = base + 0.3 * rng.standard_normal(d)
        ca[name] = cb[name] = 15
    mu = 6.0 * axis + 0.1 * rng.standard_normal(d)
    retained = confirmatory_role_set(ca, cb, 10)
    res = confirmatory_reliability(mu, a, mu, b, retained,
                                   n_perm=1000, n_boot=500, seed=1)
    assert res["cross_axis_pearson_r"] > 0.9
    assert res["nulls"]["role_correspondence_permutation"]["p_ge_observed"] < 0.01
    assert res["nulls"]["isotropic_random_direction"]["p_ge_observed"] < 0.01
    assert res["bootstrap"]["ci95"][0] > 0.5

    shuffled = {f"r{i:02d}": b[f"r{(i+7) % n:02d}"] for i in range(n)}
    res0 = confirmatory_reliability(mu, a, mu, shuffled, retained,
                                    n_perm=200, n_boot=200, seed=2)
    assert abs(res0["cross_axis_pearson_r"]) < 0.5
    print("[self-test] OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--bundle", required=True)
    a.add_argument("--counts", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n-perm", type=int, default=5000)
    a.add_argument("--n-boot", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument(
        "--bundle-revision",
        required=True,
        help="immutable 40-hex HF dataset revision the bundle was downloaded from; "
             "required so the analysed bytes are pinned",
    )
    a.add_argument(
        "--allow-dirty",
        action="store_true",
        help="DEVELOPMENT ONLY: permit provenance stamping from a dirty working tree",
    )
    sub.add_parser("self-test")
    args = ap.parse_args(argv)
    if args.mode == "self-test":
        self_test()
    else:
        analyze(args.bundle, args.counts, args.out,
                n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed,
                bundle_revision=args.bundle_revision,
                allow_dirty=args.allow_dirty)


if __name__ == "__main__":
    main()
