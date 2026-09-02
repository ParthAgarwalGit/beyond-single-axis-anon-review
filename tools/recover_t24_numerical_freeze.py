#!/usr/bin/env python3
"""Recover the T24 numerical freeze from an already-completed Qwen calibration NPZ.

Use this only when the expensive outcome-blind activation capture completed but
the downstream numerical-freeze gate aborted. It does not load Qwen or regenerate
any calibration rows. It reuses the existing calibration NPZ, invokes the current
T24 freeze runner, and writes a small receipt that distinguishes the capture Git
SHA from the later freeze/recovery Git SHA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MANIFEST_SHA = "d611e904f3b5d47c71e5ab1f3fa1a84ead6cfd5d94dba337f5c3b71aafef7793"
EXPECTED_CAPTURE_SCHEMA = "t24-qwen-calibration/1.0"
EXPECTED_MODEL_ID = "Qwen/Qwen3-32B"
EXPECTED_MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
EXPECTED_AXIS_SHA = "a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211"
EXPECTED_CAP_SHA = "6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24"
EXPECTED_REVIEWED_CONTROLS_SHA = "adf1c43cd2715676169351f69fced45d0046e70c5a74f52baa2dc5f2aa243291"
LAYERS = list(range(46, 54))


def die(msg: str) -> None:
    raise SystemExit(f"T24 RECOVERY ABORTED (fail-closed): {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_state() -> tuple[str, bool]:
    """Return HEAD SHA and source-tree dirty state."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

        porcelain = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout

    except Exception as exc:
        die(f"cannot resolve source Git state: {exc}")

    return sha, bool(porcelain.strip())


def require_clean_source_tree() -> str:
    """Fail closed before canonical T24 recovery."""
    sha, dirty = git_state()

    if dirty:
        die(
            "source Git working tree is dirty; canonical recovery "
            "requires a clean committed tree"
        )

    return sha


def scalar_text(z: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in z:
        die(f"calibration NPZ missing {key}")
    arr = np.asarray(z[key])
    if arr.size != 1:
        die(f"calibration NPZ {key} must be scalar")
    value = arr.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)



def reconcile_reviewed_controls(
    calibration: Path,
    fresh_npz: Path,
    reviewed_npz: Path,
    report: dict,
) -> tuple[dict, dict]:
    """Fail closed unless a fresh recomputation is operationally identical.

    The reviewed control artifact remains canonical. Cross-runtime float64
    projection reductions may differ at the final few ulps, so we permit only
    scalar random-threshold differences <= 1e-12, while requiring every other
    array to be byte-identical. The permitted threshold differences must also
    collapse to the same float32 representation and produce identical
    engagement masks on the frozen calibration rows.
    """

    if not reviewed_npz.exists():
        die(
            f"reviewed canonical controls missing: {reviewed_npz}"
        )

    canonical_sha = sha256_file(reviewed_npz)

    if canonical_sha != EXPECTED_REVIEWED_CONTROLS_SHA:
        die(
            "reviewed canonical controls SHA mismatch: "
            f"expected={EXPECTED_REVIEWED_CONTROLS_SHA} "
            f"observed={canonical_sha}"
        )

    fresh_sha = sha256_file(fresh_npz)

    tolerance = 1e-12

    allowed_threshold = re.compile(
        r"^(?:random_threshold|original_q25_random_threshold)_"
        r"(46|47|48|49|50|51|52|53)$"
    )

    nonexact = []
    exact_arrays = 0
    max_abs_difference = 0.0
    all_float32_equal = True

    with (
        np.load(fresh_npz, allow_pickle=False) as fresh,
        np.load(reviewed_npz, allow_pickle=False) as canonical,
        np.load(calibration, allow_pickle=False) as cal,
    ):
        if fresh.files != canonical.files:
            die(
                "fresh/canonical controls key order differs"
            )

        for key in fresh.files:
            a = np.asarray(fresh[key])
            b = np.asarray(canonical[key])

            if a.shape != b.shape or a.dtype != b.dtype:
                die(
                    f"fresh/canonical shape or dtype mismatch for {key}: "
                    f"{a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
                )

            if np.array_equal(
                a,
                b,
                equal_nan=True,
            ):
                exact_arrays += 1
                continue

            if not allowed_threshold.fullmatch(key):
                die(
                    "fresh recomputation changed a non-threshold "
                    f"canonical array: {key}"
                )

            if a.shape != () or a.dtype != np.float64:
                die(
                    f"allowed threshold has unexpected representation: {key}"
                )

            av = float(a)
            bv = float(b)

            delta = abs(av - bv)

            if delta > tolerance:
                die(
                    f"fresh/canonical threshold mismatch too large for {key}: "
                    f"delta={delta:.17g} tolerance={tolerance}"
                )

            same_float32 = (
                np.float32(av).tobytes()
                == np.float32(bv).tobytes()
            )

            if not same_float32:
                die(
                    f"fresh/canonical threshold differs at float32 for {key}"
                )

            all_float32_equal &= same_float32
            max_abs_difference = max(
                max_abs_difference,
                delta,
            )

            nonexact.append(
                {
                    "key": key,
                    "fresh": av,
                    "canonical": bv,
                    "absolute_difference": delta,
                    "float32_equal": same_float32,
                }
            )

        # Prove the differing scalar thresholds do not change which frozen
        # calibration rows engage.
        for layer in LAYERS:
            natural = np.asarray(
                cal[f"natural_{layer}"],
                dtype=np.float64,
            )

            direction = np.asarray(
                canonical[f"random_unit_{layer}"],
                dtype=np.float64,
            )

            projection = natural @ direction

            for key in (
                f"random_threshold_{layer}",
                f"original_q25_random_threshold_{layer}",
            ):
                fresh_t = float(fresh[key])
                canonical_t = float(canonical[key])

                fresh_mask = projection < fresh_t
                canonical_mask = projection < canonical_t

                if not np.array_equal(
                    fresh_mask,
                    canonical_mask,
                ):
                    die(
                        f"fresh/canonical engagement masks differ for {key}"
                    )

        # Make the freeze report match the exact canonical artifact.
        for layer in LAYERS:
            row = report["layers"][str(layer)]

            row["random_threshold"] = float(
                canonical[f"random_threshold_{layer}"]
            )

            row["sphere_radius"] = float(
                canonical[f"sphere_radius_{layer}"]
            )

            row["source_axis_threshold"] = float(
                canonical[f"source_axis_threshold_{layer}"]
            )

            sensitivity = row[
                "original_predeclared_sensitivity"
            ]

            sensitivity["random_threshold"] = float(
                canonical[
                    f"original_q25_random_threshold_{layer}"
                ]
            )

            sensitivity["sphere_radius"] = float(
                canonical[
                    f"original_q75_sphere_radius_{layer}"
                ]
            )

    reproduction = {
        "status": "PASS_OPERATIONAL_EQUIVALENCE",
        "canonical_reviewed_sha256": canonical_sha,
        "fresh_recompute_sha256": fresh_sha,
        "byte_identical_reproduction": fresh_sha == canonical_sha,
        "array_count": exact_arrays + len(nonexact),
        "exact_array_count": exact_arrays,
        "nonexact_array_count": len(nonexact),
        "nonexact_arrays": nonexact,
        "allowed_absolute_tolerance": tolerance,
        "maximum_absolute_difference": max_abs_difference,
        "all_nonthreshold_arrays_exact": True,
        "all_differing_thresholds_float32_equal": all_float32_equal,
        "all_calibration_engagement_masks_equal": True,
        "canonicalization_rule": (
            "Preserve the exact independently reviewed artifact when a "
            "fresh clean-tree recomputation differs only by <=1e-12 in "
            "float64 random-projection threshold scalars, with identical "
            "float32 values and identical frozen-calibration engagement."
        ),
    }

    # Restore exact reviewed bytes as the canonical production artifact.
    shutil.copyfile(
        reviewed_npz,
        fresh_npz,
    )

    if sha256_file(fresh_npz) != EXPECTED_REVIEWED_CONTROLS_SHA:
        die(
            "failed to restore exact reviewed canonical controls"
        )

    report["numerical_controls_npz_sha256"] = (
        EXPECTED_REVIEWED_CONTROLS_SHA
    )

    report["canonical_controls_reproduction"] = reproduction

    return report, reproduction



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration-npz", required=True)
    ap.add_argument("--assistant-axis-pt", required=True)
    ap.add_argument("--capping-config-pt", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument(
        "--canonical-controls-npz",
        default=None,
        help=(
            "Exact independently reviewed control artifact to "
            "preserve after strict fresh-recompute equivalence "
            "verification."
        ),
    )
    a = ap.parse_args(argv)

    calibration = Path(a.calibration_npz).resolve()
    axis = Path(a.assistant_axis_pt).resolve()
    cap = Path(a.capping_config_pt).resolve()
    out_npz = Path(a.out_npz).resolve()
    out_json = Path(a.out_json).resolve()
    receipt_path = Path(a.receipt).resolve()

    for p, label in [(calibration, "calibration NPZ"), (axis, "Assistant Axis"), (cap, "capping config")]:
        if not p.exists():
            die(f"missing {label}: {p}")

    if sha256_file(axis) != EXPECTED_AXIS_SHA:
        die("Assistant Axis SHA mismatch")
    if sha256_file(cap) != EXPECTED_CAP_SHA:
        die("capping config SHA mismatch")

    with np.load(calibration, allow_pickle=False) as z:
        manifest_sha = scalar_text(z, "calibration_manifest_sha256")
        capture_schema = scalar_text(z, "capture_schema")
        capture_git_sha = scalar_text(z, "capture_source_git_sha")
        if manifest_sha != EXPECTED_MANIFEST_SHA:
            die("calibration manifest SHA mismatch")
        if capture_schema != EXPECTED_CAPTURE_SCHEMA:
            die(f"capture schema mismatch: {capture_schema}")
        if not re.fullmatch(r"[0-9a-f]{40}", capture_git_sha):
            die(f"invalid capture_source_git_sha: {capture_git_sha}")
        for L in LAYERS:
            for key in (f"axis_{L}", f"natural_{L}", f"default_{L}"):
                if key not in z:
                    die(f"calibration NPZ missing {key}")

    freeze_git_sha = require_clean_source_tree()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_t24_source_control_freeze.py"),
            "--calibration-npz", str(calibration),
            "--assistant-axis-pt", str(axis),
            "--capping-config-pt", str(cap),
            "--out-npz", str(out_npz),
            "--out-json", str(out_json),
        ],
        cwd=str(ROOT),
        check=True,
    )

    report = json.loads(out_json.read_text(encoding="utf-8"))
    if report.get("status") != "FROZEN_NUMERICAL_CONTROLS":
        die("freeze runner did not produce FROZEN_NUMERICAL_CONTROLS")
    if report.get("source_git_sha") != freeze_git_sha:
        die("freeze report Git SHA does not match recovery checkout")
    if report.get("source_git_dirty") is not False:
        die("freeze report does not certify a clean source Git tree")
    if report.get("deviation", {}).get("id") != "T24_ENGAGEMENT_MATCHING_2026-08-21":
        die("freeze report is not bound to the approved engagement-matching deviation")


    if a.canonical_controls_npz:
        canonical_controls = Path(
            a.canonical_controls_npz
        ).resolve()

        report, reproduction = reconcile_reviewed_controls(
            calibration=calibration,
            fresh_npz=out_npz,
            reviewed_npz=canonical_controls,
            report=report,
        )

        # Re-write the report after canonical reconciliation so that
        # the on-disk freeze JSON and receipt bind the exact reviewed NPZ.
        out_json.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

    receipt = {
        "schema_version": "t24-calibration-recovery-receipt/1.0",
        "task": "T24",
        "status": "REAL_QWEN_CALIBRATION_CAPTURED_AND_FREEZE_RECOVERED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "capture_source_git_sha": capture_git_sha,
        "freeze_source_git_sha": freeze_git_sha,
        "freeze_source_git_dirty": False,
        "calibration_archive": report.get("calibration_archive"),
        "model": {
            "id": EXPECTED_MODEL_ID,
            "revision": EXPECTED_MODEL_REVISION,
            "dtype": "bfloat16",
            "thinking": False,
        },
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "n_items": 100,
        "layers": LAYERS,
        "calibration_npz_external_path": str(calibration),
        "calibration_npz_sha256": sha256_file(calibration),
        "numerical_controls_npz_external_path": str(out_npz),
        "numerical_controls_npz_sha256": sha256_file(out_npz),
        "freeze_json": str(out_json),
        "freeze_json_sha256": sha256_file(out_json),
        "assistant_axis_sha256": sha256_file(axis),
        "capping_config_sha256": sha256_file(cap),
        "deviation_id": "T24_ENGAGEMENT_MATCHING_2026-08-21",
        "outcome_labels_used": False,
        "expensive_capture_reused": True,
        "canonical_controls_reproduction": report.get("canonical_controls_reproduction"),
        "claim_boundary": "Recovery/freeze provenance only; causal specificity requires T29 outcomes.",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
