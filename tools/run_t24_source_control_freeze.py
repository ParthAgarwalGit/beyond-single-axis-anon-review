#!/usr/bin/env python3
"""T24 — freeze Qwen source-model matched capping controls.

This is a calibration/freeze step, not an outcome-analysis step. It consumes a
local NPZ containing, for every source-capped layer L:

  axis_L      [hidden]     released Assistant direction at layer L
  natural_L   [n,hidden]   natural calibration activations for the SAME frozen
                            common calibration corpus/serialization
  default_L   [m,hidden]   frozen default-Assistant reference activations from
                            that calibration protocol

The NPZ must also carry scalar provenance fields ``calibration_source_id`` and
``calibration_manifest_sha256``. The runner verifies those values, verifies the
released Assistant-Axis and capping-config file hashes, verifies every ``axis_L``
against the released Axis, and measures the fixed Lu source cap's engagement on
the exact same ``natural_L`` rows.

Deviation 2026-08-21: the originally frozen project controls used q25 random
lower-tail and q75 sphere-radius calibration. The first genuine Qwen calibration
showed that the fixed released source cap engaged 59.59% of layer-46 natural rows,
so the predeclared +/-0.03 engagement-matching gate correctly failed before any
T29 matched-control outcomes were generated or inspected. The approved primary
correction keeps the source treatment fixed and calibrates control engagement to
the measured source rate independently at each layer: random lower-tail quantile
q = source_rate and sphere radius quantile q = 1 - source_rate. The original
q25/q75 parameters are retained in the numerical artifact/report as an unmatched
sensitivity reference. No evaluation outcomes are accepted as input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capping_controls import (  # noqa: E402
    calibrate_lower_tail_threshold,
    calibrate_sphere_radius,
    cosine,
    default_centroid,
    engagement_rate,
    lower_tail_cap,
    orthogonal_random_direction,
    source_upper_tail_engagement,
    spherical_distance_cap,
    unit_vector,
)

LAYERS = tuple(range(46, 54))
BASE_SEED = 20260818
ORIGINAL_RANDOM_LOWER_QUANTILE = 0.25
ORIGINAL_SPHERE_RADIUS_QUANTILE = 0.75
ORTHOGONALITY_TOL = 1e-6
AXIS_MATCH_COS_TOL = 1e-7
AXIS_RAW_RTOL = 1e-5
AXIS_RAW_ATOL = 1e-7
CAPPING_VECTOR_COS_TOL = 1e-6
ENGAGEMENT_TOL = 0.03
MODEL_ID = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
SOURCE_EXPERIMENT = "layers_46:54-p0.25"
SOURCE_ARTIFACT_SHA256 = "6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24"
SOURCE_ASSISTANT_AXIS_SHA256 = "a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211"
CALIBRATION_SOURCE_ID = "T21_JBB_BEHAVIOR_ONLY_ADAPTED_NOT_LU_PERSONA_JAILBREAK"
CALIBRATION_MANIFEST_SHA256 = "d611e904f3b5d47c71e5ab1f3fa1a84ead6cfd5d94dba337f5c3b71aafef7793"
DEVIATION_ID = "T24_ENGAGEMENT_MATCHING_2026-08-21"
DEVIATION_PATH = "docs/DEVIATION_2026-08-21_T24_ENGAGEMENT_MATCHING.md"


def die(msg: str) -> None:
    raise SystemExit(f"T24 ABORTED (fail-closed): {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_git_state() -> tuple[str, bool]:
    """Return HEAD SHA and whether the source repository is dirty."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
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
            capture_output=True,
            text=True,
        ).stdout

    except Exception as exc:  # pragma: no cover
        die(f"cannot resolve source Git state: {exc}")

    return sha, bool(porcelain.strip())


def require_clean_source_tree() -> str:
    """Fail closed for canonical numerical-freeze provenance."""
    sha, dirty = source_git_state()

    if dirty:
        die(
            "source Git working tree is dirty; canonical T24 freeze "
            "requires a clean committed tree"
        )

    return sha


def source_git_sha() -> str:
    """Backward-compatible helper for non-canonical callers/tests."""
    return source_git_state()[0]


def scalar_text(z: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in z:
        die(f"calibration NPZ missing required scalar metadata: {key}")
    arr = np.asarray(z[key])
    if arr.size != 1:
        die(f"calibration metadata {key} must contain exactly one scalar value")
    value = arr.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def optional_scalar_text(z: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in z:
        return None
    arr = np.asarray(z[key])
    if arr.size != 1:
        die(f"calibration metadata {key} must contain exactly one scalar value")
    value = arr.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def matched_control_quantiles(source_rate: float) -> tuple[float, float]:
    """Return random lower-tail q and sphere radius q matching source engagement."""
    q = float(source_rate)
    if not np.isfinite(q) or not 0.0 < q < 1.0:
        die(f"source engagement must lie strictly between 0 and 1; observed={q}")
    return q, 1.0 - q


def calibration_archive_from_env() -> dict | None:
    """Return the immutable shared location of the calibration input."""
    values = {
        "repo_id": os.environ.get(
            "T24_CALIBRATION_HF_REPO", ""
        ).strip(),
        "repo_type": "dataset",
        "path": os.environ.get(
            "T24_CALIBRATION_HF_PATH", ""
        ).strip(),
        "revision": os.environ.get(
            "T24_CALIBRATION_HF_REVISION", ""
        ).strip(),
    }

    present = [
        bool(values["repo_id"]),
        bool(values["path"]),
        bool(values["revision"]),
    ]

    if any(present) and not all(present):
        die(
            "incomplete T24 calibration HF archive provenance"
        )

    if not any(present):
        return None

    if not re.fullmatch(
        r"[0-9a-f]{40}",
        values["revision"],
    ):
        die(
            "T24 calibration HF revision must be an immutable "
            "40-hex commit SHA"
        )

    return values


def torch_to_numpy(x) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        die(f"torch is required to verify released .pt artifacts: {exc}")
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def load_released_axis(path: Path) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        die(f"torch is required to load Assistant Axis: {exc}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        if "axis" not in data:
            die("released Assistant-Axis file is a dict without an 'axis' key")
        data = data["axis"]
    axis = torch_to_numpy(data).astype(np.float64, copy=False)
    if axis.ndim != 2 or axis.shape[0] <= max(LAYERS) or not np.isfinite(axis).all():
        die(f"released Assistant Axis has invalid shape/content: {axis.shape}")
    return axis


def load_source_capping_interventions(path: Path) -> dict[int, tuple[np.ndarray, float]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        die(f"torch is required to load capping config: {exc}")
    cfg = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(cfg, dict) or "experiments" not in cfg or "vectors" not in cfg:
        die("released capping config must contain 'experiments' and 'vectors'")
    exp = next((e for e in cfg["experiments"] if e.get("id") == SOURCE_EXPERIMENT), None)
    if exp is None:
        die(f"source capping experiment not found: {SOURCE_EXPERIMENT}")
    by_layer: dict[int, tuple[np.ndarray, float]] = {}
    for intervention in exp.get("interventions", []):
        if "cap" not in intervention:
            continue
        vector_name = intervention.get("vector")
        if vector_name not in cfg["vectors"]:
            die(f"source capping intervention references missing vector: {vector_name}")
        vec_data = cfg["vectors"][vector_name]
        layer = int(vec_data["layer"])
        if layer in by_layer:
            die(f"source capping experiment has duplicate cap intervention at layer {layer}")
        vector = torch_to_numpy(vec_data["vector"]).astype(np.float64, copy=False)
        threshold = float(intervention["cap"])
        if vector.ndim != 1 or not np.isfinite(vector).all() or not np.isfinite(threshold):
            die(f"source capping intervention malformed at layer {layer}")
        by_layer[layer] = (vector, threshold)
    if set(by_layer) != set(LAYERS):
        die(
            "source capping experiment layer set mismatch: "
            f"expected={list(LAYERS)} observed={sorted(by_layer)}"
        )
    return by_layer


def freeze(
    calibration_npz: Path,
    assistant_axis_pt: Path,
    capping_config_pt: Path,
    out_npz: Path,
    out_json: Path,
) -> dict:
    freeze_source_git_sha = require_clean_source_tree()

    for path, label in [
        (calibration_npz, "calibration artifact"),
        (assistant_axis_pt, "released Assistant-Axis artifact"),
        (capping_config_pt, "released capping-config artifact"),
    ]:
        if not path.exists():
            die(f"{label} missing: {path}")

    axis_file_sha = sha256_file(assistant_axis_pt)
    if axis_file_sha != SOURCE_ASSISTANT_AXIS_SHA256:
        die(f"released Assistant-Axis SHA-256 mismatch: expected={SOURCE_ASSISTANT_AXIS_SHA256} observed={axis_file_sha}")
    cap_file_sha = sha256_file(capping_config_pt)
    if cap_file_sha != SOURCE_ARTIFACT_SHA256:
        die(f"released capping-config SHA-256 mismatch: expected={SOURCE_ARTIFACT_SHA256} observed={cap_file_sha}")

    released_axis = load_released_axis(assistant_axis_pt)
    source_caps = load_source_capping_interventions(capping_config_pt)

    with np.load(calibration_npz, allow_pickle=False) as z:
        observed_source_id = scalar_text(z, "calibration_source_id")
        observed_manifest_sha = scalar_text(z, "calibration_manifest_sha256")
        capture_source_git_sha = optional_scalar_text(z, "capture_source_git_sha")
        capture_schema = optional_scalar_text(z, "capture_schema")
        if observed_source_id != CALIBRATION_SOURCE_ID:
            die(f"calibration source identity mismatch: expected={CALIBRATION_SOURCE_ID} observed={observed_source_id}")
        if observed_manifest_sha != CALIBRATION_MANIFEST_SHA256:
            die(f"calibration manifest SHA-256 mismatch: expected={CALIBRATION_MANIFEST_SHA256} observed={observed_manifest_sha}")
        if capture_source_git_sha is not None and not re.fullmatch(r"[0-9a-f]{40}", capture_source_git_sha):
            die(f"capture_source_git_sha is not a 40-hex Git SHA: {capture_source_git_sha}")

        output_arrays = {}
        layers = {}
        hidden = None

        for layer in LAYERS:
            ak, nk, dk = f"axis_{layer}", f"natural_{layer}", f"default_{layer}"
            if ak not in z or nk not in z or dk not in z:
                die(f"calibration NPZ requires {ak}, {nk}, and {dk}")
            axis = np.asarray(z[ak], dtype=np.float64)
            natural = np.asarray(z[nk], dtype=np.float64)
            default = np.asarray(z[dk], dtype=np.float64)
            if axis.ndim != 1 or natural.ndim != 2 or default.ndim != 2 or natural.shape[1] != axis.shape[0] or default.shape[1] != axis.shape[0]:
                die(f"layer {layer}: malformed axis/natural/default shapes")
            if natural.shape[0] < 100 or default.shape[0] < 20:
                die(f"layer {layer}: insufficient calibration rows natural={natural.shape[0]} default={default.shape[0]}")
            if not np.isfinite(axis).all() or not np.isfinite(natural).all() or not np.isfinite(default).all():
                die(f"layer {layer}: non-finite calibration values")
            if hidden is None:
                hidden = axis.shape[0]
            elif axis.shape[0] != hidden:
                die("hidden size changes across layers")

            released_layer = np.asarray(released_axis[layer], dtype=np.float64)
            if released_layer.shape != axis.shape:
                die(f"layer {layer}: released/calibration Axis hidden-size mismatch {released_layer.shape} vs {axis.shape}")
            axis_cos = cosine(axis, released_layer)
            axis_raw_match = bool(np.allclose(axis, released_layer, rtol=AXIS_RAW_RTOL, atol=AXIS_RAW_ATOL))
            if axis_cos < 1.0 - AXIS_MATCH_COS_TOL or not axis_raw_match:
                die(f"layer {layer}: calibration axis_L does not match released Assistant Axis (cos={axis_cos:.12f}, raw_match={axis_raw_match})")

            source_vector, source_threshold = source_caps[layer]
            if source_vector.shape != axis.shape:
                die(f"layer {layer}: source capping vector hidden-size mismatch")
            source_vector_axis_cos = cosine(source_vector, released_layer)
            if source_vector_axis_cos > -1.0 + CAPPING_VECTOR_COS_TOL:
                die(f"layer {layer}: source capping vector is not the expected negated Assistant Axis (cos={source_vector_axis_cos:.12f})")
            source_engaged = source_upper_tail_engagement(natural, source_vector, source_threshold)
            source_rate = engagement_rate(source_engaged)
            random_q, sphere_q = matched_control_quantiles(source_rate)

            axis_u = unit_vector(axis)
            random_u = orthogonal_random_direction(axis_u, base_seed=BASE_SEED, layer=layer)
            abs_cos = abs(cosine(axis_u, random_u))
            if abs_cos > ORTHOGONALITY_TOL:
                die(f"layer {layer}: |cos(axis, random)|={abs_cos} exceeds tolerance")

            random_threshold = calibrate_lower_tail_threshold(natural, random_u, quantile=random_q)
            random_out, random_engaged = lower_tail_cap(natural, random_u, random_threshold)
            random_rate = engagement_rate(random_engaged)
            if np.any(~random_engaged) and not np.allclose(random_out[~random_engaged], natural[~random_engaged], rtol=0, atol=1e-12):
                die(f"layer {layer}: random directional no-op check failed")

            center = default_centroid(default)
            sphere_radius = calibrate_sphere_radius(natural, center, quantile=sphere_q)
            sphere_out, sphere_engaged = spherical_distance_cap(natural, center, sphere_radius)
            sphere_rate = engagement_rate(sphere_engaged)
            if np.any(~sphere_engaged) and not np.allclose(sphere_out[~sphere_engaged], natural[~sphere_engaged], rtol=0, atol=1e-12):
                die(f"layer {layer}: sphere no-op check failed")
            if np.any(sphere_engaged):
                post_d = np.linalg.norm(sphere_out[sphere_engaged] - center, axis=1)
                if not np.allclose(post_d, sphere_radius, rtol=1e-9, atol=1e-9):
                    die(f"layer {layer}: radial sphere boundary check failed")

            if abs(random_rate - source_rate) > ENGAGEMENT_TOL:
                die(f"layer {layer}: engagement-matched random/source mismatch random={random_rate:.4f} source_axis={source_rate:.4f}")
            if abs(sphere_rate - source_rate) > ENGAGEMENT_TOL:
                die(f"layer {layer}: engagement-matched sphere/source mismatch sphere={sphere_rate:.4f} source_axis={source_rate:.4f}")

            original_random_threshold = calibrate_lower_tail_threshold(natural, random_u, quantile=ORIGINAL_RANDOM_LOWER_QUANTILE)
            _, original_random_engaged = lower_tail_cap(natural, random_u, original_random_threshold)
            original_random_rate = engagement_rate(original_random_engaged)
            original_sphere_radius = calibrate_sphere_radius(natural, center, quantile=ORIGINAL_SPHERE_RADIUS_QUANTILE)
            _, original_sphere_engaged = spherical_distance_cap(natural, center, original_sphere_radius)
            original_sphere_rate = engagement_rate(original_sphere_engaged)

            output_arrays[f"random_unit_{layer}"] = random_u.astype(np.float32)
            output_arrays[f"default_centroid_{layer}"] = center.astype(np.float32)
            output_arrays[f"random_threshold_{layer}"] = np.asarray(random_threshold, dtype=np.float64)
            output_arrays[f"sphere_radius_{layer}"] = np.asarray(sphere_radius, dtype=np.float64)
            output_arrays[f"source_axis_threshold_{layer}"] = np.asarray(source_threshold, dtype=np.float64)
            output_arrays[f"original_q25_random_threshold_{layer}"] = np.asarray(original_random_threshold, dtype=np.float64)
            output_arrays[f"original_q75_sphere_radius_{layer}"] = np.asarray(original_sphere_radius, dtype=np.float64)
            layers[str(layer)] = {
                "n_natural_calibration": int(natural.shape[0]),
                "n_default_reference": int(default.shape[0]),
                "hidden_size": int(axis.shape[0]),
                "axis_released_cosine": axis_cos,
                "axis_raw_allclose": axis_raw_match,
                "source_capping_vector_axis_cosine": source_vector_axis_cos,
                "source_axis_threshold": source_threshold,
                "source_axis_engagement_rate": source_rate,
                "abs_cos_axis_random": abs_cos,
                "primary_matching_rule": "engagement_matched_to_measured_source_rate",
                "random_lower_tail_quantile": random_q,
                "random_threshold": random_threshold,
                "random_engagement_rate": random_rate,
                "random_minus_source_engagement": random_rate - source_rate,
                "sphere_center": "default-Assistant centroid from default_L",
                "sphere_radius_quantile": sphere_q,
                "sphere_radius": sphere_radius,
                "sphere_engagement_rate": sphere_rate,
                "sphere_minus_source_engagement": sphere_rate - source_rate,
                "original_predeclared_sensitivity": {
                    "random_lower_tail_quantile": ORIGINAL_RANDOM_LOWER_QUANTILE,
                    "random_threshold": original_random_threshold,
                    "random_engagement_rate": original_random_rate,
                    "sphere_radius_quantile": ORIGINAL_SPHERE_RADIUS_QUANTILE,
                    "sphere_radius": original_sphere_radius,
                    "sphere_engagement_rate": original_sphere_rate,
                    "engagement_matched_to_source": False,
                },
            }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **output_arrays)
    report = {
        "schema_version": "t24-source-control-freeze/1.3",
        "task": "T24",
        "status": "FROZEN_NUMERICAL_CONTROLS",
        "source_git_sha": freeze_source_git_sha,
        "source_git_dirty": False,
        "calibration_archive": calibration_archive_from_env(),
        "interpretation_notes": {
            "engagement_gate_role": "self_consistency_regression_guard_not_independent_evidence",
            "engagement_gate_explanation": "Primary control quantiles are derived from measured source engagement. The 0.03 gate therefore checks implementation consistency, tie handling, and inequality semantics; it is not independent evidence of nontrivial control comparability.",
            "majority_engagement_layers": [46, 47, 48, 49],
            "sphere_control_asymmetry": "At layers 46-49 the matched spherical control radially projects a majority of natural rows toward the default-Assistant centroid. A null T29 sphere result is correspondingly strong, whereas a positive result is more ambiguous between intervention strength and geometry.",
        },
        "deviation": {
            "id": DEVIATION_ID,
            "path": DEVIATION_PATH,
            "trigger": "original q25/q75 controls failed the predeclared engagement-match gate on genuine Qwen calibration",
            "approved_before_t29_matched_control_outcomes": True,
            "source_treatment_changed": False,
            "primary_rule": "match each project control's engagement to measured source-cap engagement per layer",
            "original_q25_q75_retained_as_sensitivity": True,
        },
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16", "thinking": False},
        "source_axis_condition": {
            "authority": "T21 source-capping reproduction",
            "assistant_axis_sha256": axis_file_sha,
            "capping_config_sha256": cap_file_sha,
            "setting": SOURCE_EXPERIMENT,
            "layers": list(LAYERS),
            "retuned_in_t24": False,
            "engagement_measurement": "released per-layer capping vectors/thresholds applied as measurement only to the same frozen natural calibration rows",
        },
        "natural_calibration_corpus": {
            "source_id": observed_source_id,
            "manifest_sha256": observed_manifest_sha,
            "condition": "UNSTEERED",
            "outcome_labels_used": False,
            "capture_schema": capture_schema,
            "capture_source_git_sha": capture_source_git_sha,
        },
        "control_freeze": {
            "random_base_seed": BASE_SEED,
            "random_rule": "per-layer Gaussian draw; remove source-Axis component; unit normalize",
            "random_threshold_rule": "own natural projection quantile equal to measured source engagement rate",
            "sphere_center_rule": "frozen default-Assistant centroid per layer",
            "sphere_radius_rule": "natural distance-to-default-centroid quantile equal to 1 - measured source engagement rate",
            "source_treatment_engagement_measured_not_assumed": True,
            "outcome_data_used_for_calibration": False,
            "post_outcome_retuning_forbidden": True,
            "original_predeclared_sensitivity": {
                "random_lower_tail_quantile": ORIGINAL_RANDOM_LOWER_QUANTILE,
                "sphere_radius_quantile": ORIGINAL_SPHERE_RADIUS_QUANTILE,
                "role": "unmatched-engagement sensitivity/reference only",
            },
        },
        "matching_constraints": [
            "same model/revision",
            "same evaluation rows",
            "same layers 46-53",
            "same token positions",
            "same dtype",
            "same decoding",
            "same scoring",
            "same natural calibration corpus/serialization",
        ],
        "calibration_input_sha256": sha256_file(calibration_npz),
        "numerical_controls_npz": str(out_npz),
        "layers": layers,
        "validation": {
            "released_axis_sha256_check": "PASS",
            "released_capping_config_sha256_check": "PASS",
            "calibration_axis_vs_released_axis_check": "PASS",
            "source_capping_vector_orientation_check": "PASS",
            "source_axis_engagement_measured": "PASS",
            "orthogonality_tolerance_absolute_cosine": ORTHOGONALITY_TOL,
            "engagement_reference": "measured source Axis-cap rate on same natural rows",
            "engagement_absolute_tolerance": ENGAGEMENT_TOL,
            "random_directional_no_op_check": "PASS",
            "sphere_no_op_check": "PASS",
            "sphere_radial_boundary_check": "PASS",
            "primary_controls_engagement_matched": "PASS",
        },
        "claim_boundary": "Random-direction and spherical-distance caps are project controls, not Lu source methods. Primary controls are matched to source engagement incidence, not intervention magnitude or geometry. This artifact freezes operators/calibration only and contains no T29 evaluation outcomes.",
    }
    report["numerical_controls_npz_sha256"] = sha256_file(out_npz)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration-npz", required=True)
    ap.add_argument("--assistant-axis-pt", required=True, help="Pinned qwen-3-32b/assistant_axis.pt from the T21 artifact revision")
    ap.add_argument("--capping-config-pt", required=True, help="Pinned qwen-3-32b/capping_config.pt from the T21 artifact revision")
    ap.add_argument("--out-npz", default="results/t24/source_control_parameters.npz")
    ap.add_argument("--out-json", default="results/t24/source_control_freeze.json")
    a = ap.parse_args(argv)
    report = freeze(Path(a.calibration_npz), Path(a.assistant_axis_pt), Path(a.capping_config_pt), Path(a.out_npz), Path(a.out_json))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
