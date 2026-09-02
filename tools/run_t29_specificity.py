#!/usr/bin/env python
"""Run T29 paired source-model Axis-vs-control causal-specificity analysis.

Production T29 is fail-closed on three upstream authorities:

1. T21 source condition: the pinned Qwen3-32B Assistant Axis and capping config.
2. T24 numerical freeze: a real ``FROZEN_NUMERICAL_CONTROLS`` JSON plus the exact
   controls NPZ whose SHA-256 is recorded by that freeze.
3. T27 causal-judge validation: the qwen_capping gate and the scoring-run provenance
   for the automatic harmfulness labels used by T29.

The automatic scorer can produce a clean primary T29 result only when T27 marks the
harmfulness instrument ``VALIDATED``. An explicit diagnostic override may compute the
same statistics after a failed/non-gating T27 result, but the emitted report is stamped
``PROVISIONAL_PENDING_T26_T27`` and its primary result is mechanically withheld.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.causal_specificity import (  # noqa: E402
    REQUIRED_CONDITIONS,
    capability_summary,
    causal_specificity_report,
)
from src.provenance import stamp_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_ID = "Qwen/Qwen3-32B"
EXPECTED_MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
EXPECTED_AXIS_SHA256 = "a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211"
EXPECTED_CAPPING_SHA256 = "6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24"
EXPECTED_T24_STATUS = "FROZEN_NUMERICAL_CONTROLS"
EXPECTED_T24_SETTING = "layers_46:54-p0.25"
EXPECTED_T27_BRANCH = "qwen_capping"
_SHA64 = re.compile(r"^[0-9a-f]{64}$")
_SHA40 = re.compile(r"^[0-9a-f]{40}$")


def die(msg):
    raise SystemExit(f"T29 ABORTED (fail-closed): {msg}")


def _sha256_file(path):
    p = Path(path)
    if not p.exists() or not p.is_file():
        die(f"required artifact missing: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path, label):
    p = Path(path)
    if not p.exists():
        die(f"{label} missing: {p}")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        die(f"cannot parse {label} {p}: {exc}")
    if not isinstance(doc, dict):
        die(f"{label} must be a JSON object")
    return doc


def _require_sha64(value, label):
    v = str(value or "").lower()
    if not _SHA64.fullmatch(v):
        die(f"{label} missing or not a full SHA-256")
    return v


def validate_t21_source_config(path):
    """Verify the committed T21 authority carries the exact source pins T24/T29 use."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project dependency
        die(f"PyYAML is required to verify the T21 source config: {exc}")
    p = Path(path)
    if not p.exists():
        die(f"T21 source config missing: {p}")
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as exc:
        die(f"cannot parse T21 source config {p}: {exc}")
    if not isinstance(doc, dict):
        die("T21 source config is not a mapping")
    model = doc.get("model") or {}
    cap = doc.get("capping") or {}
    checks = {
        "model.id": (model.get("id"), EXPECTED_MODEL_ID),
        "model.revision": (model.get("revision"), EXPECTED_MODEL_REVISION),
        "capping.assistant_axis_sha256": (cap.get("assistant_axis_sha256"), EXPECTED_AXIS_SHA256),
        "capping.artifact_sha256": (cap.get("artifact_sha256"), EXPECTED_CAPPING_SHA256),
        "capping.experiment": (cap.get("experiment"), EXPECTED_T24_SETTING),
    }
    for label, (got, want) in checks.items():
        if got != want:
            die(f"T21 source pin mismatch for {label}: expected {want!r}, got {got!r}")
    return {
        "path": str(p),
        "sha256": _sha256_file(p),
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "assistant_axis_sha256": EXPECTED_AXIS_SHA256,
        "capping_config_sha256": EXPECTED_CAPPING_SHA256,
        "setting": EXPECTED_T24_SETTING,
    }


def validate_t24_freeze(freeze_path, controls_npz, t21):
    """Require the real numerical T24 freeze and byte-verify its controls NPZ."""
    doc = _load_json(freeze_path, "T24 numerical freeze")
    if doc.get("task") != "T24":
        die(f"T24 freeze has task={doc.get('task')!r}, expected 'T24'")
    if doc.get("status") != EXPECTED_T24_STATUS:
        die(
            f"T24 is not numerically frozen: status={doc.get('status')!r}; "
            f"required={EXPECTED_T24_STATUS!r}"
        )
    source = doc.get("source_axis_condition") or {}
    if source.get("assistant_axis_sha256") != t21["assistant_axis_sha256"]:
        die("T24 Assistant-Axis SHA does not match the frozen T21 source authority")
    if source.get("capping_config_sha256") != t21["capping_config_sha256"]:
        die("T24 capping-config SHA does not match the frozen T21 source authority")
    if source.get("setting") != t21["setting"]:
        die("T24 source setting does not match the frozen T21 source authority")
    model = doc.get("model") or {}
    if model.get("id") != t21["model_id"] or model.get("revision") != t21["model_revision"]:
        die("T24 model/revision does not match T21")

    expected_controls_sha = _require_sha64(
        doc.get("numerical_controls_npz_sha256"), "T24 numerical_controls_npz_sha256"
    )
    observed_controls_sha = _sha256_file(controls_npz)
    if observed_controls_sha != expected_controls_sha:
        die(
            "T24 controls NPZ hash mismatch: the supplied controls are not the bytes "
            "frozen by source_control_freeze.json"
        )
    for field in ("calibration_input_sha256",):
        _require_sha64(doc.get(field), f"T24 {field}")
    validation = doc.get("validation") or {}
    required_passes = (
        "released_axis_sha256_check",
        "released_capping_config_sha256_check",
        "calibration_axis_vs_released_axis_check",
        "source_capping_vector_orientation_check",
        "source_axis_engagement_measured",
    )
    for key in required_passes:
        if validation.get(key) != "PASS":
            die(f"T24 freeze validation {key} is not PASS")
    return {
        "path": str(freeze_path),
        "sha256": _sha256_file(freeze_path),
        "status": doc["status"],
        "numerical_controls_npz": str(controls_npz),
        "numerical_controls_npz_sha256": observed_controls_sha,
        "calibration_input_sha256": doc["calibration_input_sha256"],
        "source_git_sha": doc.get("source_git_sha"),
        "source_axis_condition": source,
    }


def validate_judge_binding(t27_path, scoring_report_path, data, *, allow_unvalidated=False):
    """Bind T29 labels to the Qwen scorer and T27 gate.

    A clean primary result requires a real gating T27 report whose harmfulness branch
    is VALIDATED. A non-validated branch may only proceed under the explicit diagnostic
    override, which the caller must stamp provisional and withhold as primary evidence.
    """
    t27 = _load_json(t27_path, "T27 causal-judge validation")
    scoring = _load_json(scoring_report_path, "causal judge scoring provenance")

    if t27.get("task") != "T27_causal_judge_validation":
        die("T27 artifact has the wrong task identifier")
    if t27.get("branch") != EXPECTED_T27_BRANCH:
        die(f"T27 branch must be {EXPECTED_T27_BRANCH!r}")
    if scoring.get("task") != "causal_judge_scoring" or scoring.get("branch") != EXPECTED_T27_BRANCH:
        die("judge scoring report is not the qwen_capping production scorer")

    t27_inst = t27.get("judged_instrument") or {}
    model = scoring.get("judge_model")
    provider = scoring.get("provider")
    if not model or not provider:
        die("judge scoring report lacks model/provider provenance")
    if t27_inst.get("judge_model") != model or t27_inst.get("provider") != provider:
        die("T27 validated a different judge model/provider than the scorer used by T29")
    if scoring.get("production") is not True or scoring.get("drawable_by_t26") is not True:
        die("judge scoring report is not a production/drawable Qwen scoring run")
    canary = scoring.get("canary") or {}
    if canary.get("drift_detected") is not False:
        die("judge scoring run has detected or unknown canary drift")
    source_git_sha = str(scoring.get("source_git_sha") or "").lower()
    if not _SHA40.fullmatch(source_git_sha):
        die("judge scoring report lacks a full source_git_sha")

    scored_sha = _require_sha64(
        (scoring.get("artifact_sha256") or {}).get("scored"),
        "judge scored-artifact SHA",
    )
    raw_sha = _require_sha64(
        (scoring.get("artifact_sha256") or {}).get("raw"),
        "judge raw-artifact SHA",
    )
    input_scored_sha = _require_sha64(
        data.get("source_scored_artifact_sha256"),
        "T29 input source_scored_artifact_sha256",
    )
    if input_scored_sha != scored_sha:
        die(
            "T29 input is not bound to the scored artifact in judge provenance; "
            "source_scored_artifact_sha256 differs"
        )
    if data.get("label_source") != "automatic_causal_judge":
        die("T29 production input must declare label_source='automatic_causal_judge'")

    harmful = (t27.get("outcomes") or {}).get("harmfulness") or {}
    gate = harmful.get("gate") or {}
    gating = t27.get("gating") is True
    validated = gating and gate.get("branch") == "VALIDATED"
    if not validated and not allow_unvalidated:
        die(
            "T27 has not validated the Qwen harmfulness scorer for full-corpus primary use. "
            f"gating={t27.get('gating')!r}, harmfulness_branch={gate.get('branch')!r}. "
            "Refusing a clean T29 primary result. Use --allow-unvalidated-judge only "
            "for a clearly stamped provisional diagnostic."
        )

    return {
        "validated_for_primary": validated,
        "t27_path": str(t27_path),
        "t27_sha256": _sha256_file(t27_path),
        "t27_gating": t27.get("gating"),
        "t27_harmfulness_branch": gate.get("branch"),
        "t27_primary_evidence": gate.get("primary_evidence"),
        "judge_model": model,
        "judge_provider": provider,
        "judge_endpoint": scoring.get("endpoint"),
        "judge_source_git_sha": source_git_sha,
        "judge_prompt_sha256": scoring.get("prompt_sha256"),
        "judge_prompt_file_sha256": scoring.get("prompt_file_sha256"),
        "judge_config_sha256": scoring.get("config_sha256"),
        "judge_scoring_report_path": str(scoring_report_path),
        "judge_scoring_report_sha256": _sha256_file(scoring_report_path),
        "judge_scored_artifact_sha256": scored_sha,
        "judge_raw_artifact_sha256": raw_sha,
        "canary": {
            "n": canary.get("n"),
            "start_sha256": canary.get("start_sha256"),
            "end_sha256": canary.get("end_sha256"),
            "drift_detected": canary.get("drift_detected"),
        },
    }


def _withhold_primary(report, judge):
    diagnostic = dict(report["primary_specificity"])
    report["primary_specificity"] = {
        "status": "WITHHELD_UNVALIDATED_SCORER",
        "primary_claim_allowed": False,
        "reason": (
            "T27 did not validate the automatic harmfulness scorer for full-corpus "
            "primary evidence. The inferential numbers below are retained only as an "
            "automatic-label diagnostic and must not be reported as the T29 primary result."
        ),
        "t27_harmfulness_branch": judge["t27_harmfulness_branch"],
        "automatic_label_diagnostic": diagnostic,
    }
    report["outcomes"]["primary"] = "WITHHELD_PENDING_T26_T27_VALIDATION"
    report["all_comparison_p_values_status"] = "PROVISIONAL_DIAGNOSTIC_ONLY"
    return report


def analyze(
    input_path,
    out_path,
    *,
    t21_config,
    t24_freeze,
    t24_controls_npz,
    t27_gate,
    judge_scoring_report,
    n_boot=10000,
    seed=290819,
    allow_unvalidated_judge=False,
):
    data = _load_json(input_path, "T29 outcomes")
    if "items" not in data:
        die("input JSON must contain an 'items' array")

    t21 = validate_t21_source_config(t21_config)
    t24 = validate_t24_freeze(t24_freeze, t24_controls_npz, t21)
    judge = validate_judge_binding(
        t27_gate,
        judge_scoring_report,
        data,
        allow_unvalidated=allow_unvalidated_judge,
    )

    report = causal_specificity_report(data["items"], n_boot=n_boot, seed=seed)
    if "capability" in data:
        report["capability"] = capability_summary(data["capability"])

    status = "PRIMARY_VALIDATED_SCORER" if judge["validated_for_primary"] else "PROVISIONAL_PENDING_T26_T27"
    if not judge["validated_for_primary"]:
        report = _withhold_primary(report, judge)

    out = {
        "task": "T29_source_capping_specificity",
        "status": status,
        "primary_result_eligible": bool(judge["validated_for_primary"]),
        "input": {
            "input_sha256": _sha256_file(input_path),
            "source_scored_artifact_sha256": data["source_scored_artifact_sha256"],
            "label_source": data["label_source"],
            "bootstrap_replicates": int(n_boot),
            "bootstrap_seed": int(seed),
            "required_conditions": list(REQUIRED_CONDITIONS),
        },
        "upstream_freezes": {"T21": t21, "T24": t24},
        "judge_provenance": judge,
        "report": report,
        "claim_guard": (
            "A clean T29 primary result is mechanically permitted only when the real "
            "T24 numerical freeze is byte-verified and T27 marks Qwen harmfulness VALIDATED."
        ),
    }
    out = stamp_report(out, allow_dirty=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")

    if judge["validated_for_primary"]:
        p = report["primary_specificity"]["mcnemar_exact_two_sided_p"]
        rd = report["primary_specificity"]["risk_difference"]
        ci = report["primary_specificity"]["bootstrap_95ci"]
        print(
            f"[T29 PRIMARY] axis-minus-random strict-harm RD={rd:.6f}, "
            f"95% CI=[{ci[0]:.6f}, {ci[1]:.6f}], McNemar p={p:.6g} -> {out_path}"
        )
    else:
        print(
            "[T29 PROVISIONAL] automatic-label diagnostic emitted, but the primary "
            f"result is WITHHELD because T27 harmfulness={judge['t27_harmfulness_branch']} -> {out_path}"
        )
    return out


def self_test():
    rows = []
    harms = {
        "UNSTEERED": 10,
        "ASSISTANT_AXIS_CAP_SOURCE_SETTING": 5,
        "ORTHOGONAL_RANDOM_CAP": 20,
        "SPHERICAL_DISTANCE_CAP": 15,
    }
    for condition in REQUIRED_CONDITIONS:
        for i in range(100):
            rows.append({
                "item_id": f"item-{i:03d}",
                "condition": condition,
                "harmfulness_label": "harmful" if i < harms[condition] else "refused",
                "technical_valid": True,
            })
    rep = causal_specificity_report(rows, n_boot=1000, seed=1)
    primary = rep["comparisons"]["axis_vs_random"]["strict"]
    assert primary["risk_difference"] < 0
    assert primary["bootstrap_95ci"][1] < 0
    assert primary["mcnemar_exact_two_sided_p"] < 0.001
    print("[self-test] paired statistics core OK; production bindings are tested separately")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--input", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--t21-config", default=str(ROOT / "configs" / "t21_qwen_source_capping.yaml"))
    a.add_argument("--t24-freeze", required=True,
                   help="real results/t24/source_control_freeze.json with status FROZEN_NUMERICAL_CONTROLS")
    a.add_argument("--t24-controls-npz", required=True,
                   help="source_control_parameters.npz; its bytes must match the SHA recorded by T24")
    a.add_argument("--t27-gate", required=True,
                   help="aggregate results/t27/causal_judge_validation.json for branch qwen_capping")
    a.add_argument("--judge-scoring-report", required=True,
                   help="qwen_capping causal-judge scoring provenance report used to produce the labels")
    a.add_argument("--n-boot", type=int, default=10000)
    a.add_argument("--seed", type=int, default=290819)
    a.add_argument(
        "--allow-unvalidated-judge",
        action="store_true",
        help=("diagnostic only: allow a non-VALIDATED T27 harmfulness branch; output is "
              "stamped PROVISIONAL_PENDING_T26_T27 and primary evidence is withheld"),
    )
    sub.add_parser("self-test")
    args = ap.parse_args(argv)
    if args.mode == "self-test":
        self_test()
    else:
        analyze(
            args.input,
            args.out,
            t21_config=args.t21_config,
            t24_freeze=args.t24_freeze,
            t24_controls_npz=args.t24_controls_npz,
            t27_gate=args.t27_gate,
            judge_scoring_report=args.judge_scoring_report,
            n_boot=args.n_boot,
            seed=args.seed,
            allow_unvalidated_judge=args.allow_unvalidated_judge,
        )


if __name__ == "__main__":
    main()
