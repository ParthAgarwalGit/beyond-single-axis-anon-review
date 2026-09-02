#!/usr/bin/env python3
"""Reconcile the existing DeepSeek T25 5k signed-steering corpus.

This tool is outcome-blind. It verifies design completeness, signed dose
consistency and magnitude, P50 membership, blinded condition binding,
held-out-Axis exclusion provenance, and optional causal activation provenance.
It does not regenerate outputs and never loads completion/output text into the
reconciliation corpus.

Candidate audit while T22 is not yet frozen:

  python tools/reconcile_t25_deepseek_5k.py \
    --outputs /persistent/t25/final_outputs.jsonl \
    --artifact-location 'HF_REPO@IMMUTABLE_REVISION/path/to/final_outputs.jsonl' \
    --p50-manifest design/causal_P50_assistant_proximal.json \
    --condition-map /persistent/t25/condition_map_PRIVATE.json \
    --run-provenance /persistent/t25/run_provenance.json \
    --allow-candidate-p50

Final intent is the default. It requires a FROZEN P50, a stable external raw
artifact location, run provenance binding the observed coefficient, and
held-out-Axis provenance bound to that exact P50 manifest. Candidate and final
outputs use different default filenames.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_frozen_config  # noqa: E402
from src.t25_reconciliation import (  # noqa: E402
    EXPECTED_PROMPTS,
    T25ReconciliationError,
    canonical_sha256,
    load_condition_map,
    load_harmful_questions,
    load_p50_manifest,
    load_prompt_map,
    read_jsonl,
    reconcile_rows,
    sha256_file,
    validate_activation_manifest,
    validate_external_location,
    validate_heldout_axis_provenance,
)

DEFAULT_QUESTIONS = ROOT / "design" / "causal_questions_frozen.json"
FINAL_REPORT = ROOT / "results" / "t25" / "T25_RECONCILIATION.json"
FINAL_MANIFEST = ROOT / "results" / "t25" / "T25_ARTIFACT_MANIFEST.json"
CANDIDATE_REPORT = ROOT / "results" / "t25" / "candidate" / "T25_RECONCILIATION_CANDIDATE.json"
CANDIDATE_MANIFEST = ROOT / "results" / "t25" / "candidate" / "T25_ARTIFACT_MANIFEST_CANDIDATE.json"


def die(msg: str) -> None:
    raise SystemExit(f"T25 ABORTED (fail-closed): {msg}")


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def file_record(
    path: Path | None,
    *,
    private: bool = False,
    external_location: str | None = None,
) -> dict | None:
    if path is None:
        return None
    path = Path(path)
    record = {
        "path": ("PRIVATE_EXTERNAL_ARTIFACT" if private else str(path)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if external_location is not None:
        record["external_location"] = external_location
    return record


def _load_run_provenance(
    path: Path | None,
    *,
    required: bool,
    expected_model: str,
    outputs_sha256: str,
    observed_alpha: float,
) -> dict | None:
    if path is None:
        if required:
            raise T25ReconciliationError(
                "final T25 requires --run-provenance to bind the historical run's steering magnitude"
            )
        return None
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise T25ReconciliationError("run provenance must be a JSON object")

    model = doc.get("model_id", doc.get("model"))
    if model != expected_model:
        raise T25ReconciliationError(
            f"run provenance model {model!r} != frozen primary model {expected_model!r}"
        )

    alpha = doc.get("nonzero_coefficient", doc.get("steering_alpha"))
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)):
        raise T25ReconciliationError(
            "run provenance must contain finite numeric nonzero_coefficient (or steering_alpha)"
        )
    alpha = abs(float(alpha))
    if alpha <= 0:
        raise T25ReconciliationError("run provenance nonzero coefficient must be positive")
    if not math.isclose(alpha, observed_alpha, rel_tol=1e-12, abs_tol=1e-12):
        raise T25ReconciliationError(
            f"run provenance coefficient {alpha} != observed signed-dose magnitude {observed_alpha}"
        )

    bound_outputs = doc.get("outputs_sha256")
    if required and bound_outputs != outputs_sha256:
        raise T25ReconciliationError(
            "final run provenance must bind outputs_sha256 to the exact raw T25 artifact"
        )
    if bound_outputs is not None and bound_outputs != outputs_sha256:
        raise T25ReconciliationError(
            "run provenance outputs_sha256 is bound to a different raw artifact"
        )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "model_id": model,
        "model_revision": doc.get("model_revision"),
        "nonzero_coefficient": alpha,
        "outputs_sha256_verified": bound_outputs == outputs_sha256 if bound_outputs else None,
        "mean_residual_norm": doc.get("mean_residual_norm"),
        "heldout_axis_sha256": doc.get("heldout_axis_sha256"),
        "random_control_vector_sha256": doc.get("random_control_vector_sha256"),
        "source_git_sha": doc.get("source_git_sha"),
    }


def run(args) -> tuple[dict, dict]:
    cfg, config_sha = load_frozen_config()
    conditions = tuple(cfg["steering"]["unique_conditions"])
    prompt_indices = tuple(
        int(x) for x in cfg["causal_prompt_selection"]["project_prompt_indices"]
    )
    if prompt_indices != EXPECTED_PROMPTS:
        raise T25ReconciliationError(
            f"frozen prompt indices changed: expected {EXPECTED_PROMPTS}, got {prompt_indices}"
        )

    candidate_mode = bool(args.allow_candidate_p50)
    final_intent = not candidate_mode
    external_location = validate_external_location(
        args.artifact_location, required=final_intent
    )

    if final_intent and not args.heldout_axis_provenance:
        raise T25ReconciliationError(
            "final T25 requires --heldout-axis-provenance; use --allow-candidate-p50 for an explicitly candidate audit"
        )

    roles, p50_doc, p50_sha = load_p50_manifest(
        Path(args.p50_manifest), require_frozen=final_intent
    )
    question_ids, _question_text, questions_sha = load_harmful_questions(
        Path(args.questions)
    )
    condition_map, condition_map_sha = load_condition_map(
        Path(args.condition_map) if args.condition_map else None, conditions
    )
    prompt_map, prompt_map_sha = load_prompt_map(
        Path(args.prompt_map) if args.prompt_map else None, prompt_indices
    )

    rows = read_jsonl(Path(args.outputs))
    expected_model = cfg["models"]["primary"]["model_id"]
    allowed_statuses = tuple(cfg["validity"]["labels"])
    _normalized, aggregate = reconcile_rows(
        rows,
        roles=roles,
        prompt_indices=prompt_indices,
        question_ids=question_ids,
        conditions=conditions,
        allowed_technical_statuses=allowed_statuses,
        expected_model=expected_model,
        condition_map=condition_map,
        prompt_map=prompt_map,
    )
    outputs_sha = sha256_file(Path(args.outputs))

    configured_alpha = cfg["steering"]["nonzero_coefficient"]["value"]
    observed_alpha = aggregate["observed_nonzero_coefficient"]
    if configured_alpha is not None and not math.isclose(
        abs(float(configured_alpha)), observed_alpha, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise T25ReconciliationError(
            f"observed signed-dose magnitude {observed_alpha} != active frozen config value {configured_alpha}"
        )

    run_provenance = _load_run_provenance(
        Path(args.run_provenance) if args.run_provenance else None,
        required=final_intent,
        expected_model=expected_model,
        outputs_sha256=outputs_sha,
        observed_alpha=observed_alpha,
    )

    heldout = None
    if args.heldout_axis_provenance:
        heldout = validate_heldout_axis_provenance(
            Path(args.heldout_axis_provenance),
            p50_roles=roles,
            p50_manifest_sha256=p50_sha,
            require_p50_binding=final_intent,
        )

    activation = None
    if args.activation_manifest:
        activation = validate_activation_manifest(
            Path(args.activation_manifest),
            outputs_sha256=outputs_sha,
            uid_set_sha256=aggregate["uid_set_sha256"],
        )

    technical_nonvalid = sum(
        n
        for label, n in aggregate["technical_status_counts"].items()
        if label != "valid"
    )
    finalizable = (
        final_intent
        and p50_doc.get("status") == "FROZEN"
        and external_location is not None
        and run_provenance is not None
        and run_provenance["outputs_sha256_verified"] is True
        and heldout is not None
        and heldout["p50_exclusion_verified"] is True
        and heldout["p50_manifest_binding_verified"] is True
        and aggregate["dose_contract_verified"] is True
    )
    if final_intent and not finalizable:
        # No silent downgrade: final-intent either satisfies every gate or aborts.
        raise T25ReconciliationError(
            "final-intent invocation did not satisfy every T25 finalization gate"
        )

    membership_sha_recomputed = canonical_sha256(roles)
    report = {
        "schema_version": "t25-reconciliation/1.1",
        "task": "T25",
        "status": (
            "PASS_FINAL_RECONCILIATION"
            if finalizable
            else "PASS_CANDIDATE_RECONCILIATION"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_sha": git_sha(),
        "historical_run_note": (
            "This reconciliation applies the current V4 audit contract to an existing T25 run; "
            "the report/config SHA does not assert that the historical generation itself was executed under V4."
        ),
        "method_config": {
            "path": "configs/method_frozen_v4.yaml",
            "sha256": config_sha,
        },
        "design": {
            "model": expected_model,
            "model_revision": cfg["models"]["primary"]["model_revision"],
            "roles": 50,
            "prompt_indices": list(prompt_indices),
            "harmful_question_ids": question_ids,
            "conditions": list(conditions),
            "expected_total": 5000,
            "blinded_condition_ids_preserved": bool(condition_map),
            "observed_nonzero_coefficient": observed_alpha,
            "configured_nonzero_coefficient": configured_alpha,
            "dose_contract_verified": aggregate["dose_contract_verified"],
        },
        "aggregate_reconciliation": aggregate,
        "technical_nonvalid_count": technical_nonvalid,
        "semantic_rerun_policy": (
            "semantic outputs are immutable; only technical failures may be retried under the frozen retry policy"
        ),
        "uid_provenance": {
            "stored_uids_recomputed": False,
            "reason": (
                "The existing src.ids.steering_rollout_id helper coerces question_id to int, "
                "whereas T25 uses HQ1-style harmful_question_id values. No historical T25 UID "
                "minting convention is asserted without evidence. Stored UIDs are therefore used "
                "only for uniqueness/hash provenance; factorial cell identity is independently "
                "re-derived from role, prompt index, harmful-question ID, and condition."
            ),
            "independent_factorial_key_sha256": aggregate["factorial_key_set_sha256"],
        },
        "p50": {
            "manifest_sha256": p50_sha,
            "manifest_status": p50_doc.get("status"),
            "membership_sha256_declared": p50_doc.get("membership_sha256"),
            "membership_sha256_recomputed": membership_sha_recomputed,
            "membership_hash_verified": (
                p50_doc.get("membership_sha256") == membership_sha_recomputed
            ),
            "role_set_sha256": canonical_sha256(sorted(roles)),
            "matches_outputs": (
                aggregate["role_set_sha256"] == canonical_sha256(sorted(roles))
            ),
        },
        "run_provenance": run_provenance,
        "heldout_axis_verification": heldout,
        "causal_activation_manifest": activation,
        "finalization_gate": {
            "intent": "final" if final_intent else "candidate",
            "can_be_finalized": finalizable,
            "requirements": {
                "p50_manifest_frozen": p50_doc.get("status") == "FROZEN",
                "complete_5000_factorial": aggregate["complete_factorial"] is True,
                "signed_dose_contract_verified": aggregate["dose_contract_verified"] is True,
                "raw_external_location_recorded": external_location is not None,
                "run_provenance_bound_to_raw_outputs": bool(
                    run_provenance
                    and run_provenance["outputs_sha256_verified"] is True
                ),
                "heldout_axis_p50_exclusion_verified": bool(
                    heldout and heldout["p50_exclusion_verified"]
                ),
                "heldout_axis_bound_to_exact_p50": bool(
                    heldout and heldout["p50_manifest_binding_verified"]
                ),
            },
            "note": (
                "Candidate mode is explicit and writes to candidate paths by default. "
                "Final intent fails closed instead of downgrading when any final gate is missing."
            ),
        },
        "outcome_blind": True,
        "claim_boundary": (
            "T25 establishes completion/provenance of the DeepSeek signed-steering intervention only. "
            "It does not establish causal-judge validity or Axis-vs-random outcome effects; "
            "those belong to T26/T27 and T28."
        ),
    }

    manifest = {
        "schema_version": "t25-artifact-manifest/1.1",
        "task": "T25",
        "status": "RECONCILED_FINAL" if finalizable else "RECONCILED_CANDIDATE",
        "private_raw_outputs": file_record(
            Path(args.outputs),
            private=True,
            external_location=external_location,
        ),
        "private_condition_map": (
            file_record(Path(args.condition_map), private=True)
            if args.condition_map
            else None
        ),
        "private_condition_map_sha256": condition_map_sha,
        "private_prompt_map": (
            file_record(Path(args.prompt_map), private=True)
            if args.prompt_map
            else None
        ),
        "private_prompt_map_sha256": prompt_map_sha,
        "run_provenance": file_record(Path(args.run_provenance)) if args.run_provenance else None,
        "p50_manifest": file_record(Path(args.p50_manifest)),
        "causal_questions_manifest": {
            "path": str(args.questions),
            "sha256": questions_sha,
        },
        "heldout_axis_provenance": (
            file_record(Path(args.heldout_axis_provenance))
            if args.heldout_axis_provenance
            else None
        ),
        "causal_activation_manifest": (
            file_record(Path(args.activation_manifest))
            if args.activation_manifest
            else None
        ),
        "n_final_rows": aggregate["n_rows"],
        "uid_set_sha256": aggregate["uid_set_sha256"],
        "factorial_key_set_sha256": aggregate["factorial_key_set_sha256"],
        "technical_status_counts": aggregate["technical_status_counts"],
        "observed_nonzero_coefficient": observed_alpha,
        "dose_contract_verified": True,
        "privacy_boundary": (
            "Raw generation JSONL and private condition/prompt maps are not committed. "
            "Completion/output text and causal-judge fields are discarded at ingestion. "
            "Only whitelisted frozen technical-status labels, aggregate counts, stable external "
            "location, and cryptographic provenance are emitted."
        ),
    }
    return report, manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--outputs", required=True, help="existing final T25 JSONL; do not regenerate"
    )
    ap.add_argument(
        "--artifact-location",
        default=None,
        help="stable external locator for raw outputs (HF repo@revision/path or Drive ID); required for final",
    )
    ap.add_argument("--p50-manifest", required=True)
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    ap.add_argument(
        "--condition-map",
        default=None,
        help="private blinded condition-ID -> canonical condition JSON",
    )
    ap.add_argument(
        "--prompt-map",
        default=None,
        help="private steering_prompt_id -> prompt_index JSON when rows lack prompt_index",
    )
    ap.add_argument(
        "--run-provenance",
        default=None,
        help=(
            "JSON binding model + nonzero_coefficient to the raw outputs; required for final"
        ),
    )
    ap.add_argument("--heldout-axis-provenance", default=None)
    ap.add_argument("--activation-manifest", default=None)
    ap.add_argument(
        "--allow-candidate-p50",
        action="store_true",
        help="explicit candidate mode; never writes canonical final filenames by default",
    )
    ap.add_argument("--report", default=None)
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args(argv)

    try:
        report, manifest = run(a)
    except (T25ReconciliationError, OSError, json.JSONDecodeError, KeyError) as exc:
        die(str(exc))

    candidate = bool(a.allow_candidate_p50)
    rp = Path(a.report) if a.report else (CANDIDATE_REPORT if candidate else FINAL_REPORT)
    mp = Path(a.manifest) if a.manifest else (
        CANDIDATE_MANIFEST if candidate else FINAL_MANIFEST
    )
    rp.parent.mkdir(parents=True, exist_ok=True)
    mp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    mp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(rp),
                "manifest": str(mp),
                "can_be_finalized": report["finalization_gate"]["can_be_finalized"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
