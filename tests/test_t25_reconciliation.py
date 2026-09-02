import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.t25_reconciliation import (
    EXPECTED_PROMPTS,
    EXPECTED_QUESTIONS,
    T25ReconciliationError,
    canonical_sha256,
    load_p50_manifest,
    read_jsonl,
    reconcile_rows,
    sha256_file,
    validate_heldout_axis_provenance,
)


CONDITIONS = (
    "shared_zero",
    "assistant_axis_toward",
    "assistant_axis_away",
    "random_positive",
    "random_negative",
)
ROLES = tuple(f"role_{i:02d}" for i in range(50))
TECHNICAL_STATUSES = (
    "valid",
    "empty_response",
    "generation_error",
    "truncated",
    "serialization_failure",
    "token_alignment_failure",
    "degenerate_repetition",
)
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ALPHA = 8.0
ROOT = Path(__file__).resolve().parents[1]


def doses(condition):
    return {
        "shared_zero": (0.0, 0.0),
        "assistant_axis_toward": (ALPHA, 0.0),
        "assistant_axis_away": (-ALPHA, 0.0),
        "random_positive": (0.0, ALPHA),
        "random_negative": (0.0, -ALPHA),
    }[condition]


def planted_rows(*, include_model=True):
    rows = []
    for role in ROLES:
        for prompt in EXPECTED_PROMPTS:
            for qid in EXPECTED_QUESTIONS:
                for condition in CONDITIONS:
                    uid = f"{role}:{prompt}:{qid}:{condition}"
                    axis_dose, random_dose = doses(condition)
                    row = {
                        "uid": uid,
                        "role_id": role,
                        "prompt_index": prompt,
                        "harmful_question_id": qid,
                        "condition": condition,
                        "technical_validity": "valid",
                        "axis_dose": axis_dose,
                        "random_dose": random_dose,
                        # These are deliberately present in the private source rows;
                        # reconciliation must not need them.
                        "output_text": "TOP SECRET COMPLETION",
                        "auto_harmfulness": "TOP SECRET JUDGE LABEL",
                    }
                    if include_model:
                        row["model"] = MODEL
                    rows.append(row)
    return rows


def reconcile(rows):
    return reconcile_rows(
        rows,
        roles=ROLES,
        prompt_indices=EXPECTED_PROMPTS,
        question_ids=EXPECTED_QUESTIONS,
        conditions=CONDITIONS,
        allowed_technical_statuses=TECHNICAL_STATUSES,
        expected_model=MODEL,
    )


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def p50_doc(status="FROZEN"):
    roles = list(ROLES)
    return {
        "status": status,
        "literal_assistant_role_id": "assistant",
        "membership_sha256": canonical_sha256(roles),
        "selected_roles": [
            {"rank": i + 1, "role_id": role, "assistant_side_score": 0.0}
            for i, role in enumerate(roles)
        ],
        "heldout_axis_requirement": {
            "p50_must_be_excluded_from_t24_heldout_axis": True,
        },
    }


def test_exact_5000_factorial_and_doses_pass():
    normalized, report = reconcile(planted_rows())
    assert report["n_rows"] == 5000
    assert report["n_unique_uids"] == 5000
    assert report["complete_factorial"] is True
    assert report["dose_contract_verified"] is True
    assert report["observed_nonzero_coefficient"] == ALPHA
    assert set(report["per_condition"].values()) == {1000}
    assert report["per_role_min"] == report["per_role_max"] == 100
    assert report["per_prompt"] == {str(i): 1250 for i in EXPECTED_PROMPTS}
    assert set(report["per_question"].values()) == {1000}
    assert all("completion" not in row and "output_text" not in row for row in normalized)
    assert all("auto_harmfulness" not in row for row in normalized)


def test_missing_cell_fails_closed():
    with pytest.raises(T25ReconciliationError, match="does not reconcile"):
        reconcile(planted_rows()[:-1])


def test_duplicate_cell_and_uid_fail_closed():
    rows = planted_rows()
    rows[-1] = dict(rows[0])
    with pytest.raises(T25ReconciliationError, match="does not reconcile"):
        reconcile(rows)


def test_wrong_signed_dose_fails_closed():
    rows = planted_rows()
    toward = next(r for r in rows if r["condition"] == "assistant_axis_toward")
    toward["axis_dose"] = -ALPHA
    with pytest.raises(T25ReconciliationError, match="dose columns"):
        reconcile(rows)


def test_mixed_nonzero_magnitudes_fail_closed():
    rows = planted_rows()
    toward = next(r for r in rows if r["condition"] == "assistant_axis_toward")
    toward["axis_dose"] = ALPHA + 1.0
    with pytest.raises(T25ReconciliationError, match="magnitude is not constant"):
        reconcile(rows)


def test_semantic_degenerate_status_is_retained_not_dropped():
    rows = planted_rows()
    rows[0]["technical_validity"] = "degenerate_repetition"
    _, report = reconcile(rows)
    assert report["n_rows"] == 5000
    assert report["technical_status_counts"]["degenerate_repetition"] == 1


def test_unknown_technical_status_cannot_leak_into_report():
    rows = planted_rows()
    rows[0]["technical_validity"] = "harmful_and_successful"
    with pytest.raises(T25ReconciliationError, match="not in the frozen"):
        reconcile(rows)


def test_model_identity_checked_where_present():
    rows = planted_rows()
    rows[0]["model"] = "wrong/model"
    with pytest.raises(T25ReconciliationError, match="frozen primary model"):
        reconcile(rows)


def test_blinded_condition_mapping_can_be_resolved():
    mapping = {f"blind_{i}": c for i, c in enumerate(CONDITIONS)}
    reverse = {v: k for k, v in mapping.items()}
    rows = planted_rows()
    for row in rows:
        row["condition_id"] = reverse[row.pop("condition")]
    _, report = reconcile_rows(
        rows,
        roles=ROLES,
        prompt_indices=EXPECTED_PROMPTS,
        question_ids=EXPECTED_QUESTIONS,
        conditions=CONDITIONS,
        allowed_technical_statuses=TECHNICAL_STATUSES,
        expected_model=MODEL,
        condition_map=mapping,
    )
    assert report["complete_factorial"] is True


def test_private_jsonl_ingestion_drops_completion_and_outcomes(tmp_path):
    path = tmp_path / "private.jsonl"
    write_jsonl(path, [planted_rows()[0]])
    [row] = read_jsonl(path)
    assert "output_text" not in row
    assert "completion" not in row
    assert "auto_harmfulness" not in row
    assert "judge_rationale" not in row
    assert row["axis_dose"] == 0.0


def test_p50_frozen_gate_and_membership_hash(tmp_path):
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(p50_doc("CANDIDATE_NOT_METHOD_BOUND")))
    with pytest.raises(T25ReconciliationError, match="not FROZEN"):
        load_p50_manifest(candidate, require_frozen=True)

    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(p50_doc("FROZEN")))
    roles, _, _ = load_p50_manifest(frozen, require_frozen=True)
    assert roles == list(ROLES)

    bad = p50_doc("FROZEN")
    bad["membership_sha256"] = "0" * 64
    bad_path = tmp_path / "bad_hash.json"
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(T25ReconciliationError, match="does not reproduce"):
        load_p50_manifest(bad_path, require_frozen=True)


def test_heldout_axis_requires_explicit_role_list_no_overlap_and_final_binding(tmp_path):
    p50_sha = "a" * 64
    good = tmp_path / "axis.json"
    good.write_text(
        json.dumps(
            {
                "construction_role_ids": ["other_1", "other_2"],
                "p50_manifest_sha256": p50_sha,
                "unit_axis_sha256": "b" * 64,
            }
        )
    )
    report = validate_heldout_axis_provenance(
        good,
        p50_roles=ROLES,
        p50_manifest_sha256=p50_sha,
        require_p50_binding=True,
    )
    assert report["p50_exclusion_verified"] is True
    assert report["p50_manifest_binding_verified"] is True

    unbound = tmp_path / "unbound.json"
    unbound.write_text(
        json.dumps(
            {
                "construction_role_ids": ["other_1", "other_2"],
                "unit_axis_sha256": "b" * 64,
            }
        )
    )
    with pytest.raises(T25ReconciliationError, match="p50_manifest_sha256"):
        validate_heldout_axis_provenance(
            unbound,
            p50_roles=ROLES,
            p50_manifest_sha256=p50_sha,
            require_p50_binding=True,
        )

    bad = tmp_path / "bad_axis.json"
    bad.write_text(
        json.dumps(
            {
                "construction_role_ids": [ROLES[0], "other_2"],
                "p50_manifest_sha256": p50_sha,
                "unit_axis_sha256": "b" * 64,
            }
        )
    )
    with pytest.raises(T25ReconciliationError, match="includes 1 P50"):
        validate_heldout_axis_provenance(
            bad,
            p50_roles=ROLES,
            p50_manifest_sha256=p50_sha,
            require_p50_binding=True,
        )


def test_factorial_hash_is_order_invariant():
    rows = planted_rows()
    _, a = reconcile(rows)
    _, b = reconcile(list(reversed(rows)))
    assert a["uid_set_sha256"] == b["uid_set_sha256"]
    assert a["factorial_key_set_sha256"] == b["factorial_key_set_sha256"]
    assert a["role_set_sha256"] == canonical_sha256(sorted(ROLES))


def test_cli_final_end_to_end_and_emitted_artifacts_are_leak_free(tmp_path):
    outputs = tmp_path / "outputs.jsonl"
    write_jsonl(outputs, planted_rows())
    outputs_sha = sha256_file(outputs)

    p50 = tmp_path / "p50.json"
    p50.write_text(json.dumps(p50_doc("FROZEN")))
    p50_sha = sha256_file(p50)

    heldout = tmp_path / "axis.json"
    heldout.write_text(
        json.dumps(
            {
                "construction_role_ids": ["other_1", "other_2"],
                "p50_manifest_sha256": p50_sha,
                "unit_axis_sha256": "b" * 64,
            }
        )
    )

    run_prov = tmp_path / "run_provenance.json"
    run_prov.write_text(
        json.dumps(
            {
                "model_id": MODEL,
                "nonzero_coefficient": ALPHA,
                "outputs_sha256": outputs_sha,
                "source_git_sha": "c" * 40,
            }
        )
    )

    report = tmp_path / "report.json"
    manifest = tmp_path / "manifest.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "reconcile_t25_deepseek_5k.py"),
            "--outputs",
            str(outputs),
            "--artifact-location",
            "hf://org/repo@immutable-revision/t25/final_outputs.jsonl",
            "--p50-manifest",
            str(p50),
            "--run-provenance",
            str(run_prov),
            "--heldout-axis-provenance",
            str(heldout),
            "--report",
            str(report),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    report_text = report.read_text()
    manifest_text = manifest.read_text()
    assert "PASS_FINAL_RECONCILIATION" in report_text
    assert "TOP SECRET COMPLETION" not in report_text + manifest_text
    assert "TOP SECRET JUDGE LABEL" not in report_text + manifest_text
    assert '"output_text"' not in report_text + manifest_text
    assert '"auto_harmfulness"' not in report_text + manifest_text
    assert "hf://org/repo@immutable-revision/t25/final_outputs.jsonl" in manifest_text


def test_cli_final_intent_refuses_missing_heldout_axis(tmp_path):
    outputs = tmp_path / "outputs.jsonl"
    write_jsonl(outputs, planted_rows())
    p50 = tmp_path / "p50.json"
    p50.write_text(json.dumps(p50_doc("FROZEN")))
    run_prov = tmp_path / "run_provenance.json"
    run_prov.write_text(
        json.dumps(
            {
                "model_id": MODEL,
                "nonzero_coefficient": ALPHA,
                "outputs_sha256": sha256_file(outputs),
            }
        )
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "reconcile_t25_deepseek_5k.py"),
            "--outputs",
            str(outputs),
            "--artifact-location",
            "hf://org/repo@immutable-revision/t25/final_outputs.jsonl",
            "--p50-manifest",
            str(p50),
            "--run-provenance",
            str(run_prov),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "requires --heldout-axis-provenance" in proc.stderr + proc.stdout
