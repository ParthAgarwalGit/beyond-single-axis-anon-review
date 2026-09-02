#!/usr/bin/env python3
"""T22 — freeze the DeepSeek P50 Assistant-proximal causal roles.

Input is the exploratory E80 all-valid reduction emitted by the T17 reducer.
T22 uses E80 only; it does not inspect C80/T17 results or causal outcomes.

A real frozen run writes the P50 manifest and, in the same operation, performs
the single V4 SHA succession required to (a) machine-register the pre-selection
membership deviation and (b) bind ``P50_manifest_sha256`` to the actual manifest.
Historical provenance stamps are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causal_role_selection import reconcile_observed_roles, select_p50  # noqa: E402
from src.t22_method_succession import (  # noqa: E402
    DEVIATION_ID,
    PREDECESSOR_V4_SHA,
    finalize_t22_succession,
    preflight_t22_succession,
)

DEFAULT_FLOOR = 0.95
P50_SIZE = 50
EXPECTED_LAYER = 16
EXPECTED_ARM = "translated"
EXPECTED_MEMBERSHIP = "all_valid"


def die(msg: str) -> None:
    raise SystemExit(f"T22 ABORTED (fail-closed): {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_single(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        die(f"missing reduced NPZ: {path}")
    with np.load(path, allow_pickle=True) as z:
        if "names" not in z or "matrix" not in z:
            die(f"{path} must contain names + matrix")
        names = [str(x) for x in z["names"].tolist()]
        matrix = np.asarray(z["matrix"], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(names):
        die(f"bad matrix shape in {path}")
    if len(names) != len(set(names)):
        die(f"duplicate role IDs in {path}")
    return names, matrix


def _t12_provenance_present(man: dict) -> bool:
    # PR #47's current reducer stores a per-input manifest map; older E80
    # reductions used a single SHA field. Accept either but require non-empty
    # cryptographic provenance.
    newer = man.get("t12_activation_manifests")
    if isinstance(newer, dict) and newer:
        for rec in newer.values():
            if not isinstance(rec, dict) or not rec.get("sha256"):
                return False
        return True
    older = man.get("t12_activation_manifest_sha256")
    return older not in (None, "", {})


def validate_e80_reduction(reduced: Path) -> dict:
    mp = reduced / "reduced_manifest.json"
    if not mp.exists():
        die("reduced_manifest.json missing")
    man = json.loads(mp.read_text(encoding="utf-8"))
    if man.get("layout") not in (None, "e80"):
        die(f"T22 requires exploratory E80, got layout={man.get('layout')!r}")
    if man.get("membership_rule") != EXPECTED_MEMBERSHIP:
        die("T22 requires label-independent all_valid E80 membership")
    if int(man.get("middle_block_index", -1)) != EXPECTED_LAYER:
        die("T22 requires frozen block 16")
    if not _t12_provenance_present(man):
        die("T12 activation-manifest provenance missing")
    return man


def default_mean(reduced: Path) -> tuple[np.ndarray, dict]:
    names, matrix = load_single(reduced / "default" / "all_response.npz")
    counts_path = reduced / "default" / "condition_counts_by_pool.json"
    if not counts_path.exists():
        die("default condition_counts_by_pool.json missing")
    counts_doc = json.loads(counts_path.read_text(encoding="utf-8"))
    counts = counts_doc.get("all_response")
    if not isinstance(counts, dict) or len(counts) != 5:
        die("expected exactly five default conditions")
    by_name = {str(k): v for k, v in counts.items()}
    if set(names) != set(by_name):
        die("default means/counts condition keys differ")
    fracs = {}
    for n in names:
        rec = by_name[n]
        eligible = int(rec.get("eligible", 0))
        total = int(rec.get("total", 0))
        if total <= 0:
            die(f"default condition {n!r} has invalid total")
        frac = eligible / total
        fracs[n] = frac
        if frac < DEFAULT_FLOOR:
            die(f"default condition {n!r} valid fraction {frac:.4f} < {DEFAULT_FLOOR}")
    if len(names) != 5:
        die(f"default NPZ has {len(names)} condition means, expected five")
    return matrix.mean(axis=0), fracs


def observed_t25_roles(path: Path) -> tuple[set[str], int]:
    if not path.exists():
        die(f"T25 output file does not exist: {path}")
    roles, n = set(), 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            role = row.get("role") or row.get("role_id")
            if not role:
                die("T25 row missing role/role_id")
            roles.add(str(role))
            n += 1
    return roles, n


def run(
    reduced: Path,
    out_manifest: Path,
    out_report: Path,
    t25_outputs: Path | None,
    *,
    finalize_method_succession: bool = False,
) -> dict:
    # A true freeze is one operation: validate the post-#44 predecessor before
    # emitting the real P50, then bind the resulting manifest hash into V4.
    if finalize_method_succession:
        try:
            preflight_t22_succession(ROOT)
        except ValueError as exc:
            die(str(exc))
        if out_manifest.exists() or out_report.exists():
            die("refusing to overwrite an existing T22 manifest/report during the one-step freeze")

    man = validate_e80_reduction(reduced)
    role_ids, role_means = load_single(reduced / EXPECTED_ARM / "available_all_response.npz")
    mu_default, default_fracs = default_mean(reduced)

    result = select_p50(role_ids, role_means, mu_default, p50_size=P50_SIZE)
    selected = [r["role_id"] for r in result["selected_roles"]]

    input_hashes = {
        "reduced_manifest": sha256_file(reduced / "reduced_manifest.json"),
        "role_means_npz": sha256_file(reduced / EXPECTED_ARM / "available_all_response.npz"),
        "default_means_npz": sha256_file(reduced / "default" / "all_response.npz"),
        "default_counts": sha256_file(reduced / "default" / "condition_counts_by_pool.json"),
    }
    manifest_status = "FROZEN" if finalize_method_succession else "CANDIDATE_NOT_METHOD_BOUND"
    manifest = {
        "schema_version": "t22-p50-freeze/1.1",
        "task": "T22",
        "status": manifest_status,
        "classification": "TARGET_MODEL_SPECIFIC_ADAPTATION_NOT_LU_EXACT_LIST",
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "selection_data": {
            "block": "E80",
            "prompt_arm": "USER_TRANSLATED_LU",
            "pool": "ALL_RESPONSE_TOKENS",
            "layer": EXPECTED_LAYER,
            "membership": "label_independent_technical_validity",
            "membership_reason": (
                "T07 failed the automatic role-judge validation gate; T22 therefore "
                "does not use judge-dependent score-3 selection."
            ),
        },
        "selection_rule": (
            "(mu_role - mean_over_eligible_roles(mu_role)) dot unit(v_E80), "
            "descending; exclude literal assistant; lexical role_id tie-break; take 50"
        ),
        "causal_outcomes_used_for_selection": False,
        "heldout_axis_requirement": {
            "p50_must_be_excluded_from_t24_heldout_axis": True,
            "verified_by_t22": False,
            "verification_owner": "T24/T25",
            "note": (
                "T22 records this as a downstream requirement only. The held-out Axis "
                "must verify the exclusion where that Axis is actually constructed."
            ),
        },
        "method_change_control": {
            "deviation_id": DEVIATION_ID,
            "predecessor_v4_sha256": PREDECESSOR_V4_SHA,
            "machine_registration_and_p50_hash_binding": (
                "performed by this run" if finalize_method_succession
                else "NOT PERFORMED; candidate output is not frozen"
            ),
        },
        "input_hashes": input_hashes,
        "upstream_reduction_provenance": {
            "t12_activation_manifests": man.get("t12_activation_manifests"),
            "t12_activation_manifest_sha256_legacy": man.get("t12_activation_manifest_sha256"),
            "membership_rule": man.get("membership_rule"),
            "middle_block_index": man.get("middle_block_index"),
        },
        "default_valid_fractions": default_fracs,
        **result,
    }

    reconciliation = None
    if t25_outputs is not None:
        roles, n_rows = observed_t25_roles(t25_outputs)
        reconciliation = reconcile_observed_roles(selected, roles)
        reconciliation["t25_rows_seen"] = n_rows
        reconciliation["t25_outputs_sha256"] = sha256_file(t25_outputs)
        if not reconciliation["match"]:
            die(
                "existing T25 role set does not equal selected P50: "
                f"missing={reconciliation['missing']} unexpected={reconciliation['unexpected']}"
            )

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_sha = sha256_file(out_manifest)

    succession = None
    if finalize_method_succession:
        try:
            succession = finalize_t22_succession(ROOT, manifest_sha)
        except Exception as exc:
            # Do not leave a file claiming FROZEN if V4 binding did not complete.
            out_manifest.unlink(missing_ok=True)
            die(f"method succession failed; removed unbound FROZEN manifest: {exc}")

    report = {
        "task": "T22",
        "status": "PASS_FROZEN" if finalize_method_succession else "PASS_CANDIDATE_ONLY",
        "manifest_path": str(out_manifest),
        "manifest_sha256": manifest_sha,
        "membership_sha256": manifest["membership_sha256"],
        "n_selected": len(selected),
        "assistant_excluded": "assistant" not in selected,
        "selection_boundary": manifest["boundary"],
        "t25_reconciliation": reconciliation,
        "method_succession": succession,
        "claim_boundary": (
            "P50 is a DeepSeek-specific adaptation of Lu et al.'s published ranking procedure; "
            "the source study's exact 50-role membership is unavailable. T22 does not itself "
            "verify downstream held-out-Axis exclusion."
        ),
    }
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reduced-e80", required=True)
    ap.add_argument("--out-manifest", default="design/causal_P50_assistant_proximal.json")
    ap.add_argument("--out-report", default="results/t22/selection_report.json")
    ap.add_argument("--t25-outputs", default=None)
    ap.add_argument(
        "--finalize-method-succession",
        action="store_true",
        help=(
            "required for a real FROZEN T22 result: machine-register the T22 deviation, "
            "fill V4 P50_manifest_sha256 with the emitted manifest hash, and rebind the "
            "four forward-looking V4 pins in one succession"
        ),
    )
    a = ap.parse_args(argv)
    rep = run(
        Path(a.reduced_e80),
        Path(a.out_manifest),
        Path(a.out_report),
        Path(a.t25_outputs) if a.t25_outputs else None,
        finalize_method_succession=a.finalize_method_succession,
    )
    print(json.dumps(rep, indent=2))
    if rep["status"] != "PASS_FROZEN":
        print("NOTE: candidate only; rerun with --finalize-method-succession for the real freeze.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
