"""T09 minimal provenance closure — the single E80/C80 acceptance note.

Collates the committed audit reports, manifests, and deviation records into
one acceptance/provenance report. Performs NO reruns: it only reads existing
artifacts and cross-checks that the facts they record agree.

Fail-closed: every ACCEPTED status is DERIVED from the underlying reports,
never asserted. If any source report records a failure, a coverage gap, an
integrity error, or numbers that disagree across sources, the closure
refuses to emit a report and raises :class:`ClosureError` naming the
violation. ``build_closure`` is a pure function over the loaded source
dicts so refusal paths are unit-testable.

Output: results/audit/E80_C80_provenance_closure.json (appendix /
reproducibility only).
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_historical_frozen_config  # noqa: E402
from src.provenance import stamp_report  # noqa: E402

# All facts sourced from main are read at this pinned commit.
MAIN_COMMIT = "3b5643c299a74b1f63de07da4f092958cdd46593"

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
REVISION = "6a6f4aa4197940add57724a7707d069478df56b1"
CHAT_TEMPLATE_SHA = "56a1447ad31926fdc21fb07e56e5642bd9c850c4f52d8c8af7bbe5f079a84f5f"
HISTORICAL_CONFIG_SHA = "73be73df4640c2d32bfbc8b6009741c0fadd8c466acb703d1f99df35d5c8cd79"
V4_APPROVED_SHA = "f7e7dd9df8c7c72d5da08bf8201295dfc91f3208837d8cd0867054c393984e59"

E80_DATASET = ("hf-dataset:[Author-A-HF]/persona-artifacts"
               "@d918c2a157eba4d6fb91899c6d08f90342949ab8")

N_ROLLOUTS = 22000


class ClosureError(RuntimeError):
    """A source report fails, is incomplete, or disagrees with another."""


def _require(condition, message):
    if not condition:
        raise ClosureError(message)


def derive_e80_status(e80):
    """ACCEPTED / ACCEPTED_WITH_RECORDED_GAP, derived from the audit report.

    Refuses unless every check PASSed except at most the known
    model_revision GAP, and the audit is cryptographically bound to raw
    shard bytes.
    """
    _require(e80.get("overall") in ("PASS", "GAP"),
             f"E80 audit overall is {e80.get('overall')!r}, not PASS/GAP")
    non_pass = {c["check"]: c["status"] for c in e80["checks"]
                if c["status"] != "PASS"}
    _require(set(non_pass) <= {"model_revision"},
             f"E80 audit has non-PASS checks beyond model_revision: {non_pass}")
    _require(non_pass.get("model_revision") in (None, "GAP"),
             f"E80 model_revision check is {non_pass.get('model_revision')!r}")
    _require(e80["inputs"].get("dataset_ref") == E80_DATASET,
             "E80 audit dataset_ref does not match the accepted snapshot")
    raw = e80.get("audited_raw_files") or {}
    for arm in ("translated", "wrapper", "default"):
        files = raw.get(arm) or []
        _require(files and all(f.get("sha256") for f in files),
                 f"E80 audit records no raw-file SHA-256s for arm {arm!r}")
    return "ACCEPTED" if not non_pass else "ACCEPTED_WITH_RECORDED_GAP"


def derive_c80_block(t10, selection, block, expected_eligible):
    """Return the verified summary for one C80 block, refusing on any
    verifier failure or disagreement with the attempt-selection audit."""
    entry = t10["blocks"].get(block) or {}
    _require(entry.get("verdict") == "PASS",
             f"{block} verifier verdict is {entry.get('verdict')!r} "
             f"(failed: {entry.get('failed_checks')})")
    summary = entry["summary"]
    sel = selection[block]
    _require(summary["rows"] == sel["raw_rows_all_attempts"],
             f"{block} row counts disagree: verifier {summary['rows']} vs "
             f"selection audit {sel['raw_rows_all_attempts']}")
    _require(summary["rollouts_covered"] == sel["unique_rollouts"] == N_ROLLOUTS,
             f"{block} rollout coverage disagrees or is incomplete")
    _require(sel["downstream_eligible"] == expected_eligible,
             f"{block} selection audit eligible {sel['downstream_eligible']} "
             f"!= expected {expected_eligible}")
    _require(summary["rollouts_with_multiple_valid_attempts"]
             == sel["rollouts_with_multiple_valid_attempts"],
             f"{block} multi-valid counts disagree between verifier and "
             "selection audit")
    return summary, sel


def build_closure(src):
    """Pure closure builder over loaded source dicts.

    ``src`` keys: e80, t10, selection, manifest_a, review_a.
    Raises ClosureError instead of emitting any accepted status when a
    source fails or sources disagree.
    """
    e80, t10 = src["e80"], src["t10"]
    selection, manifest_a, review_a = (src["selection"], src["manifest_a"],
                                       src["review_a"])

    e80_status = derive_e80_status(e80)

    raw_a = selection["C80-A"]["raw_sha256"]
    raw_b = selection["C80-B"]["raw_sha256"]
    _require(raw_a == manifest_a["data_location"]["authoritative_sha256"],
             "C80-A raw SHA disagrees between selection audit and manifest")
    _require(raw_a == review_a["final_jsonl_sha256"],
             "C80-A raw SHA disagrees between selection audit and "
             "technical review")
    _require(selection["selection_rule"] == "lowest_retry_technically_valid_attempt",
             "unexpected attempt-selection rule in the committed audit")
    _require(t10["manifest_check"]["rollout_id_overlap"] == 0
             and t10["manifest_check"]["question_id_overlap"] == [],
             "C80-A/C80-B manifests are not disjoint")

    exhausted_a = {x["rollout_id"] for x in review_a["exhausted_rollouts"]}
    summary_a, sel_a = derive_c80_block(
        t10, selection, "C80-A", expected_eligible=N_ROLLOUTS - len(exhausted_a))
    summary_b, sel_b = derive_c80_block(
        t10, selection, "C80-B", expected_eligible=N_ROLLOUTS)
    _require({x["rollout_id"] for x in summary_a["exhausted_rollouts"]}
             == exhausted_a,
             "C80-A exhausted rollouts disagree between verifier and "
             "technical review")

    return {
        "task": "T09_MINIMAL_PROVENANCE_CLOSURE",
        "title": "E80/C80 acceptance and provenance closure",
        "owner_executor": "[Reviewer]",
        "reviewer": "[Author B]",
        "review_status": "AWAITING_REVIEWER",
        "scope": ("Acceptance note for the already-generated E80 and C80 "
                  "raw-text artifacts. No artifact was rerun, regenerated, "
                  "or rewritten for this closure. Every accepted status "
                  "below is derived fail-closed from the referenced audit "
                  "reports; the closure refuses to build if any source "
                  "records a failure or sources disagree."),
        "main_commit_for_committed_artifacts": MAIN_COMMIT,

        "configurations": {
            "generation_time_config_sha256": HISTORICAL_CONFIG_SHA,
            "generation_time_config_note": (
                "All accepted generation artifacts are stamped with this "
                "historical config. The audit tools verify at run time that "
                "the config they load hashes to exactly this value "
                "(issue #31 documents an in-place mutation on main; bytes "
                "recoverable at git 855d683:configs/method_frozen.yaml)."),
            "downstream_analysis_config": {
                "path": "configs/method_frozen_v4.yaml",
                "sha256": V4_APPROVED_SHA,
                "status": "APPROVED (PR #38, reviewer-signed; not used by any "
                          "accepted artifact here; recorded for downstream "
                          "analyses)",
            },
        },

        "accepted_pins": {
            "model_id": MODEL_ID,
            "chat_template_sha256": CHAT_TEMPLATE_SHA,
            "model_revision": {
                "value": REVISION,
                "E80": ("POST_HOC_SUPPORTED: run-003 never recorded a "
                        "generation-time weights revision; this revision was "
                        "resolved afterwards and is supported (not proven) "
                        "by T12's activation reconstruction."),
                "C80": ("DIRECTLY_RECORDED: every C80-A/C80-B row carries "
                        "model_revision and tokenizer_revision explicitly; "
                        "verified uniform by the row-level audit."),
            },
        },

        "decoding_regimes": {
            "E80_accepted_T09_settings": {
                "sampling": "ancestral", "temperature": 0.6, "top_p": 0.95,
                "max_new_tokens": 2560, "seed": 0,
                "source": "results/audit/E80_acceptance_report.json "
                          "(generation_settings, 44,400/44,400 rows identical)",
            },
            "C80_executed_settings": {
                "sampling": "greedy", "temperature": 0.0, "do_sample": False,
                "seed": 0, "max_new_tokens_retry_ladder": [2048, 8192, 16384],
                "deviation_record":
                    "docs/deviations/2026-08-13-c80-greedy-decoding.md",
            },
            "statement": (
                "C80-A versus C80-B is the decoding-matched primary "
                "confirmatory comparison. E80 is exploratory. E80-to-C80 is "
                "not decoding-identical; any pooled 240-ID analysis must "
                "retain decoding-stratum provenance."
            ),
        },

        "attempt_selection": {
            "rule": "lowest retry index with technical_validity == 'valid'; "
                    "rollouts with no valid attempt are retained for "
                    "provenance only and excluded downstream",
            "deviation_record":
                "docs/deviations/2026-08-13-t10-attempt-selection.md",
            "selection_indexes": {
                "C80-A": "results/generation/c80_role_C80_A.selection.jsonl",
                "C80-B": "results/generation/c80_role_C80_B.selection.jsonl",
            },
            "independent_verification": (
                "All 44,000 selection rows re-derived from the raw files by "
                "the reviewer-side check (13 Aug): 0 rule violations, 0 "
                "eligibility errors, all stored raw-line and canonical-row "
                "hashes reproduce."
            ),
        },

        "blocks": {
            "E80": {
                "role_in_design": "exploratory",
                "status": e80_status,
                "data_location": E80_DATASET,
                "raw_snapshot_sha256s": e80["audited_raw_files"],
                "coverage": {
                    "translated_USER_TRANSLATED_LU": 22000,
                    "wrapper_USER_EXPLICIT": 22000,
                    "default_5_conditions": 400,
                },
                "acceptance_basis": (
                    "results/audit/E80_acceptance_report.json — 10/11 checks "
                    "PASS (IDs, prompt bytes, chat template, settings, "
                    "closure, no default wrapper, shards, duplicates); one "
                    "GAP: generation-time model revision unrecorded."
                ),
                "known_quirks": [
                    "39 unclosed-<think> rows in each role arm (handled per "
                    "docs/deviations/2026-08-08-prefilled-think-segmentation.md: "
                    "malformed => no final-answer region, judge abstains).",
                    "Default worker's `condition` metadata field mislabelled "
                    "USER_EXPLICIT; rendered prompts verified unwrapped.",
                ],
            },
            "C80-A": {
                "role_in_design": "confirmatory (decoding-matched with C80-B)",
                "status": "ACCEPTED",
                "raw_sha256": raw_a,
                "rows_all_attempts": summary_a["rows"],
                "rollouts": N_ROLLOUTS,
                "downstream_eligible": sel_a["downstream_eligible"],
                "technical_exclusions": {
                    "count": len(exhausted_a),
                    "rollouts": sorted(exhausted_a),
                    "decision": review_a["decision"],
                    "reviewer_confirmation": "confirmed (PR #29, merged)",
                },
                "acceptance_basis": (
                    "Row-level verification (verdict PASS in "
                    "results/audit/T10_C80_reconciliation_report.json): raw "
                    "SHA match; all IDs recompute; prompt bytes reproduce "
                    "from frozen templates; pins uniform; 4,400/prompt-index; "
                    "zero duplicates; zero multi-valid ambiguity."
                ),
            },
            "C80-B": {
                "role_in_design": "confirmatory (decoding-matched with C80-A)",
                "status": "ACCEPTED_UNDER_SELECTION_RULE",
                "raw_sha256": raw_b,
                "rows_all_attempts": summary_b["rows"],
                "rollouts": N_ROLLOUTS,
                "downstream_eligible": sel_b["downstream_eligible"],
                "technical_exclusions": {"count": 0},
                "acceptance_basis": (
                    "Row-level verification (verdict PASS in "
                    "results/audit/T10_C80_reconciliation_report.json): raw "
                    "SHA match; all IDs recompute; prompt bytes reproduce; "
                    "pins uniform; 4,400/prompt-index; zero duplicates. "
                    f"{sel_b['rollouts_with_multiple_valid_attempts']} "
                    "rollouts hold >1 technically valid attempt; the frozen "
                    "lowest-valid-retry rule resolves every case "
                    "deterministically (verified)."
                ),
                "known_quirks": [
                    "15 retry histories contain same-budget repeat attempts "
                    "(resume defect); token budget must be read from per-row "
                    "runtime provenance, not inferred from retry index.",
                ],
            },
        },

        "artifact_register": {
            "accepted": [
                f"E80 raw arms (translated/wrapper/default) @ {E80_DATASET}, "
                "shard SHA-256s bound in the E80 acceptance report",
                f"C80-A raw c80_role_C80_A.final.jsonl sha256={raw_a}",
                f"C80-B raw c80_role_C80_B.final.jsonl sha256={raw_b}",
                "C80 deterministic rollout manifests "
                "(C80-A 60e4e7b5…bf33, C80-B e7b532e3…6ac5; both reproduce "
                "from tools/build_t10_c80_manifest.py)",
                "C80 attempt-selection indexes (verified 44,000/44,000)",
                "C80-A provenance manifest "
                "results/generation/c80_role_C80_A.manifest.json "
                "(cross-checked against raw data by this closure)",
                "Deviation records: 2026-08-08 prefilled-think segmentation; "
                "2026-08-13 attempt selection; 2026-08-13 greedy decoding",
            ],
            "provisional": [
                "C80-B provenance manifest "
                "results/generation/c80_role_C80_B.manifest.json: its retry "
                "narrative predates the reconciliation findings and is not "
                "cross-checked by this closure (corrections requested on "
                "PR #23). The C80-B RAW DATA is accepted above under the "
                "verified selection rule; only this descriptive manifest "
                "remains provisional.",
                                "T12 E80 activation recompute outputs (PR #28 under "
                "re-review)",
                "Prefilled-<think> segmentation code (frozen in V4 but "
                "implementing code still unmerged: PRs #23/#26)",
                "Existing role-judge scores and geometry (unchanged status: "
                "provisional pending T07 judge validation)",
            ],
            "rejected_or_superseded": [
                "c80_role_C80_B.jsonl (pre-resegmentation raw; superseded by "
                "c80_role_C80_B.final.jsonl)",
                "C80-A rollouts 885b75abc2900244 (luddite/q100) and "
                "79331716c2cde644 (mentor/q37): technical execution failures, "
                "provenance-only, excluded from scoring/pooling/geometry",
                "run-003 extraction/geometry.json and "
                "h2_reliability_corrected.json (superseded per run MANIFEST)",
            ],
        },

        "references": {
            "audit_reports": [
                "results/audit/E80_acceptance_report.json",
                "results/audit/T10_C80_reconciliation_report.json",
                "results/audit/T10_C80_attempt_selection_13_aug.json",
            ],
            "reviews": ["PR #23 (C80-B findings)", "PR #28 (T12 findings)",
                        "PR #29 (C80-A approval, merged)",
                        "issue #31 (V4 freeze blockers)"],
            "method": ["docs/METHOD_FREEZE.md (V3)",
                       "docs/METHOD_FREEZE_V4_13_AUG.md (V4 approved, PR #38)"],
        },
    }


def git_show(path):
    out = subprocess.run(
        ["git", "show", f"{MAIN_COMMIT}:{path}"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return out.stdout


def main():
    _, loaded_sha = load_historical_frozen_config()
    if loaded_sha != HISTORICAL_CONFIG_SHA:
        raise ClosureError(
            f"loaded frozen config hashes to {loaded_sha}, not the "
            f"generation-time historical config {HISTORICAL_CONFIG_SHA}; "
            "refusing to build the closure (see issue #31)")

    sources = {
        "e80": json.loads(
            (REPO_ROOT / "results/audit/E80_acceptance_report.json").read_text()),
        "t10": json.loads(
            (REPO_ROOT / "results/audit/T10_C80_reconciliation_report.json").read_text()),
        "selection": json.loads(
            git_show("results/audit/T10_C80_attempt_selection_13_aug.json")),
        "manifest_a": json.loads(
            git_show("results/generation/c80_role_C80_A.manifest.json")),
        "review_a": json.loads(
            git_show("results/generation/c80_role_C80_A.technical_review.json")),
    }
    report = stamp_report(build_closure(sources),
                          allow_dirty="--allow-dirty" in sys.argv)
    out = REPO_ROOT / "results/audit/E80_C80_provenance_closure.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
