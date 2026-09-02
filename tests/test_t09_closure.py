"""T09 closure and C80 verifier: fail-closed behaviour.

The closure must refuse to emit any ACCEPTED status when a source report
records a failure, is missing its cryptographic bindings, or disagrees
with another source. The C80 verifier must detect duplicate
(rollout_id, retry) rows and fail the block verdict.
"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


closure = _load("make_t09_provenance_closure")
verifier = _load("verify_t10_c80_generation")

RAW_A = "a" * 64
RAW_B = "b" * 64
EXHAUSTED = ["885b75abc2900244", "79331716c2cde644"]


def block_summary(rows, multi_valid, exhausted_ids):
    return {
        "rows": rows, "rollouts_covered": 22000, "rollouts_missing": 0,
        "rollouts_with_multiple_valid_attempts": multi_valid,
        "exhausted_rollouts": [{"rollout_id": r} for r in exhausted_ids],
    }


def good_sources():
    e80_checks = [
        {"check": name, "status": "PASS"} for name in (
            "translated_ids", "wrapper_ids", "default_ids", "chat_template",
            "prompt_hashes", "generation_settings", "output_closure",
            "no_default_wrapper", "shard_completeness",
            "no_duplicate_or_overwritten_ids")
    ] + [{"check": "model_revision", "status": "GAP"}]
    return {
        "e80": {
            "overall": "GAP",
            "checks": e80_checks,
            "inputs": {"dataset_ref": closure.E80_DATASET},
            "audited_raw_files": {
                arm: [{"path": f"{arm}.jsonl", "bytes": 1, "sha256": "c" * 64}]
                for arm in ("translated", "wrapper", "default")
            },
        },
        "t10": {
            "manifest_check": {"rollout_id_overlap": 0,
                               "question_id_overlap": []},
            "blocks": {
                "C80-A": {"verdict": "PASS", "failed_checks": [],
                          "summary": block_summary(22286, 0, EXHAUSTED)},
                "C80-B": {"verdict": "PASS", "failed_checks": [],
                          "summary": block_summary(22735, 448, [])},
            },
        },
        "selection": {
            "selection_rule": "lowest_retry_technically_valid_attempt",
            "C80-A": {"raw_sha256": RAW_A, "raw_rows_all_attempts": 22286,
                      "unique_rollouts": 22000, "downstream_eligible": 21998,
                      "rollouts_with_multiple_valid_attempts": 0},
            "C80-B": {"raw_sha256": RAW_B, "raw_rows_all_attempts": 22735,
                      "unique_rollouts": 22000, "downstream_eligible": 22000,
                      "rollouts_with_multiple_valid_attempts": 448},
        },
        "manifest_a": {"data_location": {"authoritative_sha256": RAW_A}},
        "review_a": {
            "final_jsonl_sha256": RAW_A,
            "decision": "ACCEPT_WITH_TECHNICAL_EXCLUSIONS",
            "exhausted_rollouts": [{"rollout_id": r} for r in EXHAUSTED],
        },
    }


def test_closure_accepts_consistent_sources():
    report = closure.build_closure(good_sources())
    assert report["blocks"]["E80"]["status"] == "ACCEPTED_WITH_RECORDED_GAP"
    assert report["blocks"]["C80-A"]["status"] == "ACCEPTED"
    assert report["blocks"]["C80-B"]["status"] == "ACCEPTED_UNDER_SELECTION_RULE"


def test_closure_e80_all_pass_yields_accepted():
    src = good_sources()
    src["e80"]["overall"] = "PASS"
    src["e80"]["checks"][-1]["status"] = "PASS"
    assert closure.build_closure(src)["blocks"]["E80"]["status"] == "ACCEPTED"


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s["e80"].update(overall="FAIL"), "not PASS/GAP"),
    (lambda s: s["e80"]["checks"][0].update(status="FAIL"),
     "beyond model_revision"),
    (lambda s: s["e80"]["checks"][0].update(status="BLOCKED"),
     "beyond model_revision"),
    (lambda s: s["e80"].update(audited_raw_files={}), "no raw-file SHA-256s"),
    (lambda s: s["e80"]["audited_raw_files"]["wrapper"][0].pop("sha256"),
     "no raw-file SHA-256s"),
    (lambda s: s["e80"]["inputs"].update(dataset_ref="hf-dataset:other@rev"),
     "dataset_ref"),
    (lambda s: s["t10"]["blocks"]["C80-B"].update(verdict="FAIL"),
     "verdict"),
    (lambda s: s["t10"]["blocks"]["C80-A"]["summary"].update(rows=99),
     "row counts disagree"),
    (lambda s: s["selection"]["C80-A"].update(downstream_eligible=22000),
     "eligible"),
    (lambda s: s["selection"]["C80-B"].update(
        rollouts_with_multiple_valid_attempts=0), "multi-valid"),
    (lambda s: s["manifest_a"]["data_location"].update(
        authoritative_sha256="d" * 64), "disagrees"),
    (lambda s: s["selection"].update(selection_rule="latest_attempt"),
     "selection rule"),
    (lambda s: s["t10"]["manifest_check"].update(rollout_id_overlap=3),
     "disjoint"),
    (lambda s: s["review_a"].update(exhausted_rollouts=[]),
     "eligible|exhausted"),
])
def test_closure_refuses_bad_sources(mutate, match):
    src = copy.deepcopy(good_sources())
    mutate(src)
    with pytest.raises(closure.ClosureError, match=match):
        closure.build_closure(src)


class StubTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        return "rendered"


def test_verifier_detects_duplicate_rollout_retry_rows(tmp_path):
    manifest = {"rollouts": [{
        "rollout_id": "f" * 16, "role_id": "pirate", "question_id": 1,
        "prompt_index": 0, "block": "C80-A", "arm": "USER_TRANSLATED_LU",
        "channel": "user",
    }], "rollouts_sha256": "0" * 64}
    row = {"rollout_id": "f" * 16, "retry": 0, "role_id": "pirate",
           "question_id": 1, "prompt_index": 0, "block": "C80-A",
           "arm": "USER_TRANSLATED_LU", "channel": "user",
           "technical_validity": "valid", "finish_reason": "stop",
           "output_text": "x", "runtime_settings": {"max_new_tokens": 2048}}
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    scan = verifier.BlockScan("C80-A", manifest, StubTokenizer(), {}, {})
    scan.scan(path)
    summary = scan.summary()
    # The second identical row must be counted, not silently overwrite the
    # first in the attempts dict.
    assert summary["duplicate_rollout_retry_pairs"] == 1
    verdict, failures = verifier.block_verdict(summary)
    assert verdict == "FAIL"
    assert "duplicate_rollout_retry_pairs" in failures
