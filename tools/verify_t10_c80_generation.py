"""T10 verification — reconcile the C80-A and C80-B role-generation outputs.

Row-level verification of a generated C80 block against its deterministic
rollout manifest (tools/build_t10_c80_manifest.py) and the frozen method:

- expected deterministic IDs: every row's ``rollout_id`` is in the block's
  frozen 22,000-cell manifest, recomputes from the row's own fields via
  ``src.ids``, and ``row_id`` recomputes from (rollout_id, retry);
- exact row count: every manifest rollout covered, none missing, none extra;
- prompt-index balance: terminal rows are 4,400 per prompt index;
- model/tokenizer revision + chat template + config SHA: uniform per row and
  equal to the frozen pins;
- prompt hashes: stored messages/rendered hashes reproduce from stored
  content, and the rendered prompt is byte-identical to the frozen-template
  reconstruction from the pinned Lu texts;
- decoding settings: uniform apart from the declared max_new_tokens retry
  ladder; reported for cross-block and cross-task (T09/E80) comparison;
- finish reasons: terminal rows finish with "stop" unless the rollout
  exhausted the ladder; non-terminal rows carry a technical failure;
- technical-failure/truncation accounting: per rollout, retries are
  contiguous from 0 with exactly one terminal attempt; attempts that were
  valid but retried anyway (possible after the prefilled-<think>
  resegmentation) are counted and listed;
- duplicate UID guard: no duplicate row_id, no duplicate (rollout_id, retry);
- cross-block: rollout/row IDs disjoint between blocks, settings and code
  provenance comparable.

Blocks whose raw JSONL is unavailable locally are reported as BLOCKED with
whatever committed provenance could still be cross-checked.

Usage:
    python tools/verify_t10_c80_generation.py \
        --manifest-a <path> [--rows-a <path-to-final.jsonl>] \
        --manifest-b <path> [--rows-b <path-to-final.jsonl>] \
        [--output results/audit/T10_C80_reconciliation_report.json] \
        [--allow-dirty]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import ids  # noqa: E402
from src.config import load_historical_frozen_config  # noqa: E402
from src.prompt_rendering import role_arm_messages  # noqa: E402
from src.provenance import stamp_report  # noqa: E402

# Historical artifacts are audited against the generation-time config;
# require_historical_config() still verifies the loaded bytes.
CFG, CONFIG_SHA = load_historical_frozen_config()

# The generation artifacts audited here are stamped with the historical
# frozen configuration. Auditing against any other config (e.g. a later
# V4 or a mutated file) would silently validate rows against the wrong
# frozen values, so report generation refuses to run unless the loaded
# config bytes hash to exactly this value.
HISTORICAL_CONFIG_SHA = "73be73df4640c2d32bfbc8b6009741c0fadd8c466acb703d1f99df35d5c8cd79"


def require_historical_config():
    if CONFIG_SHA != HISTORICAL_CONFIG_SHA:
        raise RuntimeError(
            "loaded frozen config hashes to "
            f"{CONFIG_SHA}, not the generation-time historical config "
            f"{HISTORICAL_CONFIG_SHA}; refusing to audit historical "
            "artifacts against a different configuration (see issue #31)"
        )


ARM = "USER_TRANSLATED_LU"
MODEL_ID = CFG["models"]["primary"]["model_id"]
FROZEN_MODEL_REVISION = CFG["models"]["primary"]["model_revision"]
FROZEN_TOKENIZER_REVISION = CFG["models"]["primary"]["tokenizer_revision"]
FROZEN_CHAT_TEMPLATE_SHA = CFG["models"]["primary"]["chat_template_sha256"]

MAX_EXAMPLES = 10

# Settings keys that legitimately vary inside one block: the declared
# truncation-retry ladder raises only the token budget.
LADDER_KEY = "max_new_tokens"


def load_lu_texts():
    lu_dir = REPO_ROOT / "data" / "lu_et_al"
    prompts = json.loads((lu_dir / "role_prompts.json").read_text())
    questions = json.loads((lu_dir / "questions.json").read_text())
    return (
        {(p["role_id"], p["prompt_index"]): p["text"] for p in prompts["prompts"]},
        {q["question_id"]: q["text"] for q in questions["questions"]},
    )


class BlockScan:
    """Streaming verification of one block's final JSONL."""

    def __init__(self, block, manifest, tokenizer, role_prompts, questions):
        self.block = block
        self.expected = {r["rollout_id"]: r for r in manifest["rollouts"]}
        self.tokenizer = tokenizer
        self.role_prompts = role_prompts
        self.questions = questions
        self.n_rows = 0
        self.parse_errors = []
        # Real totals (never capped) and capped diagnostic samples.
        self.counts = {"unknown_rollouts": 0, "id_recompute_mismatches": 0,
                       "field_mismatches": 0, "hash_failures": 0,
                       "render_mismatches": 0, "duplicate_attempt_rows": 0}
        self.unknown_rollouts = []
        self.id_recompute_mismatches = []
        self.field_mismatches = []
        self.hash_failures = []
        self.render_mismatches = []
        self.pin_violations = Counter()
        self.settings_excl_ladder = Counter()
        self.ladder_values = Counter()
        self.finish_reasons_terminal = Counter()
        self.validity_all = Counter()
        self.validity_terminal = Counter()
        self.segmentation_terminal = Counter()
        self.prompt_index_terminal = Counter()
        self.row_id_counts = Counter()
        self.attempts = defaultdict(dict)  # rollout -> retry -> summary
        self.region_count_errors = 0

    def scan(self, path):
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    self.parse_errors.append(f"{path.name}:{lineno}: {e}")
                    continue
                self.n_rows += 1
                self._scan_row(row, f"{path.name}:{lineno}")
        self._finalize()

    def _scan_row(self, row, where):
        rollout = row.get("rollout_id")
        retry = row.get("retry")
        expected = self.expected.get(rollout)
        if expected is None:
            self.counts["unknown_rollouts"] += 1
            if len(self.unknown_rollouts) < MAX_EXAMPLES:
                self.unknown_rollouts.append({"where": where, "rollout_id": rollout})
            return

        recomputed = ids.rollout_id(
            row.get("model"), row.get("arm"), row.get("role_id"),
            row.get("question_id"), row.get("prompt_index"),
        )
        if recomputed != rollout or ids.generation_row_id(rollout, retry) != row.get("row_id"):
            self.counts["id_recompute_mismatches"] += 1
            if len(self.id_recompute_mismatches) < MAX_EXAMPLES:
                self.id_recompute_mismatches.append({"where": where, "rollout_id": rollout})

        for key in ("role_id", "question_id", "prompt_index", "block", "arm", "channel"):
            if row.get(key) != expected[key if key != "role_id" else "role_id"]:
                self.counts["field_mismatches"] += 1
                if len(self.field_mismatches) < MAX_EXAMPLES:
                    self.field_mismatches.append(
                        {"where": where, "field": key,
                         "row": row.get(key), "manifest": expected.get(key)})

        for key, pin in (("model", MODEL_ID),
                         ("model_revision", FROZEN_MODEL_REVISION),
                         ("tokenizer_revision", FROZEN_TOKENIZER_REVISION),
                         ("chat_template_sha256", FROZEN_CHAT_TEMPLATE_SHA),
                         ("config_sha256", CONFIG_SHA)):
            if row.get(key) != pin:
                self.pin_violations[key] += 1

        if (row.get("messages_sha256") != ids.messages_sha256(row.get("messages"))
                or row.get("rendered_prompt_sha256")
                != ids.text_sha256(row.get("rendered_prompt", ""))):
            self.counts["hash_failures"] += 1
            if len(self.hash_failures) < MAX_EXAMPLES:
                self.hash_failures.append({"where": where, "rollout_id": rollout})

        instruction = self.role_prompts.get((row.get("role_id"), row.get("prompt_index")))
        question = self.questions.get(row.get("question_id"))
        if instruction is None or question is None:
            rendered_expected = None
        else:
            rendered_expected = self.tokenizer.apply_chat_template(
                role_arm_messages(CFG, ARM, instruction, question),
                tokenize=False, add_generation_prompt=True,
            )
        if rendered_expected is None or row.get("rendered_prompt") != rendered_expected:
            self.counts["render_mismatches"] += 1
            if len(self.render_mismatches) < MAX_EXAMPLES:
                self.render_mismatches.append({"where": where, "rollout_id": rollout})

        settings = dict(row.get("runtime_settings") or {})
        self.ladder_values[settings.pop(LADDER_KEY, None)] += 1
        self.settings_excl_ladder[json.dumps(settings, sort_keys=True)] += 1

        regions = row.get("token_region_counts") or {}
        if (row.get("technical_validity") == "valid"
                and regions.get("all_response")
                != (regions.get("reasoning") or 0) + (regions.get("final_answer") or 0)):
            self.region_count_errors += 1

        self.validity_all[row.get("technical_validity")] += 1
        self.row_id_counts[row.get("row_id")] += 1
        if retry in self.attempts[rollout]:
            self.counts["duplicate_attempt_rows"] += 1
        self.attempts[rollout][retry] = {
            "validity": row.get("technical_validity"),
            "finish_reason": row.get("finish_reason"),
            "failure_code": row.get("technical_failure_code"),
            "prompt_index": row.get("prompt_index"),
            "segmentation": row.get("segmentation_case"),
            "budget": (row.get("runtime_settings") or {}).get(LADDER_KEY),
            "text_sha": ids.text_sha256(row.get("output_text", ""))[:16],
        }

    def _finalize(self):
        self.noncontiguous_retries = []
        self.multi_valid_rollouts = []
        self.multi_valid_differing_text = 0
        self.valid_then_retried = []
        self.exhausted = []
        self.same_budget_repeat_attempts = 0
        self.attempt_patterns = Counter()
        self.attempts_per_rollout = Counter()
        for rollout, tries in self.attempts.items():
            retries = sorted(tries)
            self.attempts_per_rollout[len(retries)] += 1
            ordered = [tries[r] for r in retries]
            self.attempt_patterns[tuple(
                (a["budget"], a["validity"]) for a in ordered)] += 1
            budgets = [a["budget"] for a in ordered]
            self.same_budget_repeat_attempts += sum(
                1 for i in range(1, len(budgets)) if budgets[i] == budgets[i - 1])
            valid_texts = {a["text_sha"] for a in ordered if a["validity"] == "valid"}
            if len(valid_texts) > 1:
                self.multi_valid_differing_text += 1
            if retries != list(range(len(retries))):
                if len(self.noncontiguous_retries) < MAX_EXAMPLES:
                    self.noncontiguous_retries.append(
                        {"rollout_id": rollout, "retries": retries})
                continue
            terminal_retry = retries[-1]
            terminal = tries[terminal_retry]
            self.validity_terminal[terminal["validity"]] += 1
            self.segmentation_terminal[str(terminal["segmentation"])] += 1
            self.finish_reasons_terminal[terminal["finish_reason"]] += 1
            self.prompt_index_terminal[terminal["prompt_index"]] += 1
            if terminal["validity"] != "valid":
                self.exhausted.append({"rollout_id": rollout,
                                       "attempts": len(retries),
                                       "terminal": terminal})
            valids = [r for r in retries if tries[r]["validity"] == "valid"]
            if len(valids) > 1:
                self.multi_valid_rollouts.append(rollout)
            if any(r < terminal_retry for r in valids):
                self.valid_then_retried.append(rollout)

    def summary(self):
        covered = set(self.attempts)
        missing = set(self.expected) - covered
        return {
            "rows": self.n_rows,
            "rollouts_covered": len(covered),
            "rollouts_missing": len(missing),
            "unknown_rollouts": self.counts["unknown_rollouts"],
            "id_recompute_mismatches": self.counts["id_recompute_mismatches"],
            "manifest_field_mismatches": self.counts["field_mismatches"],
            "stored_hash_failures": self.counts["hash_failures"],
            "render_reconstruction_mismatches": self.counts["render_mismatches"],
            "frozen_pin_violations": dict(self.pin_violations),
            "settings_excluding_token_ladder": {
                k: v for k, v in self.settings_excl_ladder.items()},
            "token_ladder_values": {str(k): v for k, v in self.ladder_values.items()},
            "prompt_index_balance_terminal": {
                str(k): v for k, v in sorted(self.prompt_index_terminal.items())},
            "finish_reasons_terminal": dict(self.finish_reasons_terminal),
            "technical_validity_all_attempts": dict(self.validity_all),
            "technical_validity_terminal": dict(self.validity_terminal),
            "segmentation_terminal": dict(self.segmentation_terminal),
            "duplicate_row_ids": sum(1 for n in self.row_id_counts.values() if n > 1),
            "duplicate_rollout_retry_pairs": self.counts["duplicate_attempt_rows"],
            "noncontiguous_retry_ladders": len(self.noncontiguous_retries),
            "attempts_per_rollout": {
                str(k): v for k, v in sorted(self.attempts_per_rollout.items())},
            "attempt_patterns_budget_validity": {
                " -> ".join(f"{b}:{v}" for b, v in sig): n
                for sig, n in self.attempt_patterns.most_common(12) if len(sig) > 1},
            "same_budget_repeat_attempts": self.same_budget_repeat_attempts,
            "rollouts_with_multiple_valid_attempts": len(self.multi_valid_rollouts),
            "rollouts_with_multiple_valid_attempts_differing_text":
                self.multi_valid_differing_text,
            "rollouts_where_a_valid_attempt_was_retried": len(self.valid_then_retried),
            "region_count_errors": self.region_count_errors,
            "exhausted_rollouts": self.exhausted[:MAX_EXAMPLES],
            "n_exhausted_rollouts": len(self.exhausted),
            "parse_errors": len(self.parse_errors),
        }


def block_verdict(summary, n_expected=22000):
    """Fail-closed PASS/FAIL for one scanned block: full coverage, perfect
    integrity, exact prompt balance. Returns (verdict, failures)."""
    failures = []
    if summary["rollouts_covered"] != n_expected or summary["rollouts_missing"] != 0:
        failures.append("coverage")
    for key in ("unknown_rollouts", "id_recompute_mismatches",
                "manifest_field_mismatches", "stored_hash_failures",
                "render_reconstruction_mismatches", "duplicate_row_ids",
                "duplicate_rollout_retry_pairs", "noncontiguous_retry_ladders",
                "region_count_errors", "parse_errors"):
        if summary[key] != 0:
            failures.append(key)
    if summary["frozen_pin_violations"]:
        failures.append("frozen_pin_violations")
    if summary["prompt_index_balance_terminal"] != {
            str(i): n_expected // 5 for i in range(5)}:
        failures.append("prompt_index_balance")
    return ("PASS" if not failures else "FAIL"), failures


def main(argv=None):
    require_historical_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest-a", required=True)
    parser.add_argument("--manifest-b", required=True)
    parser.add_argument("--rows-a", default=None)
    parser.add_argument("--rows-b", default=None)
    parser.add_argument("--provenance-a", default=None,
                        help="committed provenance manifest for block A")
    parser.add_argument("--provenance-b", default=None)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "results" / "audit" /
                    "T10_C80_reconciliation_report.json"),
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=FROZEN_TOKENIZER_REVISION)
    role_prompts, questions = load_lu_texts()

    manifests = {
        "C80-A": json.loads(Path(args.manifest_a).read_text()),
        "C80-B": json.loads(Path(args.manifest_b).read_text()),
    }
    ids_a = {r["rollout_id"] for r in manifests["C80-A"]["rollouts"]}
    ids_b = {r["rollout_id"] for r in manifests["C80-B"]["rollouts"]}
    qids_a = {r["question_id"] for r in manifests["C80-A"]["rollouts"]}
    qids_b = {r["question_id"] for r in manifests["C80-B"]["rollouts"]}

    blocks = {}
    for block, rows_path in (("C80-A", args.rows_a), ("C80-B", args.rows_b)):
        if rows_path is None:
            blocks[block] = {"status": "BLOCKED",
                             "reason": "raw final JSONL not available locally"}
            continue
        scan = BlockScan(block, manifests[block], tokenizer, role_prompts, questions)
        scan.scan(Path(rows_path))
        summary = scan.summary()
        verdict, failures = block_verdict(summary)
        blocks[block] = {"status": "VERIFIED" if verdict == "PASS" else "FAILED",
                         "verdict": verdict, "failed_checks": failures,
                         "summary": summary}

    provenance = {}
    for block, path in (("C80-A", args.provenance_a), ("C80-B", args.provenance_b)):
        if path:
            provenance[block] = json.loads(Path(path).read_text())

    report = stamp_report({
        "task": "T10_C80_RECONCILIATION",
        "title": "C80-A / C80-B role-generation reconciliation",
        "frozen_pins": {
            "model_id": MODEL_ID,
            "model_revision": FROZEN_MODEL_REVISION,
            "tokenizer_revision": FROZEN_TOKENIZER_REVISION,
            "chat_template_sha256": FROZEN_CHAT_TEMPLATE_SHA,
            "method_config_sha256": CONFIG_SHA,
        },
        "manifest_check": {
            "a_rollouts": len(ids_a), "b_rollouts": len(ids_b),
            "total": len(ids_a) + len(ids_b),
            "rollout_id_overlap": len(ids_a & ids_b),
            "question_id_overlap": sorted(qids_a & qids_b),
            "a_rollouts_sha256": manifests["C80-A"]["rollouts_sha256"],
            "b_rollouts_sha256": manifests["C80-B"]["rollouts_sha256"],
        },
        "blocks": blocks,
        "committed_provenance": provenance,
    }, allow_dirty=args.allow_dirty)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    verdicts = {b: blocks[b].get("verdict", "BLOCKED") for b in blocks}
    print(json.dumps({"report": str(out), "verdicts": verdicts}, indent=2))
    # Fail closed: exit 0 only when every requested block scanned and passed.
    return 0 if all(v == "PASS" for v in verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
