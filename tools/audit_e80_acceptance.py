"""T09 — audit and accept/reject the existing E80 raw text (run-003).

Audits the historical run-003 rollout shards against the frozen method
(configs/method_frozen.yaml). The shards predate the canonical T03 row
schema; they carry, per row: ``uid, shard, role, prompt_idx, question_id,
condition, rendered_prompt, completion, n_out_tokens, finish_reason,
closed_think, gen_max_tokens, gen_temperature, gen_top_p, gen_seed``.
Because no prompt hashes or engine token IDs were persisted, the prompt
checks are performed in their strongest available form: every stored
``rendered_prompt`` is reconstructed byte-for-byte from the pinned Lu
source artifacts, the frozen prompt templates, and the frozen tokenizer
chat template.

Eleven acceptance checks (task card):

 1. translated_ids   — 22,000 unique (role, question) cells (275 x 80) in
                       the USER_TRANSLATED_LU arm, frozen prompt_idx per cell;
 2. wrapper_ids      — the same 22,000-cell coverage for USER_EXPLICIT;
 3. default_ids      — 400 unique (question, condition) cells (80 x 5);
 4. model_revision   — the model recorded everywhere in the run equals the
                       frozen primary; generation-time weight revision
                       recorded or GAP;
 5. chat_template    — frozen tokenizer revision's chat template hash equals
                       the frozen chat_template_sha256, and every stored
                       prompt was rendered with it (byte-exact);
 6. prompt_hashes    — every rendered_prompt reproduces exactly from the
                       frozen templates + pinned Lu texts (hash-equivalent);
 7. generation_settings — one (max_tokens, temperature, top_p, seed) tuple
                       across all rows;
 8. output_closure   — stored closed_think flags agree with the completions;
                       every default condition >= the frozen 95% valid
                       fraction (role arms reported, not gated: their
                       retention gate is the validated role judge);
 9. no_default_wrapper — every DEFAULT row renders the frozen unwrapped
                       default condition, and no role instruction appears;
10. shard_completeness — declared shard sequences are present and gap-free,
                       rows sit in their declared shard, counts match the
                       run reports, every line parses;
11. no_duplicate_or_overwritten_ids — no duplicate uid and no duplicate
                       semantic cell anywhere.

Statuses: PASS, FAIL, GAP (a required fact was never recorded in the run
artifacts and cannot be reconstructed, but nothing contradicts the frozen
method), BLOCKED (audit input unavailable). Overall: FAIL > GAP > BLOCKED
> PASS.

Usage:
    python tools/audit_e80_acceptance.py \
        --data-root <path-to>/lu-replication/run-003 \
        [--output results/audit/E80_acceptance_report.json] \
        [--dataset-ref [Author-A-HF]/persona-artifacts@<revision>] \
        [--allow-dirty]
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import (block_question_ids, load_historical_frozen_config,
                        prompt_index_for)  # noqa: E402
from src.prompt_rendering import default_messages, role_arm_messages  # noqa: E402
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


BLOCK = "E80"
E80_IDS = block_question_ids(CFG)[BLOCK]
FROZEN_PROMPT_INDEX = {
    qid: prompt_index_for(CFG, E80_IDS, pos) for pos, qid in enumerate(E80_IDS)
}
MODEL_ID = CFG["models"]["primary"]["model_id"]
MODEL_DISPLAY_NAME = CFG["models"]["primary"]["display_name_for_default_template"]
FROZEN_TOKENIZER_REVISION = CFG["models"]["primary"]["tokenizer_revision"]
FROZEN_CHAT_TEMPLATE_SHA = CFG["models"]["primary"]["chat_template_sha256"]
N_DEFAULT_CONDITIONS = len(CFG["prompt_rendering"]["default_conditions"])
MIN_VALID_FRACTION_PER_DEFAULT_CONDITION = (
    CFG["role_vectors_and_axis"]["default_vector"]["minimum_valid_fraction_per_condition"]
)
CLOSE_MARKER = CFG["response_segmentation"]["closing_marker"]

# Historical run-003 layout (docs/ARTIFACT_STATUS.md and the run MANIFEST):
# arm key -> (subdirectory, frozen condition name).
ARMS = {
    "translated": ("translated", "USER_TRANSLATED_LU"),
    "wrapper": ("extraction", "USER_EXPLICIT"),
    "default": ("default", None),
}

MAX_EXAMPLES = 10


def load_lu_inputs(lu_dir):
    """Pinned Lu source artifacts: roles, per-index role prompts, questions."""
    try:
        roles = json.loads((lu_dir / "roles.json").read_text())
        prompts = json.loads((lu_dir / "role_prompts.json").read_text())
        questions = json.loads((lu_dir / "questions.json").read_text())
    except FileNotFoundError:
        return None
    return {
        "role_ids": tuple(r["role_id"] for r in roles["roles"]),
        "role_prompts": {
            (p["role_id"], p["prompt_index"]): p["text"] for p in prompts["prompts"]
        },
        "question_texts": {q["question_id"]: q["text"] for q in questions["questions"]},
    }


def load_frozen_tokenizer():
    """The frozen tokenizer, pinned to the frozen revision, with its chat
    template hash verified before any prompt is reconstructed."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=FROZEN_TOKENIZER_REVISION,
    )
    template_sha = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
    return tokenizer, template_sha


class PromptReconstructor:
    """Byte-exact expected rendered_prompt for every frozen cell."""

    def __init__(self, tokenizer, lu):
        self.tokenizer = tokenizer
        self.lu = lu

    def _render(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def role_arm(self, condition, role_id, prompt_idx, question_id):
        instruction = self.lu["role_prompts"].get((role_id, prompt_idx))
        question = self.lu["question_texts"].get(question_id)
        if instruction is None or question is None:
            return None
        return self._render(role_arm_messages(CFG, condition, instruction, question))

    def default(self, condition_index, question_id):
        question = self.lu["question_texts"].get(question_id)
        if question is None:
            return None
        return self._render(default_messages(
            CFG, condition_index, question, "user", model_name=MODEL_DISPLAY_NAME,
        ))


class ArmScan:
    """Streaming accumulator over one historical arm's rollout shards."""

    def __init__(self, arm, condition, expected_cells, reconstructor):
        self.arm = arm
        self.condition = condition
        self.expected_cells = expected_cells  # cell -> required prompt_idx (or None)
        self.reconstructor = reconstructor
        self.n_rows = 0
        self.parse_errors = []
        self.missing_field_rows = 0
        self.seen_cells = set()
        self.extra_cells = []
        self.prompt_idx_mismatches = []
        self.uid_counts = Counter()
        self.cell_counts = Counter()
        self.condition_values = Counter()
        self.settings = Counter()
        self.render_mismatches = 0
        self.render_mismatch_examples = []
        self.finish_reasons = Counter()
        self.closed_think = Counter()
        self.closed_think_flag_errors = 0
        self.empty_completions = 0
        self.shard_field_errors = 0
        self.shard_rows = {}
        self.default_condition_totals = Counter()
        self.default_condition_valid = Counter()
        self.wrapped_default_rows = []

    def scan_file(self, path, declared_shard):
        n = 0
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    self.parse_errors.append(f"{path.name}:{lineno}: {e}")
                    continue
                n += 1
                self._scan_row(row, declared_shard, f"{path.name}:{lineno}")
        self.shard_rows[path.name] = n
        self.n_rows += n

    def _scan_row(self, row, declared_shard, where):
        try:
            uid = row["uid"]
            role = row["role"]
            qid = row["question_id"]
            pidx = row["prompt_idx"]
            rendered = row["rendered_prompt"]
            completion = row["completion"]
        except KeyError:
            self.missing_field_rows += 1
            return

        self.uid_counts[uid] += 1
        self.condition_values[row.get("condition")] += 1
        self.settings[(
            row.get("gen_max_tokens"), row.get("gen_temperature"),
            row.get("gen_top_p"), row.get("gen_seed"),
        )] += 1
        if row.get("shard") != declared_shard:
            self.shard_field_errors += 1

        if self.arm == "default":
            cell = (qid, pidx)
            expected_render = self.reconstructor.default(pidx, qid)
        else:
            cell = (role, qid)
            expected_render = self.reconstructor.role_arm(self.condition, role, pidx, qid)
        self.cell_counts[cell] += 1

        if cell in self.expected_cells:
            self.seen_cells.add(cell)
            required_idx = self.expected_cells[cell]
            if required_idx is not None and pidx != required_idx:
                if len(self.prompt_idx_mismatches) < MAX_EXAMPLES:
                    self.prompt_idx_mismatches.append({
                        "where": where, "cell": list(cell),
                        "prompt_idx": pidx, "frozen": required_idx,
                    })
                expected_render = None  # cannot certify a wrong-index render
        elif len(self.extra_cells) < MAX_EXAMPLES:
            self.extra_cells.append({"where": where, "cell": list(cell)})

        if expected_render is None or rendered != expected_render:
            self.render_mismatches += 1
            if len(self.render_mismatch_examples) < MAX_EXAMPLES:
                self.render_mismatch_examples.append({
                    "where": where, "cell": list(cell),
                    "rendered_prompt_head": rendered[:160],
                })
            if self.arm == "default" and len(self.wrapped_default_rows) < MAX_EXAMPLES:
                self.wrapped_default_rows.append({
                    "where": where, "cell": list(cell),
                    "rendered_prompt_head": rendered[:160],
                })

        self.finish_reasons[row.get("finish_reason")] += 1
        closed_stored = row.get("closed_think")
        self.closed_think[closed_stored] += 1
        if closed_stored != (CLOSE_MARKER in completion):
            self.closed_think_flag_errors += 1
        if not completion.strip():
            self.empty_completions += 1

        if self.arm == "default":
            self.default_condition_totals[pidx] += 1
            valid = (
                row.get("finish_reason") == "stop"
                and closed_stored is True
                and bool(completion.strip())
            )
            if valid:
                self.default_condition_valid[pidx] += 1

    def missing_cells(self):
        return sorted(set(self.expected_cells) - self.seen_cells)


def check(name, status, expected, observed, detail=None):
    entry = {"check": name, "status": status,
             "expected": expected, "observed": observed}
    if detail is not None:
        entry["detail"] = detail
    return entry


def id_completeness_check(name, scan):
    missing = scan.missing_cells()
    extra = sum(1 for cell in scan.cell_counts if cell not in scan.expected_cells)
    ok = (scan.n_rows > 0 and not missing and not extra
          and not scan.prompt_idx_mismatches and not scan.missing_field_rows)
    return check(
        name,
        "PASS" if ok else ("BLOCKED" if scan.n_rows == 0 else "FAIL"),
        f"{len(scan.expected_cells)} unique frozen cells, frozen prompt_idx, no extras",
        {
            "rows": scan.n_rows,
            "expected_cells_covered": len(scan.seen_cells),
            "missing_cells": len(missing),
            "extra_cells": extra,
            "prompt_idx_mismatches": len(scan.prompt_idx_mismatches),
            "rows_missing_required_fields": scan.missing_field_rows,
        },
        {
            "missing_cells_sample": [list(c) for c in missing[:MAX_EXAMPLES]],
            "extra_cells_sample": scan.extra_cells,
            "prompt_idx_mismatch_sample": scan.prompt_idx_mismatches,
        },
    )


def sha256_file(path, block_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def scan_arms(data_root, lu, reconstructor, run_reports):
    scans = {}
    for arm, (subdir, condition) in ARMS.items():
        if arm == "default":
            cells = {(qid, cond): cond
                     for qid in E80_IDS for cond in range(N_DEFAULT_CONDITIONS)}
        else:
            cells = {(role, qid): FROZEN_PROMPT_INDEX[qid]
                     for role in lu["role_ids"] for qid in E80_IDS}
        scan = ArmScan(arm, condition, cells, reconstructor)
        arm_dir = data_root / subdir
        scan.raw_files = []
        for path in sorted(arm_dir.glob("rollouts_shard*.jsonl")):
            declared = int(path.stem.replace("rollouts_shard", ""))
            scan.scan_file(path, declared)
            scan.raw_files.append({
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        scans[arm] = scan
        report = run_reports.get(arm)
        scan.run_report = report
    return scans


def load_run_reports(data_root):
    reports = {}
    for arm, (subdir, _) in ARMS.items():
        arm_reports = []
        for path in sorted((data_root / subdir).glob("run_report_shard*.json")):
            arm_reports.append(json.loads(path.read_text()))
        reports[arm] = arm_reports
    manifest_path = data_root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    return reports, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "replications" / "lu-replication" / "run-003"),
        help="run-003 root containing translated/, extraction/, default/",
    )
    parser.add_argument(
        "--lu-data", default=str(REPO_ROOT / "data" / "lu_et_al"),
        help="directory holding the pinned Lu roles/role_prompts/questions",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "results" / "audit" / "E80_acceptance_report.json"),
    )
    parser.add_argument(
        "--dataset-ref", default=None,
        help="provenance label for the audited copy, e.g. hf-repo@revision",
    )
    parser.add_argument("--allow-dirty", action="store_true",
                        help="stamp from a dirty tree (development only)")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    lu = load_lu_inputs(Path(args.lu_data))

    check_names = (
        "translated_ids", "wrapper_ids", "default_ids", "model_revision",
        "chat_template", "prompt_hashes", "generation_settings",
        "output_closure", "no_default_wrapper", "shard_completeness",
        "no_duplicate_or_overwritten_ids",
    )

    require_historical_config()
    inputs = {
        "data_root": str(data_root),
        "dataset_ref": args.dataset_ref,
        "generation_time_config_sha256": HISTORICAL_CONFIG_SHA,
        "data_root_present": data_root.is_dir(),
        "lu_artifacts_present": lu is not None,
        "block": BLOCK,
        "model_id": MODEL_ID,
        "frozen_tokenizer_revision": FROZEN_TOKENIZER_REVISION,
    }

    if lu is None or not data_root.is_dir():
        reason = ("Lu source artifacts unavailable" if lu is None
                  else "run-003 data root unavailable")
        checks = [check(name, "BLOCKED", reason, None) for name in check_names]
        report = stamp_report({
            "task": "T09_E80_ACCEPTANCE",
            "title": "E80 raw-text acceptance audit (run-003)",
            "inputs": inputs,
            "checks": checks,
            "overall": "BLOCKED",
        }, allow_dirty=args.allow_dirty)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"overall": "BLOCKED", "report": str(out)}, indent=2))
        return 1

    tokenizer, observed_template_sha = load_frozen_tokenizer()
    reconstructor = PromptReconstructor(tokenizer, lu)
    run_reports, manifest = load_run_reports(data_root)
    scans = scan_arms(data_root, lu, reconstructor, run_reports)
    translated, wrapper, default = scans["translated"], scans["wrapper"], scans["default"]
    all_scans = [translated, wrapper, default]

    checks = [
        id_completeness_check("translated_ids", translated),
        id_completeness_check("wrapper_ids", wrapper),
        id_completeness_check("default_ids", default),
    ]

    # 4. model_revision — the shards themselves never recorded the model or
    # its weight revision; the run reports and manifest record the model ID
    # only. A matching ID everywhere is necessary but cannot certify the
    # weights; that fact is a GAP, not a contradiction.
    recorded_models = {
        r.get("model") for reports in run_reports.values() for r in reports
    } | ({manifest.get("model")} if manifest else set())
    recorded_models.discard(None)
    model_ok = recorded_models == {MODEL_ID}
    checks.append(check(
        "model_revision",
        "GAP" if model_ok else "FAIL",
        f"model == {MODEL_ID!r} wherever recorded; generation-time weight "
        "revision pinned",
        {
            "models_recorded_in_run_reports_and_manifest": sorted(recorded_models),
            "generation_time_model_revision_recorded": False,
            "rows_carry_model_fields": False,
        },
        "The run never recorded a weights revision. The frozen YAML pins "
        f"revision {CFG['models']['primary']['model_revision']!r} resolved "
        "post hoc (T12), whose five-row activation reconstruction against "
        "historical tensors supports — but cannot prove — that these weights "
        "produced the stored text.",
    ))

    # 5. chat_template — frozen-revision template hash plus byte-exact use.
    render_mismatches = sum(s.render_mismatches for s in all_scans)
    template_match = observed_template_sha == FROZEN_CHAT_TEMPLATE_SHA
    checks.append(check(
        "chat_template",
        "PASS" if template_match and render_mismatches == 0 else "FAIL",
        "frozen tokenizer revision's chat template hash equals the frozen "
        "chat_template_sha256 and every stored prompt renders with it",
        {
            "frozen_chat_template_sha256": FROZEN_CHAT_TEMPLATE_SHA,
            "observed_chat_template_sha256": observed_template_sha,
            "rows_not_byte_identical_to_frozen_render": render_mismatches,
        },
    ))

    # 6. prompt_hashes — strongest available form for hashless rows.
    checks.append(check(
        "prompt_hashes",
        "PASS" if render_mismatches == 0 else "FAIL",
        "every rendered_prompt reproduces byte-for-byte from the frozen "
        "templates and pinned Lu texts (no stored hashes exist to compare)",
        {
            "rows_checked": sum(s.n_rows for s in all_scans),
            "reconstruction_mismatches": render_mismatches,
        },
        {"mismatch_sample": [e for s in all_scans
                             for e in s.render_mismatch_examples][:MAX_EXAMPLES]},
    ))

    # 7. generation_settings.
    merged_settings = Counter()
    for s in all_scans:
        merged_settings.update(s.settings)
    checks.append(check(
        "generation_settings",
        "PASS" if len(merged_settings) == 1 and None not in next(
            iter(merged_settings)) else "FAIL",
        "one (max_tokens, temperature, top_p, seed) tuple across all rows",
        {"distinct_settings": {
            str(k): v for k, v in merged_settings.items()
        }},
    ))

    # 8. output_closure.
    default_fractions = {}
    closure_ok = True
    for cond in range(N_DEFAULT_CONDITIONS):
        total = default.default_condition_totals[cond]
        valid = default.default_condition_valid[cond]
        fraction = valid / total if total else None
        default_fractions[str(cond)] = {
            "rows": total, "valid": valid, "valid_fraction": fraction,
        }
        if not total or fraction < MIN_VALID_FRACTION_PER_DEFAULT_CONDITION:
            closure_ok = False
    flag_errors = sum(s.closed_think_flag_errors for s in all_scans)
    if flag_errors:
        closure_ok = False
    checks.append(check(
        "output_closure",
        "PASS" if closure_ok else "FAIL",
        "stored closed_think flags agree with completions; every default "
        f"condition >= {MIN_VALID_FRACTION_PER_DEFAULT_CONDITION:.0%} valid "
        "(stop + closed + non-empty); role-arm closure reported for "
        "judge-stage accounting",
        {
            "closed_think_flag_disagreements": flag_errors,
            "default_condition_validity": default_fractions,
            "finish_reasons": {s.arm: dict(s.finish_reasons) for s in all_scans},
            "closed_think": {s.arm: {str(k): v for k, v in s.closed_think.items()}
                             for s in all_scans},
            "empty_completions": {s.arm: s.empty_completions for s in all_scans},
        },
    ))

    # 9. no_default_wrapper — byte-exact unwrapped renders already imply no
    # wrapper; surface the default arm separately plus the metadata mislabel.
    default_condition_labels = dict(default.condition_values)
    checks.append(check(
        "no_default_wrapper",
        "PASS" if default.n_rows and default.render_mismatches == 0 else (
            "BLOCKED" if not default.n_rows else "FAIL"),
        "every DEFAULT row is byte-identical to the frozen unwrapped "
        "default-condition rendering",
        {
            "rows_checked": default.n_rows,
            "non_frozen_renders": default.render_mismatches,
            "condition_field_values": {
                str(k): v for k, v in default_condition_labels.items()
            },
        },
        {
            "wrapped_row_sample": default.wrapped_default_rows,
            "note": (
                "The default worker's `condition` metadata field is "
                "mislabelled (inherited from the extraction config); the "
                "rendered prompts themselves are what this check certifies."
                if set(default_condition_labels) - {"DEFAULT", None} else None
            ),
        },
    ))

    # 10. shard_completeness.
    parse_errors = [e for s in all_scans for e in s.parse_errors]
    shard_gaps = []
    count_mismatches = []
    for s in all_scans:
        reports = run_reports.get(s.arm, [])
        declared_n_shards = {r.get("n_shards") for r in reports}
        present = sorted(
            int(name.replace("rollouts_shard", "").replace(".jsonl", ""))
            for name in s.shard_rows
        )
        if declared_n_shards:
            n_shards = max(declared_n_shards)
            missing = sorted(set(range(n_shards)) - set(present))
            if missing:
                shard_gaps.append({"arm": s.arm, "missing_shard_indices": missing})
        expected_rows = sum(
            r.get("n_expected") or 0 for r in reports
        ) or {"translated": 22000, "wrapper": 22000, "default": 400}[s.arm]
        if s.n_rows != expected_rows:
            count_mismatches.append({
                "arm": s.arm, "expected": expected_rows, "observed": s.n_rows,
            })
    shard_ok = (not parse_errors and not shard_gaps and not count_mismatches
                and not any(s.shard_field_errors for s in all_scans))
    checks.append(check(
        "shard_completeness",
        "PASS" if shard_ok else "FAIL",
        "declared shard sequences gap-free; rows in their declared shard; "
        "counts match run reports; every line parses",
        {
            "shards": {s.arm: s.shard_rows for s in all_scans},
            "parse_errors": len(parse_errors),
            "shard_field_errors": {s.arm: s.shard_field_errors for s in all_scans},
            "count_mismatches": count_mismatches,
            "shard_sequence_gaps": shard_gaps,
        },
        {"parse_error_sample": parse_errors[:MAX_EXAMPLES]},
    ))

    # 11. no_duplicate_or_overwritten_ids.
    dup_uids = {s.arm: sum(1 for _, n in s.uid_counts.items() if n > 1)
                for s in all_scans}
    cross_arm_uids = Counter()
    for s in all_scans:
        cross_arm_uids.update(s.uid_counts.keys())
    cross_dups = sum(1 for _, n in cross_arm_uids.items() if n > 1)
    dup_cells = {s.arm: sum(1 for _, n in s.cell_counts.items() if n > 1)
                 for s in all_scans}
    checks.append(check(
        "no_duplicate_or_overwritten_ids",
        "PASS" if not any(dup_uids.values()) and not cross_dups
        and not any(dup_cells.values()) else "FAIL",
        "no duplicate uid within or across arms; no duplicate semantic cell",
        {
            "duplicate_uids_within_arm": dup_uids,
            "uids_shared_across_arms": cross_dups,
            "duplicate_cells": dup_cells,
        },
    ))

    statuses = [c["status"] for c in checks]
    overall = ("FAIL" if "FAIL" in statuses
               else "GAP" if "GAP" in statuses
               else "BLOCKED" if "BLOCKED" in statuses else "PASS")

    report = stamp_report({
        "task": "T09_E80_ACCEPTANCE",
        "title": "E80 raw-text acceptance audit (run-003)",
        "inputs": inputs,
        # Cryptographic binding: the acceptance verdict identifies these
        # exact raw bytes, not merely the dataset_ref label.
        "audited_raw_files": {
            s.arm: getattr(s, "raw_files", []) for s in scans.values()
        },
        "checks": checks,
        "overall": overall,
    }, allow_dirty=args.allow_dirty)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"overall": overall, "report": str(out)}, indent=2))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
