#!/usr/bin/env python3
"""T18 post-primary nuisance diagnostic: response-region token lengths.

This diagnostic answers a narrow question: can trivial generation-length
information reproduce the predictive signal that made the frozen T18
full-dimensional linear readout outperform the one-dimensional Assistant Axis?

Scientific contract
-------------------
* Reuse the frozen T18/G2 population: USER_TRANSLATED_LU, C80-A/B,
  label-independent technical-validity membership, 275 roles, five default
  conditions, 160 questions.
* Reuse the exact T18 5x4 doubly grouped outer folds, 4x3 grouped inner folds,
  split seed, role-balanced weighting, regularized logistic-regression C grid,
  fold-local standardization, fold-wise AUROC aggregation, and 2,000
  role-clustered bootstrap.
* Fit exactly two nuisance baselines:
    1. total generated response-token count (one scalar);
    2. reasoning-token count + final-answer-token count (two scalars).
  Total length is deliberately NOT included in the regional model because it
  is redundant with reasoning + final-answer length.
* This is descriptive sensitivity analysis. It does not change the frozen T18
  INADEQUATE verdict and introduces no post-hoc pass/fail threshold.

No model generation, activation extraction, or judge calls are performed.
Only frozen metadata and committed T18/G2 authority files are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import sklearn
from huggingface_hub import hf_hub_download, list_repo_files
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.provenance import git_dirty, stamp_report  # noqa: E402
from src.t18_readouts import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    DEFAULT_ROLE_PREFIX,
    INFERENCE_CLUSTER,
    INNER_FOLDS,
    LINEAR_C_GRID,
    OUTER_FOLDS,
    SPLIT_SEED,
    T18_CONFIG_SHA256,
    ReadoutRows,
    cluster_bootstrap,
    doubly_grouped_folds,
    run_family,
)

HF_REPO = "[Author-A-HF]/persona-artifacts"
G2_PATH = ROOT / "docs" / "G2_GEOMETRY_FREEZE.json"
T14_PATH = ROOT / "results" / "t14" / "retained_roles.json"
T18_REPORT_PATH = ROOT / "results" / "t18" / "t18_report.json"

ROLE_PREFIXES = ("t12-c80/C80-A", "t12-c80/C80-B")
DEFAULT_PREFIXES = ("t12-c80/DEFAULT-C80-A", "t12-c80/DEFAULT-C80-B")
BLOCKS = ("C80-A", "C80-B")
EXPECTED_ARM = "USER_TRANSLATED_LU"
EXPECTED_BLOCK_INDEX = 16
EXPECTED_POOL = "ALL_RESPONSE_TOKENS"
EXPECTED_MEMBERSHIP = "label_independent_technical_validity"
EXPECTED_ROLES = 275
EXPECTED_DEFAULT_CONDITIONS = 5
EXPECTED_QUESTIONS = 160

FEATURE_SPECS = {
    "total_generated_tokens": ("all_response",),
    "reasoning_plus_answer_tokens": ("reasoning", "final_answer"),
}


def die(msg: str) -> None:
    raise SystemExit(f"T18 length nuisance diagnostic ABORTED: {msg}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        die(f"required authority file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_authorities() -> dict:
    g2 = load_json(G2_PATH)
    t14 = load_json(T14_PATH)
    t18 = load_json(T18_REPORT_PATH)

    if g2.get("freeze_status") != "FROZEN":
        die("G2 geometry record is not FROZEN")
    geom = g2.get("geometry", {})
    exact = {
        "prompt_arm": EXPECTED_ARM,
        "block_index": EXPECTED_BLOCK_INDEX,
        "pool": EXPECTED_POOL,
        "membership": EXPECTED_MEMBERSHIP,
        "retained_roles": EXPECTED_ROLES,
    }
    for key, expected in exact.items():
        if geom.get(key) != expected:
            die(f"G2 {key}={geom.get(key)!r}, expected {expected!r}")
    if tuple(geom.get("blocks", [])) != BLOCKS:
        die(f"G2 blocks={geom.get('blocks')!r}, expected {BLOCKS!r}")
    compat = g2.get("t18_compatibility", {})
    if compat.get("judge_filter_for_primary_rows") != "FORBIDDEN":
        die("G2 does not forbid judge filtering for primary T18 rows")

    if t14.get("status") != "FROZEN":
        die("T14 retained-role artifact is not FROZEN")
    if t14.get("membership") != EXPECTED_MEMBERSHIP:
        die("T14 membership disagrees with G2/T18")
    if t14.get("arm") != EXPECTED_ARM:
        die("T14 prompt arm disagrees with G2/T18")
    if t14.get("block_index") != EXPECTED_BLOCK_INDEX:
        die("T14 block index disagrees with G2/T18")
    if t14.get("pool") != EXPECTED_POOL:
        die("T14 pool disagrees with G2/T18")
    retained = t14.get("retained_roles")
    if not isinstance(retained, list) or len(retained) != EXPECTED_ROLES:
        die("T14 does not contain exactly 275 retained roles")
    if len(set(retained)) != len(retained):
        die("T14 retained-role list contains duplicates")

    if not t18.get("reportable"):
        die("committed T18 report is not reportable")
    if t18.get("data", {}).get("membership") != EXPECTED_MEMBERSHIP:
        die("T18 report membership disagrees with G2")
    if t18.get("data", {}).get("judge_filter_applied") is not False:
        die("T18 report says a judge filter was applied")
    sl = t18.get("data", {}).get("slice", {})
    expected_slice = {
        "arm": EXPECTED_ARM,
        "block_index": EXPECTED_BLOCK_INDEX,
        "pool": EXPECTED_POOL,
    }
    for key, expected in expected_slice.items():
        if sl.get(key) != expected:
            die(f"T18 slice {key}={sl.get(key)!r}, expected {expected!r}")
    if tuple(sl.get("blocks", [])) != BLOCKS:
        die("T18 report blocks disagree with G2")
    if t18.get("data", {}).get("n_roles") != EXPECTED_ROLES:
        die("T18 report does not contain 275 role groups")
    if t18.get("data", {}).get("n_default_conditions") != EXPECTED_DEFAULT_CONDITIONS:
        die("T18 report does not contain five default-condition groups")
    if t18.get("data", {}).get("n_questions") != EXPECTED_QUESTIONS:
        die("T18 report does not contain 160 questions")
    if t18.get("verdict", {}).get("verdict") != "INADEQUATE":
        die("frozen T18 verdict is not INADEQUATE; inspect upstream state before running")

    revision = g2.get("input_provenance", {}).get("bundle_fetched_at_hf_revision")
    if not revision:
        die("G2 has no pinned HF dataset revision")

    return {
        "g2": g2,
        "t14": t14,
        "t18": t18,
        "retained_roles": set(map(str, retained)),
        "hf_revision": revision,
        "hashes": {
            "g2_sha256": sha256_file(G2_PATH),
            "t14_sha256": sha256_file(T14_PATH),
            "t18_report_sha256": sha256_file(T18_REPORT_PATH),
            "t18_config_sha256": T18_CONFIG_SHA256,
        },
    }


def _metadata_paths(prefix: str, revision: str, token: str | None) -> list[str]:
    files = list_repo_files(
        HF_REPO, repo_type="dataset", revision=revision, token=token
    )
    paths = sorted(
        p for p in files
        if p.startswith(prefix + "/meta_part") and p.endswith(".jsonl")
    )
    if not paths:
        die(f"no metadata shards found under {prefix} at {revision}")
    return paths


def _download_text(path: str, revision: str, token: str | None) -> tuple[Path, dict]:
    local = Path(
        hf_hub_download(
            HF_REPO,
            path,
            repo_type="dataset",
            revision=revision,
            token=token,
        )
    )
    return local, {
        "path": path,
        "bytes": local.stat().st_size,
        "sha256": sha256_file(local),
    }


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                die(f"{path}:{line_no} is not valid JSON: {e}")


def load_default_condition_map(
    block: str, revision: str, token: str | None, provenance: list[dict]
) -> dict[str, int]:
    path = f"t11-defaults/{block}/defaults.jsonl"
    local, prov = _download_text(path, revision, token)
    provenance.append(prov)
    mapping: dict[str, int] = {}
    for row in _iter_jsonl(local):
        uid = row.get("rollout_id")
        cond = row.get("default_condition_index")
        if uid in mapping:
            die(f"duplicate default rollout_id {uid!r} in {path}")
        if not isinstance(cond, int) or not (0 <= cond < EXPECTED_DEFAULT_CONDITIONS):
            die(f"invalid default condition {cond!r} for {uid!r} in {path}")
        mapping[str(uid)] = cond
    if not mapping:
        die(f"{path} produced an empty default-condition map")
    return mapping


def extract_token_counts(row: dict, source: str) -> tuple[int, int, int]:
    """Return total, reasoning, answer counts and enforce the partition."""
    trc = row.get("token_region_counts")
    if not isinstance(trc, dict):
        die(f"{source}: uid={row.get('uid')} lacks token_region_counts")
    values = {}
    for key in ("all_response", "reasoning", "final_answer"):
        value = trc.get(key)
        if value is None:
            die(f"{source}: uid={row.get('uid')} has no {key!r} token count")
        if not isinstance(value, (int, np.integer)) or int(value) < 0:
            die(
                f"{source}: uid={row.get('uid')} has invalid {key} count {value!r}"
            )
        values[key] = int(value)
    if values["all_response"] <= 0:
        die(f"{source}: uid={row.get('uid')} has zero all-response tokens")
    if values["all_response"] != values["reasoning"] + values["final_answer"]:
        die(
            f"{source}: uid={row.get('uid')} violates token partition: "
            f"all={values['all_response']}, reasoning={values['reasoning']}, "
            f"answer={values['final_answer']}"
        )
    return (
        values["all_response"],
        values["reasoning"],
        values["final_answer"],
    )


def collect_rows(authorities: dict, token: str | None) -> tuple[list[dict], list[dict]]:
    """Build the exact metadata-level T18 population, fail-closed."""
    revision = authorities["hf_revision"]
    retained = authorities["retained_roles"]
    provenance: list[dict] = []
    records: list[dict] = []

    # Role-prompted rows.
    for block, prefix in zip(BLOCKS, ROLE_PREFIXES):
        for rel in _metadata_paths(prefix, revision, token):
            local, prov = _download_text(rel, revision, token)
            provenance.append(prov)
            for row in _iter_jsonl(local):
                if row.get("validity") != "valid":
                    continue
                if not bool(row.get("has_all_response_pool")):
                    continue
                role = str(row.get("role"))
                if role not in retained:
                    continue
                qid = row.get("question_id")
                if not isinstance(qid, int):
                    die(f"{rel}: role uid={row.get('uid')} has invalid question_id")
                all_n, reason_n, answer_n = extract_token_counts(row, rel)
                records.append(
                    {
                        "uid": str(row.get("uid")),
                        "block": block,
                        "group": role,
                        "question_id": qid,
                        "label": 0,
                        "all_response": all_n,
                        "reasoning": reason_n,
                        "final_answer": answer_n,
                    }
                )

    # Default rows. Their synthetic role keys must match T18 exactly.
    for block, prefix in zip(BLOCKS, DEFAULT_PREFIXES):
        cond_map = load_default_condition_map(block, revision, token, provenance)
        for rel in _metadata_paths(prefix, revision, token):
            local, prov = _download_text(rel, revision, token)
            provenance.append(prov)
            for row in _iter_jsonl(local):
                if row.get("validity") != "valid":
                    continue
                if not bool(row.get("has_all_response_pool")):
                    continue
                uid = str(row.get("uid"))
                cond = cond_map.get(uid)
                if cond is None:
                    die(f"{rel}: default uid {uid!r} is absent from T11 defaults")
                qid = row.get("question_id")
                if not isinstance(qid, int):
                    die(f"{rel}: default uid={uid} has invalid question_id")
                all_n, reason_n, answer_n = extract_token_counts(row, rel)
                records.append(
                    {
                        "uid": uid,
                        "block": block,
                        "group": f"{DEFAULT_ROLE_PREFIX}{cond}",
                        "question_id": qid,
                        "label": 1,
                        "all_response": all_n,
                        "reasoning": reason_n,
                        "final_answer": answer_n,
                    }
                )

    # The exact T18 row unit is group x question for this C80 population.
    seen: dict[tuple[str, int], str] = {}
    for r in records:
        key = (r["group"], r["question_id"])
        if key in seen:
            die(
                f"duplicate T18 semantic row {key!r}: uids "
                f"{seen[key]!r} and {r['uid']!r}"
            )
        seen[key] = r["uid"]

    records.sort(key=lambda r: (r["group"], r["question_id"]))
    return records, provenance


def population_summary(records: list[dict]) -> dict:
    role = [r for r in records if r["label"] == 0]
    default = [r for r in records if r["label"] == 1]
    return {
        "n_rows": len(records),
        "n_role_rows": len(role),
        "n_default_rows": len(default),
        "n_roles": len({r["group"] for r in role}),
        "n_default_conditions": len({r["group"] for r in default}),
        "n_questions": len({r["question_id"] for r in records}),
        "rows_by_block_and_class": {
            block: {
                "role": sum(r["block"] == block and r["label"] == 0 for r in records),
                "default": sum(r["block"] == block and r["label"] == 1 for r in records),
            }
            for block in BLOCKS
        },
    }


def assert_exact_population(summary: dict, frozen_t18: dict) -> None:
    frozen_counts = frozen_t18["data"]["counts"]
    expected = {
        "n_rows": int(frozen_counts["kept_role"] + frozen_counts["kept_default"]),
        "n_role_rows": int(frozen_counts["kept_role"]),
        "n_default_rows": int(frozen_counts["kept_default"]),
        "n_roles": int(frozen_t18["data"]["n_roles"]),
        "n_default_conditions": int(frozen_t18["data"]["n_default_conditions"]),
        "n_questions": int(frozen_t18["data"]["n_questions"]),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            die(
                f"metadata population mismatch for {key}: "
                f"{summary.get(key)!r} != frozen T18 {value!r}"
            )


def make_rows(records: list[dict], spec_name: str) -> ReadoutRows:
    feature_names = FEATURE_SPECS[spec_name]
    X = np.asarray(
        [[float(r[name]) for name in feature_names] for r in records],
        dtype=np.float64,
    )
    roles = [r["group"] for r in records]
    questions = [r["question_id"] for r in records]
    labels = [r["label"] for r in records]
    return ReadoutRows(X, roles, questions, labels)


def assert_fold_contract(rows: ReadoutRows, frozen_t18: dict) -> list[dict]:
    """Verify exact 20 fold IDs and train/test/drop counts against PR #80."""
    folds = doubly_grouped_folds(
        rows, n_role_folds=OUTER_FOLDS[0],
        n_question_folds=OUTER_FOLDS[1], seed=SPLIT_SEED
    )
    frozen = {
        int(s["fold_id"]): s
        for s in frozen_t18["selections"]["axis_1d"]
    }
    if len(folds) != len(frozen) != 20:
        die("fold-count mismatch")
    contract = []
    for fold in folds:
        fid = int(fold["fold_id"])
        ref = frozen.get(fid)
        if ref is None:
            die(f"fold {fid} is absent from frozen T18 selections")
        observed = {
            "n_train": int(len(fold["train_idx"])),
            "n_test": int(len(fold["test_idx"])),
            "n_dropped": int(fold["n_dropped"]),
            "role_fold": int(fold["role_fold"]),
            "question_fold": int(fold["question_fold"]),
        }
        for key, value in observed.items():
            if int(ref[key]) != value:
                die(
                    f"fold {fid} {key}={value} disagrees with frozen "
                    f"T18 value {ref[key]}"
                )
        contract.append({"fold_id": fid, **observed})
    return contract


def fit_nuisance_baseline(
    rows: ReadoutRows, name: str
) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    """Use the frozen T18 full-linear family on only the nuisance features."""
    pred, fold_ids, selections = run_family(
        rows,
        "full_linear",
        folds=OUTER_FOLDS,
        inner_folds=INNER_FOLDS,
        seed=SPLIT_SEED,
    )
    boot = cluster_bootstrap(
        rows,
        pred,
        fold_ids,
        BOOTSTRAP_REPLICATES,
        SPLIT_SEED,
        INFERENCE_CLUSTER,
    )
    performance = {
        **boot,
        "name": name,
        "family_implementation": "T18 full_linear on nuisance features only",
        "n_features": int(rows.X.shape[1]),
        "feature_names": list(FEATURE_SPECS[name]),
        "selection_counts": dict(
            sorted(Counter(s["selected"] for s in selections).items())
        ),
        "selections": selections,
    }
    return performance, pred, fold_ids, selections


def fold_rows(
    rows: ReadoutRows,
    pred: np.ndarray,
    fold_ids: np.ndarray,
    selections: list[dict],
    baseline_name: str,
) -> list[dict]:
    selection = {int(s["fold_id"]): s for s in selections}
    output = []
    for fid in sorted(set(map(int, fold_ids))):
        idx = np.flatnonzero(np.asarray(fold_ids) == fid)
        if not len(idx):
            die(f"{baseline_name}: fold {fid} has no scored rows")
        auc = roc_auc_score(
            rows.y[idx], pred[idx], sample_weight=rows.weights[idx]
        )
        sel = selection[fid]
        output.append(
            {
                "baseline": baseline_name,
                "fold_id": fid,
                "role_fold": int(sel["role_fold"]),
                "question_fold": int(sel["question_fold"]),
                "n_test": int(len(idx)),
                "selected": sel["selected"],
                "inner_auroc": float(sel["inner_auroc"]),
                "outer_role_balanced_auroc": float(auc),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def clean_file_provenance(entries: list[dict]) -> list[dict]:
    """Deduplicate downloaded-file provenance without changing its content."""
    by_path = {}
    for e in entries:
        old = by_path.get(e["path"])
        if old is not None and old != e:
            die(f"same HF path produced conflicting hashes: {e['path']}")
        by_path[e["path"]] = e
    return [by_path[k] for k in sorted(by_path)]


def paper_sentence(report: dict) -> str:
    a = report["performance"]["total_generated_tokens"]
    b = report["performance"]["reasoning_plus_answer_tokens"]
    ref = report["frozen_reference"]
    return (
        "Using the same doubly grouped folds, role-balanced estimator, and "
        "inner-selected regularized logistic regression as T18, generated "
        f"response length alone achieved AUROC {a['point']:.3f} "
        f"(95% role-bootstrap CI [{a['ci_low']:.3f}, {a['ci_high']:.3f}]), "
        "while reasoning- and final-answer-token counts jointly achieved "
        f"{b['point']:.3f} (95% CI [{b['ci_low']:.3f}, {b['ci_high']:.3f}]). "
        f"For reference, the frozen Assistant-Axis and full-linear T18 "
        f"readouts achieved {ref['axis_1d_auroc']:.3f} and "
        f"{ref['full_linear_auroc']:.3f}, respectively. This post-primary "
        "nuisance diagnostic does not alter the frozen T18 verdict."
    )


def write_readme(out_dir: Path, report: dict) -> None:
    a = report["performance"]["total_generated_tokens"]
    b = report["performance"]["reasoning_plus_answer_tokens"]
    ref = report["frozen_reference"]
    text = f"""# T18 nuisance baseline: response-region lengths

## Status

**Post-primary descriptive sensitivity. The frozen T18 verdict remains `{ref['verdict']}`.**

This diagnostic asks whether the large predictive advantage of the frozen
full-dimensional T18 linear readout can be reproduced by trivial generation
length information.

## Exact design

- Same frozen T18/G2 population: {report['population']['n_rows']:,} rows.
- {report['population']['n_roles']} role groups, {report['population']['n_default_conditions']} default conditions, {report['population']['n_questions']} questions.
- Same 5 x 4 doubly grouped outer folds and 4 x 3 grouped inner folds.
- Same split seed `{report['analysis_contract']['split_seed']}`.
- Same role-balanced fitting/evaluation weights.
- Same regularized logistic-regression C grid: `{report['analysis_contract']['linear_c_grid']}`.
- Fold-local feature standardization, as in T18.
- Same mean within-fold role-balanced AUROC estimand.
- Same {report['analysis_contract']['bootstrap_replicates']:,}-replicate role-clustered bootstrap, used descriptively here.
- No new pass/fail threshold and no change to the T18 `INADEQUATE` verdict.

## Results

| Diagnostic | AUROC | Descriptive 95% role-bootstrap CI |
|---|---:|---:|
| Total generated response tokens | {a['point']:.6f} | [{a['ci_low']:.6f}, {a['ci_high']:.6f}] |
| Reasoning + final-answer token counts | {b['point']:.6f} | [{b['ci_low']:.6f}, {b['ci_high']:.6f}] |
| Frozen Assistant Axis (reference) | {ref['axis_1d_auroc']:.6f} | [{ref['axis_1d_ci95'][0]:.6f}, {ref['axis_1d_ci95'][1]:.6f}] |
| Frozen full-dimensional linear (reference) | {ref['full_linear_auroc']:.6f} | [{ref['full_linear_ci95'][0]:.6f}, {ref['full_linear_ci95'][1]:.6f}] |

## Paper-safe wording

{report['paper_safe_sentence']}

## Claim boundary

This analysis tests a simple nuisance explanation. It does not identify the
semantic content used by the full-dimensional classifier, prove that length is
causal, residualize the 4,096 activation dimensions, alter the frozen T18
population, or create a second adequacy verdict.

## Provenance

- Source Git SHA: `{report['source_git_sha']}`
- Source tree dirty: `{report['source_git_dirty']}`
- Frozen HF revision: `{report['hf']['revision']}`
- T18 config SHA-256: `{report['authorities']['t18_config_sha256']}`
- G2 SHA-256: `{report['authorities']['g2_sha256']}`
- Frozen T18 report SHA-256: `{report['authorities']['t18_report_sha256']}`
- T14 retained-role SHA-256: `{report['authorities']['t14_sha256']}`
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "t18" / "nuisance_length",
    )
    p.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token; defaults to HF_TOKEN.",
    )
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Development only. Production should run from a clean committed tree.",
    )
    args = p.parse_args(argv)

    # Capture tree state before this tool writes its own result files.
    dirty_before_outputs = git_dirty()
    if dirty_before_outputs and not args.allow_dirty:
        die(
            "working tree is dirty. Commit the diagnostic implementation first; "
            "then rerun from a clean tree. Use --allow-dirty only for development."
        )

    authorities = load_authorities()
    records, hf_files = collect_rows(authorities, args.hf_token)
    pop = population_summary(records)
    assert_exact_population(pop, authorities["t18"])

    total_rows = make_rows(records, "total_generated_tokens")
    regional_rows = make_rows(records, "reasoning_plus_answer_tokens")

    # Same ordering, labels, group keys, and questions across both diagnostics.
    if not np.array_equal(total_rows.roles, regional_rows.roles):
        die("feature builds changed row/group order")
    if not np.array_equal(total_rows.questions, regional_rows.questions):
        die("feature builds changed question order")
    if not np.array_equal(total_rows.y, regional_rows.y):
        die("feature builds changed labels")
    if total_rows.X.shape[1] != 1 or regional_rows.X.shape[1] != 2:
        die("nuisance feature dimensions violate the frozen diagnostic design")

    fold_contract = assert_fold_contract(total_rows, authorities["t18"])

    performance = {}
    all_fold_rows = []
    for name, rows in (
        ("total_generated_tokens", total_rows),
        ("reasoning_plus_answer_tokens", regional_rows),
    ):
        perf, pred, fold_ids, selections = fit_nuisance_baseline(rows, name)
        performance[name] = perf
        all_fold_rows.extend(fold_rows(rows, pred, fold_ids, selections, name))

    frozen_t18 = authorities["t18"]
    axis = frozen_t18["performance"]["axis_1d"]
    full = frozen_t18["performance"]["full_linear"]

    report = {
        "task": "T18_NUISANCE_LENGTH_DIAGNOSTIC",
        "status": "REPORTABLE_POST_PRIMARY_SENSITIVITY",
        "scientific_role": (
            "descriptive nuisance diagnostic; does not modify the frozen T18 verdict"
        ),
        "changes_frozen_t18_verdict": False,
        "post_hoc_pass_fail_threshold": None,
        "population": pop,
        "analysis_contract": {
            "outer_folds": list(OUTER_FOLDS),
            "inner_folds": list(INNER_FOLDS),
            "n_outer_folds": int(OUTER_FOLDS[0] * OUTER_FOLDS[1]),
            "split_seed": SPLIT_SEED,
            "row_weighting": "T18 role-balanced within class",
            "estimand": "mean within-fold role-balanced AUROC",
            "linear_family": "T18 full_linear implementation",
            "linear_c_grid": list(LINEAR_C_GRID),
            "standardization": "fold-local training standard deviation, as in T18",
            "bootstrap_cluster": INFERENCE_CLUSTER,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "feature_specs": {
                "total_generated_tokens": ["all_response"],
                "reasoning_plus_answer_tokens": ["reasoning", "final_answer"],
            },
            "explicitly_not_fit": [
                "total + reasoning + final_answer in one model",
                "activation residualization",
                "new T18 adequacy threshold",
            ],
        },
        "fold_contract_against_frozen_t18": {
            "status": "EXACT_MATCH",
            "folds": fold_contract,
        },
        "performance": performance,
        "frozen_reference": {
            "verdict": frozen_t18["verdict"]["verdict"],
            "axis_1d_auroc": axis["point"],
            "axis_1d_ci95": [axis["ci_low"], axis["ci_high"]],
            "full_linear_auroc": full["point"],
            "full_linear_ci95": [full["ci_low"], full["ci_high"]],
            "full_minus_axis_auroc": full["point"] - axis["point"],
        },
        "descriptive_comparisons": {
            name: {
                "nuisance_minus_axis_auroc": perf["point"] - axis["point"],
                "full_linear_minus_nuisance_auroc": full["point"] - perf["point"],
            }
            for name, perf in performance.items()
        },
        "authorities": authorities["hashes"],
        "hf": {
            "repo": HF_REPO,
            "revision": authorities["hf_revision"],
            "metadata_and_default_files": clean_file_provenance(hf_files),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "claim_boundary": [
            "Does not change the frozen T18 INADEQUATE verdict.",
            "Does not establish causal effects of response length.",
            "Does not identify which activation dimensions carry the full-linear gain.",
            "Does not establish intrinsic dimensionality or causal necessity.",
            "Automatic role-judge scores do not filter the primary rows.",
        ],
    }
    report["paper_safe_sentence"] = paper_sentence(report)
    report = stamp_report(
        report, allow_dirty=args.allow_dirty, dirty=dirty_before_outputs
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "t18_nuisance_length_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary_rows = []
    for name in FEATURE_SPECS:
        perf = report["performance"][name]
        summary_rows.append(
            {
                "diagnostic": name,
                "n_features": perf["n_features"],
                "auroc": perf["point"],
                "ci_low": perf["ci_low"],
                "ci_high": perf["ci_high"],
                "n_scored_rows": perf["n_scored_rows"],
                "bootstrap_replicates": perf["replicates_used"],
            }
        )
    summary_rows.extend(
        [
            {
                "diagnostic": "frozen_axis_1d_reference",
                "n_features": 1,
                "auroc": axis["point"],
                "ci_low": axis["ci_low"],
                "ci_high": axis["ci_high"],
                "n_scored_rows": axis["n_scored_rows"],
                "bootstrap_replicates": axis["replicates_used"],
            },
            {
                "diagnostic": "frozen_full_linear_reference",
                "n_features": 4096,
                "auroc": full["point"],
                "ci_low": full["ci_low"],
                "ci_high": full["ci_high"],
                "n_scored_rows": full["n_scored_rows"],
                "bootstrap_replicates": full["replicates_used"],
            },
        ]
    )
    write_csv(
        args.out_dir / "t18_nuisance_length_summary.csv",
        summary_rows,
        [
            "diagnostic",
            "n_features",
            "auroc",
            "ci_low",
            "ci_high",
            "n_scored_rows",
            "bootstrap_replicates",
        ],
    )
    write_csv(
        args.out_dir / "t18_nuisance_length_folds.csv",
        all_fold_rows,
        [
            "baseline",
            "fold_id",
            "role_fold",
            "question_fold",
            "n_test",
            "selected",
            "inner_auroc",
            "outer_role_balanced_auroc",
        ],
    )
    write_readme(args.out_dir, report)

    print(json.dumps({
        "report": str(report_path),
        "population": pop,
        "performance": {
            name: {
                "point": report["performance"][name]["point"],
                "ci95": [
                    report["performance"][name]["ci_low"],
                    report["performance"][name]["ci_high"],
                ],
                "selection_counts": report["performance"][name]["selection_counts"],
            }
            for name in FEATURE_SPECS
        },
        "frozen_reference": report["frozen_reference"],
        "paper_safe_sentence": report["paper_safe_sentence"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
