#!/usr/bin/env python3
"""T07 role-judge uncertainty intervals.

Adds stratified-bootstrap percentile confidence intervals around the already-frozen
T07 human-vs-automatic judge metrics. It does NOT call a judge, change labels,
change thresholds, change the T07 selected branch, or redefine any point-estimate
metric.

The production run is fail-closed on:
- exact reproduction of the frozen T07 pooled point estimates and populations;
- the final consensus and frozen T07 report hashes recorded in the T07 provenance manifest;
- the reviewed canonical T07 metric implementation hash;
- recovery of the ORIGINAL private sampling key;
- exact arm x archived-automatic-score sampling strata and the historical six-decimal inverse-probability weights;
- a clean Git tree before the bootstrap is executed.

The private key is read locally only. Item-level joins and stratum assignments are
never written to the output.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_T07_PATH = REPO_ROOT / "tools" / "run_t07_judge_validation.py"

FORMAT_VERSION = "t07-judge-uncertainty-v1"
METRIC_KEYS = (
    "score3_precision",
    "score3_recall_ipweighted",
    "macro_f1_ipweighted",
    "four_class_kappa",
)
METRIC_LABELS = {
    "score3_precision": "Score-3 precision",
    "score3_recall_ipweighted": "IPW score-3 recall",
    "macro_f1_ipweighted": "IPW four-class macro-F1",
    "four_class_kappa": "Four-class Cohen's kappa",
}
GATE_THRESHOLDS_BY_METRIC = {
    "score3_precision": 0.85,
    "score3_recall_ipweighted": 0.85,
    "macro_f1_ipweighted": 0.75,
    "four_class_kappa": 0.70,
}
HARD_FROZEN_POOLED = {
    "n_total": 300,
    "n_scored": 278,
    "n_abstained": 10,
    "n_unjudgeable": 12,
    "score3_precision": 0.555556,
    "score3_recall_ipweighted": 0.797853,
    "macro_f1_ipweighted": 0.409206,
    "four_class_kappa": 0.232956,
    "selected_branch": "measurement_limited_result",
}


def die(msg: str) -> "NoReturn":
    raise SystemExit(f"\nT07 UNCERTAINTY ABORTED (fail-closed): {msg}\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_sha() -> str:
    p = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return p.stdout.strip()


def git_dirty() -> bool:
    p = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return bool(p.stdout.strip())


def load_canonical_t07(path: Path = CANONICAL_T07_PATH):
    if not path.exists():
        die(f"canonical T07 scorer missing: {path}")
    spec = importlib.util.spec_from_file_location("_canonical_t07_validation", path)
    if spec is None or spec.loader is None:
        die(f"could not load canonical T07 scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("_metrics", "_gate", "_load_inputs", "validate"):
        if not hasattr(module, name):
            die(f"canonical T07 scorer no longer exposes required helper {name!r}")
    return module


T07 = load_canonical_t07()


def stratum_key_from_values(arm: Any, lu_score: Any) -> str:
    if arm not in {"translated", "extraction"}:
        raise ValueError(f"unexpected/missing arm {arm!r}")
    if lu_score is None:
        score_token = "None"
    else:
        if isinstance(lu_score, bool):
            raise ValueError(f"invalid lu_score {lu_score!r}")
        try:
            score_int = int(lu_score)
        except Exception as e:
            raise ValueError(f"invalid lu_score {lu_score!r}") from e
        if score_int not in (0, 1, 2, 3) or float(lu_score) != score_int:
            raise ValueError(f"invalid lu_score {lu_score!r}")
        score_token = str(score_int)
    return f"{arm}|{score_token}"


def stratum_key(row: dict[str, Any]) -> str:
    if "arm" not in row:
        raise ValueError("row missing required sampling-stratum field 'arm'")
    if "lu_score" not in row:
        raise ValueError(
            "row missing required sampling-stratum field 'lu_score'; explicit None is valid, missing is not"
        )
    return stratum_key_from_values(row["arm"], row["lu_score"])


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _consensus_ids(consensus_path: Path) -> list[str]:
    with consensus_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r.get("item_id", "") for r in rows]
    if len(rows) != 300:
        die(f"final consensus has {len(rows)} rows, expected 300")
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        die("final consensus item_id values are missing or duplicated")
    return ids


def verify_public_frozen_hashes(
    consensus_path: Path,
    frozen_report_path: Path,
    provenance_path: Path,
    canonical_t07_path: Path = CANONICAL_T07_PATH,
) -> dict[str, str]:
    prov = _json(provenance_path)
    expected_consensus = prov["final_gold_consensus"]["sha256"]
    expected_report = prov["committed_t07_result"]["sha256"]
    expected_scorer = prov["scorer"]["sha256_at_canonical_reviewed_commit"]

    actual = {
        "consensus_sha256": sha256_file(consensus_path),
        "frozen_report_sha256": sha256_file(frozen_report_path),
        "canonical_t07_scorer_sha256": sha256_file(canonical_t07_path),
    }
    expected = {
        "consensus_sha256": expected_consensus,
        "frozen_report_sha256": expected_report,
        "canonical_t07_scorer_sha256": expected_scorer,
    }
    bad = {k: (actual[k], expected[k]) for k in actual if actual[k] != expected[k]}
    if bad:
        die(
            "frozen T07 public-input/reviewed-scorer hash mismatch: "
            + "; ".join(f"{k} actual={a} expected={e}" for k, (a, e) in bad.items())
        )
    return actual


def verify_private_key_hash_binding(
    actual_key_sha256: str,
    config: dict[str, Any],
    provenance_path: Path,
    *,
    require_manifest_hash: bool,
) -> None:
    expected_cfg = config.get("sampling", {}).get("private_key_sha256")
    if not expected_cfg:
        die("uncertainty config does not bind the recovered private sampling key SHA-256")
    if actual_key_sha256 != expected_cfg:
        die(
            "private sampling key SHA-256 does not match the frozen uncertainty config "
            f"({actual_key_sha256} != {expected_cfg})"
        )

    prov = _json(provenance_path)
    expected_prov = prov.get("private_sampling_key", {}).get("sha256")
    if require_manifest_hash and not expected_prov:
        die(
            "T07 provenance manifest still has a null private-key SHA-256. "
            "After exact point-estimate reproduction, record the recovered key hash "
            "and commit that provenance-only update before bootstrapping."
        )
    if expected_prov and expected_prov != actual_key_sha256:
        die(
            "private sampling key SHA-256 disagrees with results/t07/provenance_manifest.json "
            f"({actual_key_sha256} != {expected_prov})"
        )


def verify_private_sampling_authority(
    key_path: Path,
    coverage_path: Path,
    consensus_path: Path,
) -> dict[str, Any]:
    if not key_path.exists():
        die(
            f"ORIGINAL private sampling key not found at {key_path}. "
            "Do not run a naive 300-row bootstrap and do not regenerate a different key."
        )
    key_doc = _json(key_path)
    items = key_doc.get("items")
    if not isinstance(items, dict):
        die("private sampling key has no top-level object field 'items'")

    consensus_ids = set(_consensus_ids(consensus_path))
    key_ids = set(items)
    if key_ids != consensus_ids:
        die(
            "private sampling key item IDs do not exactly match final consensus "
            f"(only_in_key={len(key_ids-consensus_ids)}, only_in_consensus={len(consensus_ids-key_ids)})"
        )

    coverage = _json(coverage_path)
    design = coverage.get("sampling_design", {})
    if design.get("method") != "stratified simple random sampling within arm x automatic-score strata":
        die(f"unexpected sampling design in coverage authority: {design.get('method')!r}")

    public_sampled = coverage["cross_tabs"]["sampled_by_stratum"]
    public_population = coverage["cross_tabs"]["population_by_stratum"]
    expected_strata = set(public_sampled)
    if len(expected_strata) != 10:
        die(f"public sampling authority has {len(expected_strata)} strata, expected 10")

    actual_sampled: Counter[str] = Counter()
    weights_by_stratum: dict[str, list[float]] = defaultdict(list)

    for item_id, rec in items.items():
        if "arm" not in rec or "lu_score" not in rec or "ip_weight" not in rec:
            die(
                f"private key item {item_id} lacks arm, lu_score, or ip_weight required by frozen T07"
            )
        try:
            sk = stratum_key_from_values(rec["arm"], rec["lu_score"])
            w = float(rec["ip_weight"])
        except Exception as e:
            die(f"private key item {item_id} has invalid stratum/weight fields: {e}")
        if not math.isfinite(w) or w <= 0:
            die(f"private key item {item_id} has invalid ip_weight={w!r}")
        actual_sampled[sk] += 1
        weights_by_stratum[sk].append(w)

    if dict(sorted(actual_sampled.items())) != dict(sorted(public_sampled.items())):
        die(
            "private-key stratum counts do not match frozen aggregate sampling authority. "
            f"actual={dict(sorted(actual_sampled.items()))}; expected={dict(sorted(public_sampled.items()))}"
        )

    if set(actual_sampled) != expected_strata:
        die("private key strata do not exactly match the ten frozen arm x automatic-score strata")

    aggregate_strata = []
    for sk in sorted(expected_strata):
        n = int(public_sampled[sk])
        N = int(public_population[sk])
        # Historical T05 authority: sampling_key_PRIVATE.json stored
        # round(population_n / sampled_n, 6), not the unrounded Python ratio.
        expected_w = round(N / n, 6)
        observed_ws = weights_by_stratum[sk]
        if len(observed_ws) != n:
            die(f"{sk}: private-key item count drift")
        if not all(w == expected_w for w in observed_ws):
            die(
                f"{sk}: frozen IP weights do not equal the historical six-decimal "
                f"serialization round({N}/{n}, 6) = {expected_w}"
            )
        if "sampled_by_stratum" in key_doc:
            if int(key_doc["sampled_by_stratum"].get(sk, -1)) != n:
                die(f"{sk}: private key sampled_by_stratum disagrees with public authority")
        if "population_by_stratum" in key_doc:
            if int(key_doc["population_by_stratum"].get(sk, -1)) != N:
                die(f"{sk}: private key population_by_stratum disagrees with public authority")

        arm, score = sk.split("|", 1)
        aggregate_strata.append(
            {
                "stratum": sk,
                "arm": arm,
                "archived_automatic_score": None if score == "None" else int(score),
                "labelled_sample_n": n,
                "source_population_n": N,
                "ip_weight": expected_w,
            }
        )

    coverage_selection_hash = design.get("selection_hash")
    key_selection_hash = key_doc.get("selection_hash")
    if coverage_selection_hash and key_selection_hash and coverage_selection_hash != key_selection_hash:
        die(
            "private key selection_hash does not match public sampling authority "
            f"({key_selection_hash} != {coverage_selection_hash})"
        )

    return {
        "private_key_sha256": sha256_file(key_path),
        "sampling_design": design["method"],
        "stratum_fields": ["arm", "archived automatic lu_score"],
        "n_strata": len(aggregate_strata),
        "aggregate_strata": aggregate_strata,
        "sampled_by_stratum": dict(sorted(actual_sampled.items())),
        "selection_hash": key_selection_hash or coverage_selection_hash,
    }


def reproduce_frozen_point_estimate(
    consensus_path: Path,
    key_path: Path,
    frozen_report_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows, _gold = T07._load_inputs(consensus_path, key_path)
    current = T07.validate(rows)
    frozen = _json(frozen_report_path)

    pooled = current["pooled"]
    frozen_pooled = frozen["pooled"]
    count_keys = ("n_total", "n_scored", "n_abstained", "n_unjudgeable")
    for k in count_keys:
        if pooled[k] != frozen_pooled[k]:
            die(f"point-estimate population mismatch for {k}: {pooled[k]} != {frozen_pooled[k]}")
        if pooled[k] != HARD_FROZEN_POOLED[k]:
            die(f"hard-frozen population mismatch for {k}: {pooled[k]} != {HARD_FROZEN_POOLED[k]}")

    m = pooled["scored_metrics"]
    fm = frozen_pooled["scored_metrics"]
    for k in METRIC_KEYS:
        if m[k] != fm[k]:
            die(f"point-estimate metric mismatch for {k}: {m[k]} != frozen {fm[k]}")
        if m[k] != HARD_FROZEN_POOLED[k]:
            die(f"hard-frozen metric mismatch for {k}: {m[k]} != {HARD_FROZEN_POOLED[k]}")

    if current["selected_branch"]["branch"] != frozen["selected_branch"]["branch"]:
        die(
            "selected T07 measurement branch does not reproduce: "
            f"{current['selected_branch']['branch']} != {frozen['selected_branch']['branch']}"
        )
    if current["selected_branch"]["branch"] != HARD_FROZEN_POOLED["selected_branch"]:
        die("selected T07 branch differs from hard-frozen measurement_limited_result")

    if current["pooled"]["gate"] != frozen["pooled"]["gate"]:
        die("pooled frozen gate object does not reproduce exactly")

    # Stronger check: all by-arm point-estimate blocks must reproduce too.
    if current["by_arm"] != frozen["by_arm"]:
        die("by-arm T07 point-estimate blocks do not reproduce exactly")

    return rows, current, frozen


def make_strata(
    rows: list[dict[str, Any]],
    expected_sizes: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        try:
            groups[stratum_key(row)].append(i)
        except ValueError as e:
            raise ValueError(f"row {i}: {e}") from e
    out = {k: np.asarray(v, dtype=np.int64) for k, v in sorted(groups.items())}
    if expected_sizes is not None:
        actual = {k: int(len(v)) for k, v in out.items()}
        exp = {k: int(v) for k, v in expected_sizes.items()}
        if actual != exp:
            raise ValueError(f"stratum sizes do not match frozen authority: actual={actual}, expected={exp}")
    return out


def resample_within_strata(
    rows: list[dict[str, Any]],
    strata: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    sampled: list[dict[str, Any]] = []
    for sk in sorted(strata):
        idx = strata[sk]
        if len(idx) == 0:
            raise ValueError(f"empty stratum {sk}")
        draw = rng.choice(idx, size=len(idx), replace=True)
        sampled.extend(rows[int(i)] for i in draw)
    return sampled


def metric_values(scored_rows: list[dict[str, Any]]) -> dict[str, float | None]:
    if not scored_rows:
        return {k: None for k in METRIC_KEYS}
    m = T07._metrics(
        [r["human_label"] for r in scored_rows],
        [r["lu_score"] for r in scored_rows],
        [r["ip_weight"] for r in scored_rows],
    )
    return {k: m[k] for k in METRIC_KEYS}


def metrics_for_population(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    scored = [
        r
        for r in rows
        if r["human_validity"] == "ok" and r["lu_score"] is not None
    ]
    return metric_values(scored)


def _new_store() -> dict[str, dict[str, list[float | None]]]:
    return {
        "pooled": {k: [] for k in METRIC_KEYS},
        "by_arm:extraction": {k: [] for k in METRIC_KEYS},
        "by_arm:translated": {k: [] for k in METRIC_KEYS},
    }


def bootstrap_distributions(
    rows: list[dict[str, Any]],
    *,
    n_replicates: int,
    seed: int,
    expected_stratum_sizes: dict[str, int],
) -> dict[str, dict[str, list[float | None]]]:
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")
    strata = make_strata(rows, expected_sizes=expected_stratum_sizes)
    rng = np.random.default_rng(seed)
    store = _new_store()

    for _ in range(n_replicates):
        rep = resample_within_strata(rows, strata, rng)
        blocks = {
            "pooled": rep,
            "by_arm:extraction": [r for r in rep if r["arm"] == "extraction"],
            "by_arm:translated": [r for r in rep if r["arm"] == "translated"],
        }
        for block_name, sub in blocks.items():
            vals = metrics_for_population(sub)
            for k in METRIC_KEYS:
                store[block_name][k].append(vals[k])
    return store


def summarize_one(
    values: Iterable[float | None],
    *,
    level: float,
    percentile_method: str,
) -> dict[str, Any]:
    values = list(values)
    valid = np.asarray(
        [float(v) for v in values if v is not None and np.isfinite(float(v))],
        dtype=float,
    )
    invalid_n = len(values) - len(valid)
    if len(valid) == 0:
        return {
            "ci95_percentile": None,
            "valid_replicates": 0,
            "undefined_or_degenerate_replicates": invalid_n,
        }
    alpha = (1.0 - level) / 2.0
    q = np.quantile(
        valid,
        [alpha, 1.0 - alpha],
        method=percentile_method,
    )
    return {
        "ci95_percentile": [round(float(q[0]), 6), round(float(q[1]), 6)],
        "valid_replicates": int(len(valid)),
        "undefined_or_degenerate_replicates": int(invalid_n),
    }


def summarize_distributions(
    distributions: dict[str, dict[str, list[float | None]]],
    *,
    level: float,
    percentile_method: str,
) -> dict[str, dict[str, Any]]:
    return {
        block: {
            metric: summarize_one(
                vals, level=level, percentile_method=percentile_method
            )
            for metric, vals in metrics.items()
        }
        for block, metrics in distributions.items()
    }


def frozen_estimates_for_block(frozen_report: dict[str, Any], block: str) -> dict[str, float]:
    if block == "pooled":
        m = frozen_report["pooled"]["scored_metrics"]
    else:
        _, arm = block.split(":", 1)
        m = frozen_report["by_arm"][arm]["scored_metrics"]
    return {k: m[k] for k in METRIC_KEYS}


def attach_estimates_and_gate(
    summary: dict[str, dict[str, Any]],
    frozen_report: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for block, metrics in summary.items():
        estimates = frozen_estimates_for_block(frozen_report, block)
        out[block] = {}
        for k, s in metrics.items():
            threshold = GATE_THRESHOLDS_BY_METRIC[k]
            out[block][k] = {
                "estimate": estimates[k],
                **s,
                "frozen_threshold": threshold,
                "frozen_point_estimate_pass": bool(estimates[k] >= threshold),
                "gate_note": "PASS/FAIL uses the frozen point estimate only; CI overlap is not a decision rule.",
            }
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_report(result: dict[str, Any]) -> str:
    u = result["uncertainty"]
    pooled = u["intervals"]["pooled"]
    lines = [
        "# T07 role-judge uncertainty intervals",
        "",
        "## Status",
        "",
        f"- Point-estimate reproduction: **{u['point_estimate_reproduction']['status']}**",
        f"- Bootstrap: **{u['bootstrap']['replicates']:,}** stratified replicates",
        f"- Seed: `{u['bootstrap']['seed']}`",
        f"- CI: percentile {int(round(u['bootstrap']['level']*100))}% (`numpy.quantile`, method `{u['bootstrap']['percentile_method']}`)",
        f"- Frozen measurement branch: `{u['selected_measurement_branch']['branch']}`",
        "- Gate rule: unchanged. Confidence-interval overlap with a threshold is not a pass/fail rule.",
        "",
        "## Frozen point-estimate reproduction",
        "",
        f"- total human-reference items: {u['point_estimate_reproduction']['populations']['n_total']}",
        f"- scored: {u['point_estimate_reproduction']['populations']['n_scored']}",
        f"- abstained: {u['point_estimate_reproduction']['populations']['n_abstained']}",
        f"- human-unjudgeable: {u['point_estimate_reproduction']['populations']['n_unjudgeable']}",
        "",
        "## Paper-facing pooled table",
        "",
        "| metric | estimate | 95% CI | frozen threshold | pass/fail |",
        "|---|---:|---:|---:|:---:|",
    ]
    for k in METRIC_KEYS:
        r = pooled[k]
        ci = r["ci95_percentile"]
        ci_text = "undefined" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        pf = "PASS" if r["frozen_point_estimate_pass"] else "FAIL"
        lines.append(
            f"| {METRIC_LABELS[k]} | {r['estimate']:.3f} | {ci_text} | {r['frozen_threshold']:.2f} | **{pf}** |"
        )

    lines += [
        "",
        "All four PASS/FAIL entries above reproduce the existing frozen point-estimate gate. "
        "The intervals report uncertainty around the frozen validation result and do not change the selected branch.",
        "",
        "## Bootstrap design",
        "",
        "- Resampling unit: labelled T06/T07 item.",
        "- Original design authority: stratified simple random sampling within `arm × archived automatic lu_score`.",
        "- Each of the 10 original strata is resampled independently with replacement.",
        "- Every replicate preserves the original labelled sample size of every stratum.",
        "- Each sampled item retains its frozen inverse-probability weight.",
        "- Scored/abstained/unjudgeable status is recomputed from each sampled item using the existing T07 definitions.",
        "- Metric computation calls the canonical `tools/run_t07_judge_validation.py` `_metrics` helper.",
        "- Score-3 precision remains unweighted; recall and four-class macro-F1 remain IP-weighted; Cohen's kappa remains unweighted.",
        "- The canonical precision/recall/F1 helper uses scikit-learn's frozen `zero_division=0` convention. "
        "Truly undefined/non-finite outputs (principally possible for kappa in degenerate replicates) are recorded as undefined, never silently replaced.",
        "",
        "### Frozen strata (aggregate only)",
        "",
        "| stratum | labelled n | source N | IP weight |",
        "|---|---:|---:|---:|",
    ]
    for s in u["sampling_authority"]["aggregate_strata"]:
        lines.append(
            f"| `{s['stratum']}` | {s['labelled_sample_n']} | {s['source_population_n']} | {s['ip_weight']:.6f} |"
        )

    lines += [
        "",
        "## Valid and undefined bootstrap replicates",
        "",
        "| block | metric | valid | undefined/degenerate |",
        "|---|---|---:|---:|",
    ]
    for block, metrics in u["intervals"].items():
        for k, r in metrics.items():
            lines.append(
                f"| {block} | {METRIC_LABELS[k]} | {r['valid_replicates']} | {r['undefined_or_degenerate_replicates']} |"
            )

    lines += [
        "",
        "## Arm-specific intervals",
        "",
    ]
    for arm in ("extraction", "translated"):
        lines += [
            f"### {arm}",
            "",
            "| metric | estimate | 95% CI |",
            "|---|---:|---:|",
        ]
        block = u["intervals"][f"by_arm:{arm}"]
        for k in METRIC_KEYS:
            r = block[k]
            ci = r["ci95_percentile"]
            ci_text = "undefined" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            lines.append(f"| {METRIC_LABELS[k]} | {r['estimate']:.3f} | {ci_text} |")
        lines.append("")

    lines += [
        "## Provenance and privacy",
        "",
        f"- source Git SHA: `{u['provenance']['source_git_sha']}`",
        f"- method/config SHA-256: `{u['provenance']['config_sha256']}`",
        f"- final consensus SHA-256: `{u['provenance']['final_consensus_sha256']}`",
        f"- frozen T07 report SHA-256: `{u['provenance']['frozen_t07_report_sha256']}`",
        f"- original private sampling key SHA-256: `{u['provenance']['private_sampling_key_sha256']}`",
        "- The private key itself, item IDs joined to labels/scores, and per-item stratum assignments are not written to this report or result JSON.",
        "",
        "## Paper implication",
        "",
        "The uncertainty intervals complete reporting around the existing T07 validation result. "
        "They do not reopen the judge gate: the production automatic role judge remains on the frozen "
        "`measurement_limited_result` branch and is not used to define confirmatory C80 membership.",
        "",
        "Human-human agreement confidence intervals are intentionally not added here; that optional secondary extension "
        "is kept separate so it cannot delay or alter the automatic-judge uncertainty task.",
        "",
    ]
    return "\n".join(lines)


def build_result(
    *,
    frozen_report: dict[str, Any],
    config: dict[str, Any],
    sampling_authority: dict[str, Any],
    interval_summary: dict[str, Any],
    public_hashes: dict[str, str],
    key_sha256: str,
    source_sha: str,
) -> dict[str, Any]:
    out = copy.deepcopy(frozen_report)
    if "uncertainty" in out:
        die("frozen report already contains an uncertainty block; refusing to overwrite it")

    pooled = frozen_report["pooled"]
    out["uncertainty"] = {
        "format_version": FORMAT_VERSION,
        "status": "COMPLETE",
        "purpose": (
            "Stratified-bootstrap uncertainty intervals around the already-frozen T07 "
            "human-vs-automatic judge metrics; no new judge calls, labels, rubric changes, "
            "threshold changes, estimand changes, or measurement-branch changes."
        ),
        "point_estimate_reproduction": {
            "status": "PASS",
            "populations": {
                k: pooled[k]
                for k in ("n_total", "n_scored", "n_abstained", "n_unjudgeable")
            },
            "metrics": {
                k: pooled["scored_metrics"][k] for k in METRIC_KEYS
            },
            "frozen_gate_all_pass": pooled["gate"]["all_pass"],
        },
        "bootstrap": {
            "replicates": int(config["bootstrap"]["replicates"]),
            "seed": int(config["bootstrap"]["seed"]),
            "level": float(config["bootstrap"]["level"]),
            "ci_method": config["bootstrap"]["ci_method"],
            "percentile_method": config["bootstrap"]["percentile_method"],
            "stratified_resampling": "within original arm x archived-automatic-score strata, with replacement",
            "preserve_labelled_n_per_stratum": True,
            "retain_frozen_item_ip_weight": True,
        },
        "sampling_authority": sampling_authority,
        "intervals": interval_summary,
        "selected_measurement_branch": copy.deepcopy(frozen_report["selected_branch"]),
        "gate_invariance": {
            "status": "UNCHANGED",
            "rule": "Frozen T07 PASS/FAIL remains based on point estimates only; CI overlap with a threshold creates no new decision rule.",
            "pooled_gate": copy.deepcopy(frozen_report["pooled"]["gate"]),
        },
        "provenance": {
            "source_git_sha": source_sha,
            "source_git_dirty_at_bootstrap_start": False,
            "config_sha256": config["_sha256"],
            "final_consensus_sha256": public_hashes["consensus_sha256"],
            "frozen_t07_report_sha256": public_hashes["frozen_report_sha256"],
            "canonical_t07_scorer_sha256": public_hashes["canonical_t07_scorer_sha256"],
            "private_sampling_key_sha256": key_sha256,
        },
        "privacy": {
            "private_key_committed": False,
            "item_level_join_committed": False,
            "per_item_stratum_assignment_committed": False,
            "aggregate_strata_only": True,
        },
    }
    return out


def run(args: argparse.Namespace) -> None:
    consensus = Path(args.consensus)
    key = Path(args.key)
    coverage = Path(args.coverage)
    frozen_report_path = Path(args.frozen_report)
    provenance = Path(args.provenance)
    config_path = Path(args.config)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    for p in (consensus, coverage, frozen_report_path, provenance, config_path):
        if not p.exists():
            die(f"required input missing: {p}")
    if not key.exists():
        die(
            f"ORIGINAL private sampling key missing: {key}. "
            "This is the only expected blocker. Stop here rather than substituting a naive bootstrap."
        )

    if git_dirty():
        die(
            "Git tree is dirty. Commit the outcome-blind uncertainty config/tool/tests first, "
            "then rerun production from that clean source SHA."
        )
    source_sha = git_sha()

    config = _json(config_path)
    config_sha = sha256_file(config_path)
    config["_sha256"] = config_sha
    if config.get("schema_version") != FORMAT_VERSION:
        die(
            f"uncertainty config schema_version={config.get('schema_version')!r}, "
            f"expected {FORMAT_VERSION!r}"
        )
    if config.get("status") != "FROZEN_BEFORE_INTERVAL_INSPECTION":
        die("uncertainty config is not marked FROZEN_BEFORE_INTERVAL_INSPECTION")
    if config["bootstrap"]["ci_method"] != "percentile":
        die("only the frozen percentile CI method is permitted")
    if config["sampling"]["stratum_fields"] != ["arm", "archived_automatic_lu_score"]:
        die("config sampling strata differ from the frozen T05/T07 design")
    if not config["sampling"].get("private_key_sha256"):
        die("config is missing the frozen private sampling key SHA-256")

    public_hashes = verify_public_frozen_hashes(
        consensus, frozen_report_path, provenance
    )
    authority = verify_private_sampling_authority(key, coverage, consensus)
    verify_private_key_hash_binding(
        authority["private_key_sha256"],
        config,
        provenance,
        require_manifest_hash=True,
    )

    rows, _current, frozen = reproduce_frozen_point_estimate(
        consensus, key, frozen_report_path
    )

    print("T07 frozen point-estimate reproduction: PASS")
    print(
        "  populations:",
        {k: frozen["pooled"][k] for k in ("n_total", "n_scored", "n_abstained", "n_unjudgeable")},
    )
    print(
        "  metrics:",
        {k: frozen["pooled"]["scored_metrics"][k] for k in METRIC_KEYS},
    )
    print("  branch:", frozen["selected_branch"]["branch"])
    print("Private sampling authority: PASS")
    print("  strata:", authority["n_strata"])
    print("  key sha256:", authority["private_key_sha256"])
    print("Bootstrap source Git SHA:", source_sha)

    distributions = bootstrap_distributions(
        rows,
        n_replicates=int(config["bootstrap"]["replicates"]),
        seed=int(config["bootstrap"]["seed"]),
        expected_stratum_sizes=authority["sampled_by_stratum"],
    )
    summary = summarize_distributions(
        distributions,
        level=float(config["bootstrap"]["level"]),
        percentile_method=config["bootstrap"]["percentile_method"],
    )
    intervals = attach_estimates_and_gate(summary, frozen)

    # Explicitly assert the frozen gate was not changed by any uncertainty work.
    if frozen["pooled"]["gate"]["all_pass"] is not False:
        die("frozen pooled gate unexpectedly passes")
    if any(intervals["pooled"][k]["frozen_point_estimate_pass"] for k in METRIC_KEYS):
        die("one or more pooled gate labels unexpectedly changed to PASS")

    result = build_result(
        frozen_report=frozen,
        config=config,
        sampling_authority=authority,
        interval_summary=intervals,
        public_hashes=public_hashes,
        key_sha256=authority["private_key_sha256"],
        source_sha=source_sha,
    )

    # Existing frozen fields must remain byte-for-byte equivalent as JSON values.
    stripped = copy.deepcopy(result)
    stripped.pop("uncertainty")
    if stripped != frozen:
        die("internal error: adding uncertainty modified an existing frozen T07 field")

    write_json(out_json, result)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_report(result), encoding="utf-8")

    print("\nT07 uncertainty bootstrap: COMPLETE")
    for k in METRIC_KEYS:
        r = result["uncertainty"]["intervals"]["pooled"][k]
        print(
            f"  {k}: estimate={r['estimate']:.6f} "
            f"CI={r['ci95_percentile']} "
            f"valid={r['valid_replicates']} "
            f"undefined={r['undefined_or_degenerate_replicates']}"
        )
    print("  gate:", "FAIL (unchanged)")
    print("  branch:", result["uncertainty"]["selected_measurement_branch"]["branch"])
    print("  wrote:", out_json)
    print("  wrote:", out_md)


def self_test() -> None:
    # Synthetic fixture: 10 exact arm x score strata, with variable human labels/validity.
    rows: list[dict[str, Any]] = []
    for arm in ("extraction", "translated"):
        for sc in (0, 1, 2, 3, None):
            for i in range(6):
                rows.append(
                    {
                        "arm": arm,
                        "lu_score": sc,
                        "ip_weight": float(1 + (0 if sc is None else sc)),
                        "human_validity": "unjudgeable" if i == 0 else "ok",
                        "human_label": int((i + (0 if sc is None else sc)) % 4),
                        "machine_validity": "closed",
                    }
                )
    expected = Counter(stratum_key(r) for r in rows)
    s = make_strata(rows, dict(expected))
    assert {k: len(v) for k, v in s.items()} == dict(expected)

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    a = resample_within_strata(rows, s, rng1)
    b = resample_within_strata(rows, s, rng2)
    assert a == b
    assert Counter(stratum_key(r) for r in a) == expected

    # IP weights are retained from sampled items, not recomputed or flattened.
    for sk in expected:
        orig = {r["ip_weight"] for r in rows if stratum_key(r) == sk}
        drawn = {r["ip_weight"] for r in a if stratum_key(r) == sk}
        assert drawn <= orig

    d1 = bootstrap_distributions(
        rows, n_replicates=25, seed=456, expected_stratum_sizes=dict(expected)
    )
    d2 = bootstrap_distributions(
        rows, n_replicates=25, seed=456, expected_stratum_sizes=dict(expected)
    )
    assert d1 == d2

    # Missing lu_score is an error; explicit None is a legitimate original stratum.
    try:
        stratum_key({"arm": "translated"})
        raise AssertionError("missing lu_score did not fail")
    except ValueError:
        pass
    assert stratum_key({"arm": "translated", "lu_score": None}) == "translated|None"

    # Undefined values are counted, never silently converted to zero.
    z = summarize_one([0.1, None, float("nan"), 0.3], level=0.95, percentile_method="linear")
    assert z["valid_replicates"] == 2
    assert z["undefined_or_degenerate_replicates"] == 2

    print("T07 uncertainty self-test: PASS")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("self-test")
    s.set_defaults(func=lambda _a: self_test())

    r = sub.add_parser("run")
    r.add_argument("--consensus", default="results/t06/gold_consensus.csv")
    r.add_argument(
        "--key",
        required=True,
        help="path to ORIGINAL sampling_key_PRIVATE.json; may be outside the repo",
    )
    r.add_argument(
        "--coverage", default="annotations/t14_gold_v3/sampling_coverage.json"
    )
    r.add_argument("--frozen-report", default="results/t07/judge_validation.json")
    r.add_argument("--provenance", default="results/t07/provenance_manifest.json")
    r.add_argument("--config", default="configs/t07_uncertainty_frozen.json")
    r.add_argument(
        "--out-json", default="results/t07/judge_validation_with_uncertainty.json"
    )
    r.add_argument(
        "--out-md", default="results/t07/T07_UNCERTAINTY_REPORT.md"
    )
    r.set_defaults(func=run)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
