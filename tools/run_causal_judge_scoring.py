#!/usr/bin/env python
"""Shared causal-judge scoring pipeline for the frozen T26/T27 validation path.

The same DeepSeek-V3/Novita judge is used for both causal branches. Production
artifacts are drawable by T26 only when the complete frozen population is present,
the repository source commit is resolvable, the hosted judge returns at least one
usable production label, and the start/end canary is both usable and stable.
Raw judge responses, including canaries, are retained for drift/reparse auditing.

Novita transport requests are paced below the provider's observed 10 requests/minute
limit. This pacing is operational only: it does not alter the frozen judge identity,
prompt, schema, temperature, validation thresholds, or causal estimand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.causal_validation import HARMFULNESS_LABELS, IDENTITY_LABELS  # noqa: E402
from src.provenance import git_dirty  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "t27_causal_judge_frozen.json"
PROMPT = ROOT / "prompts" / "causal_judge_frozen.md"
REQUIRED_KEYS = {"harmfulness", "identity", "rationale"}
N_CANARY = 50
DEFAULT_BASE_URL = "https://api.novita.ai/v3/openai"

NOVITA_PROVIDER_MODEL_ID = "deepseek/deepseek_v3"

# Operational transport controls. The Aug-23 corrected-routing production attempt
# exposed Novita's explicit "current limit 10 requests per minute" quota. A 7 s
# start-to-start interval stays below that ceiling (~8.6 requests/minute) and leaves
# margin for a rolling-window limiter. A 429 gets an additional backoff before the
# one already-frozen retry. These values do not change judge semantics.
NOVITA_OBSERVED_REQUESTS_PER_MINUTE = 10
NOVITA_MIN_REQUEST_INTERVAL_SECONDS = 7.0
NOVITA_RATE_LIMIT_BACKOFF_SECONDS = 15.0
GENERIC_REQUEST_ERROR_BACKOFF_SECONDS = 2.0


class RequestPacer:
    """Enforce a minimum start-to-start interval between hosted API attempts."""

    def __init__(self, min_interval_seconds=NOVITA_MIN_REQUEST_INTERVAL_SECONDS):
        self.min_interval_seconds = float(min_interval_seconds)
        self._last_started_at = None

    def wait(self):
        now = time.monotonic()
        if self._last_started_at is not None:
            remaining = self.min_interval_seconds - (now - self._last_started_at)
            if remaining > 0:
                time.sleep(remaining)
        self._last_started_at = time.monotonic()


def die(msg):
    raise SystemExit(f"causal scoring ABORTED (fail-closed): {msg}")


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_sha():
    """Return the exact 40-hex source commit, or None if it cannot be resolved."""
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip().lower()
    except Exception:
        return None
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None


def extract_template(prompt_md):
    text = Path(prompt_md).read_text(encoding="utf-8")
    blocks = re.findall(r"```\n(.*?)```", text, re.S)
    for block in blocks:
        if "{{COMPLETION}}" in block and "{{ROLE}}" in block:
            return block
    die(f"no template with the frozen placeholders found in {prompt_md}")


def build_prompt(template, role, role_description, question, completion):
    return (template
            .replace("{{ROLE}}", role or "")
            .replace("{{ROLE_DESCRIPTION}}", role_description or "")
            .replace("{{QUESTION}}", question or "")
            .replace("{{COMPLETION}}", completion or ""))


def parse_judge(raw):
    """Accept exactly one frozen-schema object; never coerce malformed output."""
    if raw is None:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict) or set(obj) != REQUIRED_KEYS:
        return None
    harmfulness = obj.get("harmfulness")
    identity = obj.get("identity")
    rationale = obj.get("rationale")
    if not all(isinstance(v, str) for v in (harmfulness, identity, rationale)):
        return None
    harmfulness, identity = harmfulness.strip(), identity.strip()
    if harmfulness not in HARMFULNESS_LABELS or identity not in IDENTITY_LABELS:
        return None
    return harmfulness, identity, rationale


def make_client(cfg, base_url=None):
    try:
        from openai import OpenAI
    except ImportError:
        die("pip install openai (Novita exposes an OpenAI-compatible endpoint)")
    key = os.environ.get("NOVITA_API_KEY")
    if not key:
        die("NOVITA_API_KEY is not set")
    # The scorer owns the frozen two-attempt policy and the global request pacer.
    # Disable SDK-level automatic retries so every real HTTP attempt is visible to
    # score_row(), paced, counted, and retained in the raw audit artifact.
    return OpenAI(
        api_key=key,
        base_url=base_url or DEFAULT_BASE_URL,
        max_retries=0,
    )


def judge_once(client, model, prompt, max_tokens=512, pacer=None):
    if pacer is not None:
        pacer.wait()
    result = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])
    return result.choices[0].message.content


def _is_rate_limit_error(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "ratelimit" in type(exc).__name__.lower()
        or "rate_limit" in text
        or "rate limit" in text
        or "429" in text
    )


def score_row(client, model, prompt, pacer=None):
    raws = []
    for _ in range(2):
        try:
            raw = judge_once(client, model, prompt, pacer=pacer)
        except Exception as exc:
            raw = f"__REQUEST_ERROR__: {type(exc).__name__}: {exc}"
            if _is_rate_limit_error(exc):
                time.sleep(NOVITA_RATE_LIMIT_BACKOFF_SECONDS)
            else:
                time.sleep(GENERIC_REQUEST_ERROR_BACKOFF_SECONDS)
        raws.append(raw)
        parsed = parse_judge(raw)
        if parsed is not None:
            return parsed, raws
    return None, raws


def canary_rows(rows, n=N_CANARY):
    return sorted(rows, key=lambda r: str(r.get("uid") or r.get("rollout_id")))[:n]


def run_canary(client, model, template, rows, label, raw_sink, pacer=None):
    out = []
    for row in rows:
        prompt = build_prompt(
            template, row.get("role") or row.get("role_id"),
            row.get("role_description"), row.get("question"),
            row.get("completion") or row.get("output_text"))
        parsed, raws = score_row(client, model, prompt, pacer=pacer)
        uid = row.get("uid") or row.get("rollout_id")
        raw_sink.append({"uid": uid, "phase": f"canary_{label}", "raw": raws})
        out.append({
            "uid": uid,
            "harmfulness": parsed[0] if parsed else None,
            "identity": parsed[1] if parsed else None,
        })
    payload = json.dumps(out, sort_keys=True, separators=(",", ":"))
    return {
        "phase": label,
        "n": len(out),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "labels": out,
    }


def _canary_has_usable_labels(canary):
    return any(
        row.get("harmfulness") is not None and row.get("identity") is not None
        for row in canary.get("labels", [])
    )


def _count_request_error_attempts(raw_records):
    return sum(
        1
        for record in raw_records
        for raw in record.get("raw", [])
        if isinstance(raw, str) and raw.startswith("__REQUEST_ERROR__:")
    )


def _count_rate_limit_error_attempts(raw_records):
    return sum(
        1
        for record in raw_records
        for raw in record.get("raw", [])
        if isinstance(raw, str)
        and raw.startswith("__REQUEST_ERROR__:")
        and (
            "ratelimit" in raw.lower()
            or "rate_limit" in raw.lower()
            or "rate limit" in raw.lower()
            or "429" in raw.lower()
        )
    )


PRODUCTION_CONTRACT = {
    "deepseek_steering": {"total": 5000, "per_condition": 1000},
    "qwen_capping": {"total": 400, "per_condition": 100},
}


def enforce_production_contract(branch, rows, conditions):
    problems = []
    uids = [r.get("uid") or r.get("rollout_id") for r in rows]
    if any(uid is None for uid in uids):
        problems.append("row(s) with no uid/rollout_id")
    duplicates = len(uids) - len(set(uids))
    if duplicates:
        problems.append(
            f"{duplicates} duplicate uid(s); exactly one final status per uid is required")

    per_condition = {}
    for row in rows:
        condition = row.get("condition")
        per_condition[condition] = per_condition.get(condition, 0) + 1
    unknown = [c for c in per_condition if c not in conditions]
    if unknown:
        problems.append(f"condition(s) outside the frozen inventory: {sorted(unknown)}")

    spec = PRODUCTION_CONTRACT.get(branch)
    if spec is None:
        problems.append(f"no frozen production contract for branch {branch!r}")
        return problems
    if len(rows) != spec["total"]:
        problems.append(f"{len(rows)} rows, expected exactly {spec['total']}")
    for condition in conditions:
        got = per_condition.get(condition, 0)
        if got != spec["per_condition"]:
            problems.append(
                f"condition {condition}: {got} rows, expected {spec['per_condition']}")
    return problems


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        die(f"no rows in {path}")
    return rows


def _nonproduction_reason(limit, contract_problems, drift, source_git_dirty=False,
                          instrument_unavailable=False):
    reasons = []
    if limit is not None:
        reasons.append("--limit truncates the corpus")
    if contract_problems:
        reasons.append("corpus failed the frozen production contract")
    if drift:
        reasons.append("start/end judge canary drift detected")
    if source_git_dirty:
        reasons.append("source working tree was dirty at scoring time")
    if instrument_unavailable:
        reasons.append("judge instrument returned no usable labels in the production/canary window")
    return "; ".join(reasons) or None


def score(branch, outputs, roles_path, out_path, raw_path, report_path,
          dry_run=False, limit=None, base_url=None):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    instrument = cfg["judged_instrument"]
    branches = cfg["scope"]["branches"]
    if branch not in branches:
        die(f"unknown branch {branch!r}; expected one of {sorted(branches)}")
    conditions = list(branches[branch]["conditions"])
    scientific_model = instrument["judge_model"]
    provider_model = NOVITA_PROVIDER_MODEL_ID

    template = extract_template(PROMPT)
    rubric = {}
    if roles_path:
        rubric = {
            row["role_id"]: row.get("description", "")
            for row in json.loads(Path(roles_path).read_text(encoding="utf-8"))["roles"]
        }

    rows = load_rows(outputs)
    if limit is not None:
        rows = rows[:limit]

    seen_conditions = {row.get("condition") for row in rows}
    missing = [c for c in conditions if c not in seen_conditions]
    if missing:
        die(f"branch {branch!r} is missing conditions {missing}; T26 cannot draw from "
            "a partially generated branch")

    production_candidate = limit is None
    contract_problems = enforce_production_contract(branch, rows, conditions)
    if production_candidate and contract_problems:
        die("corpus does not satisfy the frozen production contract:\n  - "
            + "\n  - ".join(contract_problems))

    source_git_sha = _git_sha()
    if production_candidate and source_git_sha is None:
        die("production scoring requires a resolvable 40-hex source git SHA")
    source_git_dirty = git_dirty()

    if dry_run:
        print(f"[dry-run] branch={branch} rows={len(rows)} conditions={sorted(seen_conditions)}")
        print(f"[dry-run] judge_model={scientific_model} provider={instrument['provider']}")
        print(f"[dry-run] provider_model_id={provider_model}")
        print(f"[dry-run] source_git_sha={source_git_sha}")
        print(
            f"[dry-run] request_pacing={NOVITA_MIN_REQUEST_INTERVAL_SECONDS:.1f}s "
            f"rate_limit_backoff={NOVITA_RATE_LIMIT_BACKOFF_SECONDS:.1f}s")
        print("[dry-run] sdk_automatic_retries_disabled=True")
        print("[dry-run] inputs valid; no tokens spent")
        return None

    endpoint = base_url or DEFAULT_BASE_URL
    client = make_client(cfg, endpoint)
    pacer = RequestPacer()
    started = datetime.now(timezone.utc).isoformat()
    can_rows = canary_rows(rows)
    scored, raws = [], []
    canary_start = run_canary(
        client, provider_model, template, can_rows, "start", raws, pacer=pacer)
    n_coverage = 0

    for i, row in enumerate(rows, 1):
        uid = row.get("uid") or row.get("rollout_id")
        role = row.get("role") or row.get("role_id")
        role_description = row.get("role_description") or rubric.get(role, "")
        question = row.get("question")
        completion = row.get("completion") or row.get("output_text")
        prompt = build_prompt(template, role, role_description, question, completion)
        parsed, raw_list = score_row(client, provider_model, prompt, pacer=pacer)
        raws.append({"uid": uid, "raw": raw_list})
        coverage_failure = parsed is None
        n_coverage += int(coverage_failure)
        scored.append({
            "uid": uid,
            "role": role,
            "condition": row.get("condition"),
            "role_description": role_description,
            "question": question,
            "completion": completion,
            "question_id": row.get("question_id"),
            "arm": row.get("arm"),
            "auto_harmfulness": parsed[0] if parsed else None,
            "auto_identity": parsed[1] if parsed else None,
            "judge_rationale": parsed[2] if parsed else None,
            "judge_coverage_failure": coverage_failure,
        })
        if i % 250 == 0:
            print(f"  scored {i}/{len(rows)}  coverage failures so far: {n_coverage}")

    canary_end = run_canary(
        client, provider_model, template, can_rows, "end", raws, pacer=pacer)
    finished = datetime.now(timezone.utc).isoformat()

    for path, payload in ((out_path, scored), (raw_path, raws)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in payload:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    drift = canary_start["sha256"] != canary_end["sha256"]
    n_changed = sum(
        1 for start, end in zip(canary_start["labels"], canary_end["labels"])
        if start != end)
    canary_start_usable = _canary_has_usable_labels(canary_start)
    canary_end_usable = _canary_has_usable_labels(canary_end)
    all_production_coverage_failed = bool(scored) and n_coverage == len(scored)
    instrument_unavailable = (
        all_production_coverage_failed or not canary_start_usable or not canary_end_usable)
    n_request_error_attempts = _count_request_error_attempts(raws)
    n_rate_limit_error_attempts = _count_rate_limit_error_attempts(raws)

    production = (
        production_candidate and not contract_problems and not drift
        and not source_git_dirty and not instrument_unavailable)
    drawable = production

    per_condition = {}
    for row in scored:
        rec = per_condition.setdefault(
            row["condition"], {"n": 0, "coverage_failures": 0})
        rec["n"] += 1
        rec["coverage_failures"] += int(row["judge_coverage_failure"])

    report = {
        "task": "causal_judge_scoring",
        "branch": branch,
        "authority": "docs/DEVIATION_2026-08-18_T27_CAUSAL_JUDGE_GATE.md",
        "judge_model": scientific_model,
        "provider": instrument["provider"],
        "provider_model_id": provider_model,
        "provider_model_id_note": (
            "Operational Novita routing identifier for the frozen DeepSeek-V3 judge; "
            "this is not a change to the scientific judge identity or T27 thresholds."),
        "endpoint": endpoint,
        "transport": {
            "request_pacing_enabled": True,
            "sdk_automatic_retries_disabled": True,
            "observed_provider_limit_requests_per_minute": NOVITA_OBSERVED_REQUESTS_PER_MINUTE,
            "min_request_interval_seconds": NOVITA_MIN_REQUEST_INTERVAL_SECONDS,
            "rate_limit_backoff_seconds": NOVITA_RATE_LIMIT_BACKOFF_SECONDS,
            "generic_request_error_backoff_seconds": GENERIC_REQUEST_ERROR_BACKOFF_SECONDS,
            "scope": "every hosted judge attempt, including canaries and retries",
            "method_boundary": (
                "Operational transport pacing only; no change to judge identity, prompt, schema, "
                "temperature, validation gate, sampling rule, or estimand."),
        },
        "production": production,
        "drawable_by_t26": drawable,
        "non_production_reason": _nonproduction_reason(
            limit, contract_problems, drift, source_git_dirty, instrument_unavailable),
        "contract_problems": contract_problems or None,
        "source_git_sha": source_git_sha,
        "source_git_dirty": source_git_dirty,
        "revision_pinnable": False,
        "revision_note": instrument["reproducibility_limitation"]["why"],
        "run_window": {"started_utc": started, "finished_utc": finished},
        "single_invocation_window": (
            "T26 validation items MUST be drawn from this scored artifact; scoring "
            "them later would characterise a possibly different instrument"),
        "prompt_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "prompt_file_sha256": _sha256_file(PROMPT),
        "config_sha256": _sha256_file(CONFIG),
        "inputs_sha256": _sha256_file(outputs),
        "n_rows": len(scored),
        "n_coverage_failures": n_coverage,
        "coverage_failure_rate": n_coverage / len(scored) if scored else None,
        "all_production_coverage_failed": all_production_coverage_failed,
        "n_request_error_attempts": n_request_error_attempts,
        "n_rate_limit_error_attempts": n_rate_limit_error_attempts,
        "per_condition": per_condition,
        "canary": {
            "n": len(can_rows),
            "start_sha256": canary_start["sha256"],
            "end_sha256": canary_end["sha256"],
            "start_usable": canary_start_usable,
            "end_usable": canary_end_usable,
            "drift_detected": drift,
            "n_items_changed": n_changed,
            "how_to_use": (
                "Re-score this same canary slice later and compare against end_sha256. "
                "A mismatch means the endpoint moved."),
        },
        "raw_responses_retained": str(raw_path),
        "artifact_sha256": {
            "scored": _sha256_file(out_path),
            "raw": _sha256_file(raw_path),
        },
        "artifact_note": (
            "The scored artifact is private and pinned by hash; item-level automatic "
            "scores joined to condition are not committed."),
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"[scoring] {len(scored)} rows, {n_coverage} coverage failures "
          f"({report['coverage_failure_rate']:.4f})")
    print(
        f"[scoring] request errors={n_request_error_attempts}, "
        f"rate-limit errors={n_rate_limit_error_attempts}")
    if instrument_unavailable:
        print("[scoring] !! JUDGE UNAVAILABLE: no usable production/canary labels; "
              "artifact marked non-production/non-drawable")
    elif drift:
        print(f"[scoring] !! CANARY DRIFT: {n_changed}/{len(can_rows)} items changed; "
              "artifact marked non-production/non-drawable")
    elif not production:
        print("[scoring] !! NON-PRODUCTION run: artifact is NOT drawable by T26")
    else:
        print(f"[scoring] canary stable across the run ({len(can_rows)} items)")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--branch", required=True,
                              choices=("deepseek_steering", "qwen_capping"))
    score_parser.add_argument("--outputs", required=True)
    score_parser.add_argument("--roles", default=str(ROOT / "data" / "lu_et_al" / "roles.json"))
    score_parser.add_argument("--out", required=True)
    score_parser.add_argument("--raw", required=True)
    score_parser.add_argument("--report", required=True)
    score_parser.add_argument("--base-url", default=None)
    score_parser.add_argument("--limit", type=int, default=None)
    score_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    score(args.branch, args.outputs, args.roles, args.out, args.raw, args.report,
          dry_run=args.dry_run, limit=args.limit, base_url=args.base_url)


if __name__ == "__main__":
    main()
