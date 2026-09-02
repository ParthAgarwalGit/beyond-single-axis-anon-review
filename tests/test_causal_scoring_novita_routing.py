"""Regression tests for Novita routing, outage handling, and request pacing."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "t27_causal_judge_frozen.json").read_text("utf-8"))
QWEN_CONDITIONS = CFG["scope"]["branches"]["qwen_capping"]["conditions"]


def _load_tool(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCORER = _load_tool("run_causal_judge_scoring_novita_test", "run_causal_judge_scoring.py")
T26 = _load_tool("run_t26_sample_novita_test", "run_t26_sample.py")


def _production_rows():
    rows = []
    for ci, condition in enumerate(QWEN_CONDITIONS):
        for i in range(100):
            rows.append({
                "uid": f"u-{ci}-{i}",
                "condition": condition,
                "role": "role",
                "role_description": "description",
                "question_id": f"q{i}",
                "question": "question",
                "completion": "completion",
            })
    return rows


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_novita_provider_route_is_separate_from_frozen_scientific_identity():
    assert CFG["judged_instrument"]["judge_model"] == "deepseek-ai/DeepSeek-V3"
    assert SCORER.NOVITA_PROVIDER_MODEL_ID == "deepseek/deepseek_v3"
    assert SCORER.NOVITA_PROVIDER_MODEL_ID != CFG["judged_instrument"]["judge_model"]


def test_request_pacer_enforces_minimum_start_interval(monkeypatch):
    clock = {"now": 100.0}
    sleeps = []

    monkeypatch.setattr(SCORER.time, "monotonic", lambda: clock["now"])

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(SCORER.time, "sleep", fake_sleep)

    pacer = SCORER.RequestPacer(min_interval_seconds=7.0)
    pacer.wait()
    assert sleeps == []

    clock["now"] += 2.0
    pacer.wait()
    assert sleeps == [pytest.approx(5.0)]

    clock["now"] += 7.0
    pacer.wait()
    assert len(sleeps) == 1


def test_rate_limit_gets_long_backoff_before_frozen_retry(monkeypatch):
    calls = {"n": 0}
    sleeps = []

    class RateLimitError(Exception):
        pass

    def fake_judge_once(client, model, prompt, max_tokens=512, pacer=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("429 RATE_LIMIT_EXCEEDED current limit 10 requests per minute")
        return json.dumps({
            "harmfulness": "refused",
            "identity": "assistant",
            "rationale": "ok",
        })

    monkeypatch.setattr(SCORER, "judge_once", fake_judge_once)
    monkeypatch.setattr(SCORER.time, "sleep", lambda seconds: sleeps.append(seconds))

    parsed, raws = SCORER.score_row(object(), "model", "prompt")

    assert parsed == ("refused", "assistant", "ok")
    assert len(raws) == 2
    assert raws[0].startswith("__REQUEST_ERROR__:")
    assert sleeps == [SCORER.NOVITA_RATE_LIMIT_BACKOFF_SECONDS]
    assert calls["n"] == 2


def test_rate_limit_error_counter_is_separate_from_all_request_errors():
    records = [
        {"raw": [
            "__REQUEST_ERROR__: RateLimitError: 429 RATE_LIMIT_EXCEEDED",
            "{\"ok\": true}",
        ]},
        {"raw": ["__REQUEST_ERROR__: TimeoutError: timeout"]},
    ]
    assert SCORER._count_request_error_attempts(records) == 2
    assert SCORER._count_rate_limit_error_attempts(records) == 1


def test_zero_usable_labels_is_nonproduction_and_nondrawable(tmp_path, monkeypatch):
    outputs = tmp_path / "qwen.jsonl"
    _write_jsonl(outputs, _production_rows())

    monkeypatch.setattr(SCORER, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(SCORER, "git_dirty", lambda: False)
    monkeypatch.setattr(SCORER, "make_client", lambda cfg, endpoint: object())
    monkeypatch.setattr(
        SCORER, "score_row",
        lambda client, model, prompt, pacer=None: (
            None, ["__REQUEST_ERROR__: NotFoundError: MODEL_NOT_FOUND"]),
    )

    def failed_canary(client, model, template, rows, label, raw_sink, pacer=None):
        labels = []
        for row in rows:
            raw_sink.append({
                "uid": row["uid"], "phase": f"canary_{label}",
                "raw": ["__REQUEST_ERROR__: NotFoundError: MODEL_NOT_FOUND"]})
            labels.append({"uid": row["uid"], "harmfulness": None, "identity": None})
        payload = json.dumps(labels, sort_keys=True, separators=(",", ":"))
        return {
            "phase": label,
            "n": len(labels),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "labels": labels,
        }

    monkeypatch.setattr(SCORER, "run_canary", failed_canary)

    report = SCORER.score(
        "qwen_capping", outputs, None,
        tmp_path / "scored.jsonl", tmp_path / "raw.jsonl", tmp_path / "report.json")

    assert report["judge_model"] == "deepseek-ai/DeepSeek-V3"
    assert report["provider_model_id"] == "deepseek/deepseek_v3"
    assert report["transport"]["request_pacing_enabled"] is True
    assert report["transport"]["min_request_interval_seconds"] == 7.0
    assert report["n_coverage_failures"] == 400
    assert report["all_production_coverage_failed"] is True
    assert report["canary"]["start_usable"] is False
    assert report["canary"]["end_usable"] is False
    assert report["production"] is False
    assert report["drawable_by_t26"] is False
    assert "no usable labels" in report["non_production_reason"]


def test_t26_rejects_legacy_report_with_100_percent_coverage_failure(tmp_path):
    outputs = tmp_path / "scored.jsonl"
    rows = []
    for ci, condition in enumerate(QWEN_CONDITIONS):
        for i in range(100):
            rows.append({
                "uid": f"u-{ci}-{i}",
                "condition": condition,
                "auto_harmfulness": None,
                "auto_identity": None,
                "judge_coverage_failure": True,
            })
    _write_jsonl(outputs, rows)

    report = {
        "task": "causal_judge_scoring",
        "branch": "qwen_capping",
        "production": True,
        "drawable_by_t26": True,
        "source_git_sha": "a" * 40,
        "source_git_dirty": False,
        "n_rows": 400,
        "n_coverage_failures": 400,
        "per_condition": {
            condition: {"n": 100, "coverage_failures": 100}
            for condition in QWEN_CONDITIONS
        },
        "canary": {"drift_detected": False},
        "artifact_sha256": {
            "scored": hashlib.sha256(outputs.read_bytes()).hexdigest(),
            "raw": "b" * 64,
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(SystemExit, match="no usable automatic labels"):
        T26._validate_scoring_report(report_path, outputs, "qwen_capping", QWEN_CONDITIONS)
