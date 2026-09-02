"""Regression guard: the causal scorer owns every retry and every paced request."""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_scorer():
    spec = importlib.util.spec_from_file_location(
        "run_causal_judge_scoring_sdk_retry_test",
        ROOT / "tools" / "run_causal_judge_scoring.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_make_client_disables_sdk_automatic_retries(monkeypatch):
    scorer = _load_scorer()
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("NOVITA_API_KEY", "test-key")

    scorer.make_client({}, "https://example.invalid/v3/openai")

    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://example.invalid/v3/openai"
    assert captured["max_retries"] == 0
