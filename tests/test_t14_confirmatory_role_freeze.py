"""T14 retained-role freeze tests for the T17 measurement-limited branch."""

import importlib.util
import json
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "freeze_t14_confirmatory_roles.py"
_spec = importlib.util.spec_from_file_location("freeze_t14_confirmatory_roles", _TOOL)
t14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t14)


def _counts():
    roles = [f"role_{i:03d}" for i in range(275)]
    a = {r: 80 for r in roles}
    b = {r: 80 for r in roles}
    a[roles[0]] = 79
    a[roles[1]] = 79
    return {
        "a": a,
        "b": b,
        "provenance": {
            "membership": "label_independent_technical_validity",
            "block_index": 16,
            "pool": "ALL_RESPONSE_TOKENS",
            "arm": "USER_TRANSLATED_LU",
            "t12_manifest_sha256": {"C80-A": "a", "C80-B": "b"},
        },
    }


def test_freeze_writes_275_role_manifest_and_expected_totals(tmp_path):
    counts = tmp_path / "counts.json"
    counts.write_text(json.dumps(_counts()), encoding="utf-8")
    authority = tmp_path / "DEVIATION_2026-08-17_CONFIRMATORY_MEMBERSHIP.md"
    authority.write_text("frozen authority\n", encoding="utf-8")
    out = tmp_path / "freeze.json"
    t14.main(["--counts", str(counts), "--authority", str(authority), "--out", str(out)])
    doc = json.loads(out.read_text("utf-8"))
    assert doc["status"] == "FROZEN"
    assert doc["retained_role_count"] == 275
    assert len(doc["retained_roles"]) == 275
    assert doc["eligible_outputs"] == {"C80-A": 21998, "C80-B": 22000}
    assert doc["threshold_per_block"] == 10


def test_freeze_rejects_score3_membership(tmp_path):
    d = _counts()
    d["provenance"]["membership"] = "score3"
    counts = tmp_path / "counts.json"
    counts.write_text(json.dumps(d), encoding="utf-8")
    authority = tmp_path / "DEVIATION_2026-08-17_CONFIRMATORY_MEMBERSHIP.md"
    authority.write_text("frozen authority\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="counts membership"):
        t14.main(["--counts", str(counts), "--authority", str(authority),
                  "--out", str(tmp_path / "freeze.json")])
