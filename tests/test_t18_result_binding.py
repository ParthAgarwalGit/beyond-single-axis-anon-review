"""Regression checks for the frozen T18 production artifact binding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "t18" / "t18_report.json"
RECEIPT = ROOT / "results" / "t18" / "t18_production_receipt.json"
README = ROOT / "results" / "t18" / "README.md"


def test_t18_report_bytes_match_frozen_receipt():
    report_bytes = REPORT.read_bytes()
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    artifact = receipt["result_artifact"]

    assert artifact["path"] == "results/t18/t18_report.json"
    assert len(report_bytes) == artifact["bytes"]
    assert hashlib.sha256(report_bytes).hexdigest() == artifact["sha256"]


def test_t18_report_provenance_matches_receipt_and_frozen_verdict():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))

    assert report["task"] == "T18"
    assert report["reportable"] is True
    assert report["source_git_dirty"] is False
    assert report["source_git_sha"] == receipt["production_source"]["source_git_sha"]
    assert report["t18_config_sha256"] == receipt["frozen_authorities"]["t18_config_sha256"]
    assert report["geometry_freeze"]["freeze_record_sha256"] == receipt["frozen_authorities"]["g2_freeze_sha256"]
    assert report["verdict"]["verdict"] == receipt["primary_result"]["verdict"] == "INADEQUATE"
    assert report["verdict"]["simultaneous_ci"] == receipt["primary_result"]["simultaneous_ci95"]
    assert report["verdict"]["max_family_difference"] == receipt["primary_result"]["max_family_difference"]


def test_t18_readme_carries_frozen_artifact_identity():
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    artifact = receipt["result_artifact"]
    text = README.read_text(encoding="utf-8")

    assert artifact["sha256"] in text
    assert f"{artifact['bytes']:,}" in text
    assert "Verdict: `INADEQUATE`" in text
