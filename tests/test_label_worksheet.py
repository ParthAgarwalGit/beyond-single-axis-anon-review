"""Regression tests for the shared T26/T06 blinded-labelling harness."""

from __future__ import annotations

import csv
import http.client
import importlib.util
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from src.causal_validation import HARMFULNESS_LABELS, IDENTITY_LABELS

ROOT = Path(__file__).resolve().parents[1]


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "label_worksheet_test_module", ROOT / "tools" / "label_worksheet.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HARNESS = _load_harness()


def _write_csv(path, rows):
    fieldnames = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _serve(path, preset_name="t26"):
    preset = HARNESS.PRESETS[preset_name]
    fieldnames, rows = HARNESS.load_rows(path, preset)
    handler = HARNESS.make_handler(path, fieldnames, rows, preset)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _request(server, method, target, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    headers = {}
    if body is not None:
        body = json.dumps(body)
        headers["Content-Type"] = "application/json"
    conn.request(method, target, body=body, headers=headers)
    response = conn.getresponse()
    payload = response.read().decode("utf-8")
    conn.close()
    return response.status, payload


def test_t26_vocabularies_match_frozen_causal_validation_contract():
    preset = HARNESS.PRESETS["t26"]
    assert tuple(preset["fields"][0]["options"]) == HARMFULNESS_LABELS
    assert tuple(preset["fields"][1]["options"]) == IDENTITY_LABELS


def test_browser_rows_exposes_only_explicit_blinded_allowlist():
    row = {
        "item_id": "C0001",
        "role": "role",
        "role_description": "description",
        "question": "question",
        "completion": "completion",
        "label_harmfulness": "",
        "label_identity": "",
        # These simulate accidental use of a private/scored file as input.
        "condition": "assistant_axis_toward",
        "auto_harmfulness": "refused",
        "auto_identity": "assistant",
        "ip_weight": "4.0",
    }
    visible = HARNESS.browser_rows([row], HARNESS.PRESETS["t26"])[0]
    assert set(visible) == {
        "item_id",
        "role",
        "role_description",
        "question",
        "completion",
        "label_harmfulness",
        "label_identity",
    }
    assert "condition" not in visible
    assert "auto_harmfulness" not in visible
    assert "auto_identity" not in visible
    assert "ip_weight" not in visible


def test_items_endpoint_does_not_leak_extra_csv_columns(tmp_path):
    worksheet = tmp_path / "worksheet.csv"
    _write_csv(
        worksheet,
        [{
            "item_id": "C0001",
            "role": "role",
            "role_description": "description",
            "question": "question",
            "completion": "completion",
            "condition": "assistant_axis_toward",
            "auto_harmfulness": "refused",
        }],
    )
    server, thread = _serve(worksheet)
    try:
        status, body = _request(server, "GET", "/items")
        assert status == 200
        row = json.loads(body)[0]
        assert row["item_id"] == "C0001"
        assert "condition" not in row
        assert "auto_harmfulness" not in row
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_off_vocabulary_post_is_rejected_and_does_not_modify_csv(tmp_path):
    worksheet = tmp_path / "worksheet.csv"
    _write_csv(
        worksheet,
        [{
            "item_id": "C0001",
            "role": "role",
            "role_description": "description",
            "question": "question",
            "completion": "completion",
        }],
    )
    before = worksheet.read_bytes()
    server, thread = _serve(worksheet)
    try:
        status, body = _request(
            server,
            "POST",
            "/save",
            {
                "item_id": "C0001",
                "label_harmfulness": "not_a_real_label",
                "label_identity": "assistant",
            },
        )
        assert status == 400
        assert json.loads(body) == {"error": "bad label_harmfulness"}

        # A rejected save must write NOTHING. Asserting the bytes are unchanged
        # is both stronger than checking individual cells and correct for a
        # worksheet that does not yet carry the label columns: load_rows() adds
        # them in memory only, and they reach disk on the first accepted save
        # (see the module docstring), so a rejected-first-save file legitimately
        # still has the original header.
        assert worksheet.read_bytes() == before
        assert not worksheet.with_suffix(".csv.tmp").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_valid_post_persists_labels_atomically(tmp_path):
    worksheet = tmp_path / "worksheet.csv"
    _write_csv(
        worksheet,
        [{
            "item_id": "C0001",
            "role": "role",
            "role_description": "description",
            "question": "question",
            "completion": "completion",
        }],
    )
    server, thread = _serve(worksheet)
    try:
        status, body = _request(
            server,
            "POST",
            "/save",
            {
                "item_id": "C0001",
                "label_harmfulness": "refused",
                "label_identity": "assistant",
            },
        )
        assert status == 200
        assert json.loads(body) == {"ok": True}

        with open(worksheet, newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        assert row["label_harmfulness"] == "refused"
        assert row["label_identity"] == "assistant"
        assert not worksheet.with_suffix(".csv.tmp").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
