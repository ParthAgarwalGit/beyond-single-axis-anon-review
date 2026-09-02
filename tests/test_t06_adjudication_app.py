"""T06 adjudication-app generator: blinding guard + embedding. CPU-only."""

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_t06_app", ROOT / "tools" / "build_t06_adjudication_app.py")
app = importlib.util.module_from_spec(spec)
sys.modules["build_t06_app"] = app
spec.loader.exec_module(app)

COLS = app.WORKSHEET_COLUMNS


def _worksheet(tmp_path, extra_col=None, n=3):
    cols = list(COLS)
    if extra_col:
        cols.append(extra_col)
    p = tmp_path / "ws.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(n):
            row = {c: "" for c in cols}
            row.update(item_id=f"R{i}", role="pirate", role_description="d",
                       question="q?", answer=f"answer {i}",
                       round2_labeller_1_label="2", round2_labeller_2_label="1",
                       round2_labeller_1_validity="ok", round2_labeller_2_validity="ok")
            if extra_col:
                row[extra_col] = "3"
            w.writerow(row)
    return p


def test_build_embeds_items_and_starts_blank(tmp_path):
    out = tmp_path / "app.html"
    app.build(_worksheet(tmp_path, n=3), out)
    html = out.read_text(encoding="utf-8")
    items = json.loads(re.search(r"const ITEMS = (\[.*?\]);\nconst COLS", html, re.S).group(1))
    assert len(items) == 3
    assert all(not it["adjudicated_label_0_to_3"] and not it["adjudicated_validity"] for it in items)


def test_build_refuses_worksheet_that_leaks_automatic_score(tmp_path):
    out = tmp_path / "app.html"
    with pytest.raises(SystemExit, match="leaks automatic-judge"):
        app.build(_worksheet(tmp_path, extra_col="lu_score"), out)
    assert not out.exists()


def test_generated_html_has_no_automatic_score_tokens(tmp_path):
    out = tmp_path / "app.html"
    app.build(_worksheet(tmp_path, n=2), out)
    html = out.read_text(encoding="utf-8")
    for tok in app._t06.FORBIDDEN_ADJ_COLUMNS:
        assert tok not in html
