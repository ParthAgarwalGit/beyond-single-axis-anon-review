"""T06 inter-labeller agreement metrics and provisional consensus. CPU-only."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "run_t06_agreement.py"
spec = importlib.util.spec_from_file_location("t06_agreement", SCRIPT)
t06 = importlib.util.module_from_spec(spec)
sys.modules["t06_agreement"] = t06
spec.loader.exec_module(t06)


def _row(item_id, label, validity="ok", role="r", desc="d", q="q", ans=None):
    return {"item_id": item_id, "role": role, "role_description": desc,
            "question": q, "answer": ans if ans is not None else f"ans-{item_id}",
            "human_validity": validity, "label_0_to_3": label}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_perfect_agreement_gives_kappa_one():
    a = ["0", "1", "2", "3", "3", "2"]
    m = t06.agreement_metrics(a, a)
    assert m["raw_agreement_4class"] == 1.0
    assert m["cohen_kappa_4class"] == 1.0
    assert m["cohen_kappa_score3_vs_rest"] == 1.0
    assert m["n"] == 6


def test_score3_vs_rest_ignores_012_distinctions():
    # differ on 0/1/2 everywhere but agree exactly on which items are 3
    a = ["0", "1", "2", "3", "3"]
    b = ["1", "2", "0", "3", "3"]
    m = t06.agreement_metrics(a, b)
    assert m["cohen_kappa_score3_vs_rest"] == 1.0     # 3-vs-rest identical
    assert m["cohen_kappa_4class"] < 1.0              # 4-class differs


def test_confusion_matrix_orientation():
    a = ["0", "1", "2", "3"]
    b = ["0", "1", "2", "3"]
    m = t06.agreement_metrics(a, b)
    cm = m["confusion_4x4_rows_l1_cols_l2_order_0123"]
    assert cm == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def test_chance_level_kappa_near_zero():
    # independent-ish labels -> kappa well below 1
    a = ["0", "1", "2", "3"] * 5
    b = ["3", "2", "1", "0"] * 5
    m = t06.agreement_metrics(a, b)
    assert m["cohen_kappa_4class"] < 0.1


def test_validity_metrics_confusion():
    v = t06.validity_metrics(["ok", "ok", "unjudgeable"], ["ok", "unjudgeable", "unjudgeable"])
    assert v["raw_agreement"] == pytest.approx(2 / 3)
    # rows=L1 [ok,unj], cols=L2 [ok,unj]
    assert v["confusion_rows_l1_cols_l2_order_ok_unjudgeable"] == [[1, 1], [0, 1]]


# --------------------------------------------------------------------------
# content join (round-2 re-blinded ids)
# --------------------------------------------------------------------------

def test_content_join_recovers_item_id():
    r1 = [_row("G000", "1"), _row("G001", "2")]
    idx = t06._index_by_content(r1)
    # round-2 with re-blinded ids but identical content for G001
    r2 = [{"item_id": "RXX", "role": "r", "role_description": "d", "question": "q",
           "answer": "ans-G001", "human_validity": "ok", "label_0_to_3": "3"}]
    got = t06._round2_by_item_id(r2, idx)
    assert got == {"G001": {"label": "3", "validity": "ok"}}


def test_content_join_rejects_nonunique_content():
    r1 = [_row("G000", "1", ans="same"), _row("G001", "2", ans="same")]
    with pytest.raises(ValueError, match="not unique"):
        t06._index_by_content(r1)


def test_content_join_recovers_mojibaked_round2_row():
    # round-1 is clean UTF-8; a round-2 file saved through Excel turned the
    # apostrophe into cp1252 mojibake. The join must still land, deterministically.
    r1 = [_row("G000", "1", ans="the role’s view")]      # ’ = U+2019
    idx = t06._index_by_content(r1)
    mojibaked = "the roleâ€™s view"            # â€™
    r2 = [{"item_id": "RXX", "role": "r", "role_description": "d", "question": "q",
           "answer": mojibaked, "human_validity": "ok", "label_0_to_3": "2"}]
    got = t06._round2_by_item_id(r2, idx)
    assert got == {"G000": {"label": "2", "validity": "ok"}}


def test_content_join_rejects_unmatched_round2_row():
    idx = t06._index_by_content([_row("G000", "1")])
    r2 = [{"item_id": "RXX", "role": "r", "role_description": "d", "question": "q",
           "answer": "no-such-answer", "human_validity": "ok", "label_0_to_3": "3"}]
    with pytest.raises(ValueError, match="no content match"):
        t06._round2_by_item_id(r2, idx)


# --------------------------------------------------------------------------
# consensus
# --------------------------------------------------------------------------

def test_partial_consensus_resolves_agreements_and_flags_residual():
    l1 = [_row("G000", "3"), _row("G001", "1"), _row("G002", "2")]
    l2 = [_row("G000", "3"), _row("G001", "0"), _row("G002", "2")]  # G001 disagrees
    report, rows = t06.compute(l1, l2)
    by = {r["item_id"]: r for r in rows}
    assert by["G000"]["status"] == "resolved" and by["G000"]["gold_label"] == "3"
    assert by["G002"]["resolution"] == "round1_agreement"
    assert by["G001"]["status"] == "ADJUDICATION_REQUIRED"
    assert by["G001"]["gold_label"] == "ADJUDICATION_REQUIRED"
    assert report["consensus"]["adjudication_required"] == 1
    assert report["consensus"]["resolved_after_round2"] == 2
    assert report["provisional"] is True


def _adj_row(item_id, adj_label, adj_validity="ok", role="r", desc="d", q="q", ans=None):
    return {"item_id": item_id, "role": role, "role_description": desc, "question": q,
            "answer": ans if ans is not None else f"ans-{item_id}",
            "round2_labeller_1_validity": "ok", "round2_labeller_2_validity": "ok",
            "round2_labeller_1_label": "2", "round2_labeller_2_label": "1",
            "adjudicated_validity": adj_validity, "adjudicated_label_0_to_3": adj_label,
            "adjudication_note": ""}


def test_adjudication_completes_consensus():
    l1 = [_row("G000", "3"), _row("G001", "2")]
    l2 = [_row("G000", "3"), _row("G001", "1")]          # G001 disagrees
    idx = t06._index_by_content(l1)
    adj_rows = [_adj_row("RX", "2", ans="ans-G001")]     # adjudicate G001 -> 2
    adj, n, pending = t06._load_adjudication(adj_rows, idx)
    assert pending == 0
    report, rows = t06.compute(l1, l2, r2_l1={"G001": {"label": "2", "validity": "ok"}},
                               r2_l2={"G001": {"label": "1", "validity": "ok"}},
                               adjudication=adj)
    by = {r["item_id"]: r for r in rows}
    assert by["G001"]["status"] == "adjudicated"
    assert by["G001"]["gold_label"] == "2"
    assert report["consensus"]["human_adjudicated"] == 1
    assert report["consensus"]["complete"] is True
    assert report["provisional"] is False


def test_blank_adjudication_leaves_pending_and_provisional():
    l1 = [_row("G000", "3"), _row("G001", "2")]
    l2 = [_row("G000", "3"), _row("G001", "1")]
    idx = t06._index_by_content(l1)
    adj_rows = [_adj_row("RX", "", adj_validity="", ans="ans-G001")]   # blank
    adj, n, pending = t06._load_adjudication(adj_rows, idx)
    assert pending == 1 and adj == {}
    report, rows = t06.compute(l1, l2, r2_l2={"G001": {"label": "1", "validity": "ok"}},
                               adjudication=adj)
    assert report["consensus"]["adjudication_pending"] == 1
    assert report["provisional"] is True


def test_adjudication_rejects_automatic_score_leak():
    idx = t06._index_by_content([_row("G000", "1")])
    bad = [{"item_id": "RX", "role": "r", "role_description": "d", "question": "q",
            "answer": "ans-G000", "adjudicated_validity": "ok",
            "adjudicated_label_0_to_3": "2", "lu_score": "3"}]
    with pytest.raises(ValueError, match="leaks automatic-judge"):
        t06._load_adjudication(bad, idx)


def test_adjudication_rejects_bad_values():
    idx = t06._index_by_content([_row("G000", "1")])
    bad = [_adj_row("RX", "9", ans="ans-G000")]          # label 9 invalid
    with pytest.raises(ValueError, match="bad adjudicated_label"):
        t06._load_adjudication(bad, idx)


def test_round2_convergence_resolves_a_disagreement():
    l1 = [_row("G000", "3"), _row("G001", "2")]
    l2 = [_row("G000", "1"), _row("G001", "2")]          # G000 disagrees (3 vs 1)
    # labeller-2 re-reviews G000 and now agrees with L1's 3
    r2_l2 = {"G000": {"label": "3", "validity": "ok"}}
    report, rows = t06.compute(l1, l2, r2_l1=None, r2_l2=r2_l2)
    by = {r["item_id"]: r for r in rows}
    assert by["G000"]["status"] == "resolved"
    assert by["G000"]["resolution"] == "round2_converged"
    assert by["G000"]["gold_label"] == "3"
    assert report["round2_updated_agreement"]["both_labellers_round2_available"] is False
