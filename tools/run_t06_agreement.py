#!/usr/bin/env python3
"""T06 — inter-labeller agreement, confusion matrices, and provisional consensus.

Implements the measurement side of T07_SCORING_SPEC 1-2 on the T05 gold set:
raw agreement, Cohen's kappa (4-class), score-3-vs-rest kappa, the full 4x4
label confusion matrix, and the validity confusion matrix — computed on **all**
300 items, never on an agreement-only subset (the spec's explicit warning).

Two rounds:
  * Round 1 is the two merged labeller files (immutable; PR #17 / PR #27).
  * Round 2 re-reviews only the round-1 disagreement items, re-blinded with
    fresh opaque ids; this tool joins them back to round-1 by content
    (role/description/question/answer are byte-identical), so no private R->G
    map is required to recompute. Pass whichever round-2 files exist.

The tool NEVER adjudicates. It builds a *partial* consensus:
  * round-1 agreements                       -> gold = the agreed value;
  * disagreements that agree after round 2   -> gold = the now-agreed value;
  * everything still in conflict             -> ADJUDICATION_REQUIRED.
Adjudication of the residual is a human step (T07_SCORING_SPEC 1) and must be
done without seeing the automatic judge score. gold_consensus stays PROVISIONAL
until (a) both labellers' round-2 files exist and (b) the residual is
adjudicated.

Outputs land under results/t06/ and never touch the merged round-1 files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import provenance

FORMAT_VERSION = "t06-agreement-v1"
LABELS = ["0", "1", "2", "3"]
VALIDITY = ["ok", "unjudgeable"]
CONTENT_KEY = ["role", "role_description", "question", "answer"]

R1_L1 = REPO_ROOT / "annotations" / "t14_gold_v3" / "labeller_1.csv"
R1_L2 = REPO_ROOT / "annotations" / "t14_gold_v3" / "labeller_2.csv"
R2_DIR = REPO_ROOT / "annotations" / "t06a_round2"


def _read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_bytes_lf(path: Path, text: str) -> str:
    if not text.endswith("\n"):
        text += "\n"
    data = text.replace("\r\n", "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _write_json_lf(path, doc):
    return _write_bytes_lf(path, json.dumps(doc, indent=2, ensure_ascii=False))


def _write_csv_lf(path, fieldnames, rows):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return _write_bytes_lf(path, buf.getvalue())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# metrics (pure)
# --------------------------------------------------------------------------

def agreement_metrics(a, b):
    """Four-class agreement between two equal-length label sequences (str 0-3)."""
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        raise ValueError("need two aligned non-empty label sequences")
    raw = float(np.mean([x == y for x, y in zip(a, b)]))
    k4 = float(cohen_kappa_score(a, b, labels=LABELS))
    a3 = [int(x == "3") for x in a]
    b3 = [int(y == "3") for y in b]
    k3 = float(cohen_kappa_score(a3, b3, labels=[0, 1]))
    cm = confusion_matrix(a, b, labels=LABELS).tolist()
    return {
        "n": len(a),
        "raw_agreement_4class": round(raw, 6),
        "cohen_kappa_4class": round(k4, 6),
        "cohen_kappa_score3_vs_rest": round(k3, 6),
        "confusion_4x4_rows_l1_cols_l2_order_0123": cm,
    }


def validity_metrics(a, b):
    a, b = list(a), list(b)
    raw = float(np.mean([x == y for x, y in zip(a, b)]))
    cm = confusion_matrix(a, b, labels=VALIDITY).tolist()
    return {
        "raw_agreement": round(raw, 6),
        "confusion_rows_l1_cols_l2_order_ok_unjudgeable": cm,
    }


# --------------------------------------------------------------------------
# round-2 join (content -> round-1 item_id)
# --------------------------------------------------------------------------

def _index_by_content(round1_rows):
    """Map the content tuple -> item_id for round-1 rows; require uniqueness so
    the round-2 content join is well defined."""
    idx = {}
    for r in round1_rows:
        k = tuple(r[c] for c in CONTENT_KEY)
        if k in idx:
            raise ValueError("round-1 content is not unique; content join is unsafe")
        idx[k] = r["item_id"]
    return idx


def _demojibake(s):
    """Reverse the Windows/Excel UTF-8->cp1252 round-trip (e.g. a labeller who
    saved the CSV through Excel turned '’' into 'a€™'). Round-1 is
    clean UTF-8, so this only ever helps a corrupted round-2 file rejoin; if it
    doesn't decode, the original string is kept."""
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def _round2_by_item_id(round2_rows, content_index):
    """Return {round1_item_id: {'label':..., 'validity':...}} for a round-2 file,
    joining by content when its ids are re-blinded (R-prefixed). Falls back to
    the demojibake'd content so a mis-encoded round-2 file still joins exactly,
    without relying on any non-unique key."""
    out = {}
    for r in round2_rows:
        k = tuple(r[c] for c in CONTENT_KEY)
        if k not in content_index:
            k = tuple(_demojibake(r[c]) for c in CONTENT_KEY)
        if k not in content_index:
            raise ValueError(f"round-2 row {r['item_id']} has no content match in round 1")
        gid = content_index[k]
        out[gid] = {"label": r["label_0_to_3"], "validity": r["human_validity"]}
    return out


# adjudication worksheet must never carry the automatic judge score
# (T07_SCORING_SPEC §1: adjudicate without seeing it).
FORBIDDEN_ADJ_COLUMNS = {"lu_score", "plan_label", "arm", "condition", "ip_weight",
                         "uid", "automatic_score", "judge_score", "machine_validity"}


def _load_adjudication(rows, content_index):
    """Return ({round1_item_id: {'label','validity'}} for FILLED rows, n_total,
    n_pending). Joins the worksheet's re-blinded ids to round-1 by content
    (demojibake-robust) and refuses a worksheet that leaks the automatic score."""
    if rows:
        cols = set(rows[0])
        leak = cols & FORBIDDEN_ADJ_COLUMNS
        if leak:
            raise ValueError(f"adjudication worksheet leaks automatic-judge columns: {sorted(leak)}")
        for req in ("adjudicated_label_0_to_3", "adjudicated_validity"):
            if req not in cols:
                raise ValueError(f"adjudication worksheet missing column {req!r}")
    out, pending = {}, 0
    for r in rows:
        k = tuple(r[c] for c in CONTENT_KEY)
        if k not in content_index:
            k = tuple(_demojibake(r[c]) for c in CONTENT_KEY)
        if k not in content_index:
            raise ValueError(f"adjudication row {r.get('item_id')} has no content match in round 1")
        gid = content_index[k]
        lab = (r.get("adjudicated_label_0_to_3") or "").strip()
        val = (r.get("adjudicated_validity") or "").strip()
        if lab and val:
            if lab not in {"0", "1", "2", "3"}:
                raise ValueError(f"adjudication {gid}: bad adjudicated_label_0_to_3 {lab!r}")
            if val not in {"ok", "unjudgeable"}:
                raise ValueError(f"adjudication {gid}: bad adjudicated_validity {val!r}")
            out[gid] = {"label": lab, "validity": val}
        elif lab or val:
            raise ValueError(f"adjudication {gid}: one field filled, the other blank")
        else:
            pending += 1
    return out, len(rows), pending


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def compute(r1_l1_rows, r1_l2_rows, r2_l1=None, r2_l2=None, adjudication=None):
    """r2_l1 / r2_l2 / adjudication are {item_id: {'label','validity'}} or None."""
    l1 = {r["item_id"]: r for r in r1_l1_rows}
    l2 = {r["item_id"]: r for r in r1_l2_rows}
    ids = sorted(set(l1) & set(l2))
    if len(ids) != len(l1) or len(l1) != len(l2):
        raise ValueError("round-1 files do not cover the same item_id set")

    a1 = [l1[i]["label_0_to_3"] for i in ids]
    b1 = [l2[i]["label_0_to_3"] for i in ids]
    v1 = [l1[i]["human_validity"] for i in ids]
    w1 = [l2[i]["human_validity"] for i in ids]

    round1 = {"label": agreement_metrics(a1, b1), "validity": validity_metrics(v1, w1)}

    # round-2-updated sequences: substitute a labeller's re-review where present
    def final(seq_ids, base, r2):
        out = []
        for i, base_val in zip(seq_ids, base):
            out.append(r2[i]["label"] if (r2 and i in r2) else base_val)
        return out

    def final_v(seq_ids, base, r2):
        out = []
        for i, base_val in zip(seq_ids, base):
            out.append(r2[i]["validity"] if (r2 and i in r2) else base_val)
        return out

    a2 = final(ids, a1, r2_l1)
    b2 = final(ids, b1, r2_l2)
    v2 = final_v(ids, v1, r2_l1)
    w2 = final_v(ids, w1, r2_l2)

    have_both_r2 = bool(r2_l1) and bool(r2_l2)
    updated = {"label": agreement_metrics(a2, b2), "validity": validity_metrics(v2, w2),
               "both_labellers_round2_available": have_both_r2}

    # consensus: agreement -> agreed value; residual -> adjudicated value if the
    # human worksheet supplies it, else ADJUDICATION_REQUIRED.
    consensus_rows, n_resolved, n_adjudicated, n_pending, n_r1_agree = [], 0, 0, 0, 0
    for k, i in enumerate(ids):
        r1_agree = (a1[k] == b1[k]) and (v1[k] == w1[k])
        resolved = (a2[k] == b2[k]) and (v2[k] == w2[k])
        adj = adjudication.get(i) if adjudication else None
        n_r1_agree += int(r1_agree)
        if resolved:
            n_resolved += 1
            gold_label, gold_validity = a2[k], v2[k]
            status = "resolved"
            resolution = "round1_agreement" if r1_agree else "round2_converged"
        elif adj:
            n_adjudicated += 1
            gold_label, gold_validity = adj["label"], adj["validity"]
            status, resolution = "adjudicated", "human_adjudicated"
        else:
            n_pending += 1
            gold_label = gold_validity = "ADJUDICATION_REQUIRED"
            status, resolution = "ADJUDICATION_REQUIRED", "residual_disagreement"
        consensus_rows.append({
            "item_id": i,
            "l1_round1_label": a1[k], "l2_round1_label": b1[k],
            "l1_final_label": a2[k], "l2_final_label": b2[k],
            "l1_round1_validity": v1[k], "l2_round1_validity": w1[k],
            "l1_final_validity": v2[k], "l2_final_validity": w2[k],
            "gold_label": gold_label, "gold_validity": gold_validity,
            "status": status, "resolution": resolution,
        })

    have_adj = adjudication is not None
    final = have_both_r2 and have_adj and n_pending == 0
    report = {
        "format_version": FORMAT_VERSION,
        "provisional": not final,
        "n_items": len(ids),
        "round1_agreement": round1,
        "round2_updated_agreement": updated,
        "round2_inputs_present": {
            "labeller_1_round2": bool(r2_l1),
            "labeller_2_round2": bool(r2_l2),
        },
        "consensus": {
            "round1_agreements_both_fields": n_r1_agree,
            "resolved_after_round2": n_resolved,
            "human_adjudicated": n_adjudicated,
            "adjudication_pending": n_pending,
            "adjudication_required": n_pending,
            "complete": final,
        },
        "adjudication_note": (
            "Residual disagreements take the gold label from the human adjudication "
            "worksheet (adjudicated without seeing the automatic judge score, "
            "T07_SCORING_SPEC §1). Consensus is FINAL only when both round-2 files "
            "exist and every residual is adjudicated; otherwise gold labels for "
            "unadjudicated residuals stay ADJUDICATION_REQUIRED."),
    }
    return report, consensus_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r1-l1", default=str(R1_L1))
    ap.add_argument("--r1-l2", default=str(R1_L2))
    ap.add_argument("--r2-l1", default=None, help="labeller-1 round-2 csv (optional)")
    ap.add_argument("--r2-l2", default=None, help="labeller-2 round-2 csv (optional)")
    ap.add_argument("--adjudication", default=None,
                    help="filled adjudication worksheet csv (optional); when every "
                         "residual is adjudicated, emits the FINAL gold_consensus.csv")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "t06"))
    a = ap.parse_args()

    r1_l1_rows = _read_csv(a.r1_l1)
    r1_l2_rows = _read_csv(a.r1_l2)
    content_index = _index_by_content(r1_l1_rows)

    inputs = {"r1_l1": a.r1_l1, "r1_l2": a.r1_l2, "r2_l1": a.r2_l1, "r2_l2": a.r2_l2,
              "adjudication": a.adjudication}
    input_hashes = {k: sha256_file(v) for k, v in inputs.items() if v and Path(v).exists()}

    r2_l1 = _round2_by_item_id(_read_csv(a.r2_l1), content_index) if a.r2_l1 else None
    r2_l2 = _round2_by_item_id(_read_csv(a.r2_l2), content_index) if a.r2_l2 else None
    adjudication = n_adj_total = n_adj_pending = None
    if a.adjudication:
        adjudication, n_adj_total, n_adj_pending = _load_adjudication(
            _read_csv(a.adjudication), content_index)

    report, consensus_rows = compute(r1_l1_rows, r1_l2_rows, r2_l1, r2_l2, adjudication)
    report["input_files"] = inputs
    report["input_file_sha256"] = input_hashes
    if a.adjudication:
        report["adjudication_worksheet"] = {"rows": n_adj_total, "pending": n_adj_pending}
    report = provenance.stamp_report(report, allow_dirty=True)

    out = Path(a.out)
    _write_json_lf(out / "agreement_report.json", report)
    fname = "gold_consensus.csv" if report["consensus"]["complete"] \
        else "gold_consensus.PROVISIONAL.csv"
    _write_csv_lf(out / fname, list(consensus_rows[0]), consensus_rows)

    r1 = report["round1_agreement"]["label"]
    ru = report["round2_updated_agreement"]["label"]
    c = report["consensus"]
    print("T06 agreement ->", out)
    print(f"  round1:  raw={r1['raw_agreement_4class']}  k4={r1['cohen_kappa_4class']}  "
          f"k3(3-vs-rest)={r1['cohen_kappa_score3_vs_rest']}")
    print(f"  updated: raw={ru['raw_agreement_4class']}  k4={ru['cohen_kappa_4class']}  "
          f"k3(3-vs-rest)={ru['cohen_kappa_score3_vs_rest']}  "
          f"(both_r2={report['round2_updated_agreement']['both_labellers_round2_available']})")
    print(f"  consensus: resolved={c['resolved_after_round2']}  "
          f"adjudicated={c['human_adjudicated']}  pending={c['adjudication_pending']}  "
          f"/ {report['n_items']}")
    print(f"  wrote {fname}  (complete={c['complete']})")
    if a.adjudication and n_adj_pending:
        print(f"  NOTE: {n_adj_pending} of {n_adj_total} worksheet rows are not yet "
              f"adjudicated; gold_consensus stays PROVISIONAL until they are filled.")


if __name__ == "__main__":
    main()
