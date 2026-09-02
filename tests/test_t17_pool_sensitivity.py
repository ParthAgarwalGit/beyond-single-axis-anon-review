"""T17 provisional pool-sensitivity: matched cohorts, has_*_pool authority, the
95% default floor, retained-role eligibility, and the reasoning-vs-answer
geometry. Pure numpy, CPU-only, deterministic. No T12 tensors, no labels."""

import numpy as np
import pytest

import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _Path

_TOOL = _Path(__file__).resolve().parents[1] / "tools" / "run_t17_pool_sensitivity.py"
_spec = _ilu.spec_from_file_location("run_t17_pool_sensitivity", _TOOL)
t17 = _ilu.module_from_spec(_spec)
_sys.modules["run_t17_pool_sensitivity"] = t17
_spec.loader.exec_module(t17)

from src.pool_sensitivity import (
    F5_POOLS,
    POOLS,
    available_case_sensitivity,
    default_mean_enforced,
    matched_f5,
    matched_triple,
    reduce_records,
)


def _rec(mode, role, cond, validity, has, vecs, uid="u"):
    return {"uid": uid, "mode": mode, "role": role, "condition": cond,
            "validity": validity, "has": has,
            "pools": {p: vecs.get(p) for p in POOLS}}


def _full(mode, role, all_v, ans_v, rea_v, **kw):
    return _rec(mode, role, "c", "valid",
               {"all_response": True, "answer": True, "reasoning": True},
               {"all_response": all_v, "answer": ans_v, "reasoning": rea_v}, **kw)


# --------------------------------------------------------------------------
# matched cohorts (review blocker #2)
# --------------------------------------------------------------------------

def test_matched_f5_uses_only_outputs_with_both_pools():
    # role a: two outputs with both all+answer, one output with all only.
    recs = [
        _full("translated", "a", [2.0], [1.0], [9.0], uid="1"),
        _full("translated", "a", [4.0], [3.0], [9.0], uid="2"),
        _rec("translated", "a", "c", "valid",
             {"all_response": True, "answer": False, "reasoning": False},
             {"all_response": [100.0]}, uid="3"),
    ]
    out = reduce_records(recs, lambda r: True)["arms"]["translated"]
    # matched f5 counts only the 2 both-pool outputs
    assert out["matched_f5"]["counts"]["a"] == 2
    assert np.allclose(out["matched_f5"]["means"]["a"]["all_response"], [3.0])   # mean(2,4)
    assert np.allclose(out["matched_f5"]["means"]["a"]["answer"], [2.0])         # mean(1,3)
    # available all-response includes the all-only output -> DIFFERENT mean
    assert out["available"]["counts"]["all_response"]["a"] == 3
    assert np.allclose(out["available"]["means"]["all_response"]["a"], [(2 + 4 + 100) / 3])


def test_matched_triple_uses_intersection_of_all_three():
    recs = [
        _full("translated", "a", [1.0], [1.0], [1.0], uid="1"),
        _rec("translated", "a", "c", "valid",   # missing reasoning
             {"all_response": True, "answer": True, "reasoning": False},
             {"all_response": [5.0], "answer": [5.0]}, uid="2"),
    ]
    out = reduce_records(recs, lambda r: True)["arms"]["translated"]
    assert out["matched_triple"]["counts"]["a"] == 1     # only the all-three output
    assert out["matched_f5"]["counts"]["a"] == 2         # both all+answer outputs


# --------------------------------------------------------------------------
# has_*_pool authority + staleness (review blocker #3)
# --------------------------------------------------------------------------

def test_stale_tensor_contradicting_flag_fails_closed():
    # metadata says has_answer_pool=false but an answer tensor is present.
    recs = [_rec("translated", "a", "c", "valid",
                 {"all_response": True, "answer": False, "reasoning": False},
                 {"all_response": [1.0], "answer": [2.0]})]
    with pytest.raises(ValueError, match="stale answer tensor"):
        reduce_records(recs, lambda r: True)


def test_flag_true_but_tensor_missing_fails_closed():
    recs = [_rec("translated", "a", "c", "valid",
                 {"all_response": True, "answer": True, "reasoning": False},
                 {"all_response": [1.0]})]   # answer flag true, no answer vec
    with pytest.raises(ValueError, match="has_answer_pool=true but no answer tensor"):
        reduce_records(recs, lambda r: True)


# --------------------------------------------------------------------------
# 95% default validity floor (review blocker #4)
# --------------------------------------------------------------------------

def test_default_floor_enforced():
    cm = {f"c{i}": np.array([float(i)]) for i in range(5)}
    good = {f"c{i}": {"total": 100, "valid": 100} for i in range(5)}
    assert np.allclose(default_mean_enforced(cm, good), [2.0])
    bad = dict(good); bad["c0"] = {"total": 100, "valid": 90}   # 90% < 95%
    with pytest.raises(ValueError, match="below the frozen 0.95 floor"):
        default_mean_enforced(cm, bad)


def test_default_requires_five_conditions():
    with pytest.raises(ValueError, match="5 default conditions"):
        default_mean_enforced({"c0": np.array([1.0])}, {"c0": {"total": 1, "valid": 1}})


def test_default_mode_counts_total_and_valid():
    recs = [
        _rec("default", None, "c0", "valid",
             {"all_response": True, "answer": False, "reasoning": False}, {"all_response": [2.0]}),
        _rec("default", None, "c0", "truncated",
             {"all_response": True, "answer": False, "reasoning": False}, {"all_response": [9.0]}),
    ]
    out = reduce_records(recs, lambda r: True)["default"]
    assert out["condition_counts"]["c0"] == {"total": 2, "valid": 1}
    assert np.allclose(out["all_response_condition_means"]["c0"], [2.0])   # truncated excluded


# --------------------------------------------------------------------------
# output membership vs role eligibility (review blocker #1)
# --------------------------------------------------------------------------

def test_output_membership_and_role_eligibility_are_separate():
    recs = [
        _full("translated", "keep", [1.0], [1.0], [1.0], uid="1"),
        _full("translated", "drop", [1.0], [1.0], [1.0], uid="2"),
    ]
    # role eligibility restricts to the retained set, independent of output rule
    out = reduce_records(recs, lambda r: True, retained_roles={"keep"})["arms"]["translated"]
    assert set(out["matched_f5"]["means"]) == {"keep"}


def test_membership_predicate_filters_outputs():
    for r in (recs := [_full("translated", "a", [0.0], [0.0], [0.0], uid="1"),
                       _full("translated", "a", [6.0], [6.0], [6.0], uid="2")]):
        r["lu_score"] = None
    recs[1]["lu_score"] = 3
    allv = reduce_records(recs, lambda r: r["validity"] == "valid")["arms"]["translated"]
    s3 = reduce_records(recs, lambda r: r["validity"] == "valid" and r.get("lu_score") == 3)["arms"]["translated"]
    assert np.allclose(allv["matched_f5"]["means"]["a"]["all_response"], [3.0])
    assert np.allclose(s3["matched_f5"]["means"]["a"]["all_response"], [6.0])


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def _planted(d=32, n=30, silent_reasoning=True, seed=1):
    rng = np.random.default_rng(seed)
    e0 = np.zeros(d); e0[0] = 1.0
    default = 6.0 * e0
    coords = np.linspace(-4, 4, n)
    roles = [f"r{i:02d}" for i in range(n)]
    answer = {r: coords[i] * e0 + rng.normal(0, 0.05, d) for i, r in enumerate(roles)}
    reasoning = ({r: default + rng.normal(0, 0.1, d) for r in roles} if silent_reasoning
                 else {r: answer[r] + rng.normal(0, 0.05, d) for r in roles})
    allr = {r: 0.5 * (answer[r] + reasoning[r]) for r in roles}
    f5 = {r: {"all_response": allr[r], "answer": answer[r]} for r in roles}
    tri = {r: {"all_response": allr[r], "answer": answer[r], "reasoning": reasoning[r]} for r in roles}
    return default, f5, tri


def test_f5_rows_sorted_columns_exact_single_reference_axis():
    dv, f5, _ = _planted()
    f5["r00"]["answer"] = f5["r00"]["all_response"].copy()   # sits on the diagonal
    res = matched_f5(dv, f5)
    ids = [r["role_id"] for r in res["rows"]]
    assert ids == sorted(ids)
    assert set(res["rows"][0]) == {"role_id", "proj_all_response", "proj_answer_only"}
    row = next(r for r in res["rows"] if r["role_id"] == "r00")
    assert row["proj_all_response"] == pytest.approx(row["proj_answer_only"], abs=1e-6)


def test_silent_reasoning_signature_matched():
    dv, f5, tri = _planted(silent_reasoning=True)
    assert matched_f5(dv, f5)["pearson_all_vs_answer"] > 0.95
    t = matched_triple(dv, tri)
    assert abs(t["pearson_all_vs_reasoning"]) < 0.5
    assert abs(t["cosine_own_axis_to_reference"]["reasoning"]) < 0.9


def test_expressive_reasoning_signature_matched():
    dv, f5, tri = _planted(silent_reasoning=False)
    t = matched_triple(dv, tri)
    assert t["pearson_all_vs_answer"] > 0.95
    assert t["pearson_all_vs_reasoning"] > 0.95


def test_reference_axis_sign_check_failure_raises():
    d = 8
    default = np.ones(d)
    f5 = {"a": {"all_response": np.ones(d), "answer": np.zeros(d)},
          "b": {"all_response": np.ones(d), "answer": np.zeros(d)}}
    with pytest.raises(ValueError):
        matched_f5(default, f5)   # role all-means equal the default -> degenerate axis


# ---------------------------------------------------------------------------
# C80 block layout (added with --layout c80 support)
# ---------------------------------------------------------------------------

def _c80_tree(root):
    """Synthetic tree matching the real C80 layout: block dirs, block-specific
    DEFAULT- dirs, reasoning stored middle-layer-only, and default metadata whose
    `condition` field is the ARM rather than the default-condition index."""
    import json as _json
    import numpy as _np
    from safetensors.numpy import save_file
    rng = _np.random.default_rng(0)
    D, NL = 32, 32
    roles = [f"r{i:02d}" for i in range(8)]
    for block in ("C80-A", "C80-B"):
        d = root / block
        d.mkdir(parents=True, exist_ok=True)
        meta, allr, ans, rea = [], {}, {}, {}
        for r in roles:
            for q in range(10):
                uid = f"{block[-1]}{r}{q}"
                base = rng.standard_normal(D)
                allr[uid] = _np.repeat(base[None, :], NL, 0).astype(_np.float16)
                ans[uid] = _np.repeat(base[None, :], NL, 0).astype(_np.float16)
                rea[uid] = base.astype(_np.float16)          # rank-1: middle layer only
                meta.append({"uid": uid, "role": r, "condition": "USER_TRANSLATED_LU",
                             "validity": "valid", "has_all_response_pool": True,
                             "has_answer_pool": True, "has_reasoning_pool": True})
        (d / "meta_part0000.jsonl").write_text(
            "".join(_json.dumps(m) + "\n" for m in meta), encoding="utf-8")
        save_file(allr, str(d / "all_response_part0000.safetensors"))
        save_file(ans, str(d / "answer_part0000.safetensors"))
        save_file(rea, str(d / "reasoning_middle_part0000.safetensors"))
        (d / "manifest.json").write_text("{}", encoding="utf-8")

        dd = root / f"DEFAULT-{block}"
        dd.mkdir(parents=True, exist_ok=True)
        dmeta, da, t11 = [], {}, []
        for cond in range(5):
            for q in range(8):
                uid = f"D{block[-1]}{cond}{q}"
                base = rng.standard_normal(D) + 3.0
                da[uid] = _np.repeat(base[None, :], NL, 0).astype(_np.float16)
                dmeta.append({"uid": uid, "role": None,
                              "condition": "USER_TRANSLATED_LU",  # the ARM, not the condition
                              "validity": "valid", "has_all_response_pool": True,
                              "has_answer_pool": False, "has_reasoning_pool": False})
                t11.append({"rollout_id": uid, "default_condition_index": cond})
        (dd / "meta_part0000.jsonl").write_text(
            "".join(_json.dumps(m) + "\n" for m in dmeta), encoding="utf-8")
        save_file(da, str(dd / "all_response_part0000.safetensors"))
        (dd / "manifest.json").write_text("{}", encoding="utf-8")
        t = root / "t11-defaults" / block
        t.mkdir(parents=True, exist_ok=True)
        (t / "defaults.jsonl").write_text(
            "".join(_json.dumps(r) + "\n" for r in t11), encoding="utf-8")
    return root


def test_c80_layout_reduces_and_recovers_five_default_conditions(tmp_path):
    """The C80 default metadata carries the arm in `condition`, so without the T11
    join every default row collapses into one bucket and the five-condition rule
    cannot be satisfied. This is the regression that blocked running T17 on C80."""
    import json as _json
    root = _c80_tree(tmp_path / "t12")
    out = tmp_path / "red"
    t17.main(["reduce", "--t12-root", str(root), "--layout", "c80",
              "--block", "C80-A", "--t11-defaults", str(root / "t11-defaults"),
              "--out", str(out)])
    counts = _json.loads((out / "DEFAULT-C80-A" / "condition_counts.json").read_text("utf-8"))
    assert sorted(counts) == ["0", "1", "2", "3", "4"]
    assert all(c["valid"] == 8 for c in counts.values())
    man = _json.loads((out / "reduced_manifest.json").read_text("utf-8"))
    assert man["layout"] == "c80" and man["arms"] == ["C80-A"]
    assert man["default_mode"] == "DEFAULT-C80-A"


def test_c80_analyze_adopts_layout_from_reduced_manifest(tmp_path):
    root = _c80_tree(tmp_path / "t12")
    red = tmp_path / "red"
    t17.main(["reduce", "--t12-root", str(root), "--layout", "c80", "--block", "C80-B",
              "--t11-defaults", str(root / "t11-defaults"), "--out", str(red)])
    t17.main(["analyze", "--reduced", str(red), "--out", str(tmp_path / "an")])
    import json as _json
    rep = _json.loads((tmp_path / "an" / "pool_sensitivity_report.json").read_text("utf-8"))
    assert rep["layout"] == "c80"
    assert "C80-B" in rep["arms"]


def test_c80_without_t11_defaults_refuses(tmp_path):
    root = _c80_tree(tmp_path / "t12")
    with pytest.raises(SystemExit):
        t17.main(["reduce", "--t12-root", str(root), "--layout", "c80",
                  "--block", "C80-A", "--out", str(tmp_path / "red")])


def test_e80_layout_is_the_unchanged_default(tmp_path):
    """Passing no --layout must keep the historical E80 contract exactly."""
    t17._apply_layout("e80")
    assert t17.ARMS == ("translated", "wrapper")
    assert t17.DEFAULT_MODE == "default"
    assert t17.PRIMARY_ARM == "translated"
