"""T26/T27 causal-judge validation and provenance-gate tests."""
import csv
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

import src.provenance as provenance
from src.causal_validation import (
    HARMFULNESS_LABELS,
    IDENTITY_LABELS,
    NON_ASSISTANT,
    _chi2_sf,
    _chi2_stat,
    binary_agreement,
    cohen_kappa,
    differential_error,
    evaluate_gate,
    multiclass_agreement,
    validate_outcome,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "t27_causal_judge_frozen.json").read_text("utf-8"))
THR = CFG["gate"]
DEEPSEEK_CONDITIONS = CFG["scope"]["branches"]["deepseek_steering"]["conditions"]
QWEN_CONDITIONS = CFG["scope"]["branches"]["qwen_capping"]["conditions"]


def _load_tool(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCORER = _load_tool("run_causal_judge_scoring_test", "run_causal_judge_scoring.py")
T26 = _load_tool("run_t26_sample_test", "run_t26_sample.py")
T27 = _load_tool("run_t27_causal_judge_validation_test", "run_t27_causal_judge_validation.py")

VALID_JUDGE = '{"harmfulness":"refused","identity":"assistant","rationale":"declined"}'


def _rows(n_per_cond=30, judge_flips=0, flip_only_in=None, seed=0):
    import random
    rng = random.Random(seed)
    rows = []
    for condition in DEEPSEEK_CONDITIONS:
        for i in range(n_per_cond):
            human = "harmful" if i % 3 == 0 else "refused"
            rows.append({
                "condition": condition,
                "ip_weight": 1.0,
                "judge_coverage_failure": False,
                "auto_harmfulness": human,
                "human_harmfulness": human,
                "auto_identity": "human_role" if i % 2 else "assistant",
                "human_identity": "human_role" if i % 2 else "assistant",
            })
    for condition in DEEPSEEK_CONDITIONS:
        k = judge_flips if flip_only_in is None or condition == flip_only_in else 0
        idx = [j for j, row in enumerate(rows) if row["condition"] == condition]
        for j in rng.sample(idx, min(k, len(idx))):
            rows[j]["auto_harmfulness"] = (
                "refused" if rows[j]["auto_harmfulness"] == "harmful" else "harmful")
    return rows


def _rows_all_labels(n_per_cond=32):
    rows = []
    for condition in DEEPSEEK_CONDITIONS:
        for i in range(n_per_cond):
            h = HARMFULNESS_LABELS[i % len(HARMFULNESS_LABELS)]
            identity = IDENTITY_LABELS[i % len(IDENTITY_LABELS)]
            rows.append({
                "condition": condition,
                "ip_weight": 1.0,
                "judge_coverage_failure": False,
                "auto_harmfulness": h,
                "human_harmfulness": h,
                "auto_identity": identity,
                "human_identity": identity,
            })
    return rows


def _production_rows(conditions, per_condition):
    rows = []
    for ci, condition in enumerate(conditions):
        for i in range(per_condition):
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


# ---------------------------------------------------------------------------
# Core statistical gate remains pinned to the reviewed definitions.
# ---------------------------------------------------------------------------


def test_chi2_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    for df in range(1, 7):
        for x in (0.5, 2.5, 9.5, 20.0):
            assert _chi2_sf(x, df) == pytest.approx(scipy_stats.chi2.sf(x, df), abs=1e-10)


def test_chi2_stat_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    import numpy as np
    table = {"a": {"x": 25, "y": 5}, "b": {"x": 20, "y": 10}, "c": {"x": 28, "y": 2}}
    obs = np.array([[table[g][lab] for lab in ("x", "y")] for g in sorted(table)])
    assert _chi2_stat(table) == pytest.approx(
        scipy_stats.chi2_contingency(obs, correction=False)[0])
    assert _chi2_stat({"a": {"correct": 30}, "b": {"correct": 30}}) == 0.0


def test_cohen_kappa_perfect_and_inverted():
    assert cohen_kappa([("a", "a"), ("b", "b")], ("a", "b")) == pytest.approx(1.0)
    assert cohen_kappa([("a", "b"), ("b", "a")], ("a", "b")) < 0


def test_perfect_full_label_judge_validates():
    result = validate_outcome(_rows_all_labels(), "harmfulness", THR)
    assert result["gate"]["part_a_pass"] is True
    assert result["gate"]["part_b_differential_bias_detected"] is False
    assert result["gate"]["branch"] == "VALIDATED"
    assert result["gate"]["primary_evidence"] == "full_corpus_automatic"


def test_absent_harmfulness_classes_count_as_zero_macro_f1():
    result = validate_outcome(_rows(), "harmfulness", THR)
    assert result["part_a_absolute_agreement"]["multiclass"]["macro_f1_ip"] == pytest.approx(0.5)
    assert result["gate"]["part_a"]["four_class_macro_f1_ip"] is False


def test_part_a_failure_keeps_human_subset_primary():
    result = validate_outcome(_rows(judge_flips=8), "harmfulness", THR)
    assert result["gate"]["part_a_pass"] is False
    assert result["gate"]["primary_evidence"] == "human_subset"
    assert result["gate"]["primary_evidence_is_full_corpus"] is False


def test_concentrated_condition_error_is_differential():
    result = validate_outcome(
        _rows(judge_flips=20, flip_only_in="assistant_axis_toward"),
        "harmfulness", THR)
    assert result["gate"]["part_b_differential_bias_detected"] is True
    assert result["gate"]["branch"] in {"DIFFERENTIAL_BIAS", "MEASUREMENT_LIMITED_BIASED"}


def test_part_b_permutation_respects_automatic_harmfulness_strata():
    import random
    from src.causal_validation import _strata_permute
    conditions = ["A", "A", "B", "B"]
    strata = {"harmful": [0, 2], "refused": [1, 3]}
    out = _strata_permute(strata, conditions, random.Random(0))
    assert sorted([out[0], out[2]]) == ["A", "B"]
    assert sorted([out[1], out[3]]) == ["A", "B"]


def test_untestable_part_b_is_not_silently_a_pass():
    part_b = {
        "accuracy_heterogeneity": {"p_ge_observed": None},
        "bias_spread_permutation": {"p_ge_observed": 1.0},
        "coverage_heterogeneity": {"p_ge_observed": 1.0},
        "bias_spread": 0.0,
    }
    part_a = {
        "binary": {"precision_gated": 1.0, "recall_ip": 1.0,
                   "kappa_gated_unweighted": 1.0},
        "multiclass": {"macro_f1_ip": 1.0, "kappa_gated_unweighted": 1.0},
    }
    result = evaluate_gate(part_a, part_b, THR, "harmfulness")
    assert result["part_b"]["accuracy_heterogeneity"] is False
    assert result["part_b_differential_bias_detected"] is True


def test_identity_gate_uses_binary_nonassistant_estimand():
    result = validate_outcome(_rows(), "identity", THR)
    assert set(result["gate"]["part_a"]) == {
        "non_assistant_precision", "non_assistant_recall_ip", "non_assistant_kappa"}


def test_ip_weighting_changes_recall_when_heavy_positive_rows_are_missed():
    rows = _rows(n_per_cond=10)
    for row in rows:
        row["ip_weight"] = 9.0 if row["condition"] == "shared_zero" else 1.0
    missed = [row for row in rows if row["condition"] == "shared_zero"
              and row["human_harmfulness"] == "harmful"]
    for row in missed[:2]:
        row["auto_harmfulness"] = "refused"
    metric = binary_agreement(
        rows, lambda x: x == "harmful", "auto_harmfulness", "human_harmfulness",
        precision_weighted=False)
    assert metric["recall_ip"] < metric["recall_unweighted"]


def test_multiclass_uses_full_frozen_label_set():
    metric = multiclass_agreement(
        _rows(n_per_cond=5), HARMFULNESS_LABELS,
        "auto_harmfulness", "human_harmfulness")
    assert metric["n_labels_in_macro"] == len(HARMFULNESS_LABELS)


def test_frozen_labels_match_t23_schema():
    schema = json.loads((ROOT / "design" / "causal_judge_schema.json").read_text("utf-8"))
    assert tuple(schema["fields"]["harmfulness"]["labels"]) == HARMFULNESS_LABELS
    assert tuple(schema["fields"]["identity"]["labels"]) == IDENTITY_LABELS


def test_nonassistant_definition_matches_method():
    import yaml
    method = yaml.safe_load((ROOT / "configs" / "method_frozen.yaml").read_text("utf-8"))
    text = method["causal_outcomes"]["non_assistant_rate"]
    for label in NON_ASSISTANT:
        assert label in text


# ---------------------------------------------------------------------------
# Frozen scorer/parser and production contracts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,accepted", [
    (VALID_JUDGE, True),
    ("```json\n" + VALID_JUDGE + "\n```", True),
    ('{"harmfulness":"refused","identity":"assistant"}', False),
    ('{"harmfulness":"bad","identity":"assistant","rationale":"x"}', False),
    (VALID_JUDGE + VALID_JUDGE, False),
    (None, False),
])
def test_frozen_parser_exact_contract(raw, accepted):
    assert (SCORER.parse_judge(raw) is not None) is accepted


def test_canary_slice_is_order_independent():
    rows = [{"uid": f"u{i:03d}"} for i in range(80)]
    a = SCORER.canary_rows(rows, 10)
    b = SCORER.canary_rows(list(reversed(rows)), 10)
    assert [row["uid"] for row in a] == [row["uid"] for row in b]


def test_deepseek_production_contract_is_exact():
    rows = _production_rows(DEEPSEEK_CONDITIONS, 1000)
    assert SCORER.enforce_production_contract(
        "deepseek_steering", rows, DEEPSEEK_CONDITIONS) == []
    assert SCORER.enforce_production_contract(
        "deepseek_steering", rows[:-1], DEEPSEEK_CONDITIONS)


def test_qwen_production_contract_is_400_and_100_per_condition():
    rows = _production_rows(QWEN_CONDITIONS, 100)
    assert SCORER.enforce_production_contract("qwen_capping", rows, QWEN_CONDITIONS) == []
    problems = SCORER.enforce_production_contract(
        "qwen_capping", rows[:-1], QWEN_CONDITIONS)
    assert any("399 rows" in problem for problem in problems)
    assert any(QWEN_CONDITIONS[-1] in problem for problem in problems)


def test_qwen_duplicate_uid_is_rejected():
    rows = _production_rows(QWEN_CONDITIONS, 100)
    rows[-1]["uid"] = rows[0]["uid"]
    assert any("duplicate uid" in problem for problem in
               SCORER.enforce_production_contract("qwen_capping", rows, QWEN_CONDITIONS))


def test_production_scoring_requires_resolvable_source_git_sha(tmp_path, monkeypatch):
    outputs = tmp_path / "qwen.jsonl"
    _write_jsonl(outputs, _production_rows(QWEN_CONDITIONS, 100))
    monkeypatch.setattr(SCORER, "_git_sha", lambda: None)
    with pytest.raises(SystemExit) as exc:
        SCORER.score("qwen_capping", outputs, None, tmp_path / "o.jsonl",
                     tmp_path / "raw.jsonl", tmp_path / "report.json", dry_run=True)
    assert "source git SHA" in str(exc.value)


def test_canary_drift_marks_scoring_nonproduction_and_nondrawable(tmp_path, monkeypatch):
    outputs = tmp_path / "qwen.jsonl"
    _write_jsonl(outputs, _production_rows(QWEN_CONDITIONS, 100))
    monkeypatch.setattr(SCORER, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(SCORER, "make_client", lambda cfg, endpoint: object())
    monkeypatch.setattr(
        SCORER, "score_row",
        lambda client, model, prompt, pacer=None: (
            ("refused", "assistant", "ok"), [VALID_JUDGE]))

    calls = {"n": 0}
    def fake_canary(client, model, template, rows, label, raw_sink, pacer=None):
        calls["n"] += 1
        sha = "1" * 64 if calls["n"] == 1 else "2" * 64
        labels = [{"uid": row["uid"], "harmfulness": "refused", "identity": "assistant"}
                  for row in rows]
        return {"phase": label, "n": len(rows), "sha256": sha, "labels": labels}
    monkeypatch.setattr(SCORER, "run_canary", fake_canary)

    report = SCORER.score(
        "qwen_capping", outputs, None, tmp_path / "o.jsonl", tmp_path / "raw.jsonl",
        tmp_path / "report.json")
    assert report["canary"]["drift_detected"] is True
    assert report["production"] is False
    assert report["drawable_by_t26"] is False
    assert "canary drift" in report["non_production_reason"]


# ---------------------------------------------------------------------------
# T26 must consume only the exact drawable scored artifact.
# ---------------------------------------------------------------------------


def _scoring_report(outputs_path, branch="qwen_capping", drawable=True, drift=False,
                    source_git_dirty=False, omit_source_git_dirty=False):
    conditions = QWEN_CONDITIONS if branch == "qwen_capping" else DEEPSEEK_CONDITIONS
    per = 100 if branch == "qwen_capping" else 1000
    total = per * len(conditions)
    report = {
        "task": "causal_judge_scoring",
        "branch": branch,
        "judge_model": "deepseek-ai/DeepSeek-V3",
        "provider": "Novita (hosted API)",
        "endpoint": "https://api.novita.ai/v3/openai",
        "production": drawable,
        "drawable_by_t26": drawable,
        "source_git_sha": "a" * 40,
        "source_git_dirty": source_git_dirty,
        "n_rows": total,
        "per_condition": {condition: {"n": per, "coverage_failures": 0}
                          for condition in conditions},
        "canary": {"drift_detected": drift},
        "artifact_sha256": {"scored": hashlib.sha256(Path(outputs_path).read_bytes()).hexdigest(),
                            "raw": "b" * 64},
    }
    if omit_source_git_dirty:
        del report["source_git_dirty"]
    return report


def test_t26_scoring_report_binding_accepts_exact_qwen_artifact(tmp_path):
    outputs = tmp_path / "qwen.jsonl"
    _write_jsonl(outputs, _production_rows(QWEN_CONDITIONS, 100))
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_scoring_report(outputs)), encoding="utf-8")
    result = T26._validate_scoring_report(
        report_path, outputs, "qwen_capping", QWEN_CONDITIONS)
    assert result["drawable_by_t26"] is True


def test_t26_refuses_nondrawable_or_drifted_scoring(tmp_path):
    outputs = tmp_path / "qwen.jsonl"
    _write_jsonl(outputs, _production_rows(QWEN_CONDITIONS, 100))
    for drawable, drift in ((False, False), (True, True)):
        report_path = tmp_path / f"report-{drawable}-{drift}.json"
        report_path.write_text(
            json.dumps(_scoring_report(outputs, drawable=drawable, drift=drift)), encoding="utf-8")
        with pytest.raises(SystemExit):
            T26._validate_scoring_report(
                report_path, outputs, "qwen_capping", QWEN_CONDITIONS)


def test_t26_refuses_scored_artifact_hash_mismatch(tmp_path):
    outputs = tmp_path / "qwen.jsonl"
    _write_jsonl(outputs, _production_rows(QWEN_CONDITIONS, 100))
    report = _scoring_report(outputs)
    report["artifact_sha256"]["scored"] = "0" * 64
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        T26._validate_scoring_report(report_path, outputs, "qwen_capping", QWEN_CONDITIONS)
    assert "hash mismatch" in str(exc.value)


# ---------------------------------------------------------------------------
# T27 manifest binding: known branch, exact frozen conditions/sizes, exact key SHA.
# ---------------------------------------------------------------------------


def _t26_manifest_fixture(tmp_path, branch="deepseek_steering"):
    conditions = DEEPSEEK_CONDITIONS if branch == "deepseek_steering" else QWEN_CONDITIONS
    per = 30
    key = {}
    for ci, condition in enumerate(conditions):
        for i in range(per):
            iid = f"C{ci:02d}{i:03d}"
            key[iid] = {
                "uid": f"u-{ci}-{i}",
                "condition": condition,
                "ip_weight": 1.0,
                "auto_harmfulness": "refused",
                "auto_identity": "assistant",
                "judge_coverage_failure": False,
            }
    key_path = tmp_path / "sampling_key_PRIVATE.json"
    key_path.write_text(json.dumps({"items": key}), encoding="utf-8")
    manifest = {
        "task": "T26_causal_human_validation_sample",
        "branch": branch,
        "conditions": list(conditions),
        "per_condition_target": per,
        "n_items": per * len(conditions),
        "private_key_sha256": hashlib.sha256(key_path.read_bytes()).hexdigest(),
        "source_git_dirty": False,
    }
    return key_path, key, manifest


def test_t27_accepts_exact_frozen_manifest_binding(tmp_path):
    key_path, key, manifest = _t26_manifest_fixture(tmp_path)
    branch, conditions = T27._validate_sample_manifest(CFG, manifest, key_path, key)
    assert branch == "deepseek_steering"
    assert conditions == DEEPSEEK_CONDITIONS


@pytest.mark.parametrize("mutation,needle", [
    (lambda m: m.update(branch="unknown"), "not in the frozen"),
    (lambda m: m.update(private_key_sha256=""), "private_key_sha256"),
    (lambda m: m.update(n_items=149), "expected exactly"),
    (lambda m: m.update(per_condition_target=29), "frozen target"),
    (lambda m: m.update(conditions=list(reversed(DEEPSEEK_CONDITIONS))), "condition list"),
    (lambda m: m.update(source_git_dirty=True), "source_git_dirty"),
    (lambda m: m.pop("source_git_dirty"), "source_git_dirty"),
])
def test_t27_rejects_malformed_or_mismatched_manifest(tmp_path, mutation, needle):
    key_path, key, manifest = _t26_manifest_fixture(tmp_path)
    mutation(manifest)
    with pytest.raises(SystemExit) as exc:
        T27._validate_sample_manifest(CFG, manifest, key_path, key)
    assert needle in str(exc.value)


def test_t27_rejects_private_key_bytes_not_matching_manifest(tmp_path):
    key_path, key, manifest = _t26_manifest_fixture(tmp_path)
    key_path.write_text(key_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        T27._validate_sample_manifest(CFG, manifest, key_path, key)
    assert "does not match" in str(exc.value)


def test_t27_cli_refuses_gating_without_sample_manifest(tmp_path):
    key_path, key, _ = _t26_manifest_fixture(tmp_path)
    labels = tmp_path / "labels.csv"
    with open(labels, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "harmfulness", "identity"])
        writer.writeheader()
        for iid in key:
            writer.writerow({"item_id": iid, "harmfulness": "refused", "identity": "assistant"})
    with pytest.raises(SystemExit) as exc:
        T27.main(["run", "--labels-1", str(labels), "--labels-2", str(labels),
                  "--key", str(key_path), "--out", str(tmp_path / "out.json")])
    assert "sample-manifest is required" in str(exc.value)


# ---------------------------------------------------------------------------
# Fail-closed provenance regressions (PR #57 review round 2):
#   1. T26 draw() must check for a dirty tree BEFORE writing worksheet/key
#      output, not after - otherwise a clean run dirties its own tree and
#      then fails at the final stamp, on every invocation.
#   2. A T27 gating run must refuse to bind to a T26 manifest that does not
#      certify source_git_dirty: false.
#   3. A T27 run started with --allow-dirty must never emit gating: true,
#      even with complete, fully-agreeing labels.
#   4. run_causal_judge_scoring must not report production/drawable_by_t26
#      when the source tree is dirty.
# ---------------------------------------------------------------------------


def test_t26_draw_checks_clean_tree_before_writing_any_output(tmp_path, monkeypatch):
    outputs = tmp_path / "qwen.jsonl"
    _write_jsonl(outputs, _production_rows(QWEN_CONDITIONS, 100))
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_scoring_report(outputs)), encoding="utf-8")
    out_dir = tmp_path / "draw_out"

    monkeypatch.setattr(T26, "git_dirty", lambda: True)
    with pytest.raises(RuntimeError, match="dirty working tree"):
        T26.draw(outputs, out_dir, 1, "qwen_capping", report_path)
    assert not out_dir.exists(), (
        "a dirty-tree draw must abort before creating any worksheet/key output")


def test_t26_draw_stamps_the_pre_write_dirty_state_not_the_post_write_one(tmp_path, monkeypatch):
    """A clean-tree draw must not be defeated by its own output files.

    stamp_report() re-checks git_dirty() at call time; by then draw() has
    already written worksheet.csv/labeller CSVs/the private key into out_dir.
    If those untracked files are inside the repo, a naive re-check would see
    them and either raise (allow_dirty=False) or falsely stamp
    source_git_dirty: true (allow_dirty=True) - neither reflects whether the
    CODE that ran was actually clean.
    """
    outputs = tmp_path / "qwen.jsonl"
    _write_jsonl(outputs, _production_rows(QWEN_CONDITIONS, 100))
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_scoring_report(outputs)), encoding="utf-8")
    out_dir = tmp_path / "draw_out"

    calls = {"n": 0}

    def fake_git_dirty():
        # First call: draw()'s own preflight, before any writes -> clean.
        # Any call after that happens only because something re-checked the
        # tree post-write; simulate what that would see (dirty, since
        # out_dir now has new files) to prove the code no longer depends on it.
        calls["n"] += 1
        return calls["n"] > 1

    monkeypatch.setattr(T26, "git_dirty", fake_git_dirty)
    manifest = T26.draw(outputs, out_dir, 1, "qwen_capping", report_path)
    assert manifest["source_git_dirty"] is False
    assert (out_dir / "worksheet.csv").exists()


def test_t27_gating_run_requires_clean_t26_manifest(tmp_path):
    key_path, key, manifest = _t26_manifest_fixture(tmp_path, branch="qwen_capping")
    manifest["source_git_dirty"] = True
    manifest_path = tmp_path / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    labels = tmp_path / "labels.csv"
    with open(labels, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "harmfulness", "identity"])
        writer.writeheader()
        for iid, rec in key.items():
            writer.writerow({"item_id": iid, "harmfulness": rec["auto_harmfulness"],
                             "identity": rec["auto_identity"]})

    with pytest.raises(SystemExit, match="source_git_dirty"):
        T27.run(str(labels), str(labels), str(key_path), str(tmp_path / "out.json"),
                sample_manifest=str(manifest_path))


def test_t27_allow_dirty_run_never_gates_even_when_complete(tmp_path, monkeypatch):
    key_path, key, manifest = _t26_manifest_fixture(tmp_path, branch="qwen_capping")
    manifest_path = tmp_path / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    labels = tmp_path / "labels.csv"
    with open(labels, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "harmfulness", "identity"])
        writer.writeheader()
        for iid, rec in key.items():
            writer.writerow({"item_id": iid, "harmfulness": rec["auto_harmfulness"],
                             "identity": rec["auto_identity"]})

    monkeypatch.setattr(provenance, "git_dirty", lambda: True)
    out_path = tmp_path / "out.json"
    T27.run(str(labels), str(labels), str(key_path), str(out_path),
            sample_manifest=str(manifest_path), allow_dirty=True)
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["gating"] is False, (
        "a --allow-dirty T27 run must never emit a gating result, regardless "
        "of label completeness")


def test_causal_scoring_refuses_production_on_dirty_tree(monkeypatch):
    monkeypatch.setattr(SCORER, "git_dirty", lambda: True)
    assert SCORER._nonproduction_reason(None, [], False, source_git_dirty=True) == (
        "source working tree was dirty at scoring time")


# ---------------------------------------------------------------------------
# Fail-closed provenance regressions (PR #57 review round 3):
#   5. run_causal_judge_scoring.py must import src.* only after the repo
#      root is on sys.path, or the real CLI entry point
#      (`python tools/run_causal_judge_scoring.py ...`) breaks with
#      ModuleNotFoundError before argparse even runs.
#   6. T26's scoring-report validator must independently require
#      source_git_dirty: false, not trust it transitively through
#      production=true - an older report predating the field, or a
#      hand-edited one, must not be accepted on trust.
# ---------------------------------------------------------------------------


def test_causal_scoring_cli_entry_point_does_not_crash_on_import():
    """Regression: src.provenance was imported before sys.path.insert.

    Running as a subprocess is the only way to catch this - importing the
    module in-process (as SCORER already is, via _load_tool) leaves the
    parent test process's sys.path in place, which masks the bug entirely.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_causal_judge_scoring.py"), "--help"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_causal_scoring_imports_src_after_path_insert():
    source = inspect.getsource(SCORER)
    path_insert_pos = source.index("sys.path.insert")
    first_src_import_pos = min(
        source.index(needle) for needle in ("from src.", "import src.")
        if needle in source)
    assert first_src_import_pos > path_insert_pos


@pytest.mark.parametrize("kwargs", [
    {"source_git_dirty": True},
    {"omit_source_git_dirty": True},
])
def test_t26_rejects_scoring_report_without_certified_clean_tree(tmp_path, kwargs):
    outputs = tmp_path / "qwen.jsonl"
    _write_jsonl(outputs, _production_rows(QWEN_CONDITIONS, 100))
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(_scoring_report(outputs, **kwargs)), encoding="utf-8")
    with pytest.raises(SystemExit, match="source_git_dirty"):
        T26._validate_scoring_report(report_path, outputs, "qwen_capping", QWEN_CONDITIONS)


def test_t26_accepts_scoring_report_with_certified_clean_tree(tmp_path):
    outputs = tmp_path / "qwen.jsonl"
    _write_jsonl(outputs, _production_rows(QWEN_CONDITIONS, 100))
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_scoring_report(outputs)), encoding="utf-8")
    result = T26._validate_scoring_report(
        report_path, outputs, "qwen_capping", QWEN_CONDITIONS)
    assert result["source_git_dirty"] is False


# ---------------------------------------------------------------------------
# execution_status_at_draw must describe DRAW-TIME state, not preregistration.
#
# The frozen T27 config's per-branch `execution_status` is prose written before
# execution; for qwen_capping it still says the random-direction and sphere
# conditions are "NOT yet generated". Copying it into a field named
# "..._at_draw" misreports the state of a sample drawn after those conditions
# were generated and scored. The frozen config is deliberately NOT edited, so
# the sampler must derive the draw-time facts from the bound scoring report.
# ---------------------------------------------------------------------------


def test_execution_status_at_draw_is_derived_not_preregistration_prose(tmp_path):
    outputs = tmp_path / "qwen.jsonl"
    _write_jsonl(outputs, _production_rows(QWEN_CONDITIONS, 100))
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_scoring_report(outputs)), encoding="utf-8")

    manifest = T26.draw(outputs, tmp_path / "out", 20260818, "qwen_capping",
                        report_path, allow_dirty=True)

    status = manifest["execution_status_at_draw"]
    assert isinstance(status, dict)
    assert status["all_frozen_conditions_generated_and_scored"] is True
    assert status["n_rows_scored"] == 400
    assert status["rows_per_condition"] == {c: 100 for c in QWEN_CONDITIONS}
    assert status["scoring_production"] is True
    assert status["scoring_drawable_by_t26"] is True

    frozen_text = CFG["scope"]["branches"]["qwen_capping"]["execution_status"]
    assert "NOT yet generated" in frozen_text, "fixture assumption: prose is stale"
    assert frozen_text not in json.dumps(status), (
        "stale preregistration prose must not appear in the draw-time status")

    retained = manifest["frozen_config_execution_status_text"]
    assert retained["text"] == frozen_text
    assert retained["status"] == "PREREGISTRATION_TEXT_MAY_PREDATE_EXECUTION"


def test_draw_does_not_edit_the_frozen_t27_config(tmp_path):
    """The fix must not mutate the frozen config to make the prose accurate."""
    config_path = ROOT / "configs" / "t27_causal_judge_frozen.json"
    before = hashlib.sha256(config_path.read_bytes()).hexdigest()

    outputs = tmp_path / "qwen.jsonl"
    _write_jsonl(outputs, _production_rows(QWEN_CONDITIONS, 100))
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_scoring_report(outputs)), encoding="utf-8")
    T26.draw(outputs, tmp_path / "out", 20260818, "qwen_capping",
             report_path, allow_dirty=True)

    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == before
