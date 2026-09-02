"""T10 checks: C80 role-generation manifest, worker, and production preflight.

CPU-only, no model. Covers both the structural manifest properties and the
production safeguards a 44,000-output frozen run needs:

- one block gives 275 roles x 80 questions = 22,000 rollouts, balanced 4,400
  per prompt index, disjoint across blocks, reconciling to 44,000;
- --strict-final requires FROZEN config, pinned revisions, a FROZEN + schema-
  compatible T08 block artifact, and frozen rendered-prompt hashes;
- preflight rejects an engine whose revisions differ from the frozen manifest;
- decoding is loaded from and checked against a frozen T09 runtime artifact;
- the rendered-prompt hash is checked BEFORE generation;
- resume rejects foreign / duplicate / malformed / corrupted / engine-mismatched
  rows;
- retryable technical failures are retried; degeneration is never retried.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder():
    return _load("build_t10", "tools/build_t10_c80_manifest.py")


@pytest.fixture(scope="module")
def worker():
    return _load("run_t10", "tools/run_t10_role_generation.py")


@pytest.fixture(scope="module")
def cfg_and_sha():
    import hashlib

    import yaml

    b = (ROOT / "configs" / "method_frozen.yaml").read_bytes()
    return yaml.safe_load(b), hashlib.sha256(b).hexdigest()


def _manifest(builder, tmp_path, block):
    return builder.build_manifest(
        ROOT, ROOT / "configs" / "method_frozen.yaml", block,
        tmp_path / f"m_{block}.json", strict_final=False,
    )


@pytest.fixture()
def manifest_b(builder, tmp_path):
    return _manifest(builder, tmp_path, "C80-B")


def _matched_engine(worker, manifest):
    """A fake engine whose revisions match the frozen manifest (passes preflight)."""
    eng = worker._dry_run_engine()
    m = manifest["model"]
    eng.model_revision = m["model_revision"]
    eng.tokenizer_revision = m["tokenizer_revision"]
    eng.chat_template_sha256 = m["chat_template_sha256"]
    return eng


# --------------------------------------------------------------------------- #
# Structural manifest properties.
# --------------------------------------------------------------------------- #
def test_one_block_is_22000_balanced(manifest_b):
    assert manifest_b["workload"]["n_outputs"] == 275 * 80 == 22000
    assert manifest_b["workload"]["both_blocks_total"] == 44000
    assert manifest_b["counts"]["per_prompt_index"] == {
        "0": 4400, "1": 4400, "2": 4400, "3": 4400, "4": 4400
    }
    ids = [r["rollout_id"] for r in manifest_b["rollouts"]]
    assert len(set(ids)) == 22000


def test_both_blocks_disjoint_and_reconcile(builder, tmp_path):
    a = _manifest(builder, tmp_path, "C80-A")
    b = _manifest(builder, tmp_path, "C80-B")
    ids_a = {r["rollout_id"] for r in a["rollouts"]}
    ids_b = {r["rollout_id"] for r in b["rollouts"]}
    assert ids_a.isdisjoint(ids_b)
    assert len(ids_a | ids_b) == 44000
    q_a = {r["question_id"] for r in a["rollouts"]}
    q_b = {r["question_id"] for r in b["rollouts"]}
    assert q_a.isdisjoint(q_b) and len(q_a) == 80 and len(q_b) == 80


def test_manifest_is_deterministic(builder, tmp_path):
    a1 = _manifest(builder, tmp_path, "C80-B")
    a2 = builder.build_manifest(
        ROOT, ROOT / "configs" / "method_frozen.yaml", "C80-B",
        tmp_path / "again.json", strict_final=False,
    )
    assert a1["rollouts_sha256"] == a2["rollouts_sha256"]


def test_invalid_block_rejected(builder, tmp_path):
    with pytest.raises(ValueError):
        builder.build_manifest(
            ROOT, ROOT / "configs" / "method_frozen.yaml", "E80",
            tmp_path / "x.json", strict_final=False,
        )


def test_role_rows_translated_arm_and_per_row_fields(worker, manifest_b, cfg_and_sha, tmp_path):
    sys.path.insert(0, str(ROOT))
    from src.schemas import validate_row

    cfg, config_sha = cfg_and_sha
    engine = worker._dry_run_engine()
    out = tmp_path / "gen.jsonl"
    worker.run(
        ROOT, manifest_b, engine, cfg, config_sha, "0" * 40,
        {"temperature": 0.0, "max_new_tokens": 8, "do_sample": False, "seed": 0},
        out, limit=15,
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 15
    for row in rows:
        assert validate_row(row, "generation")
        assert row["arm"] == "USER_TRANSLATED_LU"
        assert row["channel"] == "user"
        assert row["role_id"]
        assert row["default_condition_index"] is None
        trc = row["token_region_counts"]
        assert trc["all_response"] == trc["reasoning"] + trc["final_answer"]
        assert trc["reasoning"] > 0 and trc["final_answer"] > 0


# --------------------------------------------------------------------------- #
# strict-final gating on the T08 block artifact.
# --------------------------------------------------------------------------- #
def test_strict_final_requires_render_hashes(builder, tmp_path):
    # Config is FROZEN with pinned revisions on this branch, but no render
    # hashes are supplied => strict-final must refuse.
    with pytest.raises(ValueError, match="render-hashes"):
        builder.build_manifest(
            ROOT, ROOT / "configs" / "method_frozen.yaml", "C80-B",
            tmp_path / "sf.json", strict_final=True,
        )


def test_strict_final_requires_frozen_block_artifact(builder, tmp_path):
    # Copy the real block file but flip its status away from FROZEN, and supply
    # a render-hash file so we isolate the block-status check.
    design = tmp_path / "design"
    design.mkdir()
    block = json.loads(
        (ROOT / "design" / "questions_C80_B.json").read_text(encoding="utf-8")
    )
    block["status"] = "REVIEW_CANDIDATE"
    (design / "questions_C80_B.json").write_text(
        json.dumps(block), encoding="utf-8"
    )
    render = tmp_path / "render.jsonl"
    render.write_text("", encoding="utf-8")  # empty; block check fires first
    with pytest.raises(ValueError, match="FROZEN"):
        builder.build_manifest(
            ROOT, ROOT / "configs" / "method_frozen.yaml", "C80-B",
            tmp_path / "sf.json", strict_final=True,
            design_dir=design, render_hashes=render,
        )


def test_block_schema_incompatibility_rejected(builder, tmp_path):
    design = tmp_path / "design"
    design.mkdir()
    block = json.loads(
        (ROOT / "design" / "questions_C80_B.json").read_text(encoding="utf-8")
    )
    block["schema_version"] = "t08-question-block/9.9"
    (design / "questions_C80_B.json").write_text(json.dumps(block), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        builder.build_manifest(
            ROOT, ROOT / "configs" / "method_frozen.yaml", "C80-B",
            tmp_path / "x.json", strict_final=False, design_dir=design,
        )


# --------------------------------------------------------------------------- #
# Production preflight: engine/config/manifest agreement.
# --------------------------------------------------------------------------- #
def test_preflight_rejects_engine_revision_mismatch(worker, manifest_b, cfg_and_sha):
    _, config_sha = cfg_and_sha
    good = _matched_engine(worker, manifest_b)
    worker.preflight(manifest_b, good, config_sha, require_frozen_render_hashes=False)
    bad = copy.copy(good)
    bad.model_revision = "some-other-revision"
    with pytest.raises(worker.PreflightError, match="model_revision"):
        worker.preflight(manifest_b, bad, config_sha, require_frozen_render_hashes=False)


def test_preflight_requires_frozen_manifest_for_production(worker, manifest_b, cfg_and_sha):
    _, config_sha = cfg_and_sha
    eng = _matched_engine(worker, manifest_b)
    # DRAFT manifest without frozen render hashes must be refused in production.
    with pytest.raises(worker.PreflightError, match="FROZEN|render-prompt|rendered-prompt"):
        worker.preflight(manifest_b, eng, config_sha, require_frozen_render_hashes=True)


def test_preflight_config_mismatch(worker, manifest_b):
    eng = _matched_engine(worker, manifest_b)
    with pytest.raises(worker.PreflightError, match="method config"):
        worker.preflight(manifest_b, eng, "0" * 64, require_frozen_render_hashes=False)


# --------------------------------------------------------------------------- #
# Frozen T09 decoding.
# --------------------------------------------------------------------------- #
def test_frozen_decoding_loaded_and_checked(worker, tmp_path):
    with pytest.raises(worker.PreflightError, match="missing"):
        worker.load_frozen_decoding(tmp_path, "results/audit/E80_acceptance_report.json")
    # Present but no frozen_decoding block.
    art = tmp_path / "art.json"
    art.write_text(json.dumps({"unrelated": True}), encoding="utf-8")
    with pytest.raises(worker.PreflightError, match="frozen_decoding"):
        worker.load_frozen_decoding(tmp_path, "art.json")
    # Missing a required key.
    art.write_text(json.dumps({"frozen_decoding": {"temperature": 0.0}}), encoding="utf-8")
    with pytest.raises(worker.PreflightError, match="missing keys"):
        worker.load_frozen_decoding(tmp_path, "art.json")
    # Well-formed.
    art.write_text(json.dumps({"frozen_decoding": {
        "temperature": 0.0, "max_new_tokens": 2048, "do_sample": False, "seed": 0,
    }}), encoding="utf-8")
    decoding, sha = worker.load_frozen_decoding(tmp_path, "art.json")
    assert decoding["max_new_tokens"] == 2048 and len(sha) == 64


# --------------------------------------------------------------------------- #
# Rendered-prompt hash checked BEFORE generation.
# --------------------------------------------------------------------------- #
def test_render_hash_checked_before_generate(worker, manifest_b, cfg_and_sha):
    sys.path.insert(0, str(ROOT))
    cfg, config_sha = cfg_and_sha

    # An engine whose generate() explodes: if the render-hash check runs first,
    # we get RenderHashMismatch, not the generate error.
    def boom(prompt_ids):
        raise AssertionError("generate must not be called on a hash mismatch")

    eng = worker._dry_run_engine()
    eng.generate = boom
    rollout = dict(manifest_b["rollouts"][0], expected_rendered_prompt_sha256="f" * 64)
    with pytest.raises(worker.RenderHashMismatch):
        worker.build_generation_row(
            ROOT, rollout, eng, cfg, config_sha, cfg["models"]["primary"]["model_id"],
            "0" * 40, {"temperature": 0.0, "max_new_tokens": 8, "do_sample": False, "seed": 0},
            {(rollout["role_id"], rollout["prompt_index"]): None} if False else worker._load_role_prompt_cache(ROOT),
            {},
            retry=0, enforce_render_hash=True,
        )


def test_render_hash_match_allows_generation(worker, manifest_b, cfg_and_sha):
    sys.path.insert(0, str(ROOT))
    cfg, config_sha = cfg_and_sha
    eng = worker._dry_run_engine()
    prompt_cache = worker._load_role_prompt_cache(ROOT)
    qcache = {}
    # First render with no expected hash to learn the correct value.
    row = worker.build_generation_row(
        ROOT, manifest_b["rollouts"][0], eng, cfg, config_sha,
        cfg["models"]["primary"]["model_id"], "0" * 40,
        {"temperature": 0.0, "max_new_tokens": 8, "do_sample": False, "seed": 0},
        prompt_cache, qcache, retry=0, enforce_render_hash=False,
    )
    correct = row["rendered_prompt_sha256"]
    rollout = dict(manifest_b["rollouts"][0], expected_rendered_prompt_sha256=correct)
    row2 = worker.build_generation_row(
        ROOT, rollout, eng, cfg, config_sha, cfg["models"]["primary"]["model_id"],
        "0" * 40, {"temperature": 0.0, "max_new_tokens": 8, "do_sample": False, "seed": 0},
        prompt_cache, qcache, retry=0, enforce_render_hash=True,
    )
    assert row2["rendered_prompt_sha256"] == correct


# --------------------------------------------------------------------------- #
# Resume integrity.
# --------------------------------------------------------------------------- #
def _run_some(worker, manifest, cfg, config_sha, out, limit):
    engine = worker._dry_run_engine()
    return worker.run(
        ROOT, manifest, engine, cfg, config_sha, "0" * 40,
        {"temperature": 0.0, "max_new_tokens": 8, "do_sample": False, "seed": 0},
        out, limit=limit,
    )


def test_resume_continues_then_idempotent(worker, manifest_b, cfg_and_sha, tmp_path):
    cfg, config_sha = cfg_and_sha
    out = tmp_path / "gen.jsonl"
    first = _run_some(worker, manifest_b, cfg, config_sha, out, 10)
    assert first["rollouts_processed"] == 10
    second = _run_some(worker, manifest_b, cfg, config_sha, out, 10)
    assert second["rollouts_processed"] == 10
    assert second["skipped_already_done"] == 10
    third = _run_some(worker, manifest_b, cfg, config_sha, out, None)
    fourth = _run_some(worker, manifest_b, cfg, config_sha, out, None)
    assert fourth["rows_written"] == 0
    ids = [json.loads(x)["rollout_id"] for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(ids) == len(set(ids)) == 22000


def test_resume_rejects_foreign_rollout(worker, manifest_b, cfg_and_sha, tmp_path):
    cfg, config_sha = cfg_and_sha
    out = tmp_path / "gen.jsonl"
    _run_some(worker, manifest_b, cfg, config_sha, out, 3)
    # Append a row whose rollout_id is not in this manifest.
    lines = out.read_text(encoding="utf-8").splitlines()
    foreign = json.loads(lines[0])
    foreign["rollout_id"] = "abcdef0123456789"  # not a manifest rollout
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(foreign) + "\n")
    with pytest.raises(worker.ResumeIntegrityError, match="not in this manifest"):
        _run_some(worker, manifest_b, cfg, config_sha, out, 3)


def test_resume_rejects_corrupted_line(worker, manifest_b, cfg_and_sha, tmp_path):
    cfg, config_sha = cfg_and_sha
    out = tmp_path / "gen.jsonl"
    _run_some(worker, manifest_b, cfg, config_sha, out, 2)
    with out.open("a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
    with pytest.raises(worker.ResumeIntegrityError, match="corrupted JSON"):
        _run_some(worker, manifest_b, cfg, config_sha, out, 2)


def test_resume_rejects_duplicate_attempt(worker, manifest_b, cfg_and_sha, tmp_path):
    cfg, config_sha = cfg_and_sha
    out = tmp_path / "gen.jsonl"
    _run_some(worker, manifest_b, cfg, config_sha, out, 2)
    first_line = out.read_text(encoding="utf-8").splitlines()[0]
    with out.open("a", encoding="utf-8") as fh:
        fh.write(first_line + "\n")  # exact duplicate (rollout_id, retry)
    with pytest.raises(worker.ResumeIntegrityError, match="duplicate"):
        _run_some(worker, manifest_b, cfg, config_sha, out, 2)


def test_resume_rejects_foreign_config(worker, manifest_b, cfg_and_sha, tmp_path):
    cfg, config_sha = cfg_and_sha
    out = tmp_path / "gen.jsonl"
    _run_some(worker, manifest_b, cfg, config_sha, out, 2)
    manifest_ids = {r["rollout_id"] for r in manifest_b["rollouts"]}
    engine = worker._dry_run_engine()
    # Reading the same file with a different current config hash is a foreign
    # config: every row's stamped config_sha256 no longer matches.
    with pytest.raises(worker.ResumeIntegrityError, match="foreign config"):
        worker.load_existing_attempts(out, manifest_ids, engine, "0" * 64, root=ROOT)


def test_resume_rejects_engine_mismatch(worker, manifest_b, cfg_and_sha, tmp_path):
    cfg, config_sha = cfg_and_sha
    out = tmp_path / "gen.jsonl"
    _run_some(worker, manifest_b, cfg, config_sha, out, 2)
    manifest_ids = {r["rollout_id"] for r in manifest_b["rollouts"]}
    other = worker._dry_run_engine()
    other.model_revision = "a-different-model-revision"
    with pytest.raises(worker.ResumeIntegrityError, match="engine"):
        worker.load_existing_attempts(out, manifest_ids, other, config_sha, root=ROOT)


# --------------------------------------------------------------------------- #
# Retry handling.
# --------------------------------------------------------------------------- #
def _engine_error_then_ok(worker):
    """Fake engine: a generation_error on the first attempt, valid on retry."""
    state = {"calls": 0}
    tok = worker._CharTokenizer()
    good = "<think>ok</think>Answer."

    def generate(prompt_ids):
        state["calls"] += 1
        if state["calls"] == 1:
            return [], "", "error"  # -> generation_error (retryable)
        return [ord(c) for c in good], good, "stop"

    eng = worker.RoleEngine(
        tokenizer=tok, generate=generate,
        model_revision="dryrun-model-rev", tokenizer_revision="dryrun-tok-rev",
        chat_template_sha256=worker.hashlib.sha256(tok.chat_template.encode()).hexdigest(),
    )
    return eng


def test_retryable_failure_is_retried(worker, manifest_b, cfg_and_sha, tmp_path):
    cfg, config_sha = cfg_and_sha
    eng = _engine_error_then_ok(worker)
    out = tmp_path / "gen.jsonl"
    summary = worker.run(
        ROOT, manifest_b, eng, cfg, config_sha, "0" * 40,
        {"temperature": 0.0, "max_new_tokens": 8, "do_sample": False, "seed": 0},
        out, limit=1,
    )
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    # One rollout, two attempts: retry 0 (error) then retry 1 (valid).
    assert summary["rollouts_processed"] == 1
    assert [r["retry"] for r in rows] == [0, 1]
    assert rows[0]["technical_validity"] == "generation_error"
    assert rows[1]["technical_validity"] == "valid"
    # The resolved rollout is now DONE (its retry-1 attempt is terminal-valid),
    # so a resume never re-attempts it.
    resolved_rid = rows[0]["rollout_id"]
    manifest_ids = {r["rollout_id"] for r in manifest_b["rollouts"]}
    attempts = worker.load_existing_attempts(
        out, manifest_ids, worker._dry_run_engine(), config_sha, root=ROOT
    )
    done, next_retry, _ = worker.resume_plan(attempts, worker.DEFAULT_MAX_RETRIES)
    assert resolved_rid in done
    assert resolved_rid not in next_retry


def test_degeneration_is_never_retried(worker, manifest_b, cfg_and_sha, tmp_path):
    cfg, config_sha = cfg_and_sha
    tok = worker._CharTokenizer()
    # A degenerate output: one 8-gram repeated many times.
    unit = "abcdefgh"
    degen = unit * 40

    def generate(prompt_ids):
        return [ord(c) for c in degen], degen, "stop"

    eng = worker.RoleEngine(
        tokenizer=tok, generate=generate,
        model_revision="dryrun-model-rev", tokenizer_revision="dryrun-tok-rev",
        chat_template_sha256=worker.hashlib.sha256(tok.chat_template.encode()).hexdigest(),
    )
    out = tmp_path / "gen.jsonl"
    summary = worker.run(
        ROOT, manifest_b, eng, cfg, config_sha, "0" * 40,
        {"temperature": 0.0, "max_new_tokens": 8, "do_sample": False, "seed": 0},
        out, limit=1, max_retries=3,
    )
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    # Exactly one attempt: degeneration is terminal and never retried.
    assert len(rows) == 1
    assert rows[0]["technical_validity"] == "degenerate_repetition"
    assert rows[0]["retry"] == 0
    assert summary["exhausted_this_run"] == 0
    assert not worker._is_retryable("degenerate_repetition")
