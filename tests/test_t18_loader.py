"""The T18 loader, exercised against the repository's own row schemas.

The real activations are not available while the pipeline is being prepared,
so the loader is the one piece that cannot be checked against production data.
These tests build a small run directory whose rows pass ``src.schemas``
validation unchanged — same required fields, same cross-field rules, same
frozen block and prompt-index assignments — and then check that the loader
selects the frozen slice, consumes the frozen eligibility artifact, reads only
the primary judge region, and refuses ambiguous or corrupted input.

The fixture writes **both** T13 judge regions for every generation, with
deliberately disagreeing scores, because a loader that keyed on the generation
row alone would silently take whichever region happened to be written last.
"""

import hashlib
import json

import numpy as np
import pytest

from src import ids
from src.config import block_question_ids, load_frozen_config, prompt_index_for
from src.schemas import write_jsonl
from src.t18_readouts import DEFAULT_ROLE_PREFIX, PRIMARY_JUDGE_REGION
from tools.run_t18_dimensional_adequacy import load_rows

CFG, CONFIG_SHA = load_frozen_config()
GEOM = CFG["confirmatory_geometry"]
MODEL = CFG["models"]["primary"]["model_id"]
SENSITIVITY_REGION = "full_visible"
N_ROLES = 8
N_QUESTIONS_PER_BLOCK = 6
HIDDEN = 12


def _vector_digest(vec):
    """Matches the loader: hash the float32 bytes of the pooled vector."""
    return hashlib.sha256(
        np.ascontiguousarray(vec, dtype=np.float32).tobytes()).hexdigest()


def _generation_row(arm, role_id, block, question_id, prompt_index,
                    default_condition_index, validity="valid"):
    channel = "system" if arm == "SYSTEM_LU" else "user"
    messages = [{"role": channel, "content": f"{role_id or 'default'}"},
                {"role": "user", "content": f"q{question_id}"}]
    rendered = f"<prompt {arm} {role_id} {question_id}>"
    rollout = ids.rollout_id(MODEL, arm, role_id, question_id, prompt_index)
    prompt_tokens, output_tokens = [1, 2, 3], [4, 5, 6, 7]
    return {
        "row_id": ids.generation_row_id(rollout),
        "rollout_id": rollout,
        "model": MODEL,
        "model_revision": CFG["models"]["primary"]["model_revision"],
        "config_sha256": CONFIG_SHA,
        "git_sha": "0" * 40,
        "arm": arm,
        "channel": channel,
        "default_condition_index": default_condition_index,
        "role_id": role_id,
        "block": block,
        "question_id": question_id,
        "prompt_index": prompt_index,
        "retry": 0,
        "messages": messages,
        "rendered_prompt": rendered,
        "messages_sha256": ids.messages_sha256(messages),
        "rendered_prompt_sha256": ids.text_sha256(rendered),
        "tokenizer_revision": CFG["models"]["primary"]["tokenizer_revision"],
        "chat_template_sha256": CFG["models"]["primary"]["chat_template_sha256"],
        "runtime_settings": {"temperature": 1.0},
        "prompt_token_ids": prompt_tokens,
        "output_token_ids": output_tokens,
        "n_prompt_tokens": len(prompt_tokens),
        "n_output_tokens": len(output_tokens),
        "response_start": len(prompt_tokens),
        "response_end_exclusive": len(prompt_tokens) + len(output_tokens),
        "output_text": "<think>x</think> answer",
        "finish_reason": "stop",
        "technical_validity": validity,
        "segmentation_case": "balanced_reasoning",
    }


def build_run_dir(tmp_path, role_score=3, validity="valid", pool=None,
                  block_index=None, corrupt_vector=False, score_by_role=None,
                  sensitivity_score=0, n_questions=N_QUESTIONS_PER_BLOCK,
                  eligible_roles=None, duplicate_primary=False,
                  drop_judge_region=False, extra_eligible_role=None):
    """A minimal but schema-valid run directory over the frozen C80 blocks.

    ``score_by_role`` maps a role index to its primary-region score,
    overriding ``role_score``. ``sensitivity_score`` is what the paired
    ``full_visible`` row claims; it disagrees with the primary by default so
    that region selection is actually exercised.
    """
    pool = pool or GEOM["primary_pool"]
    block_index = GEOM["primary_layer"] if block_index is None else block_index
    arm = GEOM["primary_prompt_arm"]
    blocks = block_question_ids(CFG)
    rng = np.random.default_rng(0)
    tmp_path.mkdir(parents=True, exist_ok=True)

    generation, judge, activation, vectors = [], [], [], {}

    def judge_row(gen, score, region, retry):
        row = {
            "row_id": ids.generation_row_id(gen["rollout_id"], retry=retry),
            "rollout_id": gen["rollout_id"],
            "model": MODEL,
            "model_revision": gen["model_revision"],
            "config_sha256": gen["config_sha256"],
            "git_sha": gen["git_sha"],
            "generation_row_id": gen["row_id"],
            "judge_model": "judge-v1",
            "judge_raw_response": str(score),
            "role_score": score,
            "abstained": False,
            "judge_region": region,
        }
        if drop_judge_region:
            del row["judge_region"]
        return row

    def add(gen, label, score):
        generation.append(gen)
        if score is not None:
            judge.append(judge_row(gen, score, PRIMARY_JUDGE_REGION, 1))
            judge.append(judge_row(gen, sensitivity_score, SENSITIVITY_REGION, 3))
            if duplicate_primary:
                judge.append(judge_row(gen, score, PRIMARY_JUDGE_REGION, 4))
        vec = rng.normal(size=HIDDEN) + (3.0 if label else 0.0)
        act_id = ids.generation_row_id(gen["rollout_id"], retry=2)
        vectors[act_id] = vec.astype(np.float32)
        activation.append({
            "row_id": act_id,
            "rollout_id": gen["rollout_id"],
            "model": MODEL,
            "model_revision": gen["model_revision"],
            "config_sha256": gen["config_sha256"],
            "git_sha": gen["git_sha"],
            "generation_row_id": gen["row_id"],
            "pool": pool,
            "block_index": block_index,
            "vector_sha256": "0" * 64 if corrupt_vector else _vector_digest(vec),
            "n_pooled_tokens": 4,
        })

    for block in GEOM["primary_blocks"]:
        ids_in_block = blocks[block]
        for position in range(n_questions):
            qid = ids_in_block[position]
            pidx = prompt_index_for(CFG, ids_in_block, position)
            for r in range(N_ROLES):
                score = (score_by_role or {}).get(r, role_score)
                add(_generation_row(arm, f"role_{r:02d}", block, qid, pidx,
                                    None, validity), 0, score)
            add(_generation_row("DEFAULT", "", block, qid, pidx, pidx),
                1, None)

    if eligible_roles is None:
        eligible_roles = [f"role_{r:02d}" for r in range(N_ROLES)]
    if extra_eligible_role:
        eligible_roles = list(eligible_roles) + [extra_eligible_role]

    write_jsonl(tmp_path / "generation.jsonl", generation, "generation")
    write_jsonl(tmp_path / "judge.jsonl", judge, "judge")
    write_jsonl(tmp_path / "activations.jsonl", activation, "activation")
    np.savez(tmp_path / "vectors.npz", **vectors)
    (tmp_path / "eligible_roles.json").write_text(
        json.dumps({
            "source_task": "T15",
            "threshold": 10,
            "blocks": list(GEOM["primary_blocks"]),
            "eligible_roles_primary": eligible_roles,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    return tmp_path


# ---------------------------------------------------------------------------
# The frozen slice
# ---------------------------------------------------------------------------

def test_loader_selects_the_frozen_slice(tmp_path):
    rows, provenance = load_rows(build_run_dir(tmp_path), CFG)

    assert provenance["slice"] == {
        "pool": GEOM["primary_pool"],
        "block_index": GEOM["primary_layer"],
        "arm": GEOM["primary_prompt_arm"],
        "blocks": sorted(GEOM["primary_blocks"]),
    }
    assert provenance["n_roles"] == N_ROLES
    assert provenance["n_questions"] == 2 * N_QUESTIONS_PER_BLOCK
    assert rows.X.shape == (len(rows), HIDDEN)
    # Default rows carry the synthetic condition group, never a role name.
    assert all(r.startswith(DEFAULT_ROLE_PREFIX) for r in rows.default_roles)


def test_loader_ignores_the_wrong_layer_and_pool(tmp_path):
    off_layer = CFG["activation_extraction"]["middle_layer"][
        "off_by_one_sensitivity_block_index"]
    with pytest.raises(SystemExit, match="no rows survived"):
        load_rows(build_run_dir(tmp_path, block_index=off_layer), CFG)

    other = CFG["activation_extraction"]["sensitivity_pools"][0]
    with pytest.raises(SystemExit, match="no rows survived"):
        load_rows(build_run_dir(tmp_path / "b", pool=other), CFG)


def test_loader_requires_exactly_the_frozen_default_conditions(tmp_path):
    """The frozen default vector weights all five conditions equally, so a
    slice missing one would silently change the Axis it is meant to test."""
    with pytest.raises(SystemExit, match="exactly 5 default"):
        load_rows(build_run_dir(tmp_path, n_questions=4), CFG)


# ---------------------------------------------------------------------------
# Judge region (T13 stores two per generation)
# ---------------------------------------------------------------------------

def test_loader_reads_only_the_primary_judge_region(tmp_path):
    """The sensitivity region claims score 0 for every response while the
    primary claims 3. A loader keyed on the generation row alone would take
    whichever landed last and retain nothing."""
    rows, provenance = load_rows(
        build_run_dir(tmp_path, role_score=3, sensitivity_score=0), CFG)

    assert provenance["judge_region"] == PRIMARY_JUDGE_REGION
    assert provenance["judge_regions_present"][SENSITIVITY_REGION] > 0
    assert provenance["n_roles"] == N_ROLES, (
        "the primary region retained every role; reading the sensitivity "
        "region would have dropped them all"
    )
    assert len(rows.role_roles) == N_ROLES


def test_loader_rejects_duplicate_primary_measurements(tmp_path):
    with pytest.raises(SystemExit, match="more than one .* judge measurement"):
        load_rows(build_run_dir(tmp_path, duplicate_primary=True), CFG)


def test_loader_requires_the_region_tag(tmp_path):
    with pytest.raises(SystemExit, match="judge_region"):
        load_rows(build_run_dir(tmp_path, drop_judge_region=True), CFG)


def test_loader_keeps_only_score_three_role_rows(tmp_path):
    # Half the roles score 2 ("still an AI but some role attributes"), which
    # the frozen method excludes from role vectors; the rest score 3.
    partial = {r: 2 for r in range(N_ROLES // 2)}
    eligible = [f"role_{r:02d}" for r in range(N_ROLES // 2, N_ROLES)]
    rows, provenance = load_rows(
        build_run_dir(tmp_path, score_by_role=partial, eligible_roles=eligible),
        CFG)

    assert provenance["n_roles"] == N_ROLES - len(partial)
    assert all(int(r.split("_")[1]) >= N_ROLES // 2 for r in rows.role_roles)
    assert provenance["counts"]["kept_default"] > 0


def test_loader_drops_technically_invalid_rows(tmp_path):
    with pytest.raises(SystemExit, match="one class is empty|no retained rows"):
        load_rows(build_run_dir(tmp_path, validity="truncated"), CFG)


# ---------------------------------------------------------------------------
# Frozen eligibility artifact
# ---------------------------------------------------------------------------

def test_loader_requires_the_eligibility_artifact(tmp_path):
    run = build_run_dir(tmp_path)
    (run / "eligible_roles.json").unlink()
    with pytest.raises(SystemExit, match="eligible_roles.json is missing"):
        load_rows(run, CFG)


def test_loader_consumes_eligibility_rather_than_recomputing_it(tmp_path):
    """Roles outside the frozen set are dropped even though their rows are
    valid and score 3, so T18 cannot diagnose a role set that T15/G2 never
    described."""
    eligible = [f"role_{r:02d}" for r in range(N_ROLES - 3)]
    rows, provenance = load_rows(
        build_run_dir(tmp_path, eligible_roles=eligible), CFG)

    assert provenance["n_roles"] == len(eligible)
    assert provenance["counts"]["role_not_eligible"] > 0
    assert set(rows.role_roles) == set(eligible)


def test_loader_binds_the_eligibility_artifact_hash(tmp_path):
    run = build_run_dir(tmp_path)
    _, provenance = load_rows(run, CFG)
    expected = hashlib.sha256((run / "eligible_roles.json").read_bytes()).hexdigest()

    assert provenance["eligibility"]["eligible_roles_sha256"] == expected
    assert provenance["eligibility"]["n_eligible_roles_declared"] == N_ROLES
    assert provenance["eligibility"]["source_task"] == "T15"


def test_loader_stops_when_an_eligible_role_has_no_rows(tmp_path):
    """A frozen set naming a role the run directory never produced means the
    two artifacts describe different data, which must stop the run."""
    with pytest.raises(SystemExit, match="no retained rows"):
        load_rows(build_run_dir(tmp_path, extra_eligible_role="role_99"), CFG)


def test_loader_rejects_a_malformed_eligibility_artifact(tmp_path):
    run = build_run_dir(tmp_path)
    (run / "eligible_roles.json").write_text(
        json.dumps({"eligible_roles_primary": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="non-empty"):
        load_rows(run, CFG)

    run2 = build_run_dir(tmp_path / "dup")
    (run2 / "eligible_roles.json").write_text(
        json.dumps({"eligible_roles_primary": ["role_00", "role_00"]}),
        encoding="utf-8")
    with pytest.raises(SystemExit, match="duplicate roles"):
        load_rows(run2, CFG)


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

def test_loader_rejects_a_vector_that_does_not_match_its_hash(tmp_path):
    with pytest.raises(SystemExit, match="vector_sha256"):
        load_rows(build_run_dir(tmp_path, corrupt_vector=True), CFG)


def test_loader_reports_an_unknown_generation_reference(tmp_path):
    run = build_run_dir(tmp_path)
    path = run / "activations.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["generation_row_id"] = "f" * 16
    lines[0] = json.dumps(first, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    with pytest.raises(SystemExit, match="unknown generation row"):
        load_rows(run, CFG)


def test_loader_rejects_duplicate_row_ids(tmp_path):
    run = build_run_dir(tmp_path)
    path = run / "generation.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[0]]) + "\n",
                    encoding="utf-8", newline="\n")

    with pytest.raises(SystemExit, match="duplicate generation row_id"):
        load_rows(run, CFG)


def test_loader_rejects_rows_from_more_than_one_code_commit(tmp_path):
    run = build_run_dir(tmp_path)
    path = run / "activations.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["git_sha"] = "1" * 40
    lines[0] = json.dumps(row, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    with pytest.raises(SystemExit, match="more than one code commit"):
        load_rows(run, CFG)


def test_loaded_rows_survive_the_container_invariants(tmp_path):
    """The loader's output must satisfy every invariant the analysis assumes,
    including the label/name correspondence that the grouping relies on and
    the role-balanced weights the estimand uses."""
    rows, _ = load_rows(build_run_dir(tmp_path), CFG)
    assert set(np.unique(rows.y)) == {0, 1}
    for role in rows.role_roles:
        labels = set(rows.y[rows.roles == role])
        assert labels == {0}, f"role {role} appears with mixed labels"
    assert len(rows.default_roles) >= 1
    assert np.isclose(rows.weights[rows.y == 0].sum(),
                      rows.weights[rows.y == 1].sum())
