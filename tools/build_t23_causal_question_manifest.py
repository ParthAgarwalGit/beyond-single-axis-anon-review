#!/usr/bin/env python3
"""Freeze the T23 harmful-question source map and causal-judge schema.

T23 has two jobs:

1. Source mapping. Freeze the project-authored harmful-behaviour questions,
   their harm categories, their provenance, their target judge action, and
   their content hashes. Preserve Lu et al.'s five published introspective
   questions verbatim as the source-anchored comparison set. This half does
   not depend on P50 and is what "READY AFTER SOURCE MAPPING" refers to.

2. Causal-unit assembly (structural). Each causal unit is one role with its
   four causal role prompts crossed with the five harmful questions, carrying
   the prompt and question hashes. The concrete 50-role enumeration needs the
   frozen P50 set (T22, still blocked on G1/T15/G2). Until P50 exists this
   script validates the assembly against a supplied role list (e.g. --roles
   for a dry run) and refuses to emit a FROZEN 50-role manifest.

Everything frozen for new causal decisions is read from
configs/method_frozen_v4.yaml and never from a runtime flag: the causal prompt
indices, the five steering conditions, the P50 size and the literal Assistant
role ID all come from the downstream config, per
implementation_contract.runtime_overrides_for_frozen_fields_forbidden. The
historical configs/method_frozen.yaml is provenance-only. A FROZEN units
manifest can only be produced by passing the verified T22 P50 manifest, whose
SHA-256 must match the downstream frozen config.

Outputs:
  design/causal_questions_frozen.json         (source map + hashes)
  design/causal_units_manifest.json           (per-role units; DRAFT until P50)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

JUDGE_OBJECT_KEYS = frozenset({"harmfulness", "identity", "rationale"})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_sha256(text: str) -> str:
    return content_sha256(text)[:16]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def repo_path(path: Path | str, root: Path | None = None) -> str:
    """Repository-relative POSIX path, so frozen artifacts are byte-identical
    on Windows and Linux. Falls back to the plain POSIX form when the path is
    outside the repository (e.g. a --out-dir under a temp directory)."""
    p = Path(path)
    if root is not None:
        try:
            p = p.relative_to(root)
        except ValueError:
            pass
    return p.as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# frozen-config readers — no runtime override path exists for any of these
# --------------------------------------------------------------------------

def load_causal_prompt_indices(cfg: dict[str, Any]) -> tuple[int, ...]:
    """Read the frozen causal prompt indices.

    method_frozen_v4.yaml fixes both how many causal prompts each role gets
    and which source prompt indices they are (DEV-2026-08-10-T23-01). There is
    deliberately no flag to override this: implementation_contract forbids
    runtime overrides of frozen fields.
    """
    sel = cfg.get("causal_prompt_selection")
    if not isinstance(sel, dict):
        raise ValueError(
            "method_frozen_v4.yaml has no causal_prompt_selection block; the causal "
            "prompt indices must be frozen before causal units can be assembled"
        )
    if sel.get("runtime_override_forbidden") is not True:
        raise ValueError(
            "causal_prompt_selection.runtime_override_forbidden must be true"
        )
    indices = sel.get("project_prompt_indices")
    if not isinstance(indices, list) or not all(
        isinstance(i, int) and not isinstance(i, bool) for i in indices
    ):
        raise ValueError(
            "causal_prompt_selection.project_prompt_indices must be a list of ints"
        )
    n_expected = sel.get("n_causal_prompts_per_role")
    if n_expected != len(indices):
        raise ValueError(
            f"causal_prompt_selection declares {n_expected} prompts per role but "
            f"lists {len(indices)} indices"
        )
    if len(set(indices)) != len(indices):
        raise ValueError(f"causal prompt indices must be unique, got {indices}")
    if list(indices) != sorted(indices):
        raise ValueError(f"causal prompt indices must be ascending, got {indices}")
    if any(i < 0 for i in indices):
        raise ValueError(f"causal prompt indices must be non-negative, got {indices}")
    return tuple(indices)


def load_unique_conditions(cfg: dict[str, Any]) -> list[str]:
    """Read the five frozen steering conditions rather than hard-coding 5."""
    conditions = cfg["steering"]["unique_conditions"]
    if not isinstance(conditions, list) or not all(
        isinstance(c, str) for c in conditions
    ):
        raise ValueError("steering.unique_conditions must be a list of strings")
    if len(set(conditions)) != len(conditions):
        raise ValueError(f"steering.unique_conditions has duplicates: {conditions}")
    if not conditions:
        raise ValueError("steering.unique_conditions is empty")
    return list(conditions)


def validate_judge_schema(schema: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Fail if the causal judge labels drift from the frozen config.

    Returns the validated `distinct_from_role_judge` value so callers can
    propagate exactly what they verified instead of hard-coding true.
    """
    outcomes = cfg["causal_outcomes"]
    fields = schema["fields"]
    harm = list(fields["harmfulness"]["labels"])
    ident = list(fields["identity"]["labels"])
    if harm != list(outcomes["harmfulness_labels"]):
        raise ValueError(
            f"harmfulness labels drift: {harm} != {list(outcomes['harmfulness_labels'])}"
        )
    if ident != list(outcomes["identity_labels"]):
        raise ValueError(
            f"identity labels drift: {ident} != {list(outcomes['identity_labels'])}"
        )
    # The role judge must not be reused as the causal judge. This is a boolean
    # contract: false, "", a note string and any other non-True value all fail.
    distinct = schema.get("distinct_from_role_judge")
    if distinct is not True:
        raise ValueError(
            "judge schema field distinct_from_role_judge must be exactly true, "
            f"got {distinct!r}; the 0-3 role judge may never be reused as the "
            "causal judge"
        )
    # The parse contract is the JSON-object contract, in both files.
    parse_rules = schema.get("parse_rules", {})
    if parse_rules.get("parse_failure_is_missing_measurement") is not True:
        raise ValueError("parse_rules.parse_failure_is_missing_measurement must be true")
    if parse_rules.get("never_convert_parse_failure_to_an_outcome_label") is not True:
        raise ValueError(
            "parse_rules.never_convert_parse_failure_to_an_outcome_label must be true"
        )
    if "persistent_parse_failure_treatment" not in parse_rules:
        raise ValueError(
            "parse_rules must define persistent_parse_failure_treatment: a valid "
            "output whose judge response never parses is a coverage failure, and "
            "its denominator treatment cannot be left implicit"
        )
    return distinct


def parse_causal_judge_output(
    raw: str, schema: dict[str, Any]
) -> tuple[dict[str, str] | None, str | None]:
    """Reference implementation of the frozen judge parse contract.

    Returns (labels, None) on success, or (None, reason) on a parse failure.
    A parse failure is a MISSING MEASUREMENT: this function never falls back to
    a label, and callers must never coerce one.
    """
    harm_labels = set(schema["fields"]["harmfulness"]["labels"])
    ident_labels = set(schema["fields"]["identity"]["labels"])
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, "unparseable JSON"
    if not isinstance(obj, dict):
        return None, "not a single JSON object"
    if set(obj) != JUDGE_OBJECT_KEYS:
        missing = sorted(JUDGE_OBJECT_KEYS - set(obj))
        extra = sorted(set(obj) - JUDGE_OBJECT_KEYS)
        return None, f"key set mismatch (missing={missing}, extra={extra})"
    for field, allowed in (("harmfulness", harm_labels), ("identity", ident_labels)):
        value = obj[field]
        if not isinstance(value, str):
            return None, f"{field} is not a string"
        if value.strip() not in allowed:
            return None, f"{field} label {value!r} is outside the frozen set"
    if not isinstance(obj["rationale"], str):
        return None, "rationale is not a string"
    return {
        "harmfulness": obj["harmfulness"].strip(),
        "identity": obj["identity"].strip(),
        "rationale": obj["rationale"],
    }, None


# --------------------------------------------------------------------------
# P50 verification
# --------------------------------------------------------------------------

def verify_p50(
    manifest_path: Path,
    manifest: dict[str, Any],
    cfg: dict[str, Any],
    role_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Verify that the supplied artifact IS the frozen T22 P50 set.

    Freezing the causal design is only allowed against the real P50 manifest,
    matched by hash. This is an assertion about a verified artifact, never a
    switch that turns any 50 strings into a FROZEN manifest. Returns the
    provenance block to stamp into the units manifest.
    """
    sel = cfg["causal_role_selection"]
    expected_size = sel.get("P50_size")
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise ValueError("causal_role_selection.P50_size must be a positive int")

    expected_sha = sel.get("P50_manifest_sha256")
    if not expected_sha:
        raise ValueError(
            "causal_role_selection.P50_manifest_sha256 is still null: T22 has not "
            "frozen P50, so no FROZEN causal-units manifest may be emitted"
        )
    observed_sha = sha256_file(manifest_path)
    if observed_sha != expected_sha:
        raise ValueError(
            f"P50 manifest hash mismatch: {observed_sha} != frozen {expected_sha}"
        )

    if manifest.get("status") != "FROZEN":
        raise ValueError(
            f"P50 manifest status is {manifest.get('status')!r}, expected FROZEN"
        )
    if manifest.get("set_name") != sel.get("primary_set_name"):
        raise ValueError(
            f"P50 manifest set_name {manifest.get('set_name')!r} != "
            f"{sel.get('primary_set_name')!r}"
        )

    p50_ids = manifest.get("role_ids")
    if not isinstance(p50_ids, list) or not all(isinstance(r, str) for r in p50_ids):
        raise ValueError("P50 manifest role_ids must be a list of role-id strings")
    if len(p50_ids) != expected_size:
        raise ValueError(
            f"P50 manifest has {len(p50_ids)} roles, expected exactly {expected_size}"
        )
    if len(set(p50_ids)) != len(p50_ids):
        dupes = sorted({r for r in p50_ids if p50_ids.count(r) > 1})
        raise ValueError(f"P50 manifest has duplicate role IDs: {dupes}")

    literal_assistant = sel.get("literal_assistant_role_id")
    if not literal_assistant:
        raise ValueError(
            "causal_role_selection.literal_assistant_role_id must be set so the "
            "literal Assistant role can be excluded"
        )
    if literal_assistant in p50_ids:
        raise ValueError(
            f"P50 contains the literal Assistant role {literal_assistant!r}; "
            "causal_role_selection.selection_rule excludes it"
        )

    if role_ids is not None and list(role_ids) != list(p50_ids):
        raise ValueError(
            "--roles does not match the verified P50 manifest exactly; drop "
            "--roles or supply the P50 role list unchanged and in order"
        )

    return {
        "source": "T22 frozen P50 manifest",
        "path": repo_path(manifest_path.name),
        "sha256": observed_sha,
        "matches_frozen_config_hash": True,
        "set_name": manifest["set_name"],
        "n_roles": len(p50_ids),
        "literal_assistant_excluded": True,
        "role_ids_sha256": canonical_sha256(p50_ids),
        "role_ids": list(p50_ids),
    }


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def build_source_map(
    questions_src: dict[str, Any],
    judge_schema: dict[str, Any],
    cfg: dict[str, Any],
    config_sha: str,
    src_paths: dict[str, tuple[Path, str]],
) -> dict[str, Any]:
    distinct_from_role_judge = validate_judge_schema(judge_schema, cfg)

    rows = questions_src["questions"]
    taxonomy = list(questions_src["harm_category_taxonomy"])

    if len(rows) != 5:
        raise ValueError(f"expected 5 harmful questions, found {len(rows)}")
    categories = [r["harm_category"] for r in rows]
    if len(set(categories)) != len(categories):
        raise ValueError(f"harm categories must be disjoint, got {categories}")
    if set(categories) != set(taxonomy):
        raise ValueError("harm categories must exactly cover the declared taxonomy")
    if len(taxonomy) != len(set(taxonomy)):
        raise ValueError("harm-category taxonomy has duplicates")

    frozen_questions = []
    seen_ids: set[str] = set()
    for r in rows:
        qid = r["harmful_question_id"]
        if qid in seen_ids:
            raise ValueError(f"duplicate harmful_question_id {qid}")
        seen_ids.add(qid)
        text = r["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{qid}: empty question text")
        if r["judge_target_action"] != "refuse":
            raise ValueError(f"{qid}: target action must be refuse")
        # Provenance is enforced in the production builder, not only in tests:
        # these questions must never be attributable to Lu et al.
        if r["provenance"] != "PROJECT_AUTHORED":
            raise ValueError(
                f"{qid}: provenance must be PROJECT_AUTHORED, got {r['provenance']!r}"
            )
        frozen_questions.append({
            "harmful_question_id": qid,
            "harm_category": r["harm_category"],
            "text": text,
            "content_sha256": content_sha256(text),
            "content_sha256_short": short_sha256(text),
            "character_count": len(text),
            "word_count": len(text.split()),
            "judge_target_action": r["judge_target_action"],
            "provenance": r["provenance"],
        })

    lu_intro = questions_src["lu_introspective_questions_preserved"]
    lu_questions = [
        {"text": q, "content_sha256": content_sha256(q)}
        for q in lu_intro["questions"]
    ]

    # T23 freezes category, provenance and target action as well as text, so the
    # set hash binds the complete frozen record, not just (id, text_hash).
    set_hash_payload = [
        {
            "harmful_question_id": q["harmful_question_id"],
            "harm_category": q["harm_category"],
            "content_sha256": q["content_sha256"],
            "judge_target_action": q["judge_target_action"],
            "provenance": q["provenance"],
        }
        for q in frozen_questions
    ]

    source_map = {
        "schema_version": "t23-causal-questions-frozen/1.1",
        "task": "T23",
        "title": "Frozen harmful-question source map and causal-judge schema binding",
        "status": "REVIEW_CANDIDATE",
        "method_config": {
            "path": "configs/method_frozen_v4.yaml",
            "sha256": config_sha,
            "freeze_status": cfg.get("freeze_status"),
        },
        "input_artifacts": {
            name: {"path": repo_path(p), "sha256": sha}
            for name, (p, sha) in src_paths.items()
        },
        "harm_category_taxonomy": taxonomy,
        "n_harmful_questions": len(frozen_questions),
        "harmful_questions": frozen_questions,
        "harmful_questions_set_sha256": canonical_sha256(set_hash_payload),
        "harmful_questions_set_sha256_covers": [
            "harmful_question_id", "harm_category", "content_sha256",
            "judge_target_action", "provenance",
        ],
        "lu_introspective_questions": {
            "evidence_class": lu_intro["evidence_class"],
            "usage_rule": lu_intro["usage_rule"],
            "status_in_this_project": lu_intro["status_in_this_project"],
            "questions": lu_questions,
        },
        "causal_prompt_selection": {
            "indices": list(load_causal_prompt_indices(cfg)),
            "source": "configs/method_frozen_v4.yaml causal_prompt_selection",
            "evidence_class": cfg["causal_prompt_selection"]["evidence_class"],
            "classification": cfg["causal_prompt_selection"]["classification"],
            "runtime_override_forbidden": True,
            "deviation_record": "DEV-2026-08-10-T23-01",
        },
        "causal_judge_schema": {
            "path": repo_path(src_paths["judge_schema"][0]),
            "sha256": src_paths["judge_schema"][1],
            "harmfulness_labels": list(judge_schema["fields"]["harmfulness"]["labels"]),
            "identity_labels": list(judge_schema["fields"]["identity"]["labels"]),
            "distinct_from_role_judge": distinct_from_role_judge,
            "validation_gates_status": judge_schema["validation_gates_placeholder"]["status"],
        },
        "causal_judge_prompt": {
            "path": repo_path(src_paths["judge_prompt"][0]),
            "sha256": src_paths["judge_prompt"][1],
            "binding_note": (
                "The frozen prompt text is hash-bound here because T27 validates two "
                "independent humans against the EXACT causal prompt. Editing the prompt "
                "changes this hash and invalidates the frozen T23 source map."
            ),
        },
        "provenance_note": (
            "Harmful questions are PROJECT_AUTHORED and must never be attributed "
            "to Lu et al. The five introspective questions are VERIFIED_FROM_PAPER "
            "and used verbatim as an additional adapted set, never a substitution."
        ),
    }
    return source_map


def assemble_causal_units(
    source_map: dict[str, Any],
    role_ids: list[str],
    role_prompt_payload: dict[str, Any],
    cfg: dict[str, Any],
    source_map_sha256: str,
    p50_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble per-role causal units.

    A FROZEN manifest is emitted only when `p50_provenance` is the block
    returned by `verify_p50`. There is no other route to FROZEN: passing 50
    arbitrary role IDs produces a DRAFT dry run.
    """
    causal_prompt_indices = load_causal_prompt_indices(cfg)
    conditions = load_unique_conditions(cfg)

    if not role_ids:
        raise ValueError("no roles supplied")
    if len(set(role_ids)) != len(role_ids):
        dupes = sorted({r for r in role_ids if role_ids.count(r) > 1})
        raise ValueError(f"duplicate role IDs in the causal role list: {dupes}")

    is_p50_final = p50_provenance is not None
    if is_p50_final and list(role_ids) != list(p50_provenance["role_ids"]):
        raise ValueError("role list does not match the verified P50 provenance block")

    prompts = role_prompt_payload["prompts"]
    by_role_index: dict[tuple[str, int], dict[str, Any]] = {}
    for row in prompts:
        by_role_index[(row["role_id"], row["prompt_index"])] = row

    harmful = source_map["harmful_questions"]
    units = []
    for role_id in role_ids:
        role_prompts = []
        for idx in causal_prompt_indices:
            key = (role_id, idx)
            if key not in by_role_index:
                raise ValueError(f"role {role_id!r} has no prompt index {idx}")
            p = by_role_index[key]
            # Verify the source prompt hash before trusting the text.
            if short_sha256(p["text"]) != p["content_sha256"]:
                raise ValueError(f"role-prompt hash mismatch for {key}")
            role_prompts.append({
                "prompt_uid": p["prompt_uid"],
                "prompt_index": idx,
                "content_sha256": content_sha256(p["text"]),
            })
        unit = {
            "role_id": role_id,
            "causal_prompt_indices": list(causal_prompt_indices),
            "role_prompts": role_prompts,
            "harmful_questions": [
                {
                    "harmful_question_id": q["harmful_question_id"],
                    "harm_category": q["harm_category"],
                    "content_sha256": q["content_sha256"],
                    "judge_target_action": q["judge_target_action"],
                }
                for q in harmful
            ],
            "n_cells_pre_condition": len(role_prompts) * len(harmful),
        }
        unit["unit_sha256"] = canonical_sha256({
            "role_id": role_id,
            "prompts": [rp["content_sha256"] for rp in role_prompts],
            "questions": [q["content_sha256"] for q in harmful],
        })
        units.append(unit)

    n_conditions = len(conditions)
    n_prompts = len(causal_prompt_indices)
    n_questions = len(harmful)
    expected_outputs = len(units) * n_prompts * n_questions * n_conditions

    manifest = {
        "schema_version": "t23-causal-units/1.1",
        "task": "T23",
        "status": "FROZEN" if is_p50_final else "DRAFT_PENDING_P50",
        "p50_dependency": (
            "The concrete 50-role set is frozen by T22 (blocked on G1/T15/G2). "
            "Until then this manifest is a structural dry run over the supplied "
            "role list and must not be treated as the frozen causal design."
        ),
        "p50_provenance": p50_provenance or {
            "source": "UNVERIFIED_DRY_RUN_ROLE_LIST",
            "note": "Not the frozen P50 set; this manifest is structural only.",
        },
        "method_config_sha256": source_map["method_config"]["sha256"],
        "source_map_sha256": source_map_sha256,
        "causal_prompt_selection": dict(source_map["causal_prompt_selection"]),
        "n_roles": len(units),
        "n_causal_prompts_per_role": n_prompts,
        "n_harmful_questions": n_questions,
        "n_unique_conditions": n_conditions,
        "unique_conditions": conditions,
        "unique_conditions_sha256": canonical_sha256(conditions),
        "expected_output_count": expected_outputs,
        "harmful_questions_set_sha256": source_map["harmful_questions_set_sha256"],
        "units_sha256": canonical_sha256([u["unit_sha256"] for u in units]),
        "units": units,
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/method_frozen_v4.yaml")
    parser.add_argument(
        "--questions-source", default="design/causal_questions_source.json"
    )
    parser.add_argument(
        "--judge-schema", default="design/causal_judge_schema.json"
    )
    parser.add_argument(
        "--judge-prompt", default="prompts/causal_judge_frozen.md"
    )
    parser.add_argument("--role-prompts", default="data/lu_et_al/role_prompts.json")
    parser.add_argument(
        "--roles",
        default=None,
        help="Optional JSON file with a role-id list for a structural dry run "
             "of causal-unit assembly (stands in for P50 until T22 freezes it). "
             "Always produces a DRAFT manifest.",
    )
    parser.add_argument(
        "--p50-manifest",
        default=None,
        help="Path to the frozen T22 P50 manifest. Required for --p50-final; "
             "its SHA-256 must equal causal_role_selection.P50_manifest_sha256.",
    )
    parser.add_argument(
        "--p50-final",
        action="store_true",
        help="Assert that the verified frozen P50 set was supplied via "
             "--p50-manifest. Emits a FROZEN causal-units manifest only if the "
             "manifest hash, size, uniqueness and Assistant exclusion all pass.",
    )
    parser.add_argument("--out-dir", default="design")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    import yaml

    config_path = root / args.config
    config_bytes = config_path.read_bytes()
    cfg = yaml.safe_load(config_bytes)
    config_sha = sha256_bytes(config_bytes)

    questions_path = root / args.questions_source
    judge_path = root / args.judge_schema
    judge_prompt_path = root / args.judge_prompt
    questions_src = load_json(questions_path)
    judge_schema = load_json(judge_path)

    src_paths = {
        "questions_source": (Path(args.questions_source), sha256_file(questions_path)),
        "judge_schema": (Path(args.judge_schema), sha256_file(judge_path)),
        "judge_prompt": (Path(args.judge_prompt), sha256_file(judge_prompt_path)),
    }
    source_map = build_source_map(
        questions_src, judge_schema, cfg, config_sha, src_paths
    )

    out_dir = root / args.out_dir
    source_map_path = out_dir / "causal_questions_frozen.json"
    write_json(source_map_path, source_map)
    source_map_sha = sha256_file(source_map_path)

    if args.p50_final and not args.p50_manifest:
        raise ValueError("--p50-final requires --p50-manifest (the frozen T22 artifact)")

    units_summary: dict[str, Any] = {"status": "SKIPPED_NO_ROLE_LIST"}
    role_ids: list[str] | None = None
    if args.roles:
        role_ids = load_json(root / args.roles)
        if not isinstance(role_ids, list) or not all(
            isinstance(r, str) for r in role_ids
        ):
            raise ValueError("--roles must be a JSON list of role-id strings")

    p50_provenance = None
    if args.p50_manifest:
        p50_path = root / args.p50_manifest
        p50_manifest = load_json(p50_path)
        if args.p50_final:
            p50_provenance = verify_p50(p50_path, p50_manifest, cfg, role_ids)
            role_ids = list(p50_provenance["role_ids"])
        else:
            print(
                "note: --p50-manifest supplied without --p50-final; not verifying, "
                "not freezing", file=sys.stderr,
            )

    if role_ids:
        role_prompts = load_json(root / args.role_prompts)
        manifest = assemble_causal_units(
            source_map, role_ids, role_prompts, cfg, source_map_sha, p50_provenance
        )
        units_path = out_dir / "causal_units_manifest.json"
        write_json(units_path, manifest)
        units_summary = {
            "status": manifest["status"],
            "path": repo_path(units_path, root),
            "sha256": sha256_file(units_path),
            "n_roles": manifest["n_roles"],
            "expected_output_count": manifest["expected_output_count"],
            "units_sha256": manifest["units_sha256"],
        }

    print(json.dumps({
        "status": "PASS",
        "source_map": repo_path(source_map_path, root),
        "source_map_sha256": source_map_sha,
        "method_config_sha256": config_sha,
        "n_harmful_questions": source_map["n_harmful_questions"],
        "harm_categories": source_map["harm_category_taxonomy"],
        "harmful_questions_set_sha256": source_map["harmful_questions_set_sha256"],
        "judge_schema_sha256": src_paths["judge_schema"][1],
        "judge_prompt_sha256": src_paths["judge_prompt"][1],
        "causal_prompt_indices": source_map["causal_prompt_selection"]["indices"],
        "causal_units": units_summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
