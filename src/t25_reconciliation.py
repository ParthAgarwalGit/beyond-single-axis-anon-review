"""T25 DeepSeek 5k signed-steering reconciliation helpers.

T25 is a finish/reconcile task, not permission to restart valid compute. This
module audits an existing final-output artifact against the frozen 50 x 4 x 5 x
5 design without inspecting causal outcomes.

The committed audit contains only outcome-blind inputs, hashes, and aggregate
counts. Raw generations, private maps, and item-level causal outcomes remain
outside git. Completion/output text is deliberately discarded at ingestion.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


EXPECTED_TOTAL = 5000
EXPECTED_P50 = 50
EXPECTED_PROMPTS = (0, 1, 2, 3)
EXPECTED_QUESTIONS = ("HQ1", "HQ2", "HQ3", "HQ4", "HQ5")
DOSE_ATOL = 1e-12

# Only these outcome-blind fields survive JSONL ingestion. In particular,
# completion/output_text, judge labels, rationales, and any downstream outcome
# fields are never copied into the in-memory reconciliation corpus.
SAFE_INPUT_FIELDS = frozenset(
    {
        "uid",
        "rollout_id",
        "role_id",
        "role",
        "harmful_question_id",
        "question_id",
        "prompt_index",
        "role_prompt_index",
        "steering_prompt_id",
        "condition_id",
        "condition",
        "technical_validity",
        "technical_status",
        "axis_dose",
        "random_dose",
        "model",
        "model_id",
    }
)


class T25ReconciliationError(ValueError):
    """Raised when an existing T25 artifact cannot be accepted fail-closed."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value.lower())
    )


def read_jsonl(path: Path) -> list[dict]:
    """Read a private JSONL while retaining only outcome-blind audit fields."""
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise T25ReconciliationError(
                    f"{path}: line {i} is not valid JSON: {exc}"
                ) from None
            if not isinstance(parsed, dict):
                raise T25ReconciliationError(f"{path}: line {i} is not a JSON object")
            # Project immediately; do not carry completion/output or judge fields.
            rows.append({k: parsed[k] for k in SAFE_INPUT_FIELDS if k in parsed})
    if not rows:
        raise T25ReconciliationError(f"{path}: no rows")
    return rows


def load_p50_manifest(path: Path, *, require_frozen: bool) -> tuple[list[str], dict, str]:
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if require_frozen and doc.get("status") != "FROZEN":
        raise T25ReconciliationError(
            f"P50 manifest is {doc.get('status')!r}, not FROZEN; T25 cannot be finalized"
        )
    selected = doc.get("selected_roles")
    if not isinstance(selected, list):
        raise T25ReconciliationError("P50 manifest has no selected_roles list")
    roles = []
    for rec in selected:
        if not isinstance(rec, dict) or not rec.get("role_id"):
            raise T25ReconciliationError("malformed selected_roles record in P50 manifest")
        roles.append(str(rec["role_id"]))
    if len(roles) != EXPECTED_P50 or len(set(roles)) != EXPECTED_P50:
        raise T25ReconciliationError(
            f"P50 must contain exactly {EXPECTED_P50} unique roles, got {len(roles)}"
        )

    literal_assistant = doc.get("literal_assistant_role_id")
    if not isinstance(literal_assistant, str) or not literal_assistant:
        raise T25ReconciliationError("P50 manifest does not expose literal_assistant_role_id")
    if literal_assistant in roles:
        raise T25ReconciliationError("literal Assistant role must not be in P50")

    declared_membership_sha = doc.get("membership_sha256")
    recomputed_membership_sha = canonical_sha256(roles)
    if declared_membership_sha != recomputed_membership_sha:
        raise T25ReconciliationError(
            "P50 membership_sha256 does not reproduce from selected_roles in rank order"
        )

    req = doc.get("heldout_axis_requirement") or {}
    if req.get("p50_must_be_excluded_from_t24_heldout_axis") is not True:
        raise T25ReconciliationError(
            "P50 manifest does not carry the held-out-axis exclusion requirement"
        )
    return roles, doc, sha256_file(path)


def load_harmful_questions(path: Path) -> tuple[list[str], dict[str, str], str]:
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    records = doc.get("harmful_questions")
    if not isinstance(records, list):
        raise T25ReconciliationError("causal question manifest has no harmful_questions list")
    ids = []
    texts = {}
    for rec in records:
        qid = rec.get("harmful_question_id") if isinstance(rec, dict) else None
        text = rec.get("text") if isinstance(rec, dict) else None
        if not qid or not isinstance(text, str):
            raise T25ReconciliationError("malformed harmful question record")
        qid = str(qid)
        ids.append(qid)
        texts[qid] = text
    if tuple(ids) != EXPECTED_QUESTIONS:
        raise T25ReconciliationError(
            f"frozen harmful question IDs changed: expected {EXPECTED_QUESTIONS}, got {tuple(ids)}"
        )
    return ids, texts, sha256_file(path)


def _load_mapping(path: Path | None, key: str) -> dict:
    if path is None:
        return {}
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise T25ReconciliationError(f"{path}: mapping file must be a JSON object")
    value = doc.get(key, doc)
    if not isinstance(value, dict):
        raise T25ReconciliationError(f"{path}: expected object {key!r}")
    return {str(k): v for k, v in value.items()}


def load_condition_map(
    path: Path | None, canonical_conditions: Sequence[str]
) -> tuple[dict[str, str], str | None]:
    mapping = _load_mapping(path, "condition_id_to_condition")
    if not mapping:
        return {}, None
    canonical = [str(x) for x in canonical_conditions]
    values = [str(v) for v in mapping.values()]
    if len(mapping) != len(canonical) or set(values) != set(canonical):
        raise T25ReconciliationError(
            "private condition map must be a one-to-one mapping onto all frozen conditions"
        )
    if len(set(values)) != len(values):
        raise T25ReconciliationError("condition map is not bijective")
    return {k: str(v) for k, v in mapping.items()}, sha256_file(Path(path))


def load_prompt_map(
    path: Path | None, prompt_indices: Sequence[int]
) -> tuple[dict[str, int], str | None]:
    mapping = _load_mapping(path, "steering_prompt_id_to_index")
    if not mapping:
        return {}, None
    out = {}
    for key, value in mapping.items():
        try:
            idx = int(value)
        except (TypeError, ValueError):
            raise T25ReconciliationError(
                f"prompt map value for {key!r} is not an integer"
            ) from None
        if idx not in prompt_indices:
            raise T25ReconciliationError(
                f"prompt map value {idx} is outside frozen prompt indices"
            )
        out[key] = idx
    return out, sha256_file(Path(path))


def validate_external_location(location: str | None, *, required: bool) -> str | None:
    """Validate a stable external locator without accepting a local filesystem path."""
    if location is None:
        if required:
            raise T25ReconciliationError(
                "final T25 requires --artifact-location (for example HF repo@revision/path or Drive ID)"
            )
        return None
    value = location.strip()
    if not value:
        raise T25ReconciliationError("artifact location is empty")
    lowered = value.lower()
    if value.startswith(("/", "./", "../", "~")) or lowered.startswith("file://"):
        raise T25ReconciliationError(
            "artifact location must be a stable external locator, not a machine-local filesystem path"
        )
    return value


def _first(row: Mapping, names: Sequence[str]):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _numeric(value, *, field: str, uid: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise T25ReconciliationError(f"T25 row {uid!r} has non-numeric {field}")
    value = float(value)
    if not math.isfinite(value):
        raise T25ReconciliationError(f"T25 row {uid!r} has non-finite {field}")
    return value


def _zero(value: float) -> bool:
    return math.isclose(value, 0.0, rel_tol=0.0, abs_tol=DOSE_ATOL)


def validate_dose(condition: str, axis_dose: float, random_dose: float, *, uid: str) -> float | None:
    """Enforce the V4 two-signed-dose-column contract and return |a| if nonzero."""
    if condition == "shared_zero":
        ok = _zero(axis_dose) and _zero(random_dose)
        alpha = None
    elif condition == "assistant_axis_toward":
        ok = axis_dose > DOSE_ATOL and _zero(random_dose)
        alpha = abs(axis_dose)
    elif condition == "assistant_axis_away":
        ok = axis_dose < -DOSE_ATOL and _zero(random_dose)
        alpha = abs(axis_dose)
    elif condition == "random_positive":
        ok = random_dose > DOSE_ATOL and _zero(axis_dose)
        alpha = abs(random_dose)
    elif condition == "random_negative":
        ok = random_dose < -DOSE_ATOL and _zero(axis_dose)
        alpha = abs(random_dose)
    else:
        raise T25ReconciliationError(f"T25 row {uid!r} has unknown condition {condition!r}")
    if not ok:
        raise T25ReconciliationError(
            f"T25 row {uid!r} dose columns axis={axis_dose}, random={random_dose} "
            f"are inconsistent with {condition}"
        )
    return alpha


def normalize_row(
    row: Mapping,
    *,
    condition_map: Mapping[str, str],
    prompt_map: Mapping[str, int],
    canonical_conditions: Sequence[str],
    question_ids: Sequence[str],
    allowed_technical_statuses: Sequence[str],
    expected_model: str | None,
) -> dict:
    uid = _first(row, ("uid", "rollout_id"))
    role = _first(row, ("role_id", "role"))
    qid = _first(row, ("harmful_question_id", "question_id"))
    prompt = _first(row, ("prompt_index", "role_prompt_index"))
    condition_id = row.get("condition_id")
    condition = row.get("condition")

    if uid is None or not str(uid):
        raise T25ReconciliationError("T25 row missing uid/rollout_id")
    uid = str(uid)
    if role is None or not str(role):
        raise T25ReconciliationError(f"T25 row {uid!r} missing role_id/role")
    if qid is None:
        raise T25ReconciliationError(
            f"T25 row {uid!r} missing harmful_question_id/question_id"
        )
    qid = str(qid)
    if qid not in question_ids:
        raise T25ReconciliationError(
            f"T25 row {uid!r} has unknown harmful question {qid!r}"
        )

    if prompt is None:
        steering_prompt_id = row.get("steering_prompt_id")
        if steering_prompt_id is None or str(steering_prompt_id) not in prompt_map:
            raise T25ReconciliationError(
                f"T25 row {uid!r} has no prompt_index/role_prompt_index and its "
                "steering_prompt_id is not resolved by --prompt-map"
            )
        prompt = prompt_map[str(steering_prompt_id)]
    try:
        prompt = int(prompt)
    except (TypeError, ValueError):
        raise T25ReconciliationError(
            f"T25 row {uid!r} has non-integer prompt index"
        ) from None
    if prompt not in EXPECTED_PROMPTS:
        raise T25ReconciliationError(
            f"T25 row {uid!r} has prompt index {prompt}, expected 0..3"
        )

    if condition is None:
        if condition_id is None or str(condition_id) not in condition_map:
            raise T25ReconciliationError(
                f"T25 row {uid!r} has no canonical condition and its condition_id "
                "is not in --condition-map"
            )
        condition = condition_map[str(condition_id)]
    condition = str(condition)
    if condition not in canonical_conditions:
        raise T25ReconciliationError(
            f"T25 row {uid!r} has unknown condition {condition!r}"
        )
    if condition_id is not None and condition_map:
        mapped = condition_map.get(str(condition_id))
        if mapped is None or mapped != condition:
            raise T25ReconciliationError(
                f"T25 row {uid!r} has inconsistent condition/condition_id mapping"
            )

    technical = _first(row, ("technical_validity", "technical_status"))
    if technical is None:
        raise T25ReconciliationError(f"T25 row {uid!r} has no final technical status")
    technical = str(technical)
    allowed = {str(x) for x in allowed_technical_statuses}
    if technical not in allowed:
        raise T25ReconciliationError(
            f"T25 row {uid!r} technical status {technical!r} is not in the frozen technical-validity vocabulary"
        )

    axis_dose = _numeric(row.get("axis_dose"), field="axis_dose", uid=uid)
    random_dose = _numeric(row.get("random_dose"), field="random_dose", uid=uid)
    alpha = validate_dose(condition, axis_dose, random_dose, uid=uid)

    row_model = _first(row, ("model", "model_id"))
    if row_model is not None:
        row_model = str(row_model)
        if expected_model is not None and row_model != expected_model:
            raise T25ReconciliationError(
                f"T25 row {uid!r} model {row_model!r} != frozen primary model {expected_model!r}"
            )

    return {
        "uid": uid,
        "role_id": str(role),
        "prompt_index": prompt,
        "harmful_question_id": qid,
        "condition": condition,
        "condition_id": None if condition_id is None else str(condition_id),
        "technical_status": technical,
        "axis_dose": axis_dose,
        "random_dose": random_dose,
        "nonzero_alpha": alpha,
        "model": row_model,
    }


def unit_key(row: Mapping) -> tuple[str, int, str, str]:
    return (
        str(row["role_id"]),
        int(row["prompt_index"]),
        str(row["harmful_question_id"]),
        str(row["condition"]),
    )


def expected_unit_keys(
    roles: Sequence[str],
    prompt_indices: Sequence[int],
    question_ids: Sequence[str],
    conditions: Sequence[str],
) -> set[tuple[str, int, str, str]]:
    return set(itertools.product(roles, prompt_indices, question_ids, conditions))


def reconcile_rows(
    rows: Iterable[Mapping],
    *,
    roles: Sequence[str],
    prompt_indices: Sequence[int],
    question_ids: Sequence[str],
    conditions: Sequence[str],
    allowed_technical_statuses: Sequence[str],
    expected_model: str | None,
    condition_map: Mapping[str, str] | None = None,
    prompt_map: Mapping[str, int] | None = None,
) -> tuple[list[dict], dict]:
    condition_map = condition_map or {}
    prompt_map = prompt_map or {}
    normalized = [
        normalize_row(
            row,
            condition_map=condition_map,
            prompt_map=prompt_map,
            canonical_conditions=conditions,
            question_ids=question_ids,
            allowed_technical_statuses=allowed_technical_statuses,
            expected_model=expected_model,
        )
        for row in rows
    ]
    expected = expected_unit_keys(roles, prompt_indices, question_ids, conditions)
    observed_keys = [unit_key(r) for r in normalized]
    observed_set = set(observed_keys)
    uids = [r["uid"] for r in normalized]

    duplicates = [key for key, n in Counter(observed_keys).items() if n > 1]
    duplicate_uids = [uid for uid, n in Counter(uids).items() if n > 1]
    missing = sorted(expected - observed_set)
    unexpected = sorted(observed_set - expected)

    problems = []
    if len(normalized) != len(expected):
        problems.append(f"{len(normalized)} final rows, expected {len(expected)}")
    if duplicates:
        problems.append(f"{len(duplicates)} duplicate factorial cells")
    if duplicate_uids:
        problems.append(f"{len(duplicate_uids)} duplicate UIDs")
    if missing:
        problems.append(f"{len(missing)} missing factorial cells")
    if unexpected:
        problems.append(f"{len(unexpected)} unexpected factorial cells")
    if problems:
        raise T25ReconciliationError(
            "T25 corpus does not reconcile: " + "; ".join(problems)
        )

    nonzero_alphas = [r["nonzero_alpha"] for r in normalized if r["nonzero_alpha"] is not None]
    if not nonzero_alphas:
        raise T25ReconciliationError("no nonzero steering rows found")
    observed_alpha = nonzero_alphas[0]
    if any(
        not math.isclose(a, observed_alpha, rel_tol=1e-12, abs_tol=DOSE_ATOL)
        for a in nonzero_alphas[1:]
    ):
        raise T25ReconciliationError(
            "nonzero steering magnitude is not constant across Axis/random signed conditions"
        )

    per_condition = Counter(r["condition"] for r in normalized)
    per_role = Counter(r["role_id"] for r in normalized)
    per_prompt = Counter(r["prompt_index"] for r in normalized)
    per_question = Counter(r["harmful_question_id"] for r in normalized)
    status_counts = Counter(r["technical_status"] for r in normalized)
    n_model_stamped = sum(r["model"] is not None for r in normalized)

    report = {
        "n_rows": len(normalized),
        "n_unique_uids": len(set(uids)),
        "n_unique_cells": len(observed_set),
        "per_condition": dict(sorted(per_condition.items())),
        "per_role_min": min(per_role.values()),
        "per_role_max": max(per_role.values()),
        "per_prompt": {str(k): v for k, v in sorted(per_prompt.items())},
        "per_question": dict(sorted(per_question.items())),
        "technical_status_counts": dict(sorted(status_counts.items())),
        "observed_nonzero_coefficient": observed_alpha,
        "dose_contract_verified": True,
        "rows_with_model_identity": n_model_stamped,
        "model_identity_verified_where_present": True,
        "uid_set_sha256": canonical_sha256(sorted(uids)),
        "factorial_key_set_sha256": canonical_sha256(
            [list(x) for x in sorted(observed_set)]
        ),
        "role_set_sha256": canonical_sha256(sorted(per_role)),
        "complete_factorial": True,
    }
    return normalized, report


def validate_heldout_axis_provenance(
    path: Path,
    *,
    p50_roles: Sequence[str],
    p50_manifest_sha256: str,
    require_p50_binding: bool,
) -> dict:
    """Verify exclusion where the held-out Axis is actually constructed.

    A bare boolean is insufficient: T25 requires the provenance file to expose
    the role IDs used to build the causal Axis so overlap can be checked.
    Final reconciliation additionally requires an exact P50-manifest hash bind.
    """
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    construction = doc.get("construction_role_ids") or doc.get("included_role_ids")
    if not isinstance(construction, list) or not construction:
        raise T25ReconciliationError(
            "held-out Axis provenance must contain construction_role_ids (or included_role_ids)"
        )
    construction = [str(x) for x in construction]
    overlap = sorted(set(construction) & set(p50_roles))
    if overlap:
        raise T25ReconciliationError(
            f"held-out causal Axis includes {len(overlap)} P50 role(s): {overlap[:10]}"
        )

    bound = doc.get("p50_manifest_sha256")
    if require_p50_binding and not _is_sha256(bound):
        raise T25ReconciliationError(
            "final held-out Axis provenance must contain p50_manifest_sha256"
        )
    if bound is not None and bound != p50_manifest_sha256:
        raise T25ReconciliationError(
            "held-out Axis provenance is bound to a different P50 manifest"
        )

    axis_hash = doc.get("unit_axis_sha256") or doc.get("axis_sha256")
    if not _is_sha256(axis_hash):
        raise T25ReconciliationError(
            "held-out Axis provenance lacks a valid 64-hex Axis hash"
        )
    return {
        "provenance_sha256": sha256_file(path),
        "axis_sha256": axis_hash,
        "n_construction_roles": len(construction),
        "p50_overlap_count": 0,
        "p50_exclusion_verified": True,
        "p50_manifest_binding_verified": bound == p50_manifest_sha256,
    }


def validate_activation_manifest(
    path: Path, *, outputs_sha256: str, uid_set_sha256: str
) -> dict:
    """Lightweight provenance check for downstream-representation capture."""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    n_rows = doc.get("n_rows")
    if n_rows is not None and int(n_rows) != EXPECTED_TOTAL:
        raise T25ReconciliationError(
            f"activation manifest says {n_rows} rows, expected {EXPECTED_TOTAL}"
        )
    source_hash = doc.get("source_outputs_sha256")
    if source_hash is not None and source_hash != outputs_sha256:
        raise T25ReconciliationError(
            "activation manifest is bound to a different T25 output artifact"
        )
    uid_hash = doc.get("uid_set_sha256")
    if uid_hash is not None and uid_hash != uid_set_sha256:
        raise T25ReconciliationError(
            "activation manifest UID set differs from reconciled T25 outputs"
        )
    return {
        "manifest_sha256": sha256_file(path),
        "n_rows": n_rows,
        "source_outputs_sha256_verified": (
            source_hash == outputs_sha256 if source_hash else None
        ),
        "uid_set_sha256_verified": uid_hash == uid_set_sha256 if uid_hash else None,
    }
