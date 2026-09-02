"""Guard the immutable historical V3 config and the approved downstream V4.

``configs/method_frozen.yaml`` is historical provenance: its SHA-256 is stamped
into the T08 question blocks and the T10/C80 generation manifests and must never
change in place. ``configs/method_frozen_v4.yaml`` is the approved downstream
method configuration after PR #38. These tests turn either kind of config drift
into a loud CPU-only failure.

See ``docs/deviations/2026-08-14-method-frozen-config-thrash.md`` for the incident
that motivated this guard.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_CONFIG = ROOT / "configs" / "method_frozen.yaml"
DOWNSTREAM_CONFIG = ROOT / "configs" / "method_frozen_v4.yaml"

CANONICAL_V3_SHA = "73be73df4640c2d32bfbc8b6009741c0fadd8c466acb703d1f99df35d5c8cd79"
CANONICAL_V4_SHA = "f7e7dd9df8c7c72d5da08bf8201295dfc91f3208837d8cd0867054c393984e59"

# Historical artifacts that intentionally retain the V3 provenance stamp.
V3_STAMPING_ARTIFACTS = [
    ("design/questions_C80_A.json", ("method_config", "sha256")),
    ("design/questions_C80_B.json", ("method_config", "sha256")),
    (
        "results/generation/c80_role_C80_A.manifest.json",
        ("provenance", "method_config_sha256"),
    ),
    (
        "results/generation/c80_role_C80_B.manifest.json",
        ("provenance", "method_config_sha256"),
    ),
]

# T23 was still REVIEW_CANDIDATE when PR #38 migrated it to V4, so this
# downstream binding intentionally stamps the approved V4 SHA.
V4_STAMPING_ARTIFACTS = [
    ("design/causal_questions_frozen.json", ("method_config", "sha256")),
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dig(obj, keys):
    for key in keys:
        obj = obj[key]
    return obj


def test_historical_v3_config_matches_canonical_sha():
    observed = _sha(HISTORICAL_CONFIG)
    assert observed == CANONICAL_V3_SHA, (
        f"configs/method_frozen.yaml hashes to {observed[:16]} but the immutable "
        f"historical value is {CANONICAL_V3_SHA[:16]}. Do not add downstream "
        "decisions to the historical file; use method_frozen_v4.yaml."
    )


def test_downstream_v4_config_matches_approved_sha():
    observed = _sha(DOWNSTREAM_CONFIG)
    assert observed == CANONICAL_V4_SHA, (
        f"configs/method_frozen_v4.yaml hashes to {observed[:16]} but reviewer-"
        f"approved PR #38 froze {CANONICAL_V4_SHA[:16]}. Any method change now "
        "requires the declared change-control/deviation process."
    )


@pytest.mark.parametrize("path", [HISTORICAL_CONFIG, DOWNSTREAM_CONFIG])
def test_frozen_configs_have_no_crlf_bytes(path):
    assert b"\r" not in path.read_bytes(), f"{path.name} contains CR bytes"


@pytest.mark.parametrize("rel_path,key_path", V3_STAMPING_ARTIFACTS)
def test_historical_artifacts_still_stamp_v3(rel_path, key_path):
    path = ROOT / rel_path
    if not path.exists():
        pytest.skip(f"{rel_path} not present on this branch")
    stamped = _dig(json.loads(path.read_text(encoding="utf-8")), key_path)
    assert stamped == CANONICAL_V3_SHA, (
        f"{rel_path} stamps {stamped[:16]} but historical generation/question "
        f"provenance must remain {CANONICAL_V3_SHA[:16]}."
    )


@pytest.mark.parametrize("rel_path,key_path", V4_STAMPING_ARTIFACTS)
def test_downstream_artifacts_stamp_v4(rel_path, key_path):
    path = ROOT / rel_path
    stamped = _dig(json.loads(path.read_text(encoding="utf-8")), key_path)
    assert stamped == CANONICAL_V4_SHA, (
        f"{rel_path} stamps {stamped[:16]} but downstream T23 decisions are "
        f"bound to approved V4 {CANONICAL_V4_SHA[:16]}."
    )


# ---------------------------------------------------------------------------
# Method-body invariance across SHA successions (PR #44 review requirement 1)
# ---------------------------------------------------------------------------

# SHA-256 over the canonical JSON serialisation of the parsed V4 config with
# `deviation_records` removed. Registering a deviation appends to that list and
# therefore changes the FILE hash while leaving this one alone; changing any actual
# method field changes this one too.
#
# If this constant has to be updated, that is by definition a method change, and it
# requires a deviation record whose `affected_frozen_fields` names the changed fields
# plus explicit reviewer approval. Updating it to silence a failure is the exact
# failure mode this guard exists to prevent.
METHOD_BODY_SHA = "17b0dd2f3db6343adca0ce454bd2a7386a027378934653fb41dc62b7f5e2d32a"


def _method_body_sha(path):
    import hashlib
    import json as _json
    import yaml as _yaml
    cfg = _yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg.pop("deviation_records", None)
    canonical = _json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_registering_a_deviation_does_not_change_any_method_field():
    """The V4 file hash moved from 2fe1dbbb to f7e7dd9d when DEV-2026-08-17-T15-01 was
    registered. That succession is authorised only because it was a
    registration-only change: no frozen field value moved. This asserts that property
    directly rather than trusting the commit message, and it stands as a guard for
    every future deviation registration."""
    assert _method_body_sha(DOWNSTREAM_CONFIG) == METHOD_BODY_SHA, (
        "the method body changed, not just deviation_records. A file-SHA succession "
        "is only a registration-only change if this hash is unaffected. If a field "
        "genuinely changed, add a deviation record naming it in affected_frozen_fields "
        "and obtain reviewer approval before updating METHOD_BODY_SHA."
    )


def test_every_registered_deviation_declares_its_authorisation():
    """A deviation record is the authorisation for a change, so it must say what it
    touched, whether outcomes were already visible, and who approved it."""
    import yaml as _yaml
    records = _yaml.safe_load(
        Path(DOWNSTREAM_CONFIG).read_text(encoding="utf-8"))["deviation_records"]
    assert records, "deviation_records is the machine-readable source of truth"
    for r in records:
        for field in ("id", "decision_date", "task", "reason",
                      "affected_frozen_fields", "outcomes_already_inspected",
                      "reviewer", "reviewer_approval_status"):
            assert field in r, f"{r.get('id')} is missing {field}"
        assert isinstance(r["outcomes_already_inspected"], bool)
