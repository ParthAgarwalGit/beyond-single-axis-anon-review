"""T22 method-freeze succession support.

The real T22 freeze must make two V4 changes together:
(1) register the pre-selection E80-membership deviation in the machine-readable
``deviation_records`` list and (2) fill the allowed-later
``causal_role_selection.P50_manifest_sha256`` field with the hash of the actual
50-role manifest.  This module applies those as one reviewed SHA succession and
rebinds only the forward-looking pins documented in METHOD_FREEZE_V4_13_AUG.md.

Historical provenance stamps are deliberately untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

PREDECESSOR_V4_SHA = "f7e7dd9df8c7c72d5da08bf8201295dfc91f3208837d8cd0867054c393984e59"
PREDECESSOR_METHOD_BODY_SHA = "17b0dd2f3db6343adca0ce454bd2a7386a027378934653fb41dc62b7f5e2d32a"
DEVIATION_ID = "DEV-2026-08-18-T22-01"
DEVIATION_DOC = "docs/deviations/2026-08-18-t22-p50-membership.md"

CONFIG_REL = Path("configs/method_frozen_v4.yaml")
INTEGRITY_TEST_REL = Path("tests/test_method_frozen_integrity.py")
CAUSAL_SOURCE_MAP_REL = Path("design/causal_questions_frozen.json")
FREEZE_DOC_REL = Path("docs/METHOD_FREEZE_V4_13_AUG.md")
PROVENANCE_CLOSURE_REL = Path("tools/make_t09_provenance_closure.py")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_method_body_sha(text: str) -> str:
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError("V4 config must parse to a mapping")
    cfg.pop("deviation_records", None)
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise ValueError(f"{label}: expected exactly one occurrence, found {n}")
    return text.replace(old, new, 1)


def _deviation_yaml_block() -> str:
    return f"""- id: {DEVIATION_ID}
  decision_date: '2026-08-18'
  recorded_on: '2026-08-19'
  task: T22
  reason: >-
    T07 failed every predeclared automatic role-judge validation criterion, so
    validated score-3 E80 membership is unavailable for primary P50 selection.
    Before the real P50 ranking was emitted, {DEVIATION_DOC} fixed the primary
    selection population to technically valid E80 USER_TRANSLATED_LU outputs
    with the block-16 all-response pool present. The ranking equation,
    literal-Assistant exclusion, lexical tie-break, size 50, outcome blindness,
    and downstream held-out-axis requirement are unchanged.
  affected_frozen_fields:
  - causal_role_selection.selection_data
  - causal_role_selection.P50_manifest_sha256
  affected_artifacts:
  - {DEVIATION_DOC}
  - design/causal_P50_assistant_proximal.json
  - results/t22/selection_report.json
  - configs/method_frozen_v4.yaml
  - docs/METHOD_FREEZE_V4_13_AUG.md
  outcomes_already_inspected: false
  outcomes_inspected_note: >-
    No causal outcome is used by T22. The membership decision was recorded in
    the dated deviation document before the P50 ranking/list was produced. The
    machine-readable registration is committed together with the resulting P50
    manifest hash so V4 needs only one SHA succession for T22.
  reruns_performed: none
  reviewer: [Author-A-GitHub]
  reviewer_approval_status: PENDING_GITHUB_PR_REVIEW
  reviewer_approval_source: 'PR #49'
"""


def inspect_succession_state(root: Path) -> dict:
    """Inspect whether the checkout is ready for, or already contains, T22 succession."""
    root = Path(root)
    config_text = (root / CONFIG_REL).read_text(encoding="utf-8")
    cfg = yaml.safe_load(config_text)
    records = cfg.get("deviation_records") or []
    ids = {r.get("id") for r in records if isinstance(r, dict)}
    return {
        "v4_sha256": sha256_bytes(config_text.encode("utf-8")),
        "method_body_sha256": canonical_method_body_sha(config_text),
        "p50_manifest_sha256": cfg.get("causal_role_selection", {}).get("P50_manifest_sha256"),
        "t22_deviation_registered": DEVIATION_ID in ids,
    }


def preflight_t22_succession(root: Path) -> dict:
    """Fail before P50 output is written if this checkout cannot take the one-step succession."""
    state = inspect_succession_state(root)
    if state["t22_deviation_registered"]:
        raise ValueError(
            "T22 deviation is already registered before this real-data run; refuse a split succession"
        )
    if state["p50_manifest_sha256"] not in (None, ""):
        raise ValueError(
            "P50_manifest_sha256 is already filled before this real-data run; refuse to overwrite it"
        )
    if state["v4_sha256"] != PREDECESSOR_V4_SHA:
        raise ValueError(
            "T22 requires the exact post-#44 V4 predecessor before the real P50 freeze. "
            f"Expected {PREDECESSOR_V4_SHA}, observed {state['v4_sha256']}. "
            "Sync/rebase the stacked branch first rather than overwriting a newer freeze."
        )
    if state["method_body_sha256"] != PREDECESSOR_METHOD_BODY_SHA:
        raise ValueError(
            "unexpected predecessor method-body SHA: "
            f"{state['method_body_sha256']} != {PREDECESSOR_METHOD_BODY_SHA}"
        )
    return state


def prepare_config_successor(config_text: str, p50_manifest_sha256: str) -> tuple[str, str]:
    """Return the V4 successor text and its method-body SHA.

    The membership method change is registered as a deviation rather than
    silently rewriting the old ``selection_data`` prose. The only direct
    allowed-later field fill is the actual P50 manifest hash.
    """
    if len(p50_manifest_sha256) != 64 or any(c not in "0123456789abcdef" for c in p50_manifest_sha256):
        raise ValueError("P50 manifest SHA must be lowercase 64-hex")

    cfg = yaml.safe_load(config_text)
    if not isinstance(cfg, dict):
        raise ValueError("V4 config must parse to a mapping")
    current = cfg.get("causal_role_selection", {}).get("P50_manifest_sha256")
    ids = {r.get("id") for r in (cfg.get("deviation_records") or []) if isinstance(r, dict)}
    if current not in (None, ""):
        raise ValueError(f"P50_manifest_sha256 already filled: {current}")
    if DEVIATION_ID in ids:
        raise ValueError("T22 deviation already registered; refuse split succession")

    successor = _replace_once(
        config_text,
        "  P50_manifest_sha256: null\n",
        f"  P50_manifest_sha256: {p50_manifest_sha256}\n",
        label="P50 manifest fill",
    )
    marker = "\nv4_amendment:\n"
    if successor.count(marker) != 1:
        raise ValueError("cannot locate unique v4_amendment insertion point")
    successor = successor.replace(marker, "\n" + _deviation_yaml_block() + marker, 1)

    parsed = yaml.safe_load(successor)
    if parsed["causal_role_selection"]["P50_manifest_sha256"] != p50_manifest_sha256:
        raise ValueError("successor failed to bind the P50 manifest hash")
    recs = [r for r in parsed.get("deviation_records", []) if r.get("id") == DEVIATION_ID]
    if len(recs) != 1:
        raise ValueError("successor must register exactly one T22 deviation")
    return successor, canonical_method_body_sha(successor)


def _prepare_forward_pin_updates(
    root: Path,
    old_v4_sha: str,
    new_v4_sha: str,
    old_body_sha: str,
    new_body_sha: str,
) -> dict[Path, str]:
    updates: dict[Path, str] = {}

    p = root / INTEGRITY_TEST_REL
    text = p.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        f'CANONICAL_V4_SHA = "{old_v4_sha}"',
        f'CANONICAL_V4_SHA = "{new_v4_sha}"',
        label="integrity V4 pin",
    )
    text = _replace_once(
        text,
        f'METHOD_BODY_SHA = "{old_body_sha}"',
        f'METHOD_BODY_SHA = "{new_body_sha}"',
        label="integrity method-body pin",
    )
    updates[p] = text

    p = root / CAUSAL_SOURCE_MAP_REL
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("method_config", {}).get("sha256") != old_v4_sha:
        raise ValueError("causal source map does not point at predecessor V4 SHA")
    doc["method_config"]["sha256"] = new_v4_sha
    updates[p] = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"

    p = root / PROVENANCE_CLOSURE_REL
    text = p.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        f'V4_APPROVED_SHA = "{old_v4_sha}"',
        f'V4_APPROVED_SHA = "{new_v4_sha}"',
        label="provenance-closure V4 pin",
    )
    updates[p] = text

    p = root / FREEZE_DOC_REL
    text = p.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        f"**V4 SHA-256:** `{old_v4_sha}`",
        f"**V4 SHA-256:** `{new_v4_sha}`",
        label="freeze-doc current V4 SHA",
    )
    prior_row = (
        "| `f7e7dd9df8c7c72d…` | PR #44 review, under `DEV-2026-08-17-T15-01` | "
        "**registration-only**: a deviation record was appended; no frozen field value changed |"
    )
    new_row = (
        f"| `{new_v4_sha[:16]}…` | PR #49 review, under `{DEVIATION_ID}` | "
        "T22 succession: register the pre-selection membership deviation and fill the allowed-later "
        "P50 manifest hash in one change; no historical provenance stamp rewritten |"
    )
    text = _replace_once(text, prior_row, prior_row + "\n" + new_row, label="freeze-doc succession row")
    old_body_sentence = (
        "config with `deviation_records` removed is `17b0dd2f3db6343a…` on both sides of the\n"
        "succession above, and `test_registering_a_deviation_does_not_change_any_method_field`\n"
        "holds it there."
    )
    new_body_sentence = (
        "config with `deviation_records` removed is `17b0dd2f3db6343a…` across the PR #44\n"
        "registration-only succession. The T22 succession then fills the explicitly allowed-later\n"
        "`causal_role_selection.P50_manifest_sha256`, moving the current method-body hash to\n"
        f"`{new_body_sha[:16]}…`; the integrity test is rebound to that reviewed value."
    )
    text = _replace_once(text, old_body_sentence, new_body_sentence, label="freeze-doc body-hash note")
    updates[p] = text
    return updates


def finalize_t22_succession(root: Path, p50_manifest_sha256: str) -> dict:
    """Apply the single T22 V4 succession after the real manifest hash exists."""
    root = Path(root)
    preflight_t22_succession(root)
    config_path = root / CONFIG_REL
    config_text = config_path.read_text(encoding="utf-8")
    successor_text, successor_body_sha = prepare_config_successor(config_text, p50_manifest_sha256)
    successor_v4_sha = sha256_bytes(successor_text.encode("utf-8"))
    updates = _prepare_forward_pin_updates(
        root,
        PREDECESSOR_V4_SHA,
        successor_v4_sha,
        PREDECESSOR_METHOD_BODY_SHA,
        successor_body_sha,
    )

    yaml.safe_load(successor_text)
    for path, text in updates.items():
        if path.suffix == ".json":
            json.loads(text)
        if "\r" in text:
            raise ValueError(f"prepared successor would introduce CR bytes in {path}")

    config_path.write_text(successor_text, encoding="utf-8", newline="\n")
    for path, text in updates.items():
        path.write_text(text, encoding="utf-8", newline="\n")

    return {
        "status": "APPLIED_PENDING_GIT_COMMIT_AND_REVIEW",
        "predecessor_v4_sha256": PREDECESSOR_V4_SHA,
        "successor_v4_sha256": successor_v4_sha,
        "predecessor_method_body_sha256": PREDECESSOR_METHOD_BODY_SHA,
        "successor_method_body_sha256": successor_body_sha,
        "p50_manifest_sha256": p50_manifest_sha256,
        "registered_deviation": DEVIATION_ID,
        "forward_pins_rebound": [
            str(INTEGRITY_TEST_REL),
            str(CAUSAL_SOURCE_MAP_REL),
            str(FREEZE_DOC_REL),
            str(PROVENANCE_CLOSURE_REL),
        ],
        "historical_provenance_rewritten": False,
    }
