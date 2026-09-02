#!/usr/bin/env python3
"""Build T31 figure/table templates and the T32 claim-branch scaffold.

TODO T31 requires the nine minimum figures, each with **saved source data**.
This builder freezes the *contracts* - for every figure panel and table it
declares the exact source-data columns, how those columns map onto the chart,
which task produces them, and which T32 claim branch(es) they support - and
creates a header-only source CSV per panel so the schema is visible before any
result exists.

It fabricates no numbers, and it never destroys any. Templates are **created
only when missing**: once a source CSV has real rows, rerunning the builder
keeps them and only re-validates the header. A frozen `claims_T32.json` or
`figures_manifest.json` is likewise never rewritten back to a draft state.

Figures render only once their source CSVs have rows; until then each is
AWAITING_RESULTS. T32 claims are DRAFT and are frozen only at T32 (after T28),
gate by gate.

Outputs:
  design/figures_manifest.json
  design/claims_T32.json
  design/figures/sources/<figure_id>[__<panel_id>].csv   (header-only contracts)
  design/tables/sources/<table_id>.csv                   (header-only contracts)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

# Chart types the renderer actually implements. A contract may not declare a
# chart type outside this set: the renderer fails closed rather than falling
# back to a generic bar, which used to report rendered:true for a plot that had
# silently misinterpreted its columns.
SUPPORTED_CHART_TYPES = (
    "dag", "scatter", "bar", "grouped_bar", "dose_response", "heatmap",
)

# Required render_spec roles per chart type. Every named column must exist in
# the panel's required_columns, so a contract cannot promise a chart it cannot
# draw from its own saved data.
CHART_SPEC_ROLES = {
    "dag": ("node", "label", "status", "depends_on"),
    "scatter": ("x", "y"),
    "bar": ("category", "value"),
    "grouped_bar": ("group", "category", "value"),
    "dose_response": ("series", "x", "y"),
    "heatmap": ("row", "col", "value"),
}
OPTIONAL_SPEC_ROLES = ("label", "err_low", "err_high", "n")

# Gate outcomes a claim branch may require. A branch describing a failed or
# non-specific result must say so, rather than listing the gate it contradicts.
GATE_STATES = ("PASS", "FAIL", "INCOMPLETE", "NOT_APPLICABLE")

# --------------------------------------------------------------------------- #
# Figure contracts (TODO T31 minimum figures, in order).
#
# Every figure is a list of panels. Most have exactly one, and keep the flat
# `design/figures/sources/<figure_id>.csv` path. A multi-panel figure gets one
# source CSV per panel, because a figure must be reproducible from its own
# saved data - a single confusion-matrix table cannot reproduce a retention
# panel drawn beside it.
# --------------------------------------------------------------------------- #
FIGURES = [
    {
        "figure_id": "F1_evidence_chain",
        "title": "Evidence chain",
        "supports_claims": ["validated_transfer", "negative_result"],
        "panels": [{
            "panel_id": None,
            "chart_type": "dag",
            "required_columns": ["node_id", "label", "status", "depends_on"],
            "render_spec": {"node": "node_id", "label": "label",
                            "status": "status", "depends_on": "depends_on"},
            "producing_task": "T00/T30 (task DAG)",
        }],
    },
    {
        "figure_id": "F2_channel_2x2",
        "title": "Two-model x two-channel comparison",
        # The two-model panel is exactly where an incomplete Llama arm shows,
        # so one_model_pilot is evidenced here rather than left unillustrated.
        "supports_claims": ["prompt_conditioned_geometry",
                            "security_effect_without_identity_specificity",
                            "one_model_pilot"],
        "panels": [{
            "panel_id": None,
            "chart_type": "grouped_bar",
            "required_columns": ["model", "channel", "full_role_rate",
                                 "ci_low", "ci_high", "n"],
            # `channel` is a category, never a numeric axis.
            "render_spec": {"group": "model", "category": "channel",
                            "value": "full_role_rate", "err_low": "ci_low",
                            "err_high": "ci_high", "n": "n"},
            # Attributed to the frozen analysis-registry entry. The concrete
            # results filename is not fixed anywhere in the repository yet, so
            # naming one here would be an unverified claim.
            "producing_task": "T19 (analysis_registry P2_CHANNEL_INTERACTION)",
            "producing_artifact": "NOT_YET_DEFINED",
        }],
    },
    {
        "figure_id": "F3_role_retention_judge_confusion",
        "title": "Role retention and judge confusion",
        "supports_claims": ["validated_transfer"],
        "notes": (
            "Two panels with two separate saved sources. The confusion panel "
            "alone cannot reproduce the retention panel, so retention has its "
            "own contract rather than being implied."
        ),
        "panels": [
            {
                "panel_id": "confusion",
                "chart_type": "heatmap",
                "required_columns": ["human_label", "judge_label", "count"],
                "render_spec": {"row": "human_label", "col": "judge_label",
                                "value": "count"},
                "producing_task": "T06/T07 (human vs judge labels)",
            },
            {
                "panel_id": "retention",
                "chart_type": "bar",
                "required_columns": ["role_id", "n_eligible", "n_score3",
                                     "retention_rate", "ci_low", "ci_high"],
                "render_spec": {"category": "role_id", "value": "retention_rate",
                                "err_low": "ci_low", "err_high": "ci_high"},
                "producing_task": "T14 (role retention / eligibility)",
            },
        ],
    },
    {
        "figure_id": "F4_c80_reliability",
        "title": "C80-A / C80-B cross-axis role-projection reliability",
        "supports_claims": ["validated_transfer", "reliable_but_not_causally_specific"],
        "panels": [{
            "panel_id": None,
            "chart_type": "scatter",
            "required_columns": ["role_id", "proj_on_axis_B", "proj_on_axis_A"],
            "render_spec": {"x": "proj_on_axis_B", "y": "proj_on_axis_A",
                            "label": "role_id"},
            "producing_task": "T16 (geometry_confirmatory_v3.json)",
        }],
    },
    {
        "figure_id": "F5_all_response_vs_answer_only",
        "title": "All-response versus answer-only geometry",
        "supports_claims": ["measurement_limited_result"],
        "panels": [{
            "panel_id": None,
            "chart_type": "scatter",
            "required_columns": ["role_id", "proj_all_response", "proj_answer_only"],
            "render_spec": {"x": "proj_all_response", "y": "proj_answer_only",
                            "label": "role_id"},
            "producing_task": "T15/T17 (pool sensitivities)",
        }],
    },
    {
        "figure_id": "F6_wrapper_vs_translated",
        "title": "Wrapper versus translated geometry",
        "supports_claims": ["prompt_conditioned_geometry"],
        "panels": [{
            "panel_id": None,
            "chart_type": "scatter",
            "required_columns": ["role_id", "proj_translated", "proj_wrapper"],
            "render_spec": {"x": "proj_translated", "y": "proj_wrapper",
                            "label": "role_id"},
            "producing_task": "T15 (wrapper E80 sensitivity)",
        }],
    },
    {
        "figure_id": "F7_axis_vs_random_causal",
        "title": "Axis versus random causal effects",
        "supports_claims": ["security_effect_without_identity_specificity",
                            "reliable_but_not_causally_specific"],
        "notes": (
            "shared_zero is ONE condition, stored ONCE, with direction="
            "'shared_zero'. Do not duplicate it per direction merely to draw two "
            "curves: the frozen two-slope encoding (analysis_registry P4/P5) sets "
            "axis_dose = random_dose = 0 on that single row, and both curves are "
            "drawn through it at render time. Duplicating the row would "
            "double-count the shared zero in any count taken from this source."
        ),
        "panels": [{
            "panel_id": None,
            "chart_type": "dose_response",
            "required_columns": ["condition", "direction", "signed_coefficient",
                                 "strict_harmful_rate", "ci_low", "ci_high", "n"],
            "render_spec": {"series": "direction", "x": "signed_coefficient",
                            "y": "strict_harmful_rate", "err_low": "ci_low",
                            "err_high": "ci_high", "n": "n"},
            "producing_task": "T28 (primary causal statistics)",
        }],
    },
    {
        "figure_id": "F8_harm_identity_by_steering",
        "title": "Harm and identity by steering condition",
        "supports_claims": ["security_effect_without_identity_specificity"],
        "panels": [{
            "panel_id": None,
            "chart_type": "grouped_bar",
            "required_columns": ["condition", "outcome", "rate", "ci_low",
                                 "ci_high", "n"],
            # Long form: one row per (condition, outcome) so the grouped bar has
            # a real category column instead of two parallel numeric columns.
            "render_spec": {"group": "outcome", "category": "condition",
                            "value": "rate", "err_low": "ci_low",
                            "err_high": "ci_high", "n": "n"},
            "producing_task": "T28 (harm + identity models)",
        }],
    },
    {
        "figure_id": "F9_degeneration_by_condition",
        "title": "Degeneration by condition",
        "supports_claims": ["measurement_limited_result", "negative_result"],
        "panels": [{
            "panel_id": None,
            "chart_type": "bar",
            "required_columns": ["condition", "degenerate_rate", "empty_rate", "n"],
            "render_spec": {"category": "condition", "value": "degenerate_rate",
                            "n": "n"},
            "producing_task": "T25/T28 (degeneration outcome)",
        }],
    },
]

# --------------------------------------------------------------------------- #
# Table contracts (T31 result tables), validated as strictly as figures.
# --------------------------------------------------------------------------- #
TABLES = [
    {
        "table_id": "T_role_judge_validation",
        "title": "Role-judge validation metrics",
        "required_columns": ["metric", "value", "gate_threshold", "passes"],
        "producing_task": "T07 (P1_ROLE_JUDGE_GATE)",
        "supports_claims": ["validated_transfer"],
    },
    {
        "table_id": "T_channel_calibration",
        "title": "Channel calibration full-role rates",
        "required_columns": ["model", "channel", "full_role_rate", "ci_low",
                             "ci_high", "n"],
        "producing_task": "T19 (analysis_registry P2_CHANNEL_INTERACTION)",
        "producing_artifact": "NOT_YET_DEFINED",
        "supports_claims": ["prompt_conditioned_geometry", "one_model_pilot"],
    },
    {
        "table_id": "T_confirmatory_reliability",
        "title": "Confirmatory reliability and nulls",
        "required_columns": ["statistic", "estimate", "ci_low", "ci_high",
                             "null_mean", "null_p", "n_roles"],
        "producing_task": "T16 (P3_CONFIRMATORY_GEOMETRY)",
        "supports_claims": ["validated_transfer", "reliable_but_not_causally_specific"],
    },
    {
        "table_id": "T_causal_specificity",
        "title": "Causal specificity (Axis vs random)",
        "required_columns": ["outcome", "beta_axis", "beta_random",
                             "beta_difference", "ci_low", "ci_high", "holm_p"],
        "producing_task": "T28 (P4/P5)",
        "supports_claims": ["security_effect_without_identity_specificity",
                            "reliable_but_not_causally_specific"],
    },
]

# --------------------------------------------------------------------------- #
# T32 core evidence branches (venue emphasis from TODO T32).
#
# Each branch declares the gate OUTCOME it represents, not merely which gates
# are relevant. A branch describing a non-specific causal result requires the
# specificity gate to FAIL; listing it as a gate that must pass described the
# opposite result.
# --------------------------------------------------------------------------- #
CLAIM_BRANCHES = {
    "validated_transfer": {
        "claim": (
            "A behaviourally validated Assistant contrast direction transfers to "
            "the reasoning-distilled model and reproduces across untouched "
            "question blocks."
        ),
        "gate_expectations": {
            "P1_ROLE_JUDGE_GATE": "PASS",
            "G1": "PASS",
            "P3_CONFIRMATORY_GEOMETRY": "PASS",
            "G2": "PASS",
        },
    },
    "prompt_conditioned_geometry": {
        "claim": (
            "The instruction channel changes measured role expression "
            "behaviourally, and separately the recovered direction differs "
            "between prompt regimes geometrically. These are two distinct "
            "pieces of evidence and are reported separately, never merged into "
            "one channel-effect claim."
        ),
        "gate_expectations": {
            "P2_CHANNEL_INTERACTION": "PASS",
            "G2": "PASS",
        },
        "evidence_components": {
            "behavioural_channel_interaction": (
                "P2_CHANNEL_INTERACTION: model x channel interaction in valid "
                "score-3 probability (T19, table T_channel_calibration, F2)."
            ),
            "prompt_regime_geometry": (
                "Wrapper-versus-translated role-projection difference "
                "(T15 sensitivity, F6). Geometric, not behavioural."
            ),
        },
    },
    "reliable_but_not_causally_specific": {
        "claim": (
            "Confirmatory geometry reproduces, but the causal effect of the "
            "Assistant Axis is not distinguishable from a norm-matched random "
            "direction."
        ),
        "gate_expectations": {
            "P3_CONFIRMATORY_GEOMETRY": "PASS",
            "P4_CAUSAL_HARMFULNESS_SPECIFICITY": "FAIL",
        },
    },
    "security_effect_without_identity_specificity": {
        "claim": (
            "Steering changes harmful compliance with an Axis-versus-random "
            "difference, while the identity mechanism shows no such specificity."
        ),
        "gate_expectations": {
            "P4_CAUSAL_HARMFULNESS_SPECIFICITY": "PASS",
            "P5_CAUSAL_IDENTITY_SPECIFICITY": "FAIL",
        },
    },
    "measurement_limited_result": {
        "claim": (
            "The available evidence does not separate representational structure "
            "from measurement choices (pooling region, judge coverage), so no "
            "structural conclusion is drawn in either direction."
        ),
        "gate_expectations": {
            "G2": "INCOMPLETE",
        },
    },
    "one_model_pilot": {
        "claim": (
            "Only the DeepSeek arm is complete; the Llama control is an "
            "incomplete pilot and is not reported as a cross-model result."
        ),
        "gate_expectations": {
            "T20": "INCOMPLETE",
            "T21": "INCOMPLETE",
        },
    },
    "negative_result": {
        "claim": (
            "One or more primary gates fail; the strongest claim narrows to a "
            "rigorous transfer/sensitivity audit or a negative result."
        ),
        "gate_expectations": {
            "ANY_PRIMARY_GATE": "FAIL",
        },
        "eligibility_rule": (
            "Eligible only when at least one primary analysis-registry gate "
            "(P1-P5) or blocking gate (G1/G2) actually FAILS. An empty gate list "
            "would have made this branch universally eligible, including when "
            "every gate passed."
        ),
    },
}

SELECTION_RULE = (
    "Exactly one branch is frozen at T32. A branch is eligible only when every "
    "entry in its gate_expectations matches the observed gate outcome. "
    "Eligibility is evaluated after T28; judge-dependent branches may not be "
    "evaluated before T06-T07-G1."
)

VENUE_EMPHASIS = {
    "NeurReps": "PRIMARY: geometry of neural representations, independent reconstruction, prompt/token-region dependence, layerwise dynamics, linear-vs-multidimensional structure, causal tests of geometric meaning.",
    "FLMSec": "secondary: channel integrity, jailbreak harm, reproducible security evaluation, Axis vs random.",
    "IAB": "secondary: interpretable behaviour categories, identity mechanism, runtime behaviour.",
    "VerifyAgents": "secondary: verifier validity, human agreement, silent evaluation regressions.",
    "EIML": "secondary: construct validity, uncertainty, measurement artifacts vs representational structure.",
}


class ContractError(ValueError):
    """A figure/table contract is internally inconsistent."""


class WouldDestroyResults(RuntimeError):
    """Refusing to overwrite populated or frozen artifacts."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def panel_source_csv(figure_id: str, panel_id: str | None, kind: str = "figures") -> str:
    stem = figure_id if panel_id is None else f"{figure_id}__{panel_id}"
    return f"design/{kind}/sources/{stem}.csv"


def read_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh), [])


def count_data_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as fh:
        return max(0, sum(1 for _ in csv.reader(fh)) - 1)


def ensure_source_csv(path: Path, columns: list[str]) -> str:
    """Create a header-only template, or validate an existing file in place.

    Never truncates. Once real rows exist, rerunning the builder must not
    silently delete them, so an existing file is only header-checked.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            csv.writer(fh, lineterminator="\n").writerow(columns)
        return "created"
    header = read_header(path)
    if header != columns:
        raise WouldDestroyResults(
            f"{path.name} already exists with header {header}, which does not "
            f"match the contract {columns}. Refusing to rewrite it: migrate the "
            "existing data explicitly, or delete the file if it is an empty "
            "template you intend to replace."
        )
    return "kept_with_rows" if count_data_rows(path) else "kept_empty"


def write_json_protected(path: Path, value: Any, frozen_statuses: tuple[str, ...]) -> str:
    """Write JSON unless the file on disk is in a frozen state."""
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        status = existing.get("status")
        if status in frozen_statuses:
            raise WouldDestroyResults(
                f"{path.name} is {status}; refusing to rewrite it back to a "
                "draft/template state. Freezing is a one-way step."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    return "written"


# --------------------------------------------------------------------------- #
# Contract validation
# --------------------------------------------------------------------------- #

def _validate_columns(owner: str, cols: list[str]) -> None:
    if not cols or len(cols) != len(set(cols)):
        raise ContractError(f"{owner}: empty or non-unique columns {cols}")
    if not all(isinstance(c, str) and c for c in cols):
        raise ContractError(f"{owner}: column names must be non-empty strings")


def _validate_render_spec(owner: str, chart_type: str, spec: dict[str, str],
                          cols: list[str]) -> None:
    if chart_type not in SUPPORTED_CHART_TYPES:
        raise ContractError(
            f"{owner}: chart_type {chart_type!r} is not implemented by the "
            f"renderer; supported types are {list(SUPPORTED_CHART_TYPES)}"
        )
    required = CHART_SPEC_ROLES[chart_type]
    missing = [r for r in required if r not in spec]
    if missing:
        raise ContractError(f"{owner}: render_spec missing roles {missing}")
    unknown_roles = set(spec) - set(required) - set(OPTIONAL_SPEC_ROLES)
    if unknown_roles:
        raise ContractError(f"{owner}: unknown render_spec roles {sorted(unknown_roles)}")
    bad = [f"{role}={col}" for role, col in spec.items() if col not in cols]
    if bad:
        raise ContractError(
            f"{owner}: render_spec references columns absent from the saved-data "
            f"contract: {bad}"
        )


def validate() -> None:
    known_claims = set(CLAIM_BRANCHES)

    seen_fig: set[str] = set()
    seen_sources: set[str] = set()
    for fig in FIGURES:
        fid = fig["figure_id"]
        if fid in seen_fig:
            raise ContractError(f"duplicate figure id {fid}")
        seen_fig.add(fid)
        if not fig.get("title"):
            raise ContractError(f"{fid}: missing title")
        bad = set(fig["supports_claims"]) - known_claims
        if bad:
            raise ContractError(f"{fid} references unknown claims {bad}")
        panels = fig["panels"]
        if not panels:
            raise ContractError(f"{fid}: no panels")
        if len(panels) > 1 and any(p["panel_id"] is None for p in panels):
            raise ContractError(f"{fid}: multi-panel figures need a panel_id each")
        for panel in panels:
            owner = f"{fid}[{panel['panel_id'] or 'main'}]"
            _validate_columns(owner, panel["required_columns"])
            if not panel.get("producing_task"):
                raise ContractError(f"{owner}: missing producing_task")
            _validate_render_spec(owner, panel["chart_type"],
                                  panel["render_spec"], panel["required_columns"])
            src = panel_source_csv(fid, panel["panel_id"])
            if src in seen_sources:
                raise ContractError(f"duplicate source path {src}")
            seen_sources.add(src)

    if len(FIGURES) != 9:
        raise ContractError(f"TODO T31 requires 9 minimum figures, have {len(FIGURES)}")

    # Tables are validated as strictly as figures.
    seen_tbl: set[str] = set()
    for tbl in TABLES:
        tid = tbl["table_id"]
        if tid in seen_tbl:
            raise ContractError(f"duplicate table id {tid}")
        seen_tbl.add(tid)
        if not tbl.get("title"):
            raise ContractError(f"{tid}: missing title")
        if not tbl.get("producing_task"):
            raise ContractError(f"{tid}: missing producing_task")
        _validate_columns(tid, tbl["required_columns"])
        bad = set(tbl["supports_claims"]) - known_claims
        if bad:
            raise ContractError(f"{tid} references unknown claims {bad}")
        src = panel_source_csv(tid, None, kind="tables")
        if src in seen_sources:
            raise ContractError(f"duplicate source path {src}")
        seen_sources.add(src)

    # Claim branches must state gate OUTCOMES, and none may be unconditional.
    for name, branch in CLAIM_BRANCHES.items():
        if not branch.get("claim"):
            raise ContractError(f"claim branch {name}: missing claim text")
        expectations = branch.get("gate_expectations")
        if not expectations:
            raise ContractError(
                f"claim branch {name}: gate_expectations must be non-empty, "
                "otherwise the branch is eligible under every outcome"
            )
        bad_states = {g: s for g, s in expectations.items() if s not in GATE_STATES}
        if bad_states:
            raise ContractError(
                f"claim branch {name}: gate states {bad_states} outside {GATE_STATES}"
            )
    referenced = {c for f in FIGURES for c in f["supports_claims"]}
    referenced |= {c for t in TABLES for c in t["supports_claims"]}
    orphan = known_claims - referenced
    if orphan:
        raise ContractError(f"claim branches with no supporting figure/table: {orphan}")


def contracts_payload(fig_records, tbl_records) -> list[Any]:
    """Canonical representation of the COMPLETE contract set.

    Everything that defines what a figure or table promises is included -
    columns, chart types, render specs, producing tasks, claim mappings and
    source paths. Only `status`, which changes as results arrive, is excluded.
    """
    def strip(record):
        return {k: v for k, v in sorted(record.items()) if k != "status"}

    return [
        {"figures": [strip(f) for f in fig_records]},
        {"tables": [strip(t) for t in tbl_records]},
    ]


def build(root: Path) -> dict[str, Any]:
    validate()
    design = root / "design"
    template_actions: dict[str, str] = {}

    fig_records = []
    for fig in FIGURES:
        panels = []
        for panel in fig["panels"]:
            src = panel_source_csv(fig["figure_id"], panel["panel_id"])
            template_actions[src] = ensure_source_csv(
                root / src, panel["required_columns"]
            )
            panels.append({
                **panel,
                "source_csv": src,
                "columns_sha256": canonical_sha256(panel["required_columns"]),
            })
        fig_records.append({
            **{k: v for k, v in fig.items() if k != "panels"},
            "n_panels": len(panels),
            "panels": panels,
            "status": "AWAITING_RESULTS",
            "saved_source_data_required": True,
        })

    tbl_records = []
    for tbl in TABLES:
        src = panel_source_csv(tbl["table_id"], None, kind="tables")
        template_actions[src] = ensure_source_csv(root / src, tbl["required_columns"])
        tbl_records.append({
            **tbl,
            "source_csv": src,
            "columns_sha256": canonical_sha256(tbl["required_columns"]),
            "status": "AWAITING_RESULTS",
        })

    figures_manifest = {
        "schema_version": "t31-figures-manifest/2.0",
        "task": "T31",
        "status": "TEMPLATES_AWAITING_RESULTS",
        "note": (
            "Contracts only. No figure carries fabricated data. Each panel "
            "renders only once its own source CSV has rows. Every panel and "
            "table declares its saved source data, how those columns map onto "
            "the chart, its producing task, and the T32 claim branch(es) it "
            "supports."
        ),
        "supported_chart_types": list(SUPPORTED_CHART_TYPES),
        "n_figures": len(fig_records),
        "n_panels": sum(f["n_panels"] for f in fig_records),
        "figures": fig_records,
        "n_tables": len(tbl_records),
        "tables": tbl_records,
        "contracts_sha256": canonical_sha256(
            contracts_payload(fig_records, tbl_records)
        ),
        "contracts_sha256_covers": (
            "Complete figure and table contracts - columns, chart types, render "
            "specs, producing tasks, claim mappings, source paths - excluding "
            "only the volatile `status` field."
        ),
    }
    write_json_protected(
        design / "figures_manifest.json", figures_manifest, ("FROZEN",)
    )

    claims = {
        "schema_version": "t32-claims/2.0",
        "task": "T32",
        "status": "DRAFT_NOT_FROZEN",
        "freeze_rule": (
            "Frozen at T32 after T28, one branch per outcome. The permitted "
            "final claim narrows to the branch whose gate expectations all "
            "match (TODO T32 / 'Strongest permitted final claim')."
        ),
        "selection_rule": SELECTION_RULE,
        "gate_states": list(GATE_STATES),
        "core_evidence_branches": CLAIM_BRANCHES,
        "venue_emphasis": VENUE_EMPHASIS,
        "gate_dependent_do_not_freeze_before": (
            "T06-T07-G1 for judge-dependent branches; T28 for causal branches."
        ),
    }
    write_json_protected(
        design / "claims_T32.json", claims, ("FROZEN", "FROZEN_AT_T32"),
    )
    return {
        "figures_manifest": figures_manifest,
        "claims": claims,
        "template_actions": template_actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = build(root)
    actions = out["template_actions"]
    print(json.dumps({
        "status": "PASS",
        "n_figures": out["figures_manifest"]["n_figures"],
        "n_panels": out["figures_manifest"]["n_panels"],
        "n_tables": out["figures_manifest"]["n_tables"],
        "contracts_sha256": out["figures_manifest"]["contracts_sha256"],
        "claims_status": out["claims"]["status"],
        "n_claim_branches": len(out["claims"]["core_evidence_branches"]),
        "templates_created": sorted(k for k, v in actions.items() if v == "created"),
        "templates_kept_with_rows": sorted(
            k for k, v in actions.items() if v == "kept_with_rows"
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
