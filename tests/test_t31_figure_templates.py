"""T31/T32 checks: figure/table contracts and claim-branch scaffold (CPU-only).

Locks the properties a reviewer needs:

- the nine minimum figures exist, each panel with a non-empty, unique
  source-data contract, a producing task, and a render_spec that only names
  columns the contract actually saves;
- tables are validated as strictly as figures;
- the builder never destroys populated source data or un-freezes a frozen
  claims/manifest file;
- the renderer refuses absent, header-only, or contract-mismatched source data,
  and fails closed on an unimplemented chart type instead of drawing a
  placeholder and reporting success;
- claim branches encode gate OUTCOMES, and none is unconditionally eligible;
- the contracts checksum covers the whole contract, not just figure columns.
"""

import copy
import csv
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
    return _load("build_t31", "tools/build_t31_figure_templates.py")


@pytest.fixture(scope="module")
def renderer():
    return _load("render_t31", "tools/render_t31_figure.py")


@pytest.fixture()
def built(builder, tmp_path):
    # Build into an isolated root so the test never depends on committed output.
    (tmp_path / "design").mkdir()
    return builder.build(tmp_path), tmp_path


def _write_csv(path, header, *rows):
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------- #
# contracts
# --------------------------------------------------------------------------- #

def test_nine_figures_with_valid_contracts(built):
    manifest = built[0]["figures_manifest"]
    assert manifest["n_figures"] == 9
    for fig in manifest["figures"]:
        assert fig["title"]
        assert fig["status"] == "AWAITING_RESULTS"
        assert fig["saved_source_data_required"] is True
        assert fig["panels"]
        for panel in fig["panels"]:
            cols = panel["required_columns"]
            assert cols and len(cols) == len(set(cols))
            assert panel["producing_task"]
            assert panel["chart_type"] in manifest["supported_chart_types"]


def test_every_render_spec_names_only_saved_columns(built):
    for fig in built[0]["figures_manifest"]["figures"]:
        for panel in fig["panels"]:
            cols = set(panel["required_columns"])
            for role, column in panel["render_spec"].items():
                assert column in cols, f"{fig['figure_id']}.{role} -> {column}"


def test_tables_are_validated_as_strictly_as_figures(built, builder):
    manifest = built[0]["figures_manifest"]
    assert manifest["n_tables"] == 4
    for tbl in manifest["tables"]:
        assert tbl["title"] and tbl["producing_task"]
        cols = tbl["required_columns"]
        assert cols and len(cols) == len(set(cols))
        assert tbl["source_csv"].startswith("design/tables/sources/")
        assert tbl["columns_sha256"]

    # And the validator actually rejects a bad table, not just a bad figure.
    original = copy.deepcopy(builder.TABLES)
    try:
        builder.TABLES[0]["required_columns"] = ["metric", "metric"]
        with pytest.raises(builder.ContractError, match="non-unique"):
            builder.validate()
        builder.TABLES[0]["required_columns"] = original[0]["required_columns"]
        builder.TABLES[0]["producing_task"] = ""
        with pytest.raises(builder.ContractError, match="producing_task"):
            builder.validate()
    finally:
        builder.TABLES[:] = original


def test_table_source_templates_are_header_only(built):
    _, root = built
    files = list((root / "design" / "tables" / "sources").glob("*.csv"))
    assert len(files) == 4
    for f in files:
        with f.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh))
        assert len(rows) == 1, f"{f.name} must be header-only"
        assert rows[0]


def test_figure_source_templates_are_header_only(built):
    _, root = built
    files = list((root / "design" / "figures" / "sources").glob("*.csv"))
    # 9 figures, one of which (F3) has two panels.
    assert len(files) == 10
    for f in files:
        with f.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh))
        assert len(rows) == 1, f"{f.name} must be header-only (no fabricated data)"
        assert rows[0]


def test_f3_can_reproduce_both_panels_it_promises(built):
    fig = next(
        f for f in built[0]["figures_manifest"]["figures"]
        if f["figure_id"] == "F3_role_retention_judge_confusion"
    )
    assert fig["n_panels"] == 2
    by_panel = {p["panel_id"]: p for p in fig["panels"]}
    assert set(by_panel) == {"confusion", "retention"}
    # The retention panel needs retention/eligibility data of its own; the
    # confusion table alone cannot produce it.
    retention_cols = set(by_panel["retention"]["required_columns"])
    assert {"role_id", "n_eligible", "n_score3", "retention_rate"} <= retention_cols
    assert "T14" in by_panel["retention"]["producing_task"]


def test_f7_documents_the_single_shared_zero(built):
    fig = next(
        f for f in built[0]["figures_manifest"]["figures"]
        if f["figure_id"] == "F7_axis_vs_random_causal"
    )
    assert "shared_zero" in fig["notes"]
    assert "Do not duplicate" in fig["notes"]
    assert "direction" in fig["panels"][0]["required_columns"]


def test_f2_does_not_name_an_unverified_producing_artifact(built):
    fig = next(
        f for f in built[0]["figures_manifest"]["figures"]
        if f["figure_id"] == "F2_channel_2x2"
    )
    panel = fig["panels"][0]
    assert "P2_CHANNEL_INTERACTION" in panel["producing_task"]
    assert panel["producing_artifact"] == "NOT_YET_DEFINED"
    assert "two_model_two_channel_results.json" not in json.dumps(fig)


def test_figures_map_to_real_claims(built):
    manifest = built[0]["figures_manifest"]
    claims = set(built[0]["claims"]["core_evidence_branches"])
    for fig in manifest["figures"]:
        assert set(fig["supports_claims"]) <= claims
    for tbl in manifest["tables"]:
        assert set(tbl["supports_claims"]) <= claims


# --------------------------------------------------------------------------- #
# contracts checksum
# --------------------------------------------------------------------------- #

def test_contracts_sha_covers_more_than_figure_columns(built, builder, tmp_path):
    baseline = built[0]["figures_manifest"]["contracts_sha256"]
    original_tables = copy.deepcopy(builder.TABLES)
    original_figs = copy.deepcopy(builder.FIGURES)

    def rebuild():
        root = tmp_path / f"r{len(list(tmp_path.iterdir()))}"
        (root / "design").mkdir(parents=True)
        return builder.build(root)["figures_manifest"]["contracts_sha256"]

    try:
        # A table schema change must move the checksum.
        builder.TABLES[0]["required_columns"] = [
            *original_tables[0]["required_columns"], "extra"
        ]
        assert rebuild() != baseline
        builder.TABLES[:] = copy.deepcopy(original_tables)

        # So must a chart type, a producing task, and a claim mapping.
        builder.FIGURES[3]["panels"][0]["chart_type"] = "bar"
        builder.FIGURES[3]["panels"][0]["render_spec"] = {
            "category": "role_id", "value": "proj_on_axis_A"
        }
        assert rebuild() != baseline
        builder.FIGURES[:] = copy.deepcopy(original_figs)

        builder.FIGURES[3]["panels"][0]["producing_task"] = "T99"
        assert rebuild() != baseline
        builder.FIGURES[:] = copy.deepcopy(original_figs)

        builder.FIGURES[3]["supports_claims"] = ["negative_result"]
        assert rebuild() != baseline
    finally:
        builder.FIGURES[:] = original_figs
        builder.TABLES[:] = original_tables


# --------------------------------------------------------------------------- #
# the builder must not destroy results
# --------------------------------------------------------------------------- #

def test_rebuild_keeps_populated_source_rows(built, builder):
    _, root = built
    src = root / "design" / "figures" / "sources" / "F4_c80_reliability.csv"
    _write_csv(src, ["role_id", "proj_on_axis_B", "proj_on_axis_A"],
               ["pirate", "0.31", "0.29"], ["sage", "0.12", "0.15"])
    before = src.read_text(encoding="utf-8")

    out = builder.build(root)
    assert src.read_text(encoding="utf-8") == before, "real result rows were destroyed"
    assert out["template_actions"]["design/figures/sources/F4_c80_reliability.csv"] \
        == "kept_with_rows"


def test_rebuild_refuses_a_source_whose_header_drifted(built, builder):
    _, root = built
    src = root / "design" / "tables" / "sources" / "T_causal_specificity.csv"
    _write_csv(src, ["totally", "different"], ["1", "2"])
    with pytest.raises(builder.WouldDestroyResults, match="Refusing to rewrite"):
        builder.build(root)


def test_rebuild_cannot_unfreeze_the_claims_file(built, builder):
    _, root = built
    claims_path = root / "design" / "claims_T32.json"
    frozen = json.loads(claims_path.read_text(encoding="utf-8"))
    frozen["status"] = "FROZEN_AT_T32"
    frozen["frozen_branch"] = "validated_transfer"
    claims_path.write_text(json.dumps(frozen, indent=2) + "\n", newline="\n")

    with pytest.raises(builder.WouldDestroyResults, match="one-way"):
        builder.build(root)
    # The frozen file is untouched.
    assert json.loads(claims_path.read_text(encoding="utf-8"))["status"] == "FROZEN_AT_T32"


def test_rebuild_cannot_unfreeze_the_manifest(built, builder):
    _, root = built
    path = root / "design" / "figures_manifest.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    frozen["status"] = "FROZEN"
    path.write_text(json.dumps(frozen, indent=2) + "\n", newline="\n")
    with pytest.raises(builder.WouldDestroyResults):
        builder.build(root)


# --------------------------------------------------------------------------- #
# T32 claim branches
# --------------------------------------------------------------------------- #

def test_claims_are_draft_not_frozen(built):
    claims = built[0]["claims"]
    assert claims["status"] == "DRAFT_NOT_FROZEN"
    branches = claims["core_evidence_branches"]
    for required in (
        "validated_transfer", "prompt_conditioned_geometry",
        "reliable_but_not_causally_specific",
        "security_effect_without_identity_specificity",
        "measurement_limited_result", "one_model_pilot", "negative_result",
    ):
        assert required in branches
        assert branches[required]["claim"]
    assert "NeurReps" in claims["venue_emphasis"]
    assert claims["selection_rule"]


def test_branches_encode_gate_outcomes_not_just_gate_names(built):
    branches = built[0]["claims"]["core_evidence_branches"]
    states = set(built[0]["claims"]["gate_states"])
    for name, branch in branches.items():
        expectations = branch["gate_expectations"]
        assert expectations, f"{name} has no gate expectations"
        assert set(expectations.values()) <= states


def test_failure_branches_require_the_gate_to_fail(built):
    branches = built[0]["claims"]["core_evidence_branches"]
    # A "not causally specific" result requires the specificity gate to FAIL.
    assert branches["reliable_but_not_causally_specific"]["gate_expectations"][
        "P4_CAUSAL_HARMFULNESS_SPECIFICITY"] == "FAIL"
    assert branches["reliable_but_not_causally_specific"]["gate_expectations"][
        "P3_CONFIRMATORY_GEOMETRY"] == "PASS"
    # "no identity specificity" requires the identity gate to FAIL.
    assert branches["security_effect_without_identity_specificity"][
        "gate_expectations"]["P5_CAUSAL_IDENTITY_SPECIFICITY"] == "FAIL"
    assert branches["security_effect_without_identity_specificity"][
        "gate_expectations"]["P4_CAUSAL_HARMFULNESS_SPECIFICITY"] == "PASS"
    # An incomplete pilot is INCOMPLETE, not PASS.
    assert set(branches["one_model_pilot"]["gate_expectations"].values()) == {"INCOMPLETE"}


def test_negative_result_is_not_universally_eligible(built, builder):
    branch = built[0]["claims"]["core_evidence_branches"]["negative_result"]
    assert branch["gate_expectations"] == {"ANY_PRIMARY_GATE": "FAIL"}
    assert "FAILS" in branch["eligibility_rule"]

    original = copy.deepcopy(builder.CLAIM_BRANCHES["negative_result"])
    try:
        builder.CLAIM_BRANCHES["negative_result"]["gate_expectations"] = {}
        with pytest.raises(builder.ContractError, match="non-empty"):
            builder.validate()
    finally:
        builder.CLAIM_BRANCHES["negative_result"] = original


def test_prompt_conditioned_geometry_separates_its_two_evidence_types(built):
    branch = built[0]["claims"]["core_evidence_branches"]["prompt_conditioned_geometry"]
    components = branch["evidence_components"]
    assert "behavioural_channel_interaction" in components
    assert "prompt_regime_geometry" in components
    assert "never merged" in branch["claim"]


def test_measurement_limited_result_does_not_overclaim(built):
    claim = built[0]["claims"]["core_evidence_branches"][
        "measurement_limited_result"]["claim"]
    # The earlier wording ("rather than representational structure") asserted an
    # exclusion the gate does not establish.
    assert "rather than representational structure" not in claim
    assert "does not separate" in claim


# --------------------------------------------------------------------------- #
# the renderer
# --------------------------------------------------------------------------- #

def test_renderer_refuses_header_only_source(built, renderer):
    _, root = built
    status = renderer.check_ready(root, "F4_c80_reliability")
    assert status["ready"] is False
    assert "header-only" in status["reason"] or "AWAITING" in status["reason"]
    with pytest.raises(SystemExit):
        renderer.render(root, "F4_c80_reliability", root / "out.png")


def test_renderer_rejects_contract_mismatch(built, renderer):
    _, root = built
    src = root / "design" / "figures" / "sources" / "F4_c80_reliability.csv"
    _write_csv(src, ["wrong", "columns"], ["1", "2"])
    status = renderer.check_ready(root, "F4_c80_reliability")
    assert status["ready"] is False
    assert "contract" in status["reason"]


def test_renderer_accepts_wellformed_source(built, renderer):
    _, root = built
    src = root / "design" / "figures" / "sources" / "F4_c80_reliability.csv"
    _write_csv(src, ["role_id", "proj_on_axis_B", "proj_on_axis_A"],
               ["pirate", "0.31", "0.29"], ["sage", "0.12", "0.15"])
    status = renderer.check_ready(root, "F4_c80_reliability")
    assert status["ready"] is True
    assert status["n_rows"] == 2


def test_multi_panel_figure_needs_every_panel(built, renderer):
    _, root = built
    src = root / "design" / "figures" / "sources"
    _write_csv(src / "F3_role_retention_judge_confusion__confusion.csv",
               ["human_label", "judge_label", "count"], ["3", "3", "40"])
    status = renderer.check_ready(root, "F3_role_retention_judge_confusion")
    assert status["ready"] is False
    assert "retention" in status["reason"]

    _write_csv(src / "F3_role_retention_judge_confusion__retention.csv",
               ["role_id", "n_eligible", "n_score3", "retention_rate",
                "ci_low", "ci_high"],
               ["pirate", "80", "44", "0.55", "0.44", "0.66"])
    assert renderer.check_ready(root, "F3_role_retention_judge_confusion")["ready"]


def test_unsupported_chart_type_fails_closed(built, renderer):
    _, root = built
    path = root / "design" / "figures_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    fig = next(f for f in manifest["figures"] if f["figure_id"] == "F4_c80_reliability")
    fig["panels"][0]["chart_type"] = "violin_swarm_thing"
    path.write_text(json.dumps(manifest, indent=2) + "\n", newline="\n")
    _write_csv(root / "design" / "figures" / "sources" / "F4_c80_reliability.csv",
               ["role_id", "proj_on_axis_B", "proj_on_axis_A"], ["pirate", "0.3", "0.2"])

    # Never "rendered": true for a chart the renderer cannot draw.
    with pytest.raises(renderer.UnsupportedChartType):
        renderer.render(root, "F4_c80_reliability", root / "out.png")
    assert renderer.check_ready(root, "F4_c80_reliability")["ready"] is False


def test_every_declared_chart_type_is_implemented(built, renderer):
    manifest = built[0]["figures_manifest"]
    declared = {p["chart_type"] for f in manifest["figures"] for p in f["panels"]}
    assert declared <= set(renderer.CHART_RENDERERS), (
        declared - set(renderer.CHART_RENDERERS)
    )


def test_categorical_grouped_bar_is_not_read_as_numeric(built, renderer):
    # F2's second column is `channel`, a category. The old generic fallback read
    # column 2 as a number; the render_spec now names model/channel/rate roles.
    _, root = built
    _write_csv(root / "design" / "figures" / "sources" / "F2_channel_2x2.csv",
               ["model", "channel", "full_role_rate", "ci_low", "ci_high", "n"],
               ["deepseek", "user", "0.42", "0.38", "0.46", "1000"],
               ["deepseek", "system", "0.51", "0.47", "0.55", "1000"],
               ["llama", "user", "0.33", "0.29", "0.37", "1000"],
               ["llama", "system", "0.40", "0.36", "0.44", "1000"])
    assert renderer.check_ready(root, "F2_channel_2x2")["ready"] is True
    result = renderer.render(root, "F2_channel_2x2", root / "f2.png")
    if result.get("rendered"):
        assert (root / "f2.png").stat().st_size > 0
    else:
        assert "matplotlib unavailable" in result["reason"]
