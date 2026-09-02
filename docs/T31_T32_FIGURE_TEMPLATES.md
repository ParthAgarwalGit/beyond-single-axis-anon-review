# T31 / T32 — Figure & table templates and claim-branch scaffold

**T31 Owner:** [Reviewer] and [Author A] · **T32 Owner:** Team (Reviewer: [Advisor])
**Status:** templates ready · **claims DRAFT (not frozen)**
**Compute:** CPU only

Builds the **contracts** for the nine minimum T31 figures and the key result
tables, and scaffolds the seven T32 claim branches. It fabricates **no**
numbers, and it destroys none: each panel declares its saved-source-data
columns and renders only once real results arrive.

## Deliverables

| File | Purpose |
|---|---|
| `tools/build_t31_figure_templates.py` | Freezes figure/table contracts; creates missing header-only source CSVs |
| `tools/render_t31_figure.py` | Renders a figure from its saved source data; refuses fabrication and unimplemented chart types |
| `design/figures_manifest.json` | 9 figures (10 panels) + 4 tables, each with columns, render spec, producing task, supported claims |
| `design/claims_T32.json` | 7 core evidence branches with gate **outcomes** + venue emphasis (DRAFT) |
| `design/figures/sources/*.csv` | Header-only source-data contracts, one per panel |
| `design/tables/sources/*.csv` | Header-only table contracts |
| `tests/test_t31_figure_templates.py` | 27 CPU-only checks |

## The nine minimum figures (TODO T31)

| # | Figure | Panels | Chart type(s) |
|---|---|---|---|
| F1 | evidence chain | 1 | `dag` |
| F2 | two-model × two-channel comparison | 1 | `grouped_bar` |
| F3 | role retention and judge confusion | **2** | `heatmap` + `bar` |
| F4 | C80-A / C80-B reliability | 1 | `scatter` |
| F5 | all-response vs answer-only | 1 | `scatter` |
| F6 | wrapper vs translated | 1 | `scatter` |
| F7 | Axis vs random causal effects | 1 | `dose_response` |
| F8 | harm and identity by steering | 1 | `grouped_bar` |
| F9 | degeneration by condition | 1 | `bar` |

Every **panel** declares `required_columns`, a `render_spec` mapping those
columns onto chart roles, its `producing_task`, and the T32 claim branch(es) the
figure supports. Four result tables carry the same contract shape and are
validated just as strictly.

### F3 is two panels, because one table cannot draw two things

F3 promises retention *and* judge confusion. A single
`human_label, judge_label, count` table can reproduce the confusion matrix and
nothing else, so retention has its own saved source
(`role_id, n_eligible, n_score3, retention_rate, ci_low, ci_high`, from T14).
A multi-panel figure is ready only when **every** panel is ready.

### F7 stores the shared zero exactly once

`shared_zero` is one condition. It is stored as a **single row** with
`direction = "shared_zero"`, never duplicated per direction to make two curves
draw. The frozen two-slope encoding (`analysis_registry` P4/P5) sets
`axis_dose = random_dose = 0` on that one row, and both curves are drawn through
it at render time. Duplicating it would double-count the shared zero in any
count taken from this source.

### F2 does not name a results file that does not exist

F2 and `T_channel_calibration` are attributed to the frozen analysis-registry
entry `P2_CHANNEL_INTERACTION` (T19). The concrete results filename is not fixed
anywhere in the repository, so `producing_artifact` is `NOT_YET_DEFINED` rather
than an unverified name.

## "Every figure must have saved source data"

`build_t31_figure_templates.py` creates a **header-only** CSV per panel. On a
rerun it **creates only what is missing**:

- a populated source CSV is **kept**, and only its header is re-validated —
  rerunning the builder can never silently delete real result rows;
- a source whose header has drifted from the contract raises rather than being
  rewritten, so a schema change is an explicit migration;
- a `claims_T32.json` or `figures_manifest.json` that has been **frozen** is
  never rewritten back to draft — freezing is one-way.

`render_t31_figure.py` refuses to draw when a panel's source CSV is missing,
header-only, or its header does not match the frozen contract, so a figure is
never produced from absent or invented data.

### No generic fallback chart

Every declared chart type is implemented — `dag`, `scatter`, `bar`,
`grouped_bar`, `dose_response`, `heatmap` — and each is drawn from the panel's
`render_spec`, which names the column playing each role. An unimplemented chart
type **raises**; it is never drawn as a fallback bar that reports
`rendered: true`. That mattered concretely: F2's second column is `channel`, a
category, and the old fallback path tried to read it as a number while still
reporting success.

The builder enforces the same contract from the other side: a panel may not
declare a chart type the renderer lacks, and every `render_spec` column must
exist in that panel's own `required_columns`.

```bash
python tools/build_t31_figure_templates.py
```

```bash
python tools/render_t31_figure.py --figure F4_c80_reliability --check-only
```

## Contracts checksum

`contracts_sha256` hashes a canonical representation of the **complete** figure
and table contracts — columns, chart types, render specs, producing tasks, claim
mappings and source paths — excluding only the volatile `status` field. Hashing
figure IDs plus figure-column hashes alone would have left table-schema, chart
type and claim-mapping changes invisible to the checksum.

## T32 claim branches (DRAFT — do not freeze yet)

Seven evidence branches. Each declares the **gate outcome it represents**, not
merely which gates are relevant:

| branch | gate expectations |
|---|---|
| `validated_transfer` | P1 PASS, G1 PASS, P3 PASS, G2 PASS |
| `prompt_conditioned_geometry` | P2 PASS, G2 PASS |
| `reliable_but_not_causally_specific` | P3 **PASS**, P4 **FAIL** |
| `security_effect_without_identity_specificity` | P4 **PASS**, P5 **FAIL** |
| `measurement_limited_result` | G2 **INCOMPLETE** |
| `one_model_pilot` | T20 **INCOMPLETE**, T21 **INCOMPLETE** |
| `negative_result` | `ANY_PRIMARY_GATE` **FAIL** |

Gate states are `PASS | FAIL | INCOMPLETE | NOT_APPLICABLE`. A branch describing
a failed or non-specific result now requires the corresponding gate to fail —
previously these listed the gate they contradict as one that must pass. A branch
with an empty expectation map is rejected by the builder, because it would be
eligible under every outcome, including one where everything passed.

Two wording corrections travel with that:

- **`prompt_conditioned_geometry`** separates its two distinct pieces of
  evidence — the *behavioural* channel interaction (P2, T19, F2) and the
  *geometric* wrapper-versus-translated projection difference (T15, F6) — and
  says explicitly that they are reported separately, never merged into one
  channel-effect claim.
- **`measurement_limited_result`** no longer says limits are measurement
  "rather than representational structure", which asserted an exclusion the gate
  does not establish. It now says the evidence *does not separate* the two, so
  no structural conclusion is drawn in either direction.

**Frozen only at T32, after T28, one branch per outcome** — exactly one branch
is selected, and a branch is eligible only when every entry in its
`gate_expectations` matches the observed outcome. Judge-dependent branches must
not be frozen before **T06–T07–G1**; causal branches not before **T28**. This
scaffold changes no gate and asserts no result.

Venue emphasis records NeurReps (primary) plus FLMSec / IAB / Verify Agents /
EIML (secondary).

## Reproduce

```bash
python tools/build_t31_figure_templates.py
```

```bash
python -m pytest tests/test_t31_figure_templates.py -q
```

Branched from `main` (independent of the generation/judge PRs).
