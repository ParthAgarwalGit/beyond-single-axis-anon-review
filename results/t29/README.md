# T29 — Qwen source-model Axis-vs-control causal specificity

T29 is **complete** under the measurement branch fixed by T27. It compares the frozen Qwen3-32B Assistant-Axis cap with the matched orthogonal-random cap, spherical-distance cap, and unsteered baseline.

## Frozen population and upstream authorities

The experiment contains the same 100 evaluation items under four conditions, 400 rows total:

1. `unsteered_baseline`
2. `assistant_cap`
3. `random_direction_cap`
4. `sphere_proximity_cap`

The 400-row generation population is archived at immutable Hugging Face revision `e29553fe75a7d89757258e7c6c38723de1960ca9`, with SHA-256 `24a0c5a3699fd6b51a66d4be92bc054d14cb800db894c0162b65774c5667f06b`. T21 remains authoritative for the source Assistant-Axis condition and T24 freezes the matched controls.

The exact production automatic scorer artifact has SHA-256 `d7f8feb34f8c2e8e8fff57764c95ff81b8128fbfa35d87c380322dce07565bd3`. Production scoring covered all 400 rows with zero final coverage failures and a stable start/end canary.

## Measurement branch

T27 harmfulness is `MEASUREMENT_LIMITED`. The adjudicated T26 human subset is therefore the **primary harmfulness evidence**. The 400-row automatic harmfulness panel is **secondary sensitivity only**.

The human sample contains 120 independently sampled outputs, 30 per condition, stratified within `condition x automatic_harmfulness` cells. It is not a matched four-condition human panel, so it uses the frozen inverse-probability design and unpaired stratified bootstrap rather than McNemar or a paired item bootstrap.

## Human-primary result

Primary outcome: strict harmful compliance.

Primary contrast: `assistant_cap - random_direction_cap`; negative values favor the Assistant cap.

Design-weighted strict rates are:

- unsteered: 3.0%
- Assistant cap: 4.0%
- random-direction cap: 7.6923%
- spherical-distance cap: 3.0%

The primary Assistant-minus-random risk difference is **-0.036923**, with 10,000-replicate stratified-bootstrap 95% CI **[-0.110769, 0.000000]**. The point estimate is favorable to the Assistant cap, but the interval reaches zero.

Human strict Assistant-minus-sphere and Assistant-minus-unsteered are both **+0.0100**. Inclusive Assistant-minus-random is **-0.026923**, 95% CI **[-0.100769, 0.010000]**.

Frozen artifact: `results/t29/t29_human_primary.json`.

## Full-corpus automatic sensitivity

The production scored artifact has null `question_id` for all 400 rows. Its UID is nevertheless a frozen pairing identifier of the form `qwen_capping::<underlying-item>::<condition>`. The dedicated secondary runner validates that contract, requires exactly 100 common underlying items across all four conditions, and refuses coverage failures or labels outside the frozen harmfulness schema (`harmful`, `partial`, `refused`, `degenerate`).

Run from a checkout containing the completed T29 code:

```bash
python tools/run_t29_automatic_sensitivity.py \
  --scored-private [local-path]
  --out results/t29/t29_automatic_sensitivity.json
```

The runner verifies the private artifact against the committed production receipt, T26 manifest, T27 gate, matched-control execution, and human-primary result. It also reproduces T27's IP-weighted automatic strict-harm rate for every condition to tolerance `1e-12`.

Automatic condition rates are:

| condition | strict harm | inclusive harm | refusal | degeneration |
|---|---:|---:|---:|---:|
| unsteered | 2% | 4% | 96% | 0% |
| Assistant cap | 3% | 5% | 95% | 0% |
| random-direction cap | 2% | 4% | 96% | 0% |
| spherical-distance cap | 1% | 4% | 94% | 2% |

The paired automatic Assistant-minus-random strict-harm diagnostic is **+0.0100**, bootstrap 95% CI **[0.0000, 0.0300]**, exact two-sided McNemar **p = 1.0**. Assistant-minus-sphere is +0.0200, 95% CI [-0.0200, 0.0600], p = 0.625. Assistant-minus-unsteered is +0.0100, 95% CI [0.0000, 0.0300], p = 1.0.

Frozen artifact: `results/t29/t29_automatic_sensitivity.json`. It is mechanically marked `primary_result_eligible: false` and cannot override the human-primary evidence.

The older `tools/run_t29_specificity.py` remains for the originally contemplated `VALIDATED` automatic-scorer route. It is not the completed harmfulness route after T27 selected `MEASUREMENT_LIMITED`.

## Final interpretation

T29 **does not establish robust direction-specific harmfulness reduction**. The human-primary Axis-vs-random estimate is directionally favorable but imprecise, the paired full-corpus automatic sensitivity does not corroborate that direction, and the human-primary analysis does not show lower strict harm than sphere or unsteered.

This is a completed negative/limited causal-specificity result, not an unfinished experiment.

Final synthesis: `results/t29/t29_final_synthesis.json`.

Final finding and paper-safe wording: `docs/T29_FINAL_FINDING.md`.

Closure validation: `results/t29/T29_CLOSURE_VALIDATION.json`.

## Figure/table source export

```bash
python tools/export_t29_figure_source.py \
  --human results/t29/t29_human_primary.json \
  --automatic results/t29/t29_automatic_sensitivity.json \
  --out-dir design/figures/sources/t29
```

The exporter emits only aggregate data. The preferred causal figure should foreground the human-primary estimate and mark the 400-row automatic result as secondary.

## Identity boundary

Identity is separate from the T29-H primary harmfulness estimator. T27 mechanically returns `VALIDATED` for Qwen identity, but that result rests on only two human-positive items and both are adjudicator-dependent residual cases. It should not be promoted into a strong full-corpus identity claim, and T29 does not infer that identity mediates harmfulness.

## Privacy boundary

The private 400-row scored JSONL, T26 private sampling key, item IDs, completions, judge rationales, and item-level labels joined to condition are not committed. Committed T29 outputs are aggregate-only and retain the relevant frozen SHA-256 bindings.
