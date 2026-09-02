# T29 final finding — Qwen source-model causal specificity

## Status

**COMPLETE.** T29 is closed under the measurement branch fixed by T27.

T27 harmfulness is `MEASUREMENT_LIMITED`, so the adjudicated 120-item human subset is the primary harmfulness evidence. The fully matched 400-row automatic-label panel is retained only as a secondary sensitivity. Identity remains outside the T29-H primary estimator and retains the two-positive, adjudicator-dependent caveat from T27.

## Human-primary result

The frozen primary estimand is strict harmful-compliance risk difference,

`assistant_cap - random_direction_cap`,

with negative values favoring the Assistant-Axis cap. The design-weighted human rates are 4.0% for `assistant_cap` and 7.6923% for `random_direction_cap`, giving a risk difference of **-0.036923** with frozen stratified-bootstrap 95% CI **[-0.110769, 0.000000]**.

This is directional evidence in the preregistered direction, but the interval reaches zero. The corresponding human strict-harm comparisons against `sphere_proximity_cap` and `unsteered_baseline` are both **+0.0100**, so the human-primary data do not show a general harmfulness reduction relative to those controls.

Inclusive-harm sensitivity is weaker: Assistant minus random is **-0.026923**, 95% CI **[-0.100769, 0.010000]**.

## Frozen intervention diagnostics

T24 matched the primary random and spherical controls to the measured source-cap engagement incidence before T29 outcomes were inspected. Across layers 46–53, source engagement ranges from **0.4320 to 0.5959**, and both project controls match the source engagement rate exactly at each layer in the frozen calibration record. This supports an incidence-matched comparison, not equality of intervention magnitude or geometry.

The automatic full-corpus degeneration diagnostic is 0% for unsteered, Assistant cap, and random-direction cap, and 2% for the spherical-distance cap. There are zero final judge coverage failures in all four conditions.

## Full-corpus automatic sensitivity

The exact production scorer artifact has SHA-256 `d7f8feb34f8c2e8e8fff57764c95ff81b8128fbfa35d87c380322dce07565bd3` and contains 400 rows, 100 per frozen condition, with zero final coverage failures. Its production UIDs encode the matched underlying item as `qwen_capping::<item>::<condition>`. `question_id` is null in this artifact, so pairing is recovered fail-closed from that production UID contract rather than fabricated from a missing field. The runner also requires role, role description, question, arm, and `question_id` to agree across the four rows for each underlying item.

Automatic rates are:

| condition | strict harm | inclusive harm | refusal | degeneration |
|---|---:|---:|---:|---:|
| unsteered | 2% | 4% | 96% | 0% |
| Assistant cap | 3% | 5% | 95% | 0% |
| random-direction cap | 2% | 4% | 96% | 0% |
| spherical-distance cap | 1% | 4% | 94% | 2% |

For the paired automatic Axis-vs-random strict-harm sensitivity, the risk difference is **+0.0100**, bootstrap 95% CI **[0.0000, 0.0300]**, with one Assistant-only harmful discordance, zero random-only harmful discordances, and exact two-sided McNemar **p = 1.0**. Axis-vs-sphere is +0.0200, 95% CI [-0.0200, 0.0600], p = 0.625. Axis-vs-unsteered is +0.0100, 95% CI [0.0000, 0.0300], p = 1.0.

These numbers are not primary evidence because T27 failed the harmfulness validation gate on IP-weighted recall and macro-F1. They are useful as a sensitivity because they do not corroborate the favorable human Axis-vs-random point estimate.

## Interpretation and claim boundary

T29 **does not establish robust causal specificity for harmfulness reduction**. The human-primary estimate is directionally favorable against the random control but imprecise, while the matched automatic sensitivity points slightly in the opposite direction and the human-primary analysis does not show lower strict harm than sphere or unsteered.

Permitted paper wording:

> On the design-weighted human subset, Assistant-Axis capping reduced estimated strict harmful compliance by 3.7 percentage points relative to the matched random-direction cap, but the bootstrap interval reached zero. The paired 400-row automatic-label sensitivity did not corroborate this pattern, and the human-primary estimate did not show lower strict harm than the spherical control or unsteered baseline. We therefore do not claim a robust direction-specific harmfulness reduction.

Do not state that the Assistant Axis generally reduces harmfulness, that it outperforms matched controls, or that identity mediates harmfulness.

## Frozen artifacts

- Primary human result: `results/t29/t29_human_primary.json`.
- Secondary automatic sensitivity: `results/t29/t29_automatic_sensitivity.json`.
- Final synthesis: `results/t29/t29_final_synthesis.json`.
- Closure validation: `results/t29/T29_CLOSURE_VALIDATION.json`.
- Paper-ready result: `docs/T29_PAPER_RESULT.md`.
- Figure/table source data: `design/figures/sources/t29/`.

The private 400-row scored JSONL remains off-repo. No item IDs, completions, judge rationales, or item-level automatic labels joined to condition are committed.
