# T18 nuisance baseline: response-region lengths

## Status

**Post-primary descriptive sensitivity. The frozen T18 verdict remains `INADEQUATE`.**

This diagnostic asks whether the large predictive advantage of the frozen
full-dimensional T18 linear readout can be reproduced by trivial generation
length information.

## Exact design

- Same frozen T18/G2 population: 44,795 rows.
- 275 role groups, 5 default conditions, 160 questions.
- Same 5 x 4 doubly grouped outer folds and 4 x 3 grouped inner folds.
- Same split seed `20260812`.
- Same role-balanced fitting/evaluation weights.
- Same regularized logistic-regression C grid: `[0.001, 0.01, 0.1, 1.0, 10.0]`.
- Fold-local feature standardization, as in T18.
- Same mean within-fold role-balanced AUROC estimand.
- Same 2,000-replicate role-clustered bootstrap, used descriptively here.
- No new pass/fail threshold and no change to the T18 `INADEQUATE` verdict.

## Results

| Diagnostic | AUROC | Descriptive 95% role-bootstrap CI |
|---|---:|---:|
| Total generated response tokens | 0.487908 | [0.459278, 0.514597] |
| Reasoning + final-answer token counts | 0.534755 | [0.520994, 0.547999] |
| Frozen Assistant Axis (reference) | 0.678327 | [0.657729, 0.698841] |
| Frozen full-dimensional linear (reference) | 0.878197 | [0.865317, 0.890051] |

## Paper-safe wording

Using the same doubly grouped folds, role-balanced estimator, and inner-selected regularized logistic regression as T18, generated response length alone achieved AUROC 0.488 (95% role-bootstrap CI [0.459, 0.515]), while reasoning- and final-answer-token counts jointly achieved 0.535 (95% CI [0.521, 0.548]). For reference, the frozen Assistant-Axis and full-linear T18 readouts achieved 0.678 and 0.878, respectively. This post-primary nuisance diagnostic does not alter the frozen T18 verdict.

## Claim boundary

This analysis tests a simple nuisance explanation. It does not identify the
semantic content used by the full-dimensional classifier, prove that length is
causal, residualize the 4,096 activation dimensions, alter the frozen T18
population, or create a second adequacy verdict.

## Provenance

- Source Git SHA: `f076aa87b856c759fed1d1994958f55f94883247`
- Source tree dirty: `False`
- Frozen HF revision: `e1885e0865d1e6d7a39a9c2585b224b30e28b3f1`
- T18 config SHA-256: `8a3c93deef7d87d92da94a7fb28413008e85eef89c81e035864ba010d7b34cdc`
- G2 SHA-256: `54459154cf3a4dbdd4d44c2f6c1504eed8fce2b7f6c38384da95a98f47112937`
- Frozen T18 report SHA-256: `7cb2fc71f821290e7fea3b0cac1885b9c7ed821d5e55c473d82dc8487b060805`
- T14 retained-role SHA-256: `3c7d373cb67f96ac21d9b6cbd9097ce184d145b9d381af1ad12b3e0675524315`
