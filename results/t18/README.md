# T18 — frozen dimensional adequacy and nonlinearity stress test

## Status

**Production complete and reportable. Verdict: `INADEQUATE`.**

T18 asks a narrow predictive question: is the single Assistant Axis an adequate summary of the frozen default-Assistant-versus-role-prompted activation distinction when both roles and questions are held out? It is diagnostic, not causal. The committed repository artifact is [`t18_report.json`](./t18_report.json), **23,535 bytes**, SHA-256 `7cb2fc71f821290e7fea3b0cac1885b9c7ed821d5e55c473d82dc8487b060805`. The exact raw JSON emitted by the GCP production runner was **32,307 bytes**, SHA-256 `d4b0eea57b1ef95e9a2591c17d84a06e1e37e21021af78ddf11974f1171cbc21`. The committed file is a whitespace-normalized serialization of the same parsed JSON object; the two byte identities are deliberately recorded separately so repository binding is unambiguous.

## Frozen confirmatory slice

- Model geometry: DeepSeek-R1-Distill-Llama-8B confirmatory C80 geometry
- Prompt arm: `USER_TRANSLATED_LU`
- Blocks: C80-A and C80-B
- Transformer block: 16
- Pool: `ALL_RESPONSE_TOKENS`
- Membership: `label_independent_technical_validity`
- Automatic role-judge filtering: **not applied**
- Genuine roles: **275**
- Default conditions: **5**
- Distinct questions: **160**
- Retained response rows: **44,795**
  - role-prompted: **43,998**
  - default-Assistant: **797**
- Wrong-slice / invalid / ineligible-role / missing-vector exclusions at the T18 loader: **0 / 0 / 0 / 0**

The loader therefore scored the full materialized frozen G2 population.

## Frozen design

The production analysis uses the predeclared T18 configuration unchanged:

- doubly grouped outer evaluation: **5 role folds × 4 question folds = 20 outer folds**;
- in every outer fold, both the held-out roles and held-out questions are unseen during training;
- hyperparameters are selected only inside the outer training data using **4 role folds × 3 question folds**;
- primary metric: mean within-fold, role-balanced AUROC;
- one-dimensional baseline: fold-local Assistant Axis;
- alternatives: PCA-linear, full-dimensional regularized linear, and restrained one-hidden-layer nonlinear readouts;
- role-clustered bootstrap: **2,000 replicates**;
- multiplicity rule: simultaneous max-over-families comparison against the Axis;
- adequacy margin: **0.02 AUROC**;
- verdicts: `ADEQUATE`, `INADEQUATE`, or `INCONCLUSIVE`.

PCA, centering, scaling, hyperparameter selection, and the Axis itself are fitted inside training folds only.

## Production result

| Readout | AUROC | Role-bootstrap 95% CI | Question-clustered diagnostic 95% CI |
| --- | ---: | ---: | ---: |
| Assistant Axis (`axis_1d`) | 0.678327 | [0.657729, 0.698841] | [0.662404, 0.699461] |
| Full-dimensional linear | 0.878197 | [0.865317, 0.890051] | [0.859528, 0.900424] |
| PCA-linear | 0.669973 | [0.652356, 0.688146] | [0.652296, 0.690469] |
| Small nonlinear | 0.666687 | [0.651761, 0.681471] | [0.647829, 0.687665] |

The strongest alternative was the full-dimensional linear readout:

- Axis AUROC: **0.678327**
- Full-linear AUROC: **0.878197**
- Full-linear minus Axis: **+0.199869**
- simultaneous max-family 95% CI: **[0.180913, 0.220654]**
- predeclared adequacy margin: **0.020000**
- verdict: **`INADEQUATE`**

The lower simultaneous bound (0.181) is far above the 0.02 margin, so the verdict is not borderline.

### Alternative families relative to the Axis

| Family | AUROC difference vs Axis | Marginal 95% CI |
| --- | ---: | ---: |
| Full linear | +0.199869 | [0.180913, 0.220654] |
| PCA linear | -0.008354 | [-0.020058, 0.003282] |
| Small nonlinear | -0.011640 | [-0.032708, 0.010216] |

Only the full-dimensional linear family improves on the Axis. The leading-PC linear readout and the restrained nonlinear readout are slightly below the Axis on the frozen primary estimand.

## Hyperparameter selections across the 20 outer folds

- Axis: `axis_1d` in **20/20** folds.
- Full linear:
  - `C=0.1` in **17/20** folds;
  - `C=0.01` in **3/20** folds.
- PCA linear: `k=32, C=0.001` in **20/20** folds.
- Small nonlinear:
  - `k=32, hidden=4, alpha=10.0`: **11/20**;
  - `k=32, hidden=4, alpha=0.1`: **6/20**;
  - `k=20, hidden=4, alpha=1.0`: **1/20**;
  - `k=32, hidden=4, alpha=1.0`: **1/20**;
  - `k=20, hidden=4, alpha=10.0`: **1/20**.

Each family scored all **44,795** rows exactly once across the 20 outer test folds. Outer training sets contain 26,875–26,879 rows; held-out test folds contain 2,238–2,240 rows; 15,676–15,680 rows per fold are deliberately unused because they share exactly one of the held-out role/question grouping dimensions.

## Diagnostics

The descriptive grouped-label permutation diagnostic has mean AUROC **0.505365** across 20 draws, close to chance. Its maximum absolute deviation from 0.5 is **0.106507**. This diagnostic is explicitly descriptive and is not used as an inferential p-value.

The production log contains sklearn L-BFGS convergence warnings for some small-nonlinear fits at the frozen `max_iter=2000`. The production process nevertheless exited successfully. These warnings do not determine the verdict: the winning family is the regularized full-dimensional linear readout, while the small-nonlinear family performs slightly below the Axis.

## Interpretation

The result supports a specific distinction between **reproducibility** and **predictive sufficiency**.

Earlier confirmatory geometry shows that Assistant-related role ordering is highly reproducible across independent question blocks. T18 shows that this reproducible one-dimensional ordering does **not** exhaust the held-out linearly decodable information distinguishing default-Assistant from role-prompted response activations. A full-dimensional regularized linear readout improves AUROC by about **0.20**.

This does **not** imply that:
- the representation has one particular higher-dimensional basis;
- T18 identifies the minimum number of dimensions required;
- nonlinear structure is necessary;
- the full 4,096-dimensional readout is a mechanistic explanation;
- the extra predictive information is causally necessary for behavior;
- the retained rows constitute behaviorally verified role enactment.

The PCA-linear result should also not be compared numerically with the T17 descriptive role-space PCA component count. T18 PCA is fitted on training **response vectors** for predictive evaluation, whereas T17 PCA is descriptive and fitted to role-level geometry. They answer different questions.

## Manuscript-ready statement

A compact result statement:

> A single Assistant direction was not predictively adequate on held-out roles and questions. The Axis achieved AUROC 0.678, while a regularized full-dimensional linear readout achieved 0.878, a gain of 0.200 AUROC with a simultaneous 95% CI of [0.181, 0.221]. The interval lies well above the predeclared 0.02 adequacy margin. PCA-based linear and restrained nonlinear readouts did not improve over the Axis.

Recommended interpretation immediately after it:

> The stable ordering captured by the Assistant Axis therefore does not exhaust the linearly decodable default-Assistant-versus-role-prompted structure. This is evidence about predictive summary adequacy, not a causal claim or an estimate of the representation's intrinsic dimensionality.

## Provenance

- Reportable: `true`
- Exact raw production report: **32,307 bytes**, SHA-256 `d4b0eea57b1ef95e9a2591c17d84a06e1e37e21021af78ddf11974f1171cbc21`
- Exact committed `results/t18/t18_report.json`: **23,535 bytes**, SHA-256 `7cb2fc71f821290e7fea3b0cac1885b9c7ed821d5e55c473d82dc8487b060805`
- Serialization relation: the committed JSON is whitespace-normalized from the raw production JSON; numerical values, selections, provenance fields, diagnostics and verdict are unchanged
- Production source Git SHA: `e24847c0659d5c5cd66af33913a6c00524a006d6`
- Production source tree dirty: `false`
- Computation backend: `multiprocessing_fork`
- Outer workers: `4`
- Production VM used for the parallel run: `n2-highmem-32` in `us-east1-b`
- G2 freeze SHA-256: `54459154cf3a4dbdd4d44c2f6c1504eed8fce2b7f6c38384da95a98f47112937`
- T18 frozen-config SHA-256: `8a3c93deef7d87d92da94a7fb28413008e85eef89c81e035864ba010d7b34cdc`
- Frozen method/config SHA-256 stamped by the runner: `f7e7dd9df8c7c72d5da08bf8201295dfc91f3208837d8cd0867054c393984e59`
- Eligible-role artifact SHA-256: `68be9fba61c9ac88da45c589d9892dbd374911efd82a2301f405d3f207ed3cea`

`tests/test_t18_result_binding.py` now fails if the exact committed report bytes drift from the hash/size recorded in [`t18_production_receipt.json`](./t18_production_receipt.json), and also checks the report's source SHA, frozen authorities and verdict against the receipt.

See [`t18_production_receipt.json`](./t18_production_receipt.json) for the machine-readable execution/result summary.

## Claim boundary for downstream use

**Supported:** the single Assistant Axis is predictively inadequate as a summary of the frozen prompt-conditioned default-versus-role activation distinction on unseen roles and unseen questions.

**Not supported:** causal necessity, a unique global orientation, a unique higher-dimensional geometry, a specific minimum dimensionality, nonlinear necessity, or behaviorally verified role enactment.
