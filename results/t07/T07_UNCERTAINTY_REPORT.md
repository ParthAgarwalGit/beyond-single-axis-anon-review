# T07 role-judge uncertainty intervals

## Status

- Point-estimate reproduction: **PASS**
- Bootstrap: **10,000** stratified replicates
- Seed: `20260830`
- CI: percentile 95% (`numpy.quantile`, method `linear`)
- Frozen measurement branch: `measurement_limited_result`
- Gate rule: unchanged. Confidence-interval overlap with a threshold is not a pass/fail rule.

## Frozen point-estimate reproduction

- total human-reference items: 300
- scored: 278
- abstained: 10
- human-unjudgeable: 12

## Paper-facing pooled table

| metric | estimate | 95% CI | frozen threshold | pass/fail |
|---|---:|---:|---:|:---:|
| Score-3 precision | 0.556 | [0.456, 0.656] | 0.85 | **FAIL** |
| IPW score-3 recall | 0.798 | [0.679, 0.918] | 0.85 | **FAIL** |
| IPW four-class macro-F1 | 0.409 | [0.365, 0.455] | 0.75 | **FAIL** |
| Four-class Cohen's kappa | 0.233 | [0.165, 0.301] | 0.70 | **FAIL** |

All four PASS/FAIL entries above reproduce the existing frozen point-estimate gate. The intervals report uncertainty around the frozen validation result and do not change the selected branch.

## Bootstrap design

- Resampling unit: labelled T06/T07 item.
- Original design authority: stratified simple random sampling within `arm × archived automatic lu_score`.
- Each of the 10 original strata is resampled independently with replacement.
- Every replicate preserves the original labelled sample size of every stratum.
- Each sampled item retains its frozen inverse-probability weight.
- Scored/abstained/unjudgeable status is recomputed from each sampled item using the existing T07 definitions.
- Metric computation calls the canonical `tools/run_t07_judge_validation.py` `_metrics` helper.
- Score-3 precision remains unweighted; recall and four-class macro-F1 remain IP-weighted; Cohen's kappa remains unweighted.
- The canonical precision/recall/F1 helper uses scikit-learn's frozen `zero_division=0` convention. Truly undefined/non-finite outputs (principally possible for kappa in degenerate replicates) are recorded as undefined, never silently replaced.

### Frozen strata (aggregate only)

| stratum | labelled n | source N | IP weight |
|---|---:|---:|---:|
| `extraction|0` | 25 | 570 | 22.800000 |
| `extraction|1` | 25 | 12392 | 495.680000 |
| `extraction|2` | 45 | 293 | 6.511111 |
| `extraction|3` | 45 | 8706 | 193.466667 |
| `extraction|None` | 10 | 39 | 3.900000 |
| `translated|0` | 25 | 1334 | 53.360000 |
| `translated|1` | 25 | 11263 | 450.520000 |
| `translated|2` | 45 | 1153 | 25.622222 |
| `translated|3` | 45 | 8211 | 182.466667 |
| `translated|None` | 10 | 39 | 3.900000 |

## Valid and undefined bootstrap replicates

| block | metric | valid | undefined/degenerate |
|---|---|---:|---:|
| pooled | Score-3 precision | 10000 | 0 |
| pooled | IPW score-3 recall | 10000 | 0 |
| pooled | IPW four-class macro-F1 | 10000 | 0 |
| pooled | Four-class Cohen's kappa | 10000 | 0 |
| by_arm:extraction | Score-3 precision | 10000 | 0 |
| by_arm:extraction | IPW score-3 recall | 10000 | 0 |
| by_arm:extraction | IPW four-class macro-F1 | 10000 | 0 |
| by_arm:extraction | Four-class Cohen's kappa | 10000 | 0 |
| by_arm:translated | Score-3 precision | 10000 | 0 |
| by_arm:translated | IPW score-3 recall | 10000 | 0 |
| by_arm:translated | IPW four-class macro-F1 | 10000 | 0 |
| by_arm:translated | Four-class Cohen's kappa | 10000 | 0 |

## Arm-specific intervals

### extraction

| metric | estimate | 95% CI |
|---|---:|---:|
| Score-3 precision | 0.622 | [0.467, 0.756] |
| IPW score-3 recall | 0.815 | [0.656, 0.964] |
| IPW four-class macro-F1 | 0.447 | [0.369, 0.518] |
| Four-class Cohen's kappa | 0.219 | [0.124, 0.314] |

### translated

| metric | estimate | 95% CI |
|---|---:|---:|
| Score-3 precision | 0.489 | [0.355, 0.644] |
| IPW score-3 recall | 0.776 | [0.595, 0.954] |
| IPW four-class macro-F1 | 0.385 | [0.326, 0.443] |
| Four-class Cohen's kappa | 0.246 | [0.153, 0.342] |

## Provenance and privacy

- source Git SHA: `cf528a3248ff33f5ea8935bcca911b3c1830ca76`
- method/config SHA-256: `c1de356c2c57f60c2cc53bea5e95231eac89e300bad7f0439497fae4e2175654`
- final consensus SHA-256: `b67d10e683d888948e42762b9b35829ed10c736d818b0bebcfab632ca66cfc44`
- frozen T07 report SHA-256: `5394754a92da31957f4204bce3a0ac429404397ca2ece91162f7c6c41f8c012d`
- original private sampling key SHA-256: `5eab10c2e9bb6ce07eee2853c692c3291ce95e162e1b4de35a3fa44964f15999`
- The private key itself, item IDs joined to labels/scores, and per-item stratum assignments are not written to this report or result JSON.

## Paper implication

The uncertainty intervals complete reporting around the existing T07 validation result. They do not reopen the judge gate: the production automatic role judge remains on the frozen `measurement_limited_result` branch and is not used to define confirmatory C80 membership.

Human-human agreement confidence intervals are intentionally not added here; that optional secondary extension is kept separate so it cannot delay or alter the automatic-judge uncertainty task.
