# T17 regional independent-recovery sensitivity

Status: **POST_PRIMARY_SENSITIVITY_COMPLETE**

This analysis asks whether reasoning-only and final-answer-only assistant directions
are each reproducible across the disjoint C80-A/C80-B question blocks. It uses the
same 275-role matched cohort in both response regions, block-specific regional
default vectors, and the same cross-axis role-projection definition as T15/T16.

## Results

| Region | C80-A/B axis cosine | Cross-built Pearson r | 95% role-bootstrap CI | Spearman |
|---|---:|---:|---:|---:|
| Reasoning | 0.924677 | 0.973794 | [0.967474, 0.978937] | 0.972108 |
| Final answer | 0.834872 | 0.960060 | [0.951272, 0.967839] | 0.960806 |

Reasoning-vs-final-answer geometry also appears separately in each block:

- C80-A reasoning/answer axis cosine: **0.584119**;
  same-role own-axis Pearson r: **0.956744**.
- C80-B reasoning/answer axis cosine: **0.586893**;
  same-role own-axis Pearson r: **0.947056**.
- Existing pooled C160 reasoning/answer axis cosine: **0.587784**;
  same-role own-axis Pearson r: **0.954637**.

## Paper-ready factual wording

Reasoning-only assistant axes estimated independently from C80-A and C80-B have
cosine similarity **0.925**, while their cross-built role projections correlate at
**r=0.974** (95% role-bootstrap CI **[0.967, 0.979]**). Final-answer axes have
cosine similarity **0.835**, with cross-built role correlation **r=0.960**
(95% CI **[0.951, 0.968]**).

These results rule out unstable regional axis estimation as a simple explanation
for the pooled reasoning-versus-answer difference: each region independently
recovers strongly across the two question blocks, while reasoning-versus-answer
axis cosine remains about 0.585 within each block. This is still a response-region
sensitivity, not a new primary result and not a causal test of reasoning-to-answer
propagation.

## Reproducibility and authority

The committed scientific authorities are:

- `tools/run_t17_regional_independent_recovery.py` — reproducible generator;
- `regional_independent_recovery.json` — canonical aggregate result/provenance;
- `regional_independent_recovery_summary.csv` — compact table source.

The generator's `analysis_plan` schema is regression-tested against the committed
JSON, so a future generator/artifact schema drift fails tests. The default-validity
floor and middle block are loaded from `configs/method_frozen.yaml`, and provenance
paths are serialized with POSIX separators for cross-platform stability.

`regional_cross_projections.csv` is optional derived role-level detail. It is written
only with `--write-role-details` and is intentionally not required as a committed
scientific authority.

## Fidelity and provenance

- C160 fidelity gate: PASS. The target values are read from the canonical committed
  `results/t17/c160/t17_c160_role_space_report.json`, rather than duplicated as
  independent hard-coded scientific constants.
- Source Git SHA recorded by the committed aggregate: `0ce58dff49d13da539ca6bfa14deca1547b37006`.
- C80-A reduced manifest SHA-256: `74ba728d52e50311675b7e8b2eacc1f09d17a937e22d045b846ca7961ee45d1b`.
- C80-B reduced manifest SHA-256: `06a30075b0559363d5724c6cf00b945cf4d44fcfe167110f5873daa62169b47c`.
- HF revision: `e1885e0865d1e6d7a39a9c2585b224b30e28b3f1`.
- Bootstrap: 2,000 role replicates, seed 0.
