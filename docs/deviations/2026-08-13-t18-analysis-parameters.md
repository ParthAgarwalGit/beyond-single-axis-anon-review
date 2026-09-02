# T18 analysis parameters frozen in a separate config

## Date

2026-08-13

## Issue

`analysis_registry.sensitivity[13]` registers T18 as "grouped multidimensional
linear and restrained nonlinear readouts", but `configs/method_frozen.yaml`
does not specify the analysis parameters that decide a T18 verdict: the
candidate grids, the estimand and its aggregation rule, the resampling unit,
the multiplicity rule, and the practical-equivalence margin.

Review of PR #30 correctly noted that the 0.02 AUROC margin was not part of any
frozen specification, and that a margin chosen after real activations were
examined would be unfalsifiable.

## Correction

The T18 analysis parameters are frozen in `configs/t18_frozen.yaml`
(`schema_version: t18-freeze-v1.0`), covering:

- the target, the role-eligibility source, and the judge region;
- the doubly grouped splitting scheme and fold counts;
- the estimand: fold-aggregated, role-balanced AUROC, with pooling of
  predictions across independently fitted outer folds explicitly forbidden;
- the candidate grids, the nonlinear parameter budget, and the requirement
  that every family able to determine a verdict is covered by the
  known-answer self-test;
- the inference rule: role clustering, 2000 replicates, a simultaneous
  max-over-families bootstrap, and the **0.02 AUROC** equivalence margin.

`src/t18_readouts.py` loads this file rather than defining the values, and
`tests/test_t18_frozen.py` fails if any hard-coded value drifts from it. Every
T18 report stamps `t18_config_sha256` alongside the method `config_sha256`.

## Why a separate file rather than `method_frozen.yaml`

Adding these fields to `configs/method_frozen.yaml` would change its
`config_sha256`. That hash is already stamped into every report issued under
the G2 freeze, and `implementation_contract.config_hash_must_be_stamped_in_every_report`
makes those stamps load-bearing, so changing it would invalidate existing
references for a diagnostic that is registered as a sensitivity.

The separate file carries its own hash and is stamped in every T18 report. If
the reviewer prefers, it can be promoted into `method_frozen.yaml` at the next
dated method revision.

## Scientific impact

No activations, generations, judge scores, geometry, or outcomes were
inspected, changed, or produced. T18 has not been run on real data; the
`--run-dir` path remains gated on the G2 geometry freeze. The parameters are
frozen **before** any real activation is examined, which is the property this
record exists to establish.
