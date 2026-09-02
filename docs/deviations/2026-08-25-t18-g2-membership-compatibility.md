# T18 G2 membership compatibility amendment

**Record ID:** DEV-2026-08-25-T18-01  
**Date:** 2026-08-25  
**Task:** T18  
**Status:** REVIEW_CANDIDATE  
**Outcome inspection:** no real T18 result has been run or inspected before this amendment.

## Decision

T18 inherits the approved primary confirmatory membership rule frozen for T15/T16 and T17 by PR #44:

> `technical_validity == valid` AND the frozen block-16 `ALL_RESPONSE_TOKENS` activation is present.

The primary T18 row set must therefore be label-independent. Automatic role-judge scores must not filter the C80 role rows used by the production dimensional-adequacy analysis.

## Why this is required

`configs/t18_frozen.yaml` was prepared before the later confirmatory membership decision and still describes the negative class as retained role-adopted responses with a score-3 target. After T07, that precondition is no longer satisfiable: the production role judge failed all four predeclared validation criteria. PR #44 therefore froze label-independent confirmatory membership before any T15/T16 outcome existed.

The canonical T15/T16 result now uses `label_independent_technical_validity`, retains all 275 roles, and makes a prompt-conditioned rather than behaviour-conditioned geometry claim. T18 is explicitly downstream of that frozen geometry. Applying a score-3 filter only inside T18 would silently change the population being diagnosed and would reintroduce the failed judge into a primary result.

## What changes

Only the T18 production loader contract changes:

- role examples are selected by technical validity, frozen primary slice, and membership in the frozen eligible-role artifact;
- role-judge scores and judge-region tags do not determine primary row membership;
- `judge.jsonl` is no longer a required production input for T18;
- the canonical target wording becomes **default-Assistant versus role-prompted responses**;
- the report records `membership = label_independent_technical_validity` and the G2 authority.

## What does not change

No estimator or decision rule changes. The following remain exactly as frozen in `configs/t18_frozen.yaml`:

- `axis_1d`, `pca_linear`, `full_linear`, and `small_nonlinear` candidate families;
- the 5 x 4 doubly grouped outer evaluation over unseen roles and unseen questions;
- the 4 x 3 inner grouped validation;
- fold-wise, role-balanced AUROC aggregation;
- feature standardisation and frozen hyperparameter grids;
- 2,000 role-clustered bootstrap replicates;
- simultaneous max-over-families multiplicity adjustment;
- the 0.02 AUROC adequacy margin;
- the `ADEQUATE / INADEQUATE / INCONCLUSIVE` verdict rule;
- the descriptive permutation diagnostic.

`configs/t18_frozen.yaml` is deliberately not edited by this amendment, preserving its pre-outcome hash and all statistical analysis parameters.

## Claim boundary

T18 answers whether a single Assistant-related direction is an adequate predictive summary of the frozen **prompt-conditioned** default-versus-role geometry on unseen roles and unseen questions.

A richer predictive readout does not establish that additional causal directions are necessary. T18 predictive PCA components are also not the same object as the T17 descriptive role-space PCA components.

## Authority

- G2 freeze record: `docs/G2_GEOMETRY_FREEZE.json`
- confirmatory membership authority: `docs/DEVIATION_2026-08-17_CONFIRMATORY_MEMBERSHIP.md`
- change-control PR: #44, merged at `3848b6ccb49bd37ccdb47d127aca688ffda60c11`
- canonical T15/T16 result: `results/t15/confirmatory_reliability.json`
- T15/T16 merge commit: `fb01dfbd8247448e494136edc3b50a8e3e7d8a14`

## Review requirement

This amendment must be independently reviewed before the real T18 production run. If rejected, no reportable T18 run should be launched until the population mismatch is resolved without looking at T18 outcomes.
