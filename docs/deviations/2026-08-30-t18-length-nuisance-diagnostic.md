# T18 post-primary nuisance baseline: response-region length — 30 Aug 2026

## Purpose

The frozen T18 result shows that a regularized full-dimensional linear readout
predicts default-Assistant versus role-prompted activations substantially better
than the one-dimensional Assistant Axis. This post-primary diagnostic asks the
cheapest direct nuisance question: can simple generation-length information
reproduce that advantage?

This diagnostic is a sensitivity only. It does not alter the frozen T18
population, estimator, candidate-family verdict, or `INADEQUATE` conclusion.

## Outcome-blind analysis contract

Before the nuisance outcomes are inspected:

1. Reuse the exact frozen G2/T18 population:
   `USER_TRANSLATED_LU`, C80-A + C80-B, block 16,
   `ALL_RESPONSE_TOKENS`, label-independent technical-validity membership,
   275 roles, five default conditions, 160 questions.
2. Recover token counts only from the exact T12 metadata revision already bound
   by G2. No generation, activation extraction, or judge call is permitted.
3. Verify the reconstructed metadata population has the exact frozen T18 row
   counts and reproduces all 20 frozen outer-fold train/test/drop counts.
4. Baseline A uses one feature: total generated response-token count.
5. Baseline B uses two features: reasoning-token count and final-answer-token
   count. Do not add total length to this model.
6. Use T18's existing `full_linear` implementation on these nuisance features:
   the same C grid, fold-local standardization, 5 x 4 outer grouping, 4 x 3
   inner selection, split seed, role-balanced weights, and mean within-fold
   AUROC.
7. Reuse T18's 2,000-replicate role-clustered bootstrap for descriptive 95%
   intervals.
8. Compare descriptively with the already-frozen Axis and full-linear AUROCs.
   There is no nuisance pass/fail threshold.
9. The frozen T18 `INADEQUATE` verdict remains authoritative regardless of this
   diagnostic's outcome.
10. Do not residualize the 4,096 activation dimensions as part of this task.

## Interpretation boundary

A weak length baseline rules out simple token count as an explanation for the
full-linear advantage. A strong length baseline requires narrower wording about
what the full-dimensional readout is exploiting. Neither result identifies a
causal mechanism or the intrinsic dimensionality of persona representations.
