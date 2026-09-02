# T24 deviation: source-engagement-matched project controls

**Date:** 2026-08-21  
**Task:** T24  
**Status:** APPROVED_PRE_OUTCOME_DEVIATION  
**Authority:** Owner/Executor pre-outcome approval; ratified by reviewer [Reviewer] (`[Reviewer]`) in PR #53 review `4990812696` on 2026-08-21

## Trigger

The originally frozen T24 project controls used a 25th-percentile lower-tail threshold for the orthogonal-random direction and a 75th-percentile distance radius for the spherical control. The protocol also required both controls to match the measured engagement rate of the fixed Lu source capping treatment within an absolute tolerance of 0.03.

The first genuine Qwen3-32B calibration capture completed all 100 frozen T21 manifest items at source commit `736b94c8b1929a1b2e54cf1c1ae535c1d022bbec`. Before any T29 matched-control outcomes were generated or inspected, the numerical-freeze gate failed at layer 46:

- original random-control engagement: `0.2500`
- measured fixed source-cap engagement: `0.5959`

This is a protocol-level calibration mismatch, not a failed activation capture. The 100/100 outcome-blind calibration shards and assembled calibration NPZ remain valid inputs.

## Decision

The source Assistant-Axis capping treatment remains unchanged. Its released vectors, thresholds, model/revision, layer set, and T21 source setting are not retuned.

For the **primary T24 matched controls**, each layer now uses the measured source-cap engagement on that same layer's frozen `natural_L` rows as the target engagement rate:

- orthogonal-random lower-tail quantile: `q_random,L = source_engagement_L`
- spherical radius quantile: `q_sphere,L = 1 - source_engagement_L`

The random direction itself, base seed, Axis-orthogonality requirement, spherical center, activation rows, model, decoding, and all other frozen construction choices remain unchanged. The same absolute engagement-match tolerance of `0.03` remains fail-closed.

Because the revised quantiles are derived from the measured source engagement, this tolerance is now primarily a **self-consistency/regression guard**, not independent evidence that the controls are otherwise well matched. It can still catch implementation mismatches such as threshold-side semantics or tie-degenerate calibration behaviour.

## Original q25/q75 controls

The originally frozen q25 random and q75 sphere parameters are retained in the numerical artifact/report as an **unmatched-engagement sensitivity/reference**. They are no longer the primary specificity controls because the genuine calibration showed that they do not match the fixed source treatment's intervention incidence on the common calibration corpus.

## Outcome-blindness and reuse

This deviation is based only on activation-calibration data and the released source capping thresholds. No harmfulness, identity, refusal, or other T29 matched-control outcome labels are used to set the revised control quantiles. No matched-control T29 outcomes were inspected before this decision.

The completed Qwen calibration capture should be reused. Re-running the 100-item model capture is neither required nor desired unless its byte-level provenance fails later checks.

## Claim boundary

The revised primary controls are matched to the **frequency of intervention engagement** on the common natural calibration rows. They are not claimed to match intervention magnitude, geometry, or all distributional effects of the source Assistant-Axis cap. Random-direction and spherical-distance controls remain project-defined controls, not Lu et al. methods.

At layers 46–49, measured source engagement exceeds 0.5, so the matched spherical control projects a majority of natural rows toward the default-Assistant centroid. This is a strong, geometry-specific intervention. A null sphere result at these layers would therefore be especially informative for specificity, while a positive sphere result would be more ambiguous because intervention strength and geometry can both contribute. T29 should interpret this asymmetry explicitly rather than treating matched incidence as matched intervention magnitude.
