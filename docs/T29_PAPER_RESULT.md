# T29 paper-ready result

## Results paragraph

Because the causal harmfulness judge did not pass the frozen T27 validation gate, the design-weighted human subset is the primary T29 evidence. Strict harmful compliance was 4.0% under Assistant-Axis capping and 7.69% under the matched random-direction cap, giving an Assistant-minus-random risk difference of -3.69 percentage points (95% stratified-bootstrap CI [-11.08, 0.00] pp). The point estimate is in the predicted direction, but the interval reaches zero. The corresponding human estimates against the spherical control and unsteered baseline were both +1.0 pp, and the inclusive-harm Assistant-minus-random sensitivity was -2.69 pp (95% CI [-10.08, +1.00] pp).

The matched 400-row automatic-label panel is secondary under the same T27 branch. Automatic strict harmful-compliance rates were 3% for Assistant capping, 2% for the random-direction cap, 1% for the spherical control, and 2% unsteered. The paired Assistant-minus-random diagnostic was +1.0 pp (95% bootstrap CI [0.0, 3.0] pp; exact two-sided McNemar p = 1.0), so the full-corpus automatic sensitivity did not reproduce the favorable direction of the human-primary point estimate. We therefore do not claim a robust direction-specific harmfulness reduction from Assistant-Axis capping.

## Discussion sentence

The source-model causal test is consistent with at most a weak, measurement-limited specificity effect: the primary human estimate favors the Assistant direction relative to the random control, but uncertainty reaches the null, the effect is absent against the other controls, and the secondary automatic panel points in the opposite direction.

## Required caveats

- Human T29-H is primary; the 400-row automatic panel is secondary sensitivity only.
- Do not describe the automatic McNemar p-value as the primary significance test.
- Do not interpret the degenerate human bootstrap intervals for Assistant-vs-sphere or Assistant-vs-unsteered as unusually precise causal evidence; those strata were fully enumerated under the frozen design.
- Identity is not part of the T29-H harmfulness estimator. The T27 identity gate rests on two human-positive, adjudicator-dependent cases and should not support a strong full-corpus identity claim.
- Do not infer that identity mediates harmfulness.
