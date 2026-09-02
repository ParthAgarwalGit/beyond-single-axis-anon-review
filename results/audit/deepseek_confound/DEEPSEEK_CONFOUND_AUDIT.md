# DeepSeek confound/audit bundle — post-primary sensitivities

**Scope.** Three CPU-only checks against the frozen C80 recovery result.
No new generation, no model inference, no judging, no re-extraction, no
change to C80-A/B membership, role eligibility, decoding, block, response
pool, or Axis orientation. These are post-primary sensitivities; nothing
here modifies or redefines T15/T16.

**Fidelity gate (precondition).** The primary axes were reconstructed from
the frozen inputs (T17 reduced role/default means, frozen five-condition
default construction, T14 275-role set) and reproduce the frozen headline
cross-axis Pearson **r = 0.9765281** (|Δ| < 1e-6 vs
`results/t15/confirmatory_reliability.json`). All sensitivities below are
computed against this verified reconstruction. Exact input hashes, the
pinned T12-C80 metadata revision (`e1885e08…`), commands, and output hashes
are in `provenance.json` (clean-tree stamped).

---

## 1. Response-length confound — `response_length_sensitivity.{json,csv}`

**Question.** Is role position along the cross-built Assistant direction
associated with mean generated-response length, and does the cross-question-set
role ordering survive removing the linear association with length?

**Frozen inputs.** Per-row `token_region_counts.all_response` (generated
content tokens — the identical token set the representations pool) from the
pinned T12-C80 metadata, restricted to the same rows the frozen reduce used
(`validity == valid` with the all-response pool); cross-built role scores
(A-on-B-axis / B-on-A-axis) from the verified reconstruction.

**Result.**

| block | Pearson(len, score) | Spearman | OLS R² | reasoning-tokens r | answer-tokens r |
|---|---|---|---|---|---|
| C80-A | 0.595 | 0.610 | 0.354 | 0.508 | 0.561 |
| C80-B | 0.612 | 0.618 | 0.374 | 0.567 | 0.534 |

Residualized recovery (score ~ length residuals, A vs B):
**Pearson r = 0.8783, Spearman 0.8549**, vs frozen headline 0.9765
(Δ = −0.098).

**Reading.** Mean response length is a real, moderate correlate of Axis
position (~35–37% of score variance per block) — roles scoring further toward
the default-Assistant side tend to produce longer responses (positive
correlations and slopes in both blocks). After removing the linear length
association in both blocks independently, the cross-block role ordering
remains strong (r ≈ 0.88). The recovery is therefore not reducible to a
length artifact, but length association must be *reported*, not dismissed:
this check bounds only the simple linear confound and says nothing about
whether length itself is partially Axis-caused or vice versa.

**Materiality for the paper claim:** does not overturn the recovery claim;
requires one honest sentence where the paper currently admits the check was
absent, plus the residualized r as the supporting number.

## 2. Default-condition stability — `default_condition_sensitivity.{json,csv}`

**Question.** Is the result materially dependent on one particular default
prompt, or is the equal-weight five-condition default reference stable?

**Frozen inputs.** The five per-condition default centroids per block
(frozen T17 DEFAULT reductions), the frozen primary axes, frozen role means.

**Result.** Per block: all pairwise condition-centroid cosines ≥ **0.993**;
projections of the five centroids on the opposite block's frozen Axis span
a range of **0.089** (SD 0.034) for C80-A and **0.106** (SD 0.044) for
C80-B on the representation scale. Rebuilding the
Axis with each single condition as μD: axis-vs-primary cosine 0.755–0.931,
recovery r **0.9497–0.9773** across all five (headline 0.9765). Leave-one-
condition-out: axis cosine ≥ 0.972, recovery r 0.9746–0.9780. **Zero sign-
check failures across all ten variants.**

**Reading.** The five default conditions are geometrically close and no
single condition is load-bearing: even the most divergent single-condition
Axis (condition 3, "You are {model_name}." — cosine ≈ 0.76–0.78 to primary)
still yields r = 0.9715 recovery. The equal-weight reference is stable;
the recovery statistic is insensitive to the default-prompt choice at the
±0.027 level worst-case.

**Materiality:** strengthens the claim; no change needed beyond citing the
range.

## 3. Proper held-out-role reconstruction — `heldout_role_recovery.{json,csv}`

**Question.** Does the recovered ordering hold when each role is genuinely
excluded from *both* axes used to score it (full rebuild, not projection-
pair deletion)?

**Frozen inputs.** Frozen role means and default means; for each of the 275
roles, both axes rebuilt from the remaining 274 roles; the held-out role
scored on the opposite block's rebuilt axis.

**Result.** Held-out recovery across 275 score pairs: **Pearson r = 0.9758,
Spearman 0.9753** (headline 0.9765; Δ = −0.0008). Leave-one-role-out axes
are nearly identical to the full axes: cosine ≥ **0.99981** in every case
(median > 0.9999); most influential roles: `infant`, `caveman`,
`coral_reef`. Zero sign-check failures.

**Reading.** The ordering is not driven by any evaluated role's own
contribution to the axes; with 275 equally weighted roles, no single role
moves the direction. This is the stronger version of the appendix LOO and
can replace it.

**Materiality:** strengthens the claim; supersedes the projection-pair LOO
number in the appendix.

---

## Overall interpretation (3–5 lines)

None of the three checks materially weakens the DeepSeek recovery claim.
The genuine finding is the length association (~0.6 correlation with Axis
position): the paper should state it and report the residualized recovery
r = 0.878 as the bound on the linear-length-free ordering. Default-reference
and held-out-role checks came back strongly stable (worst-case recovery
0.9497 and 0.9758 respectively) and can be cited as one-line robustness
results. No frozen choice was retuned; all numbers derive from committed
frozen artifacts under the verified fidelity gate.
