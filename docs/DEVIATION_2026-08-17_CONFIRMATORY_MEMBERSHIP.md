# Deviation record — confirmatory output-membership rule

**Record ID:** DEV-2026-08-17-T15-01
**Date:** 2026-08-17
**Author:** [Author A]
**Reviewer:** [Reviewer] (change-control authority for #41)
**Reviewer approval status:** PENDING_GITHUB_PR_REVIEW (PR #44) — not self-approved
**Governs:** T15/T16 (confirmatory Axis reliability), T17 (pool geometry)
**Required by:** `configs/method_frozen.yaml → implementation_contract.change_control`
**Machine-readable registration:** `configs/method_frozen_v4.yaml → deviation_records`
(that list is the declared source of truth; this document is its long form)

---

## 1. Were outcomes inspected before this record?

**No.** At the time of writing, **no confirmatory cross-axis statistic has been
computed under either membership rule** — not for C80-A/C80-B, not for any resplit,
not provisionally. `results/t15/` contains no report, and no T17 reduction has been
run against the real corpus. This record is therefore a pre-registration, not a
post-hoc justification. Any reader can verify this from the commit order: this record
precedes the first commit of `results/t15/confirmatory_reliability.json`.

## 2. What changes

The primary confirmatory role-vector membership rule changes from

> `technical_validity == valid` **AND** segmentation permits a non-empty role-judge
> input **AND** automatic role score == 3 under a human-validated frozen judge

to

> `technical_validity == valid` **AND** the frozen block-16 all-response pool is
> present

and the per-block eligibility threshold (≥10 per 80-ID block) is evaluated on
**technically-valid output counts** instead of score-3 counts.

**On the dropped middle clause.** The superseded rule required that segmentation
permit a non-empty role-judge input. That clause existed to guarantee something
*judgeable*; under a label-independent rule nothing is judged, so it has no operative
role and is deliberately not carried over. This keeps the pre-registered text
identical to the implemented predicate, which enforces exactly
`validity == "valid" AND has_all_response_pool`.

Dropping it selects the same outputs. Verified directly against the C80 metadata:
adding the judgeable-input requirement excludes **zero** further rows in either block
(C80-A 21,998 either way, C80-B 22,000 either way), because every technically valid
C80 row is `balanced_reasoning` and carries an answer pool. The two predicates are
therefore extensionally identical on this corpus, and no retained output, role, or
count changes.

## 3. Why — the frozen precondition is unmet

The third clause of `validity.primary_role_vector_outputs_require` is **conditional**:
it requires a score of 3 *under a human-validated frozen judge*. T07 was that
validation, and the judge failed the predeclared gate on all four criteria
(`results/t07/judge_validation.json`):

| metric | observed | gate |
|---|---|---|
| score-3 precision | 0.556 | ≥ 0.85 |
| score-3 recall (ip) | 0.798 | ≥ 0.85 |
| macro-F1 (ip) | 0.409 | ≥ 0.75 |
| four-class κ | 0.233 | ≥ 0.70 |

No human-validated judge exists, so the clause cannot be satisfied as written.
Dropping it follows the freeze's own conditional structure rather than overriding it.

Three supporting diagnostics were run before this decision (all on committed or
read-only inputs; see `tools/run_boundary_diagnostic.py`):

1. **A more permissive boundary does not rescue the judge.** At score ≥2 the judge
   reaches precision 0.781, ip-weighted recall 0.726, ip-weighted macro-F1 0.782, and
   binary κ 0.570. This is an **exploratory comparison against the same numerical
   thresholds**, not a re-application of the frozen gate: collapsing the rubric at
   ≥2 makes macro-F1 and κ *binary* quantities, whereas the frozen criteria are
   four-class (macro-F1 0.409, four-class κ 0.233, both in
   `results/t07/judge_validation.json`). Precision and recall condition on the same
   positive class in both places and are the only directly comparable quantities;
   **both miss at the permissive boundary** (0.781 vs ≥0.85; 0.726 vs ≥0.85), so the
   conclusion — that moving the boundary does not recover a usable filter — rests
   only on the comparable quantities. Of
   the judge's 90 score-3 calls, 18 are human score-1 (no meaningful role expression),
   while 40 true human score-3 outputs are called score-2.
2. **No scored coverage exists for the confirmatory corpus.** The archived production
   scores from the judge **evaluated in T07** (`lu-replication/run-003/{translated,
   extraction}`) have **zero uid overlap** with C80-A/C80-B, and C80 metadata carries
   no `lu_score` field. Those scores are not described as validated: T07 characterized
   that judge against human consensus and it failed every predeclared criterion. A
   judge-filtered confirmatory analysis would additionally require a *fresh* run whose
   model revision is unrecoverable (`configs/t07_judge_frozen.json`:
   `model_revision: null`, `NOT_RECORDED_IN_RECOVERABLE_ARCHIVED_ARTIFACTS`), i.e. an
   instrument that has never been characterized at all.
3. **The label-independent rule introduces no outcome-dependent selection.** The
   design generates 80 outputs per role per block (275 × 80 = 22,000). Technical
   validity removes two of them in C80-A (21,998/22,000; C80-B is 22,000/22,000; all
   `balanced_reasoning`), so at most two roles contribute 79 rather than 80. All 275
   roles clear the ≥10 threshold in both blocks and retention is 275/275 = 100%. The
   two exclusions are technical failures, independent of what the response said.

## 4. Methodological consequence (the substantive reason, not just the permissive one)

Score-3 filtering would select, for each role, the subset of outputs a *single fixed
judge* deemed most role-expressive. That selection is systematic and is repeated
identically in C80-A and C80-B. Because the judge's biases are themselves
reproducible across the two halves, a judge-filtered cross-axis correlation can
absorb variance attributable to judge behaviour rather than to model role geometry —
precisely the confound a split-half reliability claim must exclude. The
label-independent rule cannot produce this artifact: every role contributes all of its
technically valid outputs in each block — 80 by design, less the two technical
exclusions in C80-A — and nothing is removed on the basis of what the response said.

## 5. Claim scope — what is gained and lost

- **Retained:** the confirmatory claim that role-conditioned representations occupy
  reproducible positions along a single shared direction across decoding-matched
  halves built from *different question blocks*.
- **Lost:** the claim is now **prompt-conditioned**, not behaviour-conditioned. It
  states that roles the model was *prompted* into occupy reproducible axis positions,
  not that roles the model *successfully enacted* do. The manuscript must say this
  plainly and must not describe the retained outputs as "in-role".
- **Expected direction of bias:** including non-role-expressing outputs dilutes role
  means toward the grand mean, which **attenuates** the cross-axis correlation. The
  reliability conclusion therefore rests on separation from the three frozen nulls,
  not on the raw magnitude of r.

## 6. Affected artifacts

- `src/geometry.py`, `src/confirmatory_geometry.py` — unchanged; membership is an
  input, not a property of the geometry code.
- `tools/run_t15_confirmatory.py`, `tools/gpu_cells/t15_confirmatory_reduce.py` —
  membership rule and eligibility counts.
- `tools/run_t17_pool_sensitivity.py` — the `all_valid` membership path is promoted
  from provisional to primary; `score3` remains available as a sensitivity path.
- `docs/RESULTS_DRAFT.md` — Pillar A framing and the limitations section.

## 7. Analyses rerun

**None required.** No confirmatory analysis had been run under the superseded rule,
so there is nothing to recompute or withdraw.

## 8. Pre-declared sensitivities

Declared here, before any confirmatory number exists:

1. **Membership sensitivity on E80** — on the E80 corpus, where archived scores from
   the judge characterized in T07 do exist, report `cos(v_score3, v_all_valid)` at
   block 16. This
   bounds empirically how far the membership rule moves the Axis and bridges the
   Pillar A / Pillar B axis constructions.
2. **Judge-filtered C80 appendix (explicitly unvalidated)** — a fresh T13 judge run
   over C80 may be reported in an appendix. Its instrument has no recoverable
   revision and no validation; it is reported for descriptive completeness only and
   **must not** be presented as a confirmatory or robustness result.
3. Existing frozen sensitivities are unaffected: 300 frozen question resplits, the
   drop-question-227 duplicate check, and the off-by-one block-15 check.
