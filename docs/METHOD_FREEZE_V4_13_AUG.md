# Method Freeze V4 — 13 August 2026

**Task:** T02

**Owner/Executor:** [Author B]

**Reviewer:** [Reviewer]

**Status:** REVIEW_CANDIDATE

**Downstream config:** `configs/method_frozen_v4.yaml`

**V4 SHA-256:** `f7e7dd9df8c7c72d5da08bf8201295dfc91f3208837d8cd0867054c393984e59`

### SHA succession

| SHA | Authorised by | Nature of change |
|---|---|---|
| `2fe1dbbbacdfec06…` | PR #38 (reviewer-approved) | original V4 freeze |
| `f7e7dd9df8c7c72d…` | PR #44 review, under `DEV-2026-08-17-T15-01` | **registration-only**: a deviation record was appended; no frozen field value changed |

A SHA change without a written succession line is indistinguishable from corruption,
so every future change to this file adds a row here.

The registration-only claim is machine-checked, not asserted: the SHA-256 of the parsed
config with `deviation_records` removed is `17b0dd2f3db6343a…` on both sides of the
succession above, and `test_registering_a_deviation_does_not_change_any_method_field`
holds it there.

### Rebinding rule for a SHA succession

When this file's SHA changes, rebind **forward-looking** pins and **never rewrite
historical provenance stamps**. Forward-looking pins (the integrity test constant, the
T23 causal source map, this document, the provenance-closure generator) assert what the
current approved config *is*, so they must track the succession. Historical stamps
inside already-produced artifacts record which config was actually in effect when that
artifact was generated; rewriting them would falsify the audit trail rather than update
it, and an artifact stamped with a superseded SHA is correct as written.
**Historical base config:** `configs/method_frozen.yaml`
**Historical base SHA-256:** `73be73df4640c2d32bfbc8b6009741c0fadd8c466acb703d1f99df35d5c8cd79`

## Preservation rule
The historical `configs/method_frozen.yaml` remains byte-for-byte unchanged
because its SHA is already stamped into generated artifacts. New downstream
analysis uses V4 while preserving historical raw-generation provenance.

## New V4 decisions
1. `response_segmentation.prompt_prefills_opening_marker: true` for the primary
   DeepSeek chat template.
2. One common lowest-valid-retry C80 attempt-selection rule.
3. Explicit C80 greedy-decoding deviation from E80/T09.
4. All-response/reasoning/final-answer comparison is a main paper analysis,
   while all-response remains the Lu-comparable primary pool.
5. Exact PCA verification record is required.
6. Primary post-intervention representation movement uses one fixed
   pre-steering PCA basis.
7. Reasoning-only versus answer-only intervention timing is optional sensitivity
   only and must be frozen before outcome inspection.
8. Causal activation capture is required.
9. T18 Owner/Executor is [Author A].
10. Internal completion target is 23 August; 24 August is submission buffer.
11. T23 causal-prompt selection and P50 manifest guards are downstream V4
    amendments. They must not be written into the historical V3 artifact.

## T23 migration amendment — 15 August 2026
The T23 fields `causal_prompt_selection` and the P50 guard fields under
`causal_role_selection` were initially written into the historical V3 config in
place. Issue #31 identified that as a provenance error. The historical file has
therefore been restored byte-for-byte to SHA-256
`73be73df4640c2d32bfbc8b6009741c0fadd8c466acb703d1f99df35d5c8cd79`, and
the T23 fields now live in `configs/method_frozen_v4.yaml` only.

The migration is recorded as **DEV-2026-08-10-T23-01** and documented in
`docs/deviations/2026-08-15-t23-causal-config-migration.md`. The decision date
remains 10 August 2026, the migration was recorded on 15 August 2026, and
`outcomes_already_inspected: false`. Independent review is assigned to
**[Reviewer]** through the Issue #31 fix PR; approval metadata remains pending and
must not be self-stamped before that GitHub review is submitted.

T23 builders, tests, and the committed causal-question source map now use V4 for
new decisions. Historical generation artifacts keep their original V3 config
hashes; those provenance stamps are not rewritten.

## C80 reconciliation
| Block | Rollouts | Eligible | Technical exclusions | >1 valid | Different valid text |
|---|---:|---:|---:|---:|---:|
| C80-A | 22,000 | 21,998 | 2 | 0 | 0 |
| C80-B | 22,000 | 22,000 | 0 | 448 | 443 |

## Claim boundaries
V4 does not permit claims that:
- E80 and C80 used identical decoding;
- PCA defines the Assistant Axis;
- 70% variance proves intrinsic dimensionality;
- the prefilled-`<think>` correction changed model behavior;
- the two C80-A technical exclusions are semantic persona failures;
- raw C80 generations were regenerated or rewritten during reconciliation.
