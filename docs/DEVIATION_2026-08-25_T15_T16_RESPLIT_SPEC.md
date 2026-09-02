# T15/T16 — 300 repeated frozen-question resplits: missing specification, frozen now

Date: 2026-08-25
Record ID: `DEV-2026-08-25-T15T16-01`
Scope: the predeclared `repeated_frozen_question_resplits: 300` sensitivity under `confirmatory_geometry.uncertainty` in `configs/method_frozen.yaml` / `configs/method_frozen_v4.yaml` / `configs/t12_runtime_resolved.yaml`

## What was already frozen, and what was not

T02 fixes the **original** C80-A/C80-B split deterministically — this is not a gap. `configs/method_frozen.yaml`, `confirmatory_split_algorithm`:

1. remove the frozen E80 IDs from the full 0..239 source-ID space;
2. place IDs 7 and 227 together in C80-A;
3. sort the remaining IDs by descending source-text character length, ties by ascending ID;
4. assign each ID to the block with the lower current character total, ties toward C80-A, enforcing 80 IDs per block.

`all_blocks_assignment_sha256: 6f0213dd56073d3ce87b835e8c7b8a7759ab9b53e78486bced689e3f0c1e6172` is the resulting assignment hash. Verified directly against `configs/method_frozen.yaml:330`.

T02 also fixes the count of the resplit sensitivity — `300` — but **not** how those 300 alternative 80/80 partitions are sampled. `docs/RESULTS_DRAFT.md` records this as outstanding: *"The predeclared 300 repeated frozen-question resplits remain an outstanding sensitivity and are not represented as completed."* No seed, no sampling algorithm, no constraint specification exists anywhere in the repo. This note supplies that missing specification, before any resplit result is inspected.

## Why a specification is needed, not just a seed

"300 random 80/80 splits" is underspecified on its own: a naive split could produce a prompt-index-imbalanced partition (defeating the reason `prompt_index` exists — see `prompt_assignment.rationale` in the frozen config) or separate the duplicate-text pair. The constraints below make the sampling procedure exact and reproducible, not just seeded.

## Frozen specification

**1. Universe.** The 160 C80 question IDs only (C80-A ∪ C80-B). E80 is excluded.

**2. Held fixed.** 275-role membership, block 16, all-response pool, the technical-validity rule, default construction, Axis orientation, and all existing generated rows. No new model generation. No question, role, prompt, or activation choice changes.

**3. Prompt-index is a fixed per-question attribute, not recomputed.** The original rule (`prompt_index = zero_based_position_within_frozen_block mod 5`) only applies at extraction time. Since generation is not repeated, each question's `prompt_index` is whatever was recorded for it in `design/questions_C80_A.json` / `design/questions_C80_B.json` and is treated as fixed. Verified directly from those files: exactly 32 of the 160 questions carry each `prompt_index` value in `{0,1,2,3,4}` — five strata of 32.

**4. Duplicate pair falls in different strata.** Verified directly: `question_id=7` has `prompt_index=0`; `question_id=227` has `prompt_index=2`. The "keep together" constraint is therefore a coupling *across* two strata, not a within-stratum rule.

**5. Sampling procedure**, applied identically for all 300 draws:

For each draw, partition the 160 questions into two labelled halves (`half_1`, `half_2`) by processing the five `prompt_index` strata in ascending order (`0,1,2,3,4`), each independently, using one `numpy.random.Generator` per draw that persists across all five strata (so the five per-stratum draws are one continuous, reproducible random stream, not five independently seeded ones):

- for the stratum containing `question_id=7`: force `7` into `half_1`; draw 15 more from the remaining 31 members of that stratum (via `rng.permutation` of the sorted remainder, first 15) to complete `half_1`'s 16; the other 16 form `half_2`;
- for the stratum containing `question_id=227`: force `227` into `half_1`; same procedure;
- for the three strata containing neither: `rng.permutation` the sorted 32-member stratum; first 16 → `half_1`, remaining 16 → `half_2`.

This guarantees, by construction rather than by rejection sampling: exactly 80 IDs per half; exactly 16 IDs from each `prompt_index` stratum in each half; `7` and `227` always in the same half. `half_1` vs `half_2` is an arbitrary label with no relation to the historical C80-A/C80-B naming — the primary statistic (cross-axis role-projection Pearson r) is symmetric under swapping the two halves, so this labelling choice cannot affect the result.

**6. Master seed is provenance-derived, not chosen from results.** First 8 hex digits of `all_blocks_assignment_sha256` = `6f0213dd` = `1862407133` (decimal; verified: `int("6f0213dd", 16) == 1862407133`).

**7. Child seeds.** `numpy.random.SeedSequence(1862407133).spawn(300)`. Draw `i` (`i = 0..299`) uses child `i`, which is fully determined by `(entropy=1862407133, spawn_key=(i,))` — confirmed empirically: every child's `.entropy` equals the master's; only `.spawn_key` differs. The manifest records `spawn_key` per draw; the shared entropy is recorded once. This is sufficient to exactly reconstruct every draw's RNG stream.

**8. The historical observed split is retained, separately marked.** The actual C80-A/C80-B membership (read from the frozen design files, not generated by this procedure) is carried in the manifest under `observed_split`, explicitly distinct from the 300 generated draws — not draw 0, not resampled into the distribution, a separate labelled record for comparison.

## What this note does not do

It does not compute, inspect, or estimate the resplit geometry statistic. That is a separate step (`tools/run_t15_t16_resplits.py`, not yet written), which must consume the manifest this note's generator produces as a frozen input, and must not run before the manifest is committed and hashed.

## Freeze discipline

This specification is written and the manifest generator built and verified for determinism **before** any resplit geometry is computed. No outcome — resplit distribution, cross-axis correlation, axis cosine — has been inspected in writing this note. `configs/method_frozen.yaml` and `configs/method_frozen_v4.yaml` are not edited; T02's original split algorithm and hash are unchanged.
