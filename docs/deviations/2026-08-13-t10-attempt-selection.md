# T10 common attempt-selection rule

## Date
2026-08-13

## Scope
This amendment applies to the already-generated C80-A and C80-B translated-arm
role-generation artifacts. It does not change question IDs, prompts, model
weights, raw generations, or activation values.

## Problem
The raw C80 files retain retry attempts under a shared semantic `rollout_id`.
C80-B contains cases in which more than one attempt is technically valid. A
downstream pipeline therefore needs one predeclared attempt-selection rule.

Reconciled artifacts:
- C80-A: 22,286 physical rows; 22,000 semantic rollouts; 21,998 downstream eligible; 2 exhausted technical failures.
- C80-B: 22,735 physical rows; 22,000 semantic rollouts; 22,000 downstream eligible.
- C80-B has 448 rollouts with more than one technically valid attempt; 443 have different valid generated texts.

## Frozen rule
For each semantic `rollout_id`, select the **lowest retry index whose
`technical_validity == "valid"`**.

If no technically valid attempt exists, retain the highest-retry row only for
technical-failure provenance. Exclude that rollout from downstream role
scoring, activation pooling, retention, and geometry.

## Consequence
- C80-A: 21,998 / 22,000 eligible; 2 technical exclusions.
- C80-B: 22,000 / 22,000 eligible; 0 technical exclusions.

Raw SHA-256 values remain:
- C80-A: `c7229c926066271eb99bc6b647ee8a7446e67071d0b60d5c6592d42566f1238c`
- C80-B: `2d82dbe1ef2ab49068d8f2ad5f544db2da4bd3e1d7d44ec89415d0b8bb84c1b8`

Machine-readable audit:
`results/audit/T10_C80_attempt_selection_13_aug.json`.

Selection indexes:
- `results/generation/c80_role_C80_A.selection.jsonl`
- `results/generation/c80_role_C80_B.selection.jsonl`

## C80-B retry-counter provenance
Fifteen C80-B histories extend through retry index 4. In those histories,
retry 3 records `max_new_tokens=2048` and retry 4 records
`max_new_tokens=8192`. Twelve retry-3 rows were already valid; three were
truncated and became valid at retry 4.

Therefore token budget must be read from per-row runtime provenance, not
inferred from retry index alone.

## Claim boundary
This fixes attempt-selection semantics only. It does not make C80 decoding
identical to E80 decoding and does not rewrite raw generations.
