# T18 production run after G2

This note is the canonical production handoff for T18 after the approved label-independent confirmatory membership decision.

## Precondition

Do not inspect a real T18 result until this compatibility PR is independently reviewed.

The production analysis must use:

- `USER_TRANSLATED_LU`;
- transformer block 16;
- `ALL_RESPONSE_TOKENS`;
- C80-A and C80-B;
- the frozen 275-role G2 eligible set;
- all technically valid role-prompted rows in that slice;
- all five technically valid default-Assistant conditions.

Automatic role-judge scores must not filter the primary T18 rows.

## Why the production entry point differs from PR #30

PR #30 froze and validated the statistical analysis before the later T15/T16 confirmatory membership decision. Its original loader still applies a score-3 role-judge filter. T07 failed the predeclared judge gate, and PR #44 subsequently froze label-independent confirmatory membership before any T15/T16 outcome existed.

T18 is downstream of that G2 geometry, so the production loader must diagnose the same population. The statistical machinery itself remains unchanged.

## Preflight

From a clean checkout of the reviewed branch:

```bash
python -m pytest \
  tests/test_t18_readouts.py \
  tests/test_t18_loader.py \
  tests/test_t18_frozen.py \
  tests/test_t18_g2_compat.py -q

python tools/run_t18_g2_dimensional_adequacy.py \
  --self-test \
  --out /tmp/t18_self_test.json
```

The self-test must pass all known-answer worlds before the real run.

## Required run-directory layout

The G2-compatible production loader requires four files:

| File | Purpose |
|---|---|
| `generation.jsonl` | frozen generation metadata, arm, block, role/question IDs, default condition, technical validity |
| `activations.jsonl` | block-16 all-response activation metadata and vector hashes |
| `vectors.npz` | activation `row_id` to pooled vector |
| `eligible_roles.json` | frozen G2/T15 eligible-role set |

`judge.jsonl` may be present for provenance or later descriptive sensitivities, but it is not read by the primary G2-compatible T18 loader.

The eligible-role artifact must declare exactly the 275 roles frozen by G2. Do not reconstruct a different role set from automatic scores.

## Real production command

Run from a clean tree without `--allow-unfrozen` and without `--allow-dirty`:

```bash
python tools/run_t18_g2_dimensional_adequacy.py \
  --run-dir runs/confirmatory \
  --out results/t18/t18_report.json
```

Do not change the frozen seed, 2,000 bootstrap replicates, fold definitions, candidate grids, role balancing, simultaneous max-over-families inference, or 0.02 AUROC margin.

## Required review checks after execution

Before the result enters the manuscript, verify:

1. `reportable: true` and G2 freeze hash present;
2. `membership: label_independent_technical_validity`;
3. `judge_filter_applied: false`;
4. 275 role groups and five default-condition groups survive;
5. C80-A/C80-B, block 16, `ALL_RESPONSE_TOKENS`, and `USER_TRANSLATED_LU` are stamped in the slice;
6. source git SHA, method-config SHA, T18-config SHA, eligible-role SHA, and G2 SHA are recorded;
7. the verdict is exactly `ADEQUATE`, `INADEQUATE`, or `INCONCLUSIVE` under the frozen simultaneous interval rule;
8. T18 predictive PCA component selections are not compared numerically with T17 role-space PCA component counts;
9. any richer-readout gain is described as predictive summary adequacy, not causal necessity.

## Manuscript wording boundary

Use **role-prompted** or **prompt-conditioned** for the T18 negative class. Do not call the primary C80 rows behaviourally verified role enactments.
