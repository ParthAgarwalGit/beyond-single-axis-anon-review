# T07 — the automatic role judge does not clear the retention gate

**Date:** 2026-08-16 · **Owner:** [Author A] · **Reviewer:** [Author B]  
**Status:** result in; downstream branch decision needed  
**Artifacts:** `results/t07/judge_validation.json`, `results/t06/gold_consensus.csv`

## TL;DR

The archived production role judge (`Qwen/Qwen2.5-32B-Instruct-AWQ`), evaluated
against the final adjudicated T06 human gold (300 items), **fails all four
predeclared validation thresholds**. It cannot serve as the automatic
large-scale role-retention filter for T14.

The supported reading is **measurement-limited**. The automatic judge fails the
frozen gate, while moderate inter-human agreement (κ = 0.556) shows that the
role-expression boundary itself is also difficult to measure. We therefore do
not interpret judge-human disagreement as judge error in isolation. Downstream
retention must be human-backed and frozen before confirmatory geometry is run.

## What was tested

The validation compares the **archived production scores** against the final T06
human consensus using the predeclared T07 scoring specification. The archived
production reference records the judge model family, prompt construction,
answer-region extraction, parser, temperature, max tokens, and seed convention.

Important provenance limitation: the exact historical Hugging Face commit of
`Qwen/Qwen2.5-32B-Instruct-AWQ` was **not recorded in the recoverable repository
artifacts**. We therefore describe this as validation of archived production
scores, not as a fresh bit-for-bit rerun of a revision-pinned judge model.

Predeclared gate on the scored population:

- score-3 precision ≥ 0.85
- score-3 recall ≥ 0.85
- macro-F1 ≥ 0.75
- four-class κ ≥ 0.70

Recall and macro-F1 are IP-weighted for corpus validity. Judge abstention is
reported separately.

## Results (pooled scored, n = 278)

| metric | value | threshold |
|---|---:|---:|
| score-3 precision | **0.556** | ≥ 0.85 |
| score-3 recall (IP-weighted) | **0.798** | ≥ 0.85 |
| macro-F1 (IP-weighted) | **0.409** | ≥ 0.75 |
| four-class κ | **0.233** | ≥ 0.70 |

- Populations: scored 278, abstained 10 (**3.5%** of human-ok), unjudgeable 12.
- By arm, both fail: translated κ 0.246; extraction/explicit-wrapper κ 0.219.
- Unweighted score-3 recall is 0.521; the corpus-valid IP-weighted value is 0.798,
  which still misses the frozen threshold.
- Sensitivity set (n = 60): **16 items (27%)** cross the 2/3 retention boundary
  when the full reasoning trace is shown rather than answer-only text.

## Interpretation

1. **The automatic judge fails the predeclared gate.** Its score-3 precision,
   recall, macro-F1, and four-class κ all miss the frozen thresholds, so it is not
   validated as the blanket retention filter.
2. **The human boundary is also noisy.** Human-human κ = 0.556 is moderate and
   remains below 0.70, showing that score-2 versus score-3 role expression is
   genuinely difficult to measure consistently.
3. **Answer-only and full-response views differ materially.** The 27% 2↔3 flip
   rate on the paired sensitivity subset means the visible response region can
   change the retention decision for a substantial fraction of boundary cases.

## Implications

- **T14 → T15/T16:** automatic-judge-only retention is not admissible. Freeze a
  human-backed retention rule before constructing the confirmatory Axes.
- **T13:** blanket rescoring with this judge should not define retention. It is
  optional for descriptive or appendix analysis only, with the measurement
  limitation stated explicitly.
- **Paper:** report the failed validation as a measurement-validity result and
  narrow claims accordingly.

## Decisions needed

1. Freeze the exact **human-backed T14 retention rule**.
2. Decide whether T13 is worth running for descriptive/appendix use.
3. Add a compact measurement-validity result and limitation to the manuscript.

## Reproduce

```bash
python tools/run_t07_judge_validation.py run \
    --consensus results/t06/gold_consensus.csv \
    --key annotations/t14_gold_v3/sampling_key_PRIVATE.json \
    --sensitivity annotations/t14_gold_v3/sensitivity_labeller_1.csv \
    --out results/t07
```

The private key stays local. Only the aggregate validation artifact is committed.
Gold provenance and inputs are recorded in `results/t06/agreement_report.json`.
