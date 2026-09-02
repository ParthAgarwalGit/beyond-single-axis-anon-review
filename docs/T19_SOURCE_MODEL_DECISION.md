# T19 source-model feasibility decision

## Status

`DONE — CLOSED BY T21 PRODUCTION EXECUTION`

T19 was a time-boxed source-model and compute go/no-go for the post-PI source-capping branch. Its purpose was to choose a Lu et al. source model and determine whether source-comparable execution was feasible. No additional T19 benchmark is required.

## Decision

The selected source model is `Qwen/Qwen3-32B` at revision `9216db5781bf21249d130ec9da846c4624c16137`.

The decisive evidence is the completed T21 production execution rather than a separate synthetic feasibility benchmark:

- BF16 execution completed without 4-bit/8-bit quantization;
- thinking was disabled;
- the released Lu et al. capping artifact and source setting were used;
- the source setting was `layers_46:54-p0.25` (layers 46–53);
- the public safety evaluation completed 200/200 generations/judgments;
- the Lu-sized public capability battery completed 6,224/6,224 generations.

See PR #45 and its committed T21 provenance artifacts for the exact model revision, Lu source commit, capping artifact revision/hash, axis hash, execution notebooks, judge provenance, capability results, and claim boundaries.

## Why Qwen rather than Gemma

Qwen3-32B is the source-model target for the capping reproduction because the released Lu et al. artifacts include an actual Qwen3-32B activation-capping configuration. Gemma 2 27B remains a source-study geometry model, but it is not the primary source-capping reproduction target for this branch.

## Precision boundary

A 4-bit load may be useful as an engineering smoke test, but it is not evidence of source-comparable feasibility. The final source-capping evidence uses BF16. Any retained T19 utility must distinguish `BF16_SOURCE_COMPARABLE_FEASIBLE` from quantized smoke-only operation.

## Closure rule

Do not spend further submission-week compute benchmarking Gemma, Llama 3.3 70B, or alternative Qwen configurations under T19. New source-model experiments, if any, belong to the later intervention tasks and must have an independent scientific reason.
