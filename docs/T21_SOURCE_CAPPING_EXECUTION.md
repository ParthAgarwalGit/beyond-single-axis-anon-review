# T21 source-capping execution and reproducibility record

## Purpose

T21 tests whether the released Lu et al. Qwen3-32B Assistant-Axis capping setting can be reproduced with the public intervention artifacts, while preserving a strict provenance boundary where the source paper/release does not publish enough information for an exact evaluation replay.

## Frozen intervention

- Source repository: `safety-research/assistant-axis`
- Source commit: `a98961956072224eaf244eb289d6c01700b63795`
- Model: `Qwen/Qwen3-32B`
- Model revision: `9216db5781bf21249d130ec9da846c4624c16137`
- Precision: BF16; no 4-bit/8-bit quantization
- Thinking: disabled
- Capping artifact: `lu-christina/assistant-axis-vectors`, revision `3b3b788432ad33e3a28d9ff08e88a530c0740814`
- Capping file SHA-256: `6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24`
- Experiment: `layers_46:54-p0.25` (layers 46–53)
- Released Assistant Axis SHA-256: `a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211`

The capping-vector orientation was checked statically and dynamically. The released config vector is sign-opposed to the released Assistant Axis and the source upper-cap operation moves activations toward the released Assistant direction. No manual sign flip is applied.

## Capability battery

Execution notebook reviewed: `T21_FULL_LU_SIZED_CAPABILITY_BATTERY_QWEN32B_BF16_v2.ipynb` (SHA-256 recorded in `results/t21/ARTIFACT_MANIFEST.json`).

The frozen public manifest contains 3,112 items per condition: IFEval 541, MMLU-Pro 1,400, GSM8K 1,000, EQ-Bench 171. Manifest SHA-256: `7217d6021d77d9dc2091601dbe85a2974ccaa610b97fc77e90007732f9931f55`. Two conditions yield 6,224 completed generations. Exact Lu membership for the subsampled MMLU-Pro/GSM8K items is not claimed; the notebook's deterministic freeze seed is `210817`.

A superseded 24-item smoke/diagnostic capability check produced MMLU-Pro accuracies of 0.125 unsteered and 0.208 capped; those values were discarded from T21 conclusions because the diagnostic used a different lightweight prompt/parser path and the later full battery achieved >99.6% MMLU-Pro parseability, making the compact result unsuitable as a capability estimate.

Canonical results are in `results/t21/t21_full_capability_report.json` and `results/t21/t21_full_capability_summary.csv`. `results/t21/t21_capability_uncertainty.json` records aggregate-count uncertainty checks: approximate 95% intervals for the capped-minus-unsteered binary deltas are IFEval `[-0.0460, 0.0423]`, MMLU-Pro `[-0.0438, 0.0238]`, and GSM8K `[-0.0208, 0.0148]`. These are conservative aggregate-only Newcombe/Wilson intervals because paired discordance counts are not committed. EQ-Bench is continuous, so a binomial interval is not applicable; a paired interval would require per-item scores. The reported 2.2619% sum of relative reductions is therefore descriptive and does not have a defensible combined confidence interval from the committed aggregates alone.

## Safety evaluation and judge

Execution notebook reviewed: `T21_03_SOURCE_MAX_MATCH_DEEPSEEK_V3_D2_2_v5_NOVITA.ipynb` (SHA-256 recorded in `results/t21/ARTIFACT_MANIFEST.json`).

The frozen adapted safety manifest contains 100 JBB harmful behaviors, SHA-256 `d611e904f3b5d47c71e5ab1f3fa1a84ead6cfd5d94dba337f5c3b71aafef7793`. This is intentionally labeled `JBB_BEHAVIOR_ONLY_ADAPTED_NOT_LU_PERSONA_JAILBREAK`.

The judge run uses:
- paper judge name: `deepseek-v3`
- model ID: `deepseek-ai/DeepSeek-V3`
- recorded public model-repo revision: `e815299b0bcbac849fa540c768ef21845365c9eb`
- provider: `novita` (our reproducibility choice; source provider undisclosed)
- faithful Appendix D.2.2 transcription
- first 512 source-model response tokens
- no explicitly supplied temperature/top-p/max-token values because the source paper does not disclose them

All 200 safety generations were judged successfully. The corrected canonical report records `judge_prompt_fidelity = FAITHFUL_TRANSCRIPTION_SOURCE_APPENDIX_D2.2`.

## External raw artifacts

Large JSONL/checkpoint artifacts remain on persistent storage and are intentionally not committed because the repository ignores generation JSONLs/checkpoints. The committed summary artifacts contain the row counts, frozen-manifest hashes, source/model/config pins, results, and claim boundaries required for review. See `results/t21/ARTIFACT_MANIFEST.json`.

## Closure decision

T21 is complete with status `PARTIAL / EVALUATION-LIMITED`: intervention reproduction is successful, capability preservation is broadly consistent with the source claim, and the adapted behavior-only safety test does not show a strict-harm reduction. No post-hoc tuning or safety-set replacement is performed after observing this result.
