# GPU notebook cells (T12 / T11 / T19)

Self-contained GPU/notebook helpers used during the endgame execution path. These
files are operational reproducibility utilities, not the authority for scientific
results; the reviewed `src/`, `tools/`, frozen configs, and committed result/
provenance records remain canonical.

**One-time setup cell** (run first on a fresh machine):
```python
%pip install -q torch transformers accelerate safetensors huggingface_hub pyyaml sentencepiece bitsandbytes vllm ninja
```
(`bitsandbytes` is only for optional quantized smoke checks; `vllm` + `ninja` are
for T11. vLLM may pin its own torch, so T11 is safest in a fresh runtime.)

| Cell | Task | Model | Output / status |
|---|---|---|---|
| `t12_c80_activation_extract.py` | Teacher-force stored completions; pool all-response/answer @ 32 layers + reasoning @ block 16 | DeepSeek-R1-Distill-Llama-8B | `t12-c80/<BLOCK>/`; exact frozen attempt selection + hash-verified resume |
| `t11_default_generation.py` | Reproducibility rerun for five frozen default conditions × C80 questions | DeepSeek-R1-Distill-Llama-8B | defaults to `t11-defaults-repro/<BLOCK>/`; fail-closed canonical IDs/rendering/schema/provenance |
| `t19_source_model_feasibility.py` | Historical source-model feasibility utility | Qwen3-32B | T19 is already closed by completed T21 BF16 production execution; no new T19 run is required |

## Important: do not regenerate production data for this PR

The production C80 activation/default artifacts already exist and have been used
by the reviewed downstream analyses. The fixes in this PR make these retained GPU
helpers safe for a future reproducibility rerun; they do **not** request or justify
rewriting the existing production artifacts.

T11 therefore defaults to a separate `t11-defaults-repro/<BLOCK>/` prefix. A real
rerun also requires `EXECUTION_GIT_SHA` to be filled with the reviewed 40-hex
source commit. This prevents a cleanup rerun from silently overwriting the
already-analysed defaults or producing an unbound new execution.

## T12 fail-closed behavior

The extraction helper now implements the frozen T09/T10 attempt-selection rule
explicitly: choose the lowest-retry technically valid attempt; if none is valid,
retain the highest-retry failed attempt for technical-failure provenance only.
Those failed fallback rows are not pooled.

Resume no longer treats the existence of `meta_part*.jsonl` as completion. A part
is skipped only when its `part_manifest_*.json` parses and every referenced
metadata/tensor file exists and matches the recorded SHA-256. Missing, corrupt, or
partially uploaded files therefore fail closed instead of being silently accepted.
Newly written parts are checked through the same verifier before they are treated
as complete.

## T11 canonical parity and provenance

The T11 helper is bound to the current DEFAULT-row contract:

- exactly the frozen 80 question IDs for the selected C80 block, crossed with all
  five default conditions;
- canonical 16-hex rollout IDs using the same semantic fields as
  `src.ids.rollout_id(model, "DEFAULT", "", question_id, condition_index)`;
- `arm="DEFAULT"`, `role_id=""`, user channel, and
  `prompt_index == default_condition_index`;
- frozen model/tokenizer revision and chat-template SHA, canonical user-channel
  default rendering, rendered-prompt/message hashes, native prompt token IDs, and
  generation-row boundary/count invariants;
- exact source C80 artifact SHA + row count and frozen block-ID-list SHA before
  question recovery, so the recovered question bytes are bound to the committed
  source artifact rather than accepted from an arbitrary rollout file;
- existing HF rows are fully validated before being skipped; a mere
  `(question_id, condition)` match is not enough;
- the active V4 method hash is the new execution `config_sha256`; the historical
  C80 V3 hash is retained only as separately named
  `matched_c80_generation_config_sha256` comparability provenance.

The helper refuses a real rerun until `EXECUTION_GIT_SHA` is set, and it requires
the final output to close the exact 80 × 5 = 400-cell grid with no duplicates.

## Regression coverage

`tests/test_gpu_cells_contracts.py` is CPU-only and checks the two branches of the
T12 attempt-selection rule, duplicate retry rejection, missing/corrupt part-file
resume failures, frozen C80 block-ID hashes, and parity of the embedded T11
rollout/generation IDs with current `src.ids`.

## T19 closure

T19 is `DONE — CLOSED BY T21 PRODUCTION EXECUTION`. The selected source model is
`Qwen/Qwen3-32B`, and the completed T21 run demonstrates source-comparable BF16
feasibility directly. See `docs/T19_SOURCE_MODEL_DECISION.md` and PR #45.

The retained T19 utility defaults to Qwen3-32B BF16. A 4-bit load is reported only
as `SMOKE_ONLY_NOT_SOURCE_COMPARABLE`; it must not be interpreted as evidence that
the Lu-comparable production setup fits.

## Reproducibility run order

For a true future rerun, the intended ordering is:

1. T12 C80 role activation extraction;
2. canonical/parity-checked T11 default generation;
3. T12 extraction over the corresponding default outputs;
4. downstream reduction from a pinned immutable HF revision.

No production rerun is needed merely to merge these helper-code fixes.
