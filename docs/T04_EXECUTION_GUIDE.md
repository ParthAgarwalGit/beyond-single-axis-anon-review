# T04 — Formal hook and pooling validation

## What this task proves

T04 checks that the code reads and changes the intended residual stream before any large activation extraction or causal steering run.

The frozen method requires:

- hook target: `model.model.layers[L]`;
- hooked tensor: the first tensor in the block output, before the model's final norm;
- primary block: `16`;
- sensitivity block: `15`;
- mapping: block `L` output corresponds to `hidden_states[L+1]`;
- primary pool: all generated response content tokens;
- sensitivity pools: reasoning tokens and final-answer tokens;
- prompt-prefill positions are never steered;
- generated-token decode positions are steered.

The current canonical code already defines the layer mapping, response segmentation, and arithmetic pooling. T04 validates those definitions against the actual model runtime. The frozen configuration explicitly keeps production blocked until this validation passes.

## Files in this toolkit

Copy these files into the matching folders in the repository:

```text
src/runtime_hooks.py
tools/run_t04_hook_and_pooling_validation.py
tests/test_hook_and_pooling.py
docs/T04_EXECUTION_GUIDE.md
notebooks/t04/T04_hook_and_pooling_validation.ipynb
```

The GPU run creates the required deliverables:

```text
results/validation/hook_validation_v3.json
results/validation/pooling_validation_v3.json
```

## 1. Create the branch

From the latest `main`:

```bash
git checkout main
git pull
git checkout -b t04/hook-pooling-validation
```

## 2. Confirm T03 is present

These existing canonical files must already be in the repository:

```text
configs/method_frozen.yaml
src/config.py
src/generation.py
src/activation_extraction.py
src/prompt_rendering.py
src/steering.py
```

Confirm the YAML still specifies:

```text
primary_block_index: 16
off_by_one_sensitivity_block_index: 15
module_path_template: model.model.layers[L]
mapping: layers[L] output corresponds to output_hidden_states[L+1]
```

Do not change frozen scientific fields simply to make a test pass.

## 3. Add the toolkit files

Place the files at the exact paths shown above. Do not put them under an extra nested toolkit folder inside the repository.

## 4. Install the short-run environment

On Colab or an A100 worker:

```bash
pip install -U \
  "transformers>=4.44" \
  accelerate \
  huggingface_hub \
  pyyaml \
  pytest
```

Prefer the project's pinned environment when it is ready. Record any environment difference in the PR.

## 5. Run the CPU-only tests first

These do not load the 8B model:

```bash
pytest -q tests/test_hook_and_pooling.py
```

Expected result:

- the hook unit tests pass;
- the hand-written segmentation and pooling tests pass;
- the final report test skips because the GPU reports do not exist yet.

## 6. Run the GPU validation

From the repository root:

```bash
python tools/run_t04_hook_and_pooling_validation.py \
  --config configs/method_frozen.yaml \
  --dtype bfloat16 \
  --attn-implementation eager \
  --max-new-tokens 128
```

The script resolves the model and tokenizer revisions to immutable Hugging Face commit SHAs and records them in the reports. For production, the reviewed immutable revisions should be copied into the runtime provenance rather than relying on an unpinned `main`.

The default non-zero coefficient is **validation-only**. It proves that the hook changes the residual stream and downstream logits. It is not the causal steering coefficient and must not be selected using identity or harmfulness outcomes.

## 7. Run the full test file again

```bash
pytest -q tests/test_hook_and_pooling.py
```

The JSON report test should now pass.

## What each validation checks

### Unhooked baseline

The script generates one short deterministic completion with no hook.

### Read-only hook

The same prompt is run with a hook that only reads block 16. Logits and generated token IDs must match the unhooked baseline.

### Zero-vector steering

The steering code path runs with an exact zero vector on decode calls. Logits and generated token IDs must match the baseline.

### Non-zero steering

A fixed seeded unit direction is added at block 16 on generated-token decode calls only. The prompt prefill must remain unchanged. Downstream logits after the first sampled response token must change.

The script also compares generated token IDs. They may remain unchanged when the same token is still the greedy argmax, even though the logits changed.

### Layer indexing

During one teacher-forced pass, read-only hooks capture blocks 15 and 16. The script checks:

```text
model.model.layers[15] output == hidden_states[16]
model.model.layers[16] output == hidden_states[17]
```

within the declared same-path tolerance.

### Prompt and response boundary

The generation start index must equal the stored prompt-token count. Padding and terminal EOS IDs are excluded from the response content pool.

### Pooling

The code checks all-response, reasoning, and final-answer spans on hand-written token sequences. It also checks that:

- arithmetic pooling equals a direct mean over the selected block-output rows;
- prompt positions cannot enter a response pool;
- malformed reasoning keeps the all-response pool but disables reasoning and answer pools;
- an empty answer after `</think>` is handled explicitly;
- multiple balanced reasoning pairs use the final closing marker, as frozen.

### Teacher-forced stored completion

The stored completion is processed in two ways:

1. one full prompt-plus-completion forward pass;
2. one token at a time with KV caching.

Block-16 activations for the same completion must match within the declared BF16 tolerance. The final-answer pooled vectors must also match.

If the short generated completion ends with malformed or unclosed reasoning, the script preserves it for the hook comparisons but uses a clearly recorded direct-answer completion for the answer-pool teacher-forcing check.

### Hook timing

The report records:

- prompt-prefill call count;
- generated-token decode call count;
- maximum prefill delta;
- maximum decode delta.

For non-zero steering, the required result is:

```text
prefill delta = 0
decode delta > 0
```

This also confirms that the first sampled response token is unsteered. Steering begins when that token is fed back into the model and can affect the second and later sampled tokens.

## Review the reports

Open:

```text
results/validation/hook_validation_v3.json
results/validation/pooling_validation_v3.json
```

[Reviewer] should inspect more than the top-level `PASS`. Review:

- resolved model and tokenizer revisions;
- chat-template hash;
- Git SHA and dirty status;
- dtype and attention implementation;
- exact tolerances;
- per-step logit comparisons;
- block-15 and block-16 mapping;
- response spans and token counts;
- teacher-forcing differences;
- hook timing;
- the validation-only scaling note.

## What to do if a check fails

Do not loosen tolerances immediately.

1. Confirm the same model revision, tokenizer revision, dtype, attention implementation, prompt IDs, and seed were used.
2. Confirm the hook is on `model.model.layers[16]`, not the MLP submodule and not `hidden_states[16]`.
3. Confirm the hook returns `None` for read-only calls and prompt-prefill calls.
4. Confirm zero steering runs through the same decode-time code path as non-zero steering.
5. Confirm the first changed non-zero logit step is at least step 1. Step 0 comes from the unmodified prompt prefill.
6. For teacher-forcing differences, first reproduce with eager attention in the pinned environment.
7. Record and justify any tolerance change. Never silently edit the result JSON.

## Completion criteria

T04 is complete only when:

- CPU tests pass;
- both JSON reports say `PASS`;
- read-only and zero steering reproduce the baseline;
- non-zero steering changes the intended residual and later logits;
- block 16 maps to `hidden_states[17]`;
- block 15 maps to `hidden_states[16]`;
- prompt and response boundaries are correct;
- all three pools are validated;
- teacher-forced answer activations match within tolerance;
- [Reviewer] reviews the reports and code.

## Commit and PR

Commit message:

```text
T04: validate hook placement, timing, and token pooling
```

PR title:

```text
T04: Formal hook and pooling validation
```

In the PR, include:

- both result JSON files;
- terminal output showing all tests passed;
- exact GPU and environment;
- a short reviewer note from [Reviewer] confirming the layer mapping, masks, and timing.
