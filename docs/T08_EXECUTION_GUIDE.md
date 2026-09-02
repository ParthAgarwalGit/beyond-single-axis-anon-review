# T08 execution guide — freeze the 240-ID, three-block manifest

## Purpose

T08 preserves the existing E80 IDs, materializes the T02-frozen C80-A and
C80-B assignments, keeps duplicate IDs 7 and 227 together, applies the same
position-modulo-five prompt assignment for every role, and freezes exact
rendered-prompt hashes before generation.

The latest TODO requires four committed question-manifest files and forbids
inspection of C80-A versus C80-B geometry before both confirmatory blocks are
complete.

## Files in this toolkit

```text
tools/build_t08_question_manifest.py
tools/build_t08_render_hashes.py
tests/test_t08_question_manifest.py
```

The four required outputs are created under `design/`.

## Two-stage freeze

### Stage A — question/block freeze

This stage requires only the merged T01 source artifact and the T02 method
configuration.

```bash
python tools/build_t08_question_manifest.py
pytest -q tests/test_t08_question_manifest.py
```

This creates:

```text
design/questions_E80.json
design/questions_C80_A.json
design/questions_C80_B.json
design/questions_240_manifest.json
```

When the model/tokenizer/chat-template fields remain unpinned, the combined
manifest deliberately reports:

```text
status: DRAFT_PRE_RENDER
ready_for_generation: false
```

That is not a failure. It prevents the team from claiming that prompt bytes
were frozen when only question assignments were frozen.

### Stage B — exact rendered-prompt hash freeze

Before any C80 generation:

1. `configs/method_frozen.yaml` must say `freeze_status: FROZEN`.
2. The primary model must have non-null:
   - `model_revision`;
   - `tokenizer_revision`;
   - `chat_template_sha256`.
3. T03's corrected canonical modules must be present:
   - `src/config.py`;
   - `src/ids.py`;
   - `src/prompt_rendering.py`.
4. Install the pinned Transformers environment.

Run:

```bash
python tools/build_t08_render_hashes.py
```

Expected output:

```text
design/rendered_prompt_hashes_USER_TRANSLATED_LU.jsonl
```

It contains exactly:

```text
275 roles × 240 questions = 66,000 rows
```

Each row stores the role/question assignment, deterministic rollout ID,
message-list hash, exact rendered-prompt hash, prompt token count, revisions,
chat-template hash, and method-config hash.

Then rebuild the four manifests in strict mode:

```bash
python tools/build_t08_question_manifest.py   --render-manifest design/rendered_prompt_hashes_USER_TRANSLATED_LU.jsonl   --strict-final

pytest -q tests/test_t08_question_manifest.py
```

The combined manifest must then report:

```text
status: FROZEN
ready_for_generation: true
```

## Review checks for [Author A]

- E80 IDs exactly match the prior frozen E80.
- C80-A and C80-B each contain 80 IDs.
- The three blocks are disjoint and cover IDs 0–239.
- IDs 7 and 227 are both in C80-A.
- Every prompt index appears 16 times in each block and 48 times overall.
- Source question text hashes match the pinned T01 artifact.
- Block ID-list and assignment hashes match T02.
- Render manifest contains 66,000 unique rollout IDs.
- Every question has 275 role rows.
- One model revision, tokenizer revision, chat-template hash, and method hash
  are used throughout.
- No C80 geometry was inspected while building or reviewing the manifest.

## Branch and PR

Create the branch from updated `main` only after T01 and T02 are merged:

```text
t08/question-manifest
```

Suggested commit:

```text
design(t08): freeze 240-ID three-block question manifest
```

Suggested PR title:

```text
design(t08): freeze E80/C80-A/C80-B extraction manifest
```

Do not start T10 C80 generation until the combined manifest says
`ready_for_generation: true`.
