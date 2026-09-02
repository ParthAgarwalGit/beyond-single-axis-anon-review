# T10 — C80 translated-arm role generation (prepared, not launched)

**Owner/Executor:** [Author B] and [Author A] · **Reviewer:** [Reviewer]
**Status:** READY (prepared) · **BLOCKED ON T02–T04–T08** for launch
**Compute:** Working shared A100 / Colab Pro+ (production); CPU only for the dry run

Generates the translated-arm role-conditioned outputs for the two untouched
confirmatory blocks:

```
275 roles × 160 question IDs (C80-A + C80-B) = 44,000 new outputs
```

split into two 80-question halves of **22,000** each. The two halves are the two
blocks; they run the **same** builder and worker, differing only in `--block`,
so the 44,000 IDs reconcile.

- **[Author A]'s half:** C80-B (22,000)
- **[Author B]'s half:** C80-A (22,000)

(Block-to-person assignment is a coordination choice, not a scientific one —
confirm with [Author B]; the code is symmetric.)

Each role–question pair uses exactly one role prompt, at the question's frozen
positional prompt index (T08 `prompt_assignment`), so each block is balanced
**4,400 per prompt index**.

## Deliverables

| File | Purpose |
|---|---|
| `tools/build_t10_c80_manifest.py` | Deterministic 22,000-rollout block manifest (`--block`) |
| `tools/run_t10_role_generation.py` | Role-generation worker (resume, schema-checked) |
| `tests/test_t10_c80_generation.py` | 6 CPU-only checks |

## Deterministic fingerprints

```bash
python tools/build_t10_c80_manifest.py --block C80-A
python tools/build_t10_c80_manifest.py --block C80-B
```

Expected (with the T08 branch's frozen config + block files):
- each block: `n_outputs = 22000`, `both_blocks_total = 44000`, per-prompt-index all `4400`
- `C80-A rollouts_sha256 = 60e4e7b5984a47db22e6e4b5def97832e7269c4d57863102449f4f9bad27bf33`
- `C80-B rollouts_sha256 = e7b532e3816e47489d2d6c48cf43da7f16cace4c15f878f00d8c8c6a5eb86ac5`

The two blocks are disjoint in both rollout IDs and question IDs, and reconcile
to 44,000. `--strict-final` requires `freeze_status: FROZEN` and pinned
revisions.

## Per-row fields (TODO T10)

Each row carries: deterministic rollout ID, block, role, question ID, prompt
index, channel, rendered-prompt hash, model/tokenizer revision, seed and
decoding settings (`runtime_settings`), output text, finish reason,
`token_region_counts` (all-response / reasoning / final-answer token counts),
and `technical_failure_code` (the frozen validity label, or `null` when valid).
Rows are validated against the canonical generation schema before any byte is
written.

## Worker

Deterministic rollout IDs come from `src.ids.rollout_id(model,
USER_TRANSLATED_LU, role_id, question_id, prompt_index)`. Role-prompt and
question texts are read from their frozen artifacts and hash-checked (question
lookup cached per block), so a drifted prompt or question cannot silently change
an input. Segmentation and validity labelling use the canonical generation code.

### Production preflight (all checked before any forward pass)

1. **Engine identity** — `preflight()` verifies the actual engine's
   `model_revision`, `tokenizer_revision`, and `chat_template_sha256` against the
   frozen manifest; a mismatch aborts.
2. **Frozen decoding** — `load_frozen_decoding()` loads decoding from the frozen
   **T09 runtime artifact** (`frozen_decoding` block in
   `results/audit/E80_acceptance_report.json`) and checks the required keys; an
   arbitrary `runtime_settings` dict is not accepted in production, and the
   artifact SHA is stamped into every row.
3. **Frozen rendered-prompt hash** — each rollout's freshly rendered prompt hash
   is verified against the manifest's `expected_rendered_prompt_sha256`
   **before** `engine.generate()`; a mismatch raises `RenderHashMismatch` and no
   forward pass is spent.

### Resume integrity

`load_existing_attempts()` validates every existing row against the canonical
generation schema and **rejects** the run on any: corrupted JSON line, foreign
`config_sha256`, `rollout_id` outside this manifest, foreign arm, engine-revision
mismatch, or duplicate `(rollout_id, retry)`.

### Retry handling

Retryable **technical** failures (`generation_error`, `truncated`,
`serialization_failure`, `token_alignment_failure`) are retried under the same
rollout ID with an incrementing `retry` counter, up to `--max-retries` (default
3). **Semantic degeneration is never retried** (`degenerate_repetition`,
`empty_response`, and `valid` are terminal); `degenerate_repetition` is
explicitly guarded. A rollout counts as done only when it reaches a terminal
outcome or exhausts its retry budget.

### `--strict-final` (manifest builder)

Requires `freeze_status: FROZEN`, pinned model/tokenizer/chat-template
revisions, the T08 block artifact itself **FROZEN and schema-compatible**
(`t08-question-block/1.0`), and a frozen T08 rendered-prompt hash file
(`--render-hashes`) so every rollout carries its `expected_rendered_prompt_sha256`.

Dry run (CPU, no model — a wiring test that bypasses the T09 requirement):

```bash
python tools/build_t10_c80_manifest.py --block C80-B --out /tmp/m.json
python tools/run_t10_role_generation.py --manifest /tmp/m.json --dry-run --limit 20
```

Production generation is **not** auto-wired: the worker loads frozen decoding,
then raises rather than launching. Launch waits on **T04**, **strict-final T08**,
and the **T09 runtime freeze**.

## Launch preconditions

- **T04 hook/pooling validation** — merged to `main` (#18). ✅
- **T08 240-ID manifest** — merged to `main` (#20); the C80 block files are
  `FROZEN` (`t08-question-block/1.0`). ✅
- **strict-final T08 render hashes** — the translated-arm rendered-prompt hash
  JSONL is a model-dependent artifact (produced by `build_t08_render_hashes.py`)
  and is gitignored, so it is not on `main`. Generate it, then pass it to the
  builder via `--render-hashes` to build a `FROZEN` T10 manifest. ⏳
- **T09 runtime freeze** — `results/audit/E80_acceptance_report.json` with a
  `frozen_decoding` block; not yet available. Until then production decoding is
  provisional. ⏳

So the remaining gate before a frozen 22k run is **T09 decoding** plus
**generating the strict-final render-hash file**; everything else this PR needs
is on `main`.

## Base

Rebased directly onto `main` now that T04 (#18) and T08 (#20) are merged. T10
depends on nothing from the still-open T11 PR (#22): it does not use the DEFAULT
`default_rollout_id` / schema decoupling.

This PR carries the `.gitattributes` **yaml-LF** rule so `method_frozen.yaml`
(whose config SHA is stamped into every artifact and checked by the builder and
`preflight`) is byte-reproducible on every platform. PR #22 also adds these two
lines; whichever lands second should drop the duplicate hunk.
