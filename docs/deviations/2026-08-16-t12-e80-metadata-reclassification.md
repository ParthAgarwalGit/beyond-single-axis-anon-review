# T12 E80 metadata reclassification propagated to reports/manifest (PR #28 blocker close)

## Date
2026-08-16

## What this closes
The remaining blocker on PR #28: after the code fixes (B1/B2/B3, merged via the
review-fix PR) the seeded-unclosed reclassification existed only in
`results/t12/provenance/review_fix_real_corpus_audit.json`, while the committed
per-mode reports and `activation_manifest_v3.json` still showed the stale
`direct_answer` / `answer: 22000` classification.

## What changed (metadata only — no tensor rerun, no forward pass)
Propagated the audit into the committed artifacts:

- `results/t12/{translated,wrapper}/activation_recompute_report.json` and the
  same reports embedded in `activation_manifest_v3.json`:
  - `pool_counts.answer` 22000 → **21961** (the 39 seeded-unclosed rows have no
    answer pool);
  - `segmentation_counts`: `direct_answer: 39` → **`malformed_reasoning: 39`**;
  - `segmentation_provenance_counts`: 39 rows moved
    `frozen_segment_response` → **`prompt_seeded_unclosed_open_marker`**;
  - added `prompt_seeded_unclosed_report` and
    `frozen_boundary_prefix_check_report` (pass rate 1.0).
- `default` report: unchanged counts (0 unclosed); the two report sections added
  for consistency.

`activation_tensors_modified: false`. `validity_counts` are unchanged
(reclassification changes segmentation, not technical validity). The 78 affected
rows are 39 per role arm (translated: 38 length + 1 stop; wrapper: 39 length),
listed by uid in the audit and now in each report's
`prompt_seeded_unclosed_report.uids`.

## Off-repo metadata (completed 2026-08-24, metadata-only)
The original E80 metadata tree was recovered from the production output and
verified byte-for-byte against the committed pre-patch manifest: 174/174
`meta_part*.jsonl` files matched their recorded SHA-256 values (86 translated,
86 wrapper, 2 default).

The committed fail-closed patcher was then dry-run against that exact tree and
found exactly 78 target UIDs across 59 metadata files: 39 translated and 39
wrapper. The patch was applied locally, and a second dry-run passed idempotently.
Exactly those 59 translated/wrapper metadata files changed; the default metadata
remained untouched.

The patched 174-file metadata tree was published to
`[Author-A-HF]/persona-artifacts` under `t12-e80/` at immutable HF revision
`b0c9852cb42a17d222965ed2d27403ea4f86b2a1`.

For the 78 affected rows, the patch sets:
`has_answer_pool: false`, `has_reasoning_pool: false`,
`segmentation_case: malformed_reasoning`,
`segmentation_provenance: prompt_seeded_unclosed_open_marker`, null
answer/reasoning token counts, and the corresponding segmentation note. The
stored all-response tensors remain valid and are kept. No model forward pass and
no tensor rewrite occurred.

The complete before/after per-file hashes and target UID accounting are recorded
in `results/t12/provenance/hf_metadata_patch_receipt.json`. The same immutable
HF revision is pinned in `results/t12/provenance/production_fingerprint.json`.

Downstream is protected: the merged T17 reducer takes availability from the
`has_*_pool` flags and **fails closed** if a tensor contradicts its flag, so the
78 stale answer/reasoning tensors are no longer pool-eligible at the pinned E80
metadata revision.

## Config re-pin
`production_fingerprint.json` records `runtime_config_sha256` (7b8bd291…) ≠
`base_config_sha256` (73be73df…, the committed frozen YAML the C80 role rollouts
also carry). The difference is byte-level YAML serialization only:
`results/t12/provenance/runtime_config_diff.json` records
`semantic_config_equal: true` with zero differing fields. No configuration value
or frozen scientific field differs between the runtime-resolved and base YAML.
The extraction code state is pinned by the audit `git_sha`.

## Reproducibility
The real-corpus review audit is reproducible from the stored E80 rollout shards
with `tools/build_t12_review_fix_real_corpus_audit.py`. The script reuses the
committed T12 row-preparation/segmentation logic, loads the frozen tokenizer
revision, performs no model forward pass, verifies duplicate-free UIDs, and
recomputes the seeded-unclosed counts, UID lists, pool availability, segmentation
counts, and per-row frozen boundary prefix checks.

The external metadata patch is reproducible with
`tools/patch_t12_hf_seeded_unclosed_metadata.py`. It reads the canonical 39+39
UID lists from the committed audit, rejects unexpected pre-patch states, requires
whole-corpus coverage of exactly 78 targets, is idempotent for re-verification,
and records before/after hashes in the committed receipt.

## Verification context (owner-reported)
C80-A (21,998/22,000 valid), C80-B (22,000/22,000), and the C80 defaults
(397/400 + 400/400) reconcile with expected row counts and zero duplicate uids;
no tensor rerun is required.
