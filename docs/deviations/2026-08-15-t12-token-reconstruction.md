# T12 activation recompute: token-ID reconstruction and seeded-boundary handling

## Date
2026-08-15

## Issue
`response_tokenization.authoritative_token_ids` is frozen: primary masks must
not be reconstructed by concatenating independently retokenized strings; the
generation engine's stored token IDs are authoritative. Run-003 did **not**
persist generation-native output token IDs, so T12 reconstructs them from the
stored `rendered_prompt` + `completion`. That is a §16 method deviation and is
recorded here, together with two consequences the PR #28 review flagged.

This record covers three coupled points.

### (a) Token IDs reconstructed from stored text
- **Reason:** run-003 lacks generation-native token IDs; retokenizing the
  stored text deterministically is the only option short of regenerating 44,400
  outputs, which would change the artifacts under review.
- **Affected frozen field:** `response_tokenization.authoritative_token_ids`.
- **Affected artifacts:** all T12 pooled tensors and metadata
  (`t12-activation-recompute-v3`).
- **Outcomes inspected before decision:** none. This is a mechanical
  reconstruction, not a result.
- **Supporting evidence:** the historical five-row reconstruction gate
  (min cosine 0.99997 against the historical answer tensors) plus, from this
  change, a **per-row** frozen delimiter prefix-token check reported as a pass
  rate (see (b)/item 2) rather than relying on five rows to cover the corpus.

### (b) Prompt-seeded reasoning boundary, and its now-reported prefix check
DeepSeek-R1-Distill's chat template prefills the opening `<think>` into the
prompt (`add_generation_prompt=True`), so a well-formed output is
`reasoning</think>answer` with **zero** `<think>` and **one** `</think>` in the
completion. The frozen `segment_response` reads only the completion and would
call that an orphan close (malformed). T12 reconstructs the boundary from the
full stored serialization for these prompt-seeded rows.

The delimiter prefix-token check (`prefix_ids == content[:len(prefix_ids)]`) is
now computed **per row** on this path and reported as
`frozen_boundary_prefix_check_report.pass_rate` in each mode report, instead of
being left `None` for ~99.8% of production rows. A row that passes is an exact
reconstruction at its boundary; the pass rate scales the five-row gate to the
whole set.

### (c) Unclosed seeded reasoning is malformed, not direct answer
Because the `<think>` is seeded by the prompt, a completion with **no**
`</think>` is an **unclosed open marker**, which METHOD_FREEZE 6.2 classifies as
`malformed_reasoning`: the all-response pool is retained, but the reasoning and
final-answer pools are unavailable and the judge abstains. The prior code let
these fall through to the frozen segmenter, which saw a marker-free completion,
returned `direct_answer`, and stored the entire in-progress reasoning trace as
the **answer** tensor with `validity == "valid"`.

- **Decision:** reclassify prompt-seeded rows with no `</think>` as
  `malformed_reasoning` (provenance `prompt_seeded_unclosed_open_marker`).
  All-response pool retained; reasoning/answer pools unavailable; judge abstains.
  The row's technical `validity` label is unchanged (e.g. `truncated` when the
  budget ran out mid-trace).
- **Why it matters:** under the old behaviour these pseudo-answer tensors and
  their judge inputs could have entered the answer-pool sensitivity and the role
  judge. Reclassifying quarantines them, exactly as the malformed rule intends.
- **Reported for audit:** `prompt_seeded_unclosed_report` in each mode report
  gives the count, the `finish_reason` split, and confirms the rendered prompt
  ends with `<think>` by construction.

## Effect on already-produced artifacts
The stored **all-response** tensors are unaffected and reusable — reclassifying
only changes metadata and answer/reasoning-pool availability, not the
all-response pool. Applying this fix to the committed run is a **metadata-only,
CPU** re-run of `prepare_row` over the stored rollouts (no model, no GPU, no
regeneration); it moves the ~39 previously-`direct_answer` rows per role arm to
`malformed_reasoning` and drops their answer-pool availability. The stale answer
tensors for those rows become unreferenced (consumers filter on
`has_answer_pool`).

## Runtime config provenance (review item 3)

The exact production runtime-resolved configuration has now been recovered
from the persistent T12 production provenance directory and committed as
`configs/t12_runtime_resolved.yaml`.

Its byte-level SHA256 is:

`7b8bd2918a61cf53cf97fcc8bc45d43e5f96a3987170467a2aff5ba8631c3a3e`

which exactly matches `runtime_config_sha256` recorded in
`results/t12/provenance/production_fingerprint.json`.

The frozen base configuration has byte-level SHA256:

`73be73df4640c2d32bfbc8b6009741c0fadd8c466acb703d1f99df35d5c8cd79`.

Although the two serialized YAML files have different file hashes, parsing
both files produces identical configuration objects. A recursive semantic
comparison yields zero differing fields. Therefore no runtime configuration
value, including any frozen scientific field, differed from the frozen base
configuration.

The machine-readable comparison is committed at
`results/t12/provenance/runtime_config_diff.json`.

## Reviewer approval
Pending. This record and the code change (`tools/run_t12_activation_recompute.py`,
`tests/test_t12_activation_recompute.py`) are offered for the PR #28 reviewer to
accept; the runtime-config artifact (item 3) is owed by the run owner.

## Claim boundary
T12 provides reconstructed-token-ID activation pools with declared provenance.
The prompt-seeded reconstruction is validated by the five-row gate and the
now-reported per-row prefix check. Unclosed seeded reasoning is treated as
malformed per the freeze and does not enter the answer/reasoning sensitivities
or role judging.
