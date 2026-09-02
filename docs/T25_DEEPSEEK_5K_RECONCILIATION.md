# T25 — DeepSeek 5k signed-steering reconciliation

## Scope

T25 is a **finish/reconcile, do-not-restart** task for the DeepSeek-R1-Distill-Llama-8B signed-steering experiment. The end-stage plan keeps this run as complementary reasoning-model causal-transfer evidence if valid compute is already underway. It does not replace the source-model capping specificity branch.

The frozen design is:

- 50 P50 roles;
- 4 causal prompt indices: 0, 1, 2, 3;
- 5 project-authored harmful questions: HQ1–HQ5;
- 5 unique steering conditions: `shared_zero`, `assistant_axis_toward`, `assistant_axis_away`, `random_positive`, `random_negative`;
- V4 two-column signed-dose encoding: `axis_dose` and `random_dose`;
- one common nonzero magnitude `a` across the four signed nonzero conditions;
- 5,000 unique semantic cells total.

Raw generations and private blinded condition maps remain outside git. Git stores immutable hashes and outcome-blind aggregate reconciliation/provenance records.

## What this branch adds

- `src/t25_reconciliation.py`: outcome-blind reconciliation core, including signed-dose and P50/held-out-Axis checks.
- `tools/reconcile_t25_deepseek_5k.py`: fail-closed CLI for an existing T25 corpus.
- `tests/test_t25_reconciliation.py`: synthetic tests for the complete factorial, signed doses, duplicates/missing cells, blinding-map resolution, frozen-P50 gate, blindness hardening, and held-out-Axis exclusion/binding.

This branch does **not** fabricate a 5k result or rerun generation.

## Required external inputs

The real final reconciliation needs the existing off-repo T25 artifacts:

1. final T25 generation JSONL, one final status per planned semantic cell, carrying `axis_dose` and `random_dose`;
2. a stable external locator for that raw artifact, such as an HF repo + immutable revision + path or a persistent Drive identifier;
3. frozen P50 manifest from T22;
4. private blinded condition-ID map if raw rows do not carry canonical condition names;
5. private steering-prompt-ID to prompt-index map if raw rows do not carry prompt indices;
6. run provenance JSON binding the primary model, nonzero steering coefficient, and raw-output SHA-256;
7. held-out causal-Axis provenance exposing the exact role IDs used to construct the Axis, the Axis hash, and `p50_manifest_sha256` bound to the exact frozen P50;
8. optional causal activation manifest if downstream representations were captured.

The tool does not guess missing mappings or provenance. Missing final provenance fails closed.

### Minimal run-provenance schema

The final path requires at least:

```json
{
  "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
  "nonzero_coefficient": 8.0,
  "outputs_sha256": "<sha256 of exact raw T25 JSONL>"
}
```

`8.0` above is only an example schema value. The real file must contain the coefficient actually used by the historical run. The auditor derives the magnitude independently from the row-level signed-dose columns and requires an exact match. Optional provenance such as model revision, mean residual norm, held-out-Axis hash, random-control-vector hash, and source git SHA is carried through when present.

## Signed-dose contract

Every final semantic row must carry numeric finite `axis_dose` and `random_dose` values consistent with its canonical condition:

| condition | axis_dose | random_dose |
|---|---:|---:|
| `shared_zero` | `0` | `0` |
| `assistant_axis_toward` | `+a` | `0` |
| `assistant_axis_away` | `-a` | `0` |
| `random_positive` | `0` | `+a` |
| `random_negative` | `0` | `-a` |

All nonzero rows must share one common `|a|`. If the active V4 config has a non-null frozen coefficient, the observed magnitude must also match it. Final reconciliation additionally binds the observed magnitude to the external run-provenance file.

## Candidate reconciliation while T22 is still upstream

Candidate mode is explicit:

```bash
python tools/reconcile_t25_deepseek_5k.py \
  --outputs /persistent/t25/final_outputs.jsonl \
  --artifact-location 'HF_REPO@IMMUTABLE_REVISION/path/to/final_outputs.jsonl' \
  --p50-manifest design/causal_P50_assistant_proximal.json \
  --condition-map /persistent/t25/condition_map_PRIVATE.json \
  --prompt-map /persistent/t25/prompt_map_PRIVATE.json \
  --run-provenance /persistent/t25/run_provenance.json \
  --allow-candidate-p50
```

Candidate mode writes by default to:

- `results/t25/candidate/T25_RECONCILIATION_CANDIDATE.json`;
- `results/t25/candidate/T25_ARTIFACT_MANIFEST_CANDIDATE.json`.

It may emit `PASS_CANDIDATE_RECONCILIATION`, but it is **not T25 DONE/FROZEN**. Candidate mode may omit final-only provenance such as the held-out-Axis binding.

## Final reconciliation

Final intent is the default. After the real T22 P50 is frozen:

```bash
python tools/reconcile_t25_deepseek_5k.py \
  --outputs /persistent/t25/final_outputs.jsonl \
  --artifact-location 'HF_REPO@IMMUTABLE_REVISION/path/to/final_outputs.jsonl' \
  --p50-manifest design/causal_P50_assistant_proximal.json \
  --condition-map /persistent/t25/condition_map_PRIVATE.json \
  --prompt-map /persistent/t25/prompt_map_PRIVATE.json \
  --run-provenance /persistent/t25/run_provenance.json \
  --heldout-axis-provenance /persistent/t25/heldout_axis_provenance.json \
  --activation-manifest /persistent/t25/causal_activation_manifest.json
```

Final intent does **not** silently downgrade to candidate. If any final gate is absent or fails, the command aborts and does not emit `PASS_FINAL_RECONCILIATION`.

The final path requires all of the following:

- exactly 5,000 final rows and 5,000 unique stored UIDs;
- exactly one row for every 50 × 4 × 5 × 5 factorial cell;
- no duplicate or unexpected roles, prompts, questions, or conditions;
- technical status belongs to the frozen technical-validity vocabulary;
- every row obeys the V4 two-column signed-dose contract;
- all nonzero rows use one common magnitude `a`;
- the observed magnitude matches the run-provenance coefficient, and the active V4 coefficient too if that field is non-null;
- the raw-output SHA-256 and stable external location are recorded;
- the run-provenance `outputs_sha256` binds to that exact raw artifact;
- the output role set exactly matches the frozen P50;
- P50 `membership_sha256` is recomputed from `selected_roles` in rank order and must match;
- the held-out causal Axis exposes its construction-role IDs and has zero P50 overlap;
- the held-out Axis provenance contains `p50_manifest_sha256` equal to the exact frozen P50 manifest hash;
- the Axis hash is present;
- optional activation provenance, if supplied, is consistent with the reconciled output artifact/UID set.

## UID provenance boundary

The current repository helper `src.ids.steering_rollout_id` coerces `question_id` to an integer, while T25 uses `HQ1`–`HQ5` harmful-question IDs. Because the historical T25 UID minting convention is not established by the current helper, the auditor does **not** pretend to recompute stored UIDs.

Stored UIDs are used for uniqueness and cryptographic provenance only. Scientific cell identity is independently re-derived from `(role_id, prompt_index, harmful_question_id, condition)`, and the exact 5,000-cell factorial is checked from those fields. If the historical UID convention is later recovered from the actual run code/provenance, a separate UID-reproduction check can be added without changing the factorial correctness gate.

## Blindness and privacy hardening

The raw JSONL reader immediately projects each private row onto an allow-list of outcome-blind audit fields. `completion`, `output_text`, automatic/human judge labels, judge rationales, and downstream causal outcomes are not retained in the reconciliation corpus.

Technical-status strings are also fail-closed against the frozen technical-validity vocabulary before they can enter committed aggregate counts. This prevents a historical custom status string from smuggling semantic/outcome information into the report.

The raw artifact remains private/off-repo, but its **stable external location** is intentionally recorded so the SHA-bound artifact remains retrievable rather than depending on out-of-band knowledge.

## Retry and immutability rule

The audit never silently removes semantic degeneration or replaces completed semantic outputs. Only genuine technical failures may be retried under the already frozen retry policy. The final corpus must contain one final technical status for every planned cell.

## Committed deliverables after the real audit

Only final mode writes the canonical deliverables by default:

- `results/t25/T25_RECONCILIATION.json`;
- `results/t25/T25_ARTIFACT_MANIFEST.json`;
- any small non-private held-out-Axis/activation provenance records appropriate for git.

The raw 5k JSONL and private blinded maps remain external. Their SHA-256/byte metadata and the raw artifact's stable external location are recorded in the manifest.

## Historical V4 note

The reconciliation report is stamped with the active V4 audit config because V4 defines the current acceptance contract. That stamp does **not** claim that the existing historical T25 generation itself was executed under V4. The report states this explicitly, and the run-provenance file should preserve the actual historical generation metadata where available.

## Claim boundary

T25 answers whether the DeepSeek signed-steering intervention corpus is complete, correctly signed/scaled, correctly bound to P50 and a held-out Axis, and reproducibly auditable. It does **not** establish that the causal judge is valid or that Assistant-Axis steering changes harmfulness/identity more than random steering. T26/T27 own measurement validation; T28 owns the outcome statistics.

## Current branch status

`READY_FOR_EXTERNAL_ARTIFACT_RECONCILIATION`.

The repo-side reconciliation path is implemented. Real `T25_RECONCILIATION.json` and `T25_ARTIFACT_MANIFEST.json` cannot be truthfully emitted until the existing off-repo 5k artifact, its stable external location/run provenance, and the final T22/held-out-Axis provenance are available to the runner.
