# T23 causal-config migration into Method Freeze V4

**Recorded:** 2026-08-15  
**Decision date:** 2026-08-10  
**Task:** T23  
**Issue:** #31  
**Historical config:** `configs/method_frozen.yaml`  
**Downstream config:** `configs/method_frozen_v4.yaml`  
**Required reviewer for this amendment:** `[Reviewer]`  
**Review status:** `PENDING_GITHUB_PR_REVIEW`

## Reason

T23 had to resolve two causal-design details that the historical V3 method freeze left under-specified: which four of the five existing role-prompt indices are used for the project causal adaptation, and the guards required before a P50 causal manifest can be called frozen. The T23 implementation chose prompt indices `0,1,2,3`, recorded them as a `PROJECT_DECISION`, and added the P50 size / literal-Assistant / manifest-hash guards.

Those additions were mistakenly written into the already-frozen historical `configs/method_frozen.yaml`. That changed the bytes and invalidated the provenance contract for artifacts that were stamped with the historical SHA-256.

## Resolution

1. Restore `configs/method_frozen.yaml` byte-for-byte to the historical file from commit `855d683`.
2. Preserve its SHA-256 as `73be73df4640c2d32bfbc8b6009741c0fadd8c466acb703d1f99df35d5c8cd79`.
3. Move the T23 causal additions into `configs/method_frozen_v4.yaml`, the declared downstream configuration.
4. Re-point T23 and other new-decision consumers to V4 rather than the historical file.
5. Preserve existing raw E80/C80 generation stamps; do not rewrite their `config_sha256` fields.

## Frozen fields introduced by T23

- `causal_prompt_selection.n_causal_prompts_per_role = 4`
- `causal_prompt_selection.project_prompt_indices = [0,1,2,3]`
- `causal_prompt_selection.evidence_class = PROJECT_DECISION`
- `causal_prompt_selection.runtime_override_forbidden = true`
- `causal_role_selection.literal_assistant_role_id = assistant`
- `causal_role_selection.P50_size = 50`
- `causal_role_selection.P50_manifest_sha256 = null` until T22 freezes the P50 manifest
- the rule that no causal-units manifest may be marked `FROZEN` before the P50 manifest hash is populated and verified

## Outcome-inspection status

No causal generation had been run when these T23 fields were chosen. P50 and the nonzero steering coefficient were not frozen, and no causal outputs had been produced or judged. Therefore these fields could not have been selected using the causal outcomes.

## Affected artifacts

- `configs/method_frozen_v4.yaml`
- `design/causal_questions_frozen.json`
- `tools/build_t23_causal_question_manifest.py`
- `tests/test_t23_causal_questions.py`
- `docs/T23_CAUSAL_QUESTION_MAP.md`
- `docs/METHOD_FREEZE_V4_13_AUG.md`

Historical generation artifacts remain unchanged.

## Review and approval

The prior in-file `reviewer_approved_by: [Author B]` stamp is not treated as independent reviewer approval. This amendment deliberately records approval as pending until `[Reviewer]`, the reviewer who opened issue #31, reviews this fix PR. The GitHub review is the authoritative approval record. If the reviewer requires an in-file approval stamp, it should be added in this same PR after that review rather than being pre-filled by the owner.
