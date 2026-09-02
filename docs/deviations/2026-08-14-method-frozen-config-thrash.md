# INCIDENT: `configs/method_frozen.yaml` mutated in place after freeze

## Date
2026-08-14

## Severity
Blocker for anything that reconciles against `method_frozen.yaml` (T08, T10/C80,
T23). The file is supposed to be byte-frozen because its SHA-256 is stamped into
generated artifacts; it has instead been overwritten repeatedly, so no single
committed value matches all the artifacts that reference it.

## Evidence: five distinct `method_frozen.yaml` states on `main`
Tracing the blob across recent `origin/main` history:

| commit | subject | method_frozen.yaml sha256 (short) |
|---|---|---|
| 855d683 / 3116469 / e23c36a / 71b678c | T04 / T14 / C80-A provenance | `73be73df` |
| 9ff127e | Add files via upload | `febbcbfc` |
| ad69d12 | T23 review fixes | `cf9cf825` |
| 2a409a2 | T31/T32 review fixes | `85605c48` |
| 7caae3d / 7189db3 | Merge PR #21 (T23) / #25 (T31-T32) | `ea2c94a9` |
| 48e8563 | Freeze C80 attempt selection + T02 V4 | `73be73df` (restored) |
| 5173cf5 / 3b5643c / ff1acc7 | Merge PR #29 / Add files via upload | `ea2c94a9` (clobbered again) |

## Which artifacts require which value
- `73be73df` (no causal blocks, 24,618 bytes) is stamped into:
  - `design/questions_C80_A.json`, `design/questions_C80_B.json` (T08 blocks),
  - `results/generation/c80_role_C80_A.manifest.json` and
    `results/generation/c80_role_C80_B.manifest.json`
    (`provenance.method_config_sha256`) — i.e. the **44,000-row C80 generations**,
    which are expensive and already produced.
- `cf9cf825` is stamped into `design/causal_questions_frozen.json`
  (`method_config.sha256`) — the **T23** committed frozen map. T23's builder and
  tests also *require* the `causal_prompt_selection` and
  `causal_role_selection.literal_assistant_role_id` blocks, which exist only in
  the mutated (non-`73be73df`) configs.
- `ea2c94a9` is the value currently committed on `main`. It matches **none** of
  the stamped artifacts, which is why
  `tests/test_t23_causal_questions.py::test_committed_frozen_map_stamps_the_current_input_hashes`
  is already red on `main`.

## Root cause
Multiple tasks (T23, T31/T32) added their own blocks directly into the
supposedly frozen `configs/method_frozen.yaml`, and several "Add files via
upload" commits overwrote it with stale working copies. This violates
`docs/METHOD_FREEZE_V4_13_AUG.md` (Preservation rule: the historical
`method_frozen.yaml` must remain byte-for-byte unchanged because its SHA is
stamped into generated artifacts) and the config's own
`implementation_contract.runtime_overrides_for_frozen_fields_forbidden`.

## Why this PR does not merge `main` and does not rewrite the config
No single value of `method_frozen.yaml` satisfies T08 + T10/C80 + T23
simultaneously, so the fix is a config *split*, not a one-line revert, and it
must re-freeze T23's map. Rewriting the frozen provenance of four tasks on a
guess would be destructive and is an owner decision. This PR therefore:
- keeps `configs/method_frozen.yaml` byte-frozen at the canonical `73be73df`
  (the value stamped into the expensive C80 generations and the T08 blocks) so
  the C80 half it delivers stays self-consistent, and
- adds `tests/test_method_frozen_integrity.py`, which fails loudly if the
  working-tree config no longer matches the SHA stamped into the committed
  C80/T08 artifacts — the guard that would have caught every clobber above.

## Required remediation (T02 owner sign-off)
1. Restore `configs/method_frozen.yaml` on `main` to the canonical `73be73df`
   (protects the 44,000-row C80 generations and the T08 blocks, which cannot be
   cheaply regenerated).
2. Move the causal blocks (`causal_prompt_selection`, `causal_role_selection.*`)
   out of `method_frozen.yaml` into a task-scoped / versioned config
   (e.g. `configs/method_frozen_v4.yaml`, per the V4 governance), and repoint the
   T23 builder/tests `--config` default to it.
3. Regenerate `design/causal_questions_frozen.json` against that config and
   re-stamp; confirm
   `test_committed_frozen_map_stamps_the_current_input_hashes` passes.
4. Do the same audit for T31/T32's `85605c48` additions.
5. Keep the new integrity guard so `method_frozen.yaml` can never again drift
   from the artifacts that stamp it.

## Claim boundary
This record diagnoses the corruption and protects the C80 stamp. It does not
rewrite any task's frozen artifacts; steps 1-4 are the owner's to execute.
