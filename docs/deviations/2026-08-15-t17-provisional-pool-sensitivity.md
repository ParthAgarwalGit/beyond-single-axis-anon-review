# Historical note: superseded for primary C80 membership

This 2026-08-15 record is preserved as provenance for the earlier provisional T17 path. Its statement that score-3 membership would later become the frozen primary rule was superseded by the 2026-08-17 confirmatory-membership decision after T07 failed its validation gates. For the current primary C80 T17 branch, `all_valid` label-independent technical-validity membership is primary, an explicit frozen T14 retained-role manifest is required, and score-3 remains an unvalidated sensitivity only. No text below should be read as overriding the later change-control record.

---

# T17 pool-sensitivity run provisionally, before the role filter freezes

## Date
2026-08-15

## What this records
The all-response vs answer-only vs reasoning-only pool geometry (T17; figure
`F5_all_response_vs_answer_only`) is being computed **now**, on all
technically-valid outputs, rather than waiting for the frozen role filter that
T06 (human adjudication) and T07 (validated role judge) will produce. This lets
the reasoning experiment use the already-extracted T12 activation pools instead
of sitting idle behind the human/judge labels.

It is a deliberate, declared deviation from the frozen construction, limited to
**two** filters, which are separated in the code (review of PR #34 correctly
noted that the frozen retained-role set is not just an output predicate).

## The two substitutions
The frozen role vector averages **valid score-3 outputs**
(`role_vectors_and_axis.primary_role_output_class`) over the **frozen retained
role set** (`split_half_eligibility`: ≥10 valid score-3 outputs in *each* of
C80-A and C80-B, intersected). Both are T06/T07 output and are not frozen yet.
The provisional run substitutes:

1. **output membership** — provisional `validity == "valid"` in place of frozen
   `valid AND score-3`;
2. **role eligibility** — provisional *all roles present* in place of the frozen
   retained-role manifest.

The reducer takes these as separate inputs (`--membership`, `--retained-roles`),
so the frozen result is produced by re-running `reduce --membership score3
--retained-roles <frozen manifest>` on the same T12 pools; the analysis stage is
byte-for-byte the same code. `--membership score3` alone is NOT the frozen set.

Everything else is unchanged and enforced in code:

- default vector: technical-validity-only, 5 conditions × 0.2, with the **95%
  per-condition valid floor enforced** (`default_mean_enforced`, using the
  per-condition valid/total counts the reducer preserves);
- Axis formula `v = mu_default − mean_r(mu_role_r)`, unit normalisation, strict
  sign check;
- middle block index 16 and primary arm `USER_TRANSLATED_LU` are **read from and
  asserted against the active frozen config**, not hardcoded; equal role
  weighting.

## Matched cohorts (a correctness property, not a deviation)
A pool projection difference must reflect removing a token span, not changing
which outputs contribute. So a role's all-response and answer means in F5 are
averaged over the **same outputs** (the cohort with both pools), and the
three-way comparison uses the intersection of outputs with all three pools. The
available-case means (each pool over every output that has it) are reported only
as a labelled sensitivity. Pool availability is taken from T12's authoritative
`has_*_pool` metadata; a tensor that contradicts the flag (e.g. a stale answer
tensor after the #35 segmentation fix) fails the reduction closed. **Do not run
or commit the real provisional numbers until the #35 T12 segmentation fix is
incorporated and the affected T12 metadata is regenerated.**

## Why this is valid to report now
The reasoning question — does the persona signal that separates default-Assistant
from role-adopted outputs on the Assistant Axis survive when the reasoning trace
is dropped, and is it present in the trace itself — is well-defined over all
role-conditioned outputs, independent of how in-role the judge later rules each
one. The provisional numbers are **exploratory / sensitivity evidence** and re-run
on the retained set once it exists.

## Handling rules
1. Every T17 artifact is stamped `provisional: true` with the membership rule
   recorded in `reduced_manifest.json` and `pool_sensitivity_report.json`.
2. Provisional outputs live under `results/t17/`. They do **not** populate the
   hash-gated `design/figures/sources/F5_all_response_vs_answer_only.csv`, which
   stays `AWAITING_RESULTS` until the confirmatory geometry is frozen (same
   freeze discipline as T31/T32).
3. `--membership score3` uses the **unvalidated automatic** judge score and is
   itself still provisional; it is a sensitivity, not the frozen filter.
4. The paper reports this as a provisional/sensitivity result and does not state
   the frozen retained-role version until T06/T07 land.

## Provenance chain
- Inputs: T12 pooled activations (`t12-activation-recompute-v3`), whose per-file
  SHA-256s are recorded in `results/t12/activation_manifest_v3.json`. The pooled
  tensors themselves live off-repo (Drive/Colab), so `reduce` must run where they
  are mounted.
- Code: `src/pool_sensitivity.py`, `tools/run_t17_pool_sensitivity.py`
  (`t17-pool-sensitivity-v1`).
- Validation: `tests/test_t17_pool_sensitivity.py` and the tool's `self-test`
  subcommand (planted geometry; recovers high all-vs-answer, low all-vs-reasoning
  correlation) prove the analysis with no tensors, labels, or GPU.

## Claim boundary
This supports a provisional statement about answer-only / reasoning-only geometry
in the reasoning-distilled model. It is not the frozen confirmatory result and
must not be described as one until the role filter is frozen and the analysis is
re-run on the retained set.
