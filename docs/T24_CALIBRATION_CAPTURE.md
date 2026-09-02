# T24 real Qwen calibration capture

**Status:** NUMERICAL FREEZE COMPLETE; INDEPENDENT REVIEW PENDING. The genuine Qwen3-32B activation capture completed 100/100 frozen T21 items, the approved pre-outcome engagement-matching correction was applied, and the recovered numerical controls passed the focused T24 validation checks. The production freeze artifacts are committed in PR #53. T29 remains blocked until independent review of this freeze.

## Scope

PR #48 froze the source Axis condition, random-direction construction, spherical control, source hashes, layer set, engagement tolerance, and the rule that no matched-control outcome may be inspected before the numerical parameters are frozen. PR #53 implements the genuine Qwen3-32B calibration capture and numerical freeze.

This capture uses only the previously frozen 100-item T21 behavior-only manifest. It never loads harmfulness, refusal, identity, or other outcome labels.

## Pinned inputs

- model: `Qwen/Qwen3-32B`
- model revision: `9216db5781bf21249d130ec9da846c4624c16137`
- dtype: BF16
- quantization/offload: forbidden
- thinking: disabled with the Qwen chat template
- frozen T21 manifest SHA-256: `d611e904f3b5d47c71e5ab1f3fa1a84ead6cfd5d94dba337f5c3b71aafef7793`
- released Assistant Axis SHA-256: `a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211`
- released capping config SHA-256: `6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24`
- layers: 46 through 53 inclusive

## Capture convention

The hook is registered on `model.model.layers[L]`, matching the block-output residual site used by the source capping implementation.

For each frozen manifest item, the prompt is serialized with the model's chat template using `enable_thinking=False`. The calibration replay is deterministic and outcome-blind: `do_sample=False`, `max_new_tokens=256`, `use_cache=True`. These decoding values are used only to expose a stable natural activation distribution for pre-outcome calibration. They are not claimed to reproduce a source-paper decoding schedule.

At each layer:

- `natural_L` contains every block-output row seen by the hook during the unsteered Hugging Face `generate` call, including the prefill positions and later decode-step calls;
- `default_L` contains one row per frozen manifest item, obtained by averaging all hook-visible rows for that item before pooling across items;
- `axis_L` is copied directly from the pinned released Assistant Axis artifact.

The one-row-per-item `default_L` convention prevents a long item from receiving greater weight in the spherical-control centre merely because it contributed more hook positions. It is a **project-defined calibration choice**, not a Lu et al. method. The source Axis treatment itself is not retuned.

## 2026-08-21 engagement-matching deviation

The first genuine capture completed all 100 items at capture source commit `736b94c8b1929a1b2e54cf1c1ae535c1d022bbec`. The downstream numerical-freeze gate then measured layer-46 source-cap engagement at `0.5959` while the original q25 random control engaged `0.2500`, exceeding the frozen `0.03` tolerance.

Because this happened before any T29 matched-control outcomes were generated or inspected, the approved primary correction is outcome-blind:

- source Assistant-Axis cap stays exactly fixed;
- random direction and seed stay fixed;
- random lower-tail quantile becomes the measured source engagement rate for that layer;
- sphere centre stays fixed;
- sphere radius quantile becomes `1 - source engagement rate` for that layer;
- the same `0.03` engagement-match gate remains fail-closed;
- the original q25 random and q75 sphere parameters are retained as unmatched-engagement sensitivity/reference values.

This matches **intervention incidence**, not intervention magnitude or geometry.

## Completed production freeze

The completed 100/100 capture was reused without rerunning Qwen. Recovery produced `FROZEN_NUMERICAL_CONTROLS` with separate capture and freeze provenance:

- capture source Git SHA: `736b94c8b1929a1b2e54cf1c1ae535c1d022bbec`
- freeze source Git SHA: `a880b4102de910ef08b57bd43fd00ca87b3de48e`
- calibration NPZ SHA-256: `bb0670832e065f52a7c46dc99acf467960523f4cca888cca996347e3290d19c1`
- numerical controls NPZ SHA-256: `adf1c43cd2715676169351f69fced45d0046e70c5a74f52baa2dc5f2aa243291`

Across layers 46–53, primary random and sphere engagement exactly match the measured source-cap engagement on the finite calibration rows, while Axis/random absolute cosine remains far below the frozen `1e-6` tolerance. The focused T24 test suite passed after synchronizing PR #53 with current `main`.

## Output, recovery, and provenance

`tools/gpu_cells/t24_capture_qwen_calibration.py` writes resumable per-item NPZ shards to external persistent storage and assembles `t24_qwen_calibration.npz`.

If the expensive capture completed but the old gate aborted, `tools/recover_t24_numerical_freeze.py` reuses that existing calibration NPZ without loading Qwen or regenerating any item. It invokes the current numerical-freeze runner and records separate capture-source and freeze-source Git SHAs.

The final outputs are:

- external `t24_qwen_calibration.npz`;
- external `source_control_parameters.npz`;
- in-repo `results/t24/source_control_freeze.json`;
- in-repo `results/t24/t24_calibration_capture_receipt.json`.

Large NPZs remain on Drive/Hugging Face and are bound into the small review artifacts by SHA-256 rather than forced into Git.

## Required gate before T29

Before T29 production, independently verify:

1. capture and recovery/freeze Git SHAs are recorded separately when recovery is used;
2. source model, revision, manifest, Axis, and capping-config hashes match the pins;
3. all eight layers are present and finite;
4. `run_t24_source_control_freeze.py` reaches `FROZEN_NUMERICAL_CONTROLS`;
5. the freeze JSON records deviation ID `T24_ENGAGEMENT_MATCHING_2026-08-21` and `source_treatment_changed: false`;
6. per-layer Axis/random absolute cosine is within `1e-6`;
7. primary random and sphere engagement are each within `0.03` of measured source Axis-cap engagement;
8. original q25/q75 parameters remain present only as unmatched-engagement sensitivity/reference values;
9. external NPZ SHA-256 values match the receipt/freeze JSON.

Only after this independent review should T29 production start.


## Durable artifact locations

The exact independently reviewed numerical controls are committed at
`results/t24/source_control_parameters.npz` with SHA-256
`adf1c43cd2715676169351f69fced45d0046e70c5a74f52baa2dc5f2aa243291`.

The frozen calibration input is archived at the immutable Hugging Face dataset
revision:

- dataset: `[Author-B-HF]/[anonymized-repository-name]-artifacts`
- revision: `1bdf7ce4f5e71e487c8e13735f85490afc1cee28`
- path: `t24-calibration/t24_qwen_calibration.npz`
- SHA-256: `bb0670832e065f52a7c46dc99acf467960523f4cca888cca996347e3290d19c1`

A clean-tree CPU recomputation at `a880b4102de910ef08b57bd43fd00ca87b3de48e` produced SHA-256
`bfa90a1a7377cd32e48f55d4ade4b0c168af2396b0bb878d4fb492d422b6ef41` before canonical reconciliation. It reproduced
44 of 56 arrays byte-for-byte. The only non-byte-identical values were 12
scalar float64 random-projection thresholds, with maximum absolute difference
`5.3290705182007514e-15`.

Every non-threshold array was exact, every differing threshold had the same
float32 representation, and the fresh and canonical thresholds produced
identical engagement masks on the frozen calibration rows. The differences are
therefore recorded as cross-runtime final-ulp numerical variation rather than a
change in the intervention.

The exact previously reviewed `adf1c43cd2715676169351f69fced45d0046e70c5a74f52baa2dc5f2aa243291` artifact
remains the canonical production control object.
