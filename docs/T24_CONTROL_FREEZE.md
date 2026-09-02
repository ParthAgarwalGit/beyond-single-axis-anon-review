# T24 source-model control freeze

**Status:** protocol and operators frozen; real numerical calibration artifact still required before T29 production.

T24 no longer reopens the source Assistant-Axis condition. T21 fixed the Qwen3-32B source-comparable setting at `layers_46:54-p0.25`, BF16, no quantization, thinking disabled. The observed T21 evaluation is therefore treated as fixed evidence, not as a development signal for choosing control parameters.

## What T24 adds

The source-model specificity experiment uses four matched conditions: unsteered, the fixed T21 Assistant-Axis cap, an orthogonal random-direction cap, and a spherical distance-to-default cap.

For each layer 46 through 53, the random direction is drawn deterministically from a frozen seed, its component along that layer's released Assistant direction is removed, and it is unit-normalized. Its cap threshold remains the 25th percentile of **its own** natural projection distribution. Reusing the Assistant Axis's raw scalar threshold is forbidden because the projection scales can differ.

The spherical control is project-defined. Its center is the frozen default-Assistant centroid at each layer. Its radius remains the 75th percentile of the natural distance-to-that-centroid distribution. Activations inside the radius are unchanged; activations outside it are projected radially to the boundary.

The source treatment's engagement is no longer assumed to be 25% on our calibration corpus. The numerical-freeze runner loads the released `capping_config.pt`, extracts the actual per-layer source thresholds, applies the source operator as a **measurement only** to the same `natural_L` rows, and records the observed Assistant-Axis cap engagement rate. The random and sphere controls must each remain within the frozen absolute tolerance of 0.03 from that measured source engagement before T24 can reach `FROZEN_NUMERICAL_CONTROLS`.

## Source integrity checks added after review

The numerical freeze now fails closed unless both released source artifacts are supplied and match the T21 pins:

- `qwen-3-32b/assistant_axis.pt`, SHA-256 `a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211`;
- `qwen-3-32b/capping_config.pt`, SHA-256 `6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24`.

For every capped layer, `axis_L` must numerically match the released Assistant Axis with the same orientation. The released capping-config vector must also be collinear with and negatively oriented to the released Assistant Axis, matching the T21 orientation audit and the source implementation's upper-tail capping convention. This closes the previous gap where a mismatched calibration Axis could have been frozen silently.

## Frozen natural-calibration corpus

The common natural-calibration corpus is explicitly pinned to the T21 public safety manifest:

- source ID: `T21_JBB_BEHAVIOR_ONLY_ADAPTED_NOT_LU_PERSONA_JAILBREAK`;
- manifest SHA-256: `d611e904f3b5d47c71e5ab1f3fa1a84ead6cfd5d94dba337f5c3b71aafef7793`;
- 100 frozen items;
- condition: unsteered Qwen3-32B under the exact T21 serialization/forward path;
- no harmfulness, refusal, or other outcome label is allowed to enter calibration or row selection.

The calibration NPZ must carry the source ID and manifest SHA as scalar metadata, and the runner checks both before any numerical parameter is emitted. The same activation-row construction must be used for source treatment engagement, random control calibration, and sphere control calibration.

## Required calibration artifact

`tools/run_t24_source_control_freeze.py` deliberately cannot invent numerical thresholds, centroids, or radii. It requires a local NPZ with, for every layer `L` in 46..53:

- `axis_L`: released/source Assistant direction;
- `natural_L`: activations from the frozen common natural calibration corpus/serialization;
- `default_L`: frozen default-Assistant reference activations from the same protocol.

It also requires the two released `.pt` artifacts above. Example invocation:

```bash
python tools/run_t24_source_control_freeze.py \
  --calibration-npz /path/to/t24_qwen_calibration.npz \
  --assistant-axis-pt /path/to/qwen-3-32b/assistant_axis.pt \
  --capping-config-pt /path/to/qwen-3-32b/capping_config.pt
```

The command writes `results/t24/source_control_parameters.npz` and `results/t24/source_control_freeze.json`. The JSON is stamped with `source_git_sha`, the calibration-input hash, both released-artifact hashes, the corpus identity, the measured source engagement at every layer, the project-control engagement rates, and all validation checks.

## Matching constraints

T29 must keep model/revision, evaluation rows, layer range, token positions, dtype, decoding, scoring, and the calibration protocol matched across the four conditions. Random and sphere parameters may not be changed after their evaluation outcomes are inspected.

## What is and is not complete

The **implementation and pre-outcome freeze protocol are complete in this PR**. The repository does not currently contain the large real calibration activations needed to produce the per-layer numerical control artifact, so this PR does not fabricate those values. T24 reaches `FROZEN_NUMERICAL_CONTROLS` only after that calibration command is run on the real artifact and its two outputs are committed and reviewed.

The random and spherical caps are new project controls, not methods attributed to Lu et al. T24 itself supports a Methods claim only; causal-specificity conclusions require T29 results.