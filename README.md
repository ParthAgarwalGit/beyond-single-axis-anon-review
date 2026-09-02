# Stable Role Ordering Despite Different Directions: Testing the Assistant Axis in a Reasoning Model

**Anonymous review artifact for the paper _Stable Role Ordering Despite Different Directions: Testing the Assistant Axis in a Reasoning Model_.**

This repository contains the code, frozen configurations, aggregate results, human-annotation artifacts, and provenance records used for the submission. The reviewer-facing artifact is intentionally smaller than the internal research repository: author-identifying material, internal planning files, superseded experiments, private condition keys, selected raw sensitive outputs, and third-party material without clear redistribution permission are omitted.

## What is in this repository

The reviewer-facing source of truth is organized as follows:

| Path | Purpose |
| --- | --- |
| `src/` | Canonical analysis and intervention code used by the final pipeline |
| `configs/` | Frozen experiment and evaluation configurations |
| `results/` | Canonical result artifacts used by the paper |
| `design/` | Frozen question blocks, analysis inputs, and figure/table source data |
| `tools/` | Reproduction and verification scripts |
| `annotations/` | Human annotation artifacts that can be shared safely |
| `docs/` | Reproducibility, source-comparison, and artifact-status documentation |
| `LICENSE` | License for original code and documentation |
| `THIRD_PARTY_NOTICES.md` | Rights and redistribution notes for third-party material |

The authoritative scientific evidence is under `results/` and `design/`. Historical or superseded replication work is not part of the reviewer-facing evidence chain.

## Main evidence map

The table below maps the paper's main questions to the canonical in-repository artifacts.

| Question | Canonical artifacts | Submission-level result |
| --- | --- | --- |
| **Is the automatic role-expression measurement reliable enough to define confirmatory membership?** | `results/t06/`, `results/t07/judge_validation.json`, `results/t07/judge_validation_with_uncertainty.json` | No. The production role judge misses all four pre-specified validation criteria. Confirmatory DeepSeek geometry therefore uses label-independent technical validity rather than automatic role labels. |
| **Does role ordering reproduce across independent question sets?** | `results/t15/confirmatory_reliability.json`, `results/t15/t15_t16_resplit_results.json` | Yes. Cross-built same-role Pearson r = 0.976528, 95% CI [0.971010, 0.981079], across 275 roles. In 300 rebuilt balanced question partitions, the observed split is at the 84th percentile of the sampled distribution (median r = 0.972914). |
| **Do reasoning and final-answer representations recover the same structure?** | `results/t17/c160/`, `results/t17/regional_independent_recovery/` | The fitted directions differ more than the role ordering. Reasoning-vs-final-answer axis cosine is 0.587784, while the same-role score correlation is r = 0.954637. |
| **Is one assistant-axis coordinate an adequate predictive summary?** | `results/t18/t18_report.json`, `results/t18/README.md` | No, under the frozen predictive test. The one-dimensional axis reaches AUROC 0.678327; a regularized full-dimensional linear readout reaches 0.878197, a difference of +0.199869 with simultaneous 95% CI [0.180913, 0.220654]. The frozen verdict is `INADEQUATE`. |
| **Does source-model assistant-axis capping show robust direction-specific harmfulness reduction?** | `results/t21/`, `results/t24/`, `results/t27/`, `results/t29/` | Not established. The design-weighted human-primary Assistant-minus-random strict-harm risk difference is -3.69 percentage points with 95% CI [-11.08, 0.00]; the secondary automatic 400-row sensitivity has the opposite sign (+1.0 percentage point). |

These results support a narrow conclusion: **prompt-conditioned role ordering is highly reproducible, but the fitted direction is more conditional than the ordering, the role-expression judge is measurement-limited, one axis does not exhaust held-out linearly decodable information, and the matched causal experiment does not establish robust direction-specific harmfulness reduction.**

## Quick start

A portable environment is specified in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

pytest -q
```

`requirements.txt` provides the portable project environment anchored to the DeepSeek generation/activation stack. Exact task-specific software and hardware versions are recorded in the corresponding provenance artifacts; use those records for exact reproduction.

Most statistical analyses and tests can run on CPU. Model generation, activation extraction, and intervention runs require suitable model access and a CUDA-capable GPU.

## Reproducing the main paper analyses

Each canonical result directory contains task-specific provenance and, where applicable, an exact reproduction command. The main entry points are:

| Analysis | Entry point | Result directory |
| --- | --- | --- |
| Role-judge validation | `tools/run_t07_judge_validation.py` | `results/t07/` |
| Judge uncertainty intervals | `tools/run_t07_judge_uncertainty.py` | `results/t07/` |
| Confirmatory C80 recovery | `tools/run_t15_confirmatory.py` | `results/t15/` |
| Rebuilt 300 question partitions | `tools/run_t15_t16_resplits.py` | `results/t15/` |
| Response-region geometry | `tools/run_t17_c160_geometry.py` | `results/t17/` |
| Regional independent recovery | `tools/run_t17_regional_independent_recovery.py` | `results/t17/` |
| Predictive adequacy | `tools/run_t18_dimensional_adequacy.py` | `results/t18/` |
| Qwen matched-control human-primary analysis | `tools/run_t29_human_primary.py` | `results/t29/` |
| Qwen automatic sensitivity | `tools/run_t29_automatic_sensitivity.py` | `results/t29/` |

For exact inputs, hashes, frozen options, and command-line arguments, use the README or machine-readable provenance record inside the corresponding `results/<task>/` directory.

## Important interpretation boundaries

### DeepSeek geometry is prompt-conditioned

The production role-expression judge fails its pre-specified validation criteria. The confirmatory DeepSeek population is therefore defined by technical validity, not by automatically verified full role enactment. The paper treats the recovered structure as **prompt-conditioned role geometry**.

### High recovery does not identify one uniquely privileged global direction

Independent C80-A/C80-B axes recover nearly the same role ordering, but the observed recovery is compatible with the frozen common-orientation reference. The paper therefore does not claim that the experiment isolates a unique global assistant direction.

### Predictive adequacy is not intrinsic dimensionality

T18 asks whether a single axis score retains the predictive information in the full response activation when both roles and questions are held out. Its `INADEQUATE` verdict does not identify a minimum number of dimensions, a unique higher-dimensional basis, nonlinear necessity, or a causal mechanism.

### Qwen matched controls do not establish a robust safety effect

The harmfulness automatic scorer fails its absolute validation gate, so the design-weighted human subset is primary and the 400-row automatic panel is secondary. The human-primary interval reaches zero and the secondary automatic analysis reverses the sign of the Assistant-vs-random contrast.

## External and omitted artifacts

Some material is intentionally not redistributed in this review artifact.

### Lu et al. source material

The study uses public source material from the assistant-axis reference implementation. Four project-created JSON containers in the internal repository embed verbatim upstream text whose repository-level redistribution rights are not explicit. They are therefore omitted from this reviewer/public-release artifact.

`tools/fetch_lu_et_al_source.py` pins the upstream public commit and reconstructs those containers from source. See `THIRD_PARTY_NOTICES.md` for the exact licensing boundary.

### Private or sensitive artifacts

Private condition keys, item-to-condition mappings used for blinded causal validation, and selected raw sensitive generations are not included. Aggregate results, validation summaries, hashes, and provenance required to evaluate the paper's claims are included where redistribution is appropriate.

### Large external artifacts

Some generation or activation artifacts are too large or unsuitable for direct inclusion. Canonical in-repository reports bind the analyzed inputs by immutable revision and/or cryptographic hash. The task-specific result documentation records these dependencies.

## Figure and table provenance

Paper figure/table source data are stored under:

```text
design/figures/sources/
design/tables/sources/
```

The corresponding generation/verification tooling is under `tools/`. Final figures should be regenerated from these frozen sources rather than from notebook state.

## Repository authority

When files appear to disagree, use the following order:

1. the canonical machine-readable artifact under `results/<task>/`;
2. the task README/provenance record in the same result directory;
3. `docs/ARTIFACT_STATUS.md`;
4. older methodological or historical documentation.

Superseded exploratory material is not paper evidence.

## License

Original code, tests, tools, and documentation in this repository are released under the MIT License. Third-party material retains its original rights and is documented separately in `THIRD_PARTY_NOTICES.md`.

## Double-blind review

Author names, affiliations, acknowledgements, identifying repository metadata, and citation metadata for this repository are intentionally omitted during double-blind review. A citation file and complete author metadata can be added after the review period.
