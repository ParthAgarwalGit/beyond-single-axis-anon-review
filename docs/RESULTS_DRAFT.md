# Does the Assistant Axis Generalize? — Results (draft)

**Primary representation model:** DeepSeek-R1-Distill-Llama-8B (BF16).  
**Source-capping model:** Qwen3-32B (BF16, thinking disabled).  
**Status:** T06/T07 measurement decision frozen; T15/T16 confirmatory reconstruction complete; T17 C160 production analysis complete; T21 source-model capping baseline complete but evaluation-limited; T24 matched Qwen numerical controls and the four-condition T29 generation/scoring population are complete; T26/T27 human validation is complete; T29 causal-specificity inference is complete under the `MEASUREMENT_LIMITED` harmfulness branch; DeepSeek T28 remains separate.

This is a **source-anchored, behaviour-first transfer and construct-validity audit, not an exact replication** of Lu et al. Every numeric claim below is tied to a named committed artifact or merged PR. Pending causal conclusions remain explicitly withheld.

---

## 1. Measurement validity limits the role-enactment claim (T06 → T07)

Two humans labelled a stratified 300-item sample on the frozen 0–3 role-expression rubric, followed by blinded resolution of disagreements. The final human consensus resolves all 300 sampled items, with label distribution {0:14, 1:127, 2:61, 3:98}; 12 items are marked unjudgeable (`results/t06/gold_consensus.csv`, PR #37).

The production automatic role judge does **not** clear the frozen validation gate (`results/t07/judge_validation.json`, PR #37):

| metric | observed | gate |
|---|---:|---:|
| score-3 precision | 0.556 | ≥ 0.85 |
| score-3 recall (ip-weighted) | 0.798 | ≥ 0.85 |
| macro-F1 (ip-weighted, four-class) | 0.409 | ≥ 0.75 |
| four-class κ | 0.233 | ≥ 0.70 |

The judge scored 278 items and abstained on 10; 12 were human-unjudgeable. Per-arm four-class κ is 0.219 for extraction and 0.246 for translated outputs. In the paired answer-only versus full-response sensitivity, 16 of 60 items cross the 2↔3 boundary. The resulting measurement branch is `measurement_limited_result`.

A more permissive score ≥2 boundary also does not rescue the gate (`results/t07/boundary_diagnostic.json`). Under this **binary exploratory collapse**, precision is 0.781 and ip-weighted recall is 0.726, so **0 of the 2 comparable frozen thresholds are met**. The reported ip-weighted binary macro-F1 (0.782) and binary κ (0.570) are explicitly **not comparable** to the frozen four-class macro-F1 and κ thresholds. The archived judge scores come from the judge that failed validation and have zero UID overlap with C80; they are therefore not described as validated and are not used to select C80 outputs.

**Consequence for the confirmatory representation claim.** Before the C80 confirmatory statistic was computed, PR #44 recorded a label-independent primary membership rule: technically valid outputs with the required activation pool present. This yields 275/275 retained roles. The result therefore concerns **prompt-conditioned role representations**, not behaviorally verified successful role enactment.

---

## 2. Independent C80 blocks recover nearly the same role ordering (T15/T16)

We reconstruct the Assistant-related direction independently on C80-A and C80-B using block 16, the all-response pool, `USER_TRANSLATED_LU`, independent per-block default means, and the label-independent technical-validity membership fixed above (`results/t15/confirmatory_reliability.json`, merged PR #41).

C80-A contains **21,998** technically eligible role outputs and C80-B contains **22,000**, with **275/275 roles** retained in both blocks. The primary cross-axis same-role Pearson correlation is

**r = 0.976528, 95% bootstrap CI [0.971010, 0.981079]**

from 2,000 role-bootstrap replicates. Secondary statistics are similarly high: Spearman ρ = **0.975928**, independent-Axis cosine = **0.912713**, same-axis projection Pearson r = **0.975918**, and leave-one-role-out r ranges from **0.975671 to 0.977177**.

The result separates strongly from two null references. Under role-correspondence permutation, **0 of 5,000 null draws reached the observed statistic** (null mean -0.000471; 95th percentile 0.099511). Under the isotropic random-direction reference, **0 of 5,000 null draws reached the observed statistic** (null mean 0.003313; 95th percentile 0.356251). These are reported as zero exceedances, not literal `p = 0`.

The stronger common-orientation sign-flip reference is different. Its null mean is 0.964840 and 95th percentile is 0.984195; the observed statistic is **not separated** from this reference (`p_ge_observed = 0.3784`). The confirmatory result therefore supports highly reproducible prompt-conditioned role ordering across untouched question blocks, but it does **not** establish that the particular global Assistant orientation is uniquely privileged or that a single direction is sufficient.

The reported bootstrap and leave-one-role-out uncertainty resample the **already computed cross-projections**; they do not rebuild both Axes within each replicate. The predeclared **300 repeated frozen-question resplits** remain an outstanding sensitivity and are not represented as completed.

A predeclared E80 membership bridge further shows that membership choice matters: `cos(v_score3, v_all_valid) = 0.785438`, and the corresponding role-projection Pearson correlation is **0.698940** over 253 common roles. This bridge inherits E80's recorded provenance caveat: E80 is `ACCEPTED_WITH_RECORDED_GAP` because its generation-time model revision was not directly recorded. Score-3 and all-technically-valid E80 constructions should therefore not be treated as interchangeable or as confirmatory C80 evidence.

---

## 3. Role space is multidimensional, while role ordering is stable across response regions (T17)

The merged T17 production run forms C160 by exact count-weighted union of the C80-A/C80-B role means, uses centered unstandardized role-space PCA, and keeps the Assistant Axis fixed rather than redefining it from PCA (`results/t17/c160/t17_c160_role_space_report.json`, merged PR #55).

| region | PC1 variance explained | PCs to 70% | \|Axis·PC1\| |
|---|---:|---:|---:|
| all-response (primary) | **0.1932** | **18** | **0.6952** |
| answer-only (sensitivity) | 0.2405 | 14 | 0.6007 |
| reasoning (sensitivity) | 0.2001 | 19 | 0.7637 |

Thus the primary all-response PC1 explains only **19.3%** of role-space variance, and **18 components** are needed to reach 70%. The Assistant Axis is substantially but not fully aligned with PC1. This argues against treating the role space as effectively one-dimensional.

Cross-region comparisons show a complementary pattern:

| pair | Axis cosine | same-role projection r |
|---|---:|---:|
| all-response ↔ answer | 0.7655 | 0.9575 |
| all-response ↔ reasoning | 0.9624 | 0.9957 |
| answer ↔ reasoning | **0.5878** | **0.9546** |

Answer and reasoning therefore yield materially different Axis directions, while same-role ordering remains highly correlated. The all-response ↔ reasoning agreement is partly mechanical because reasoning tokens dominate the all-response pool by token mass; the answer ↔ reasoning comparison is the more informative region contrast. Reasoning-vs-answer results are **block-16 only** because the stored reasoning representation is middle-layer only; there is no reasoning-region layer sweep.

PCA remains descriptive. These region differences support response-region dependence in the representation, but they do not show that reasoning training caused the difference.

---

## 4. The public Qwen source-capping intervention is reproducible, but the adapted safety evaluation shows no strict-harm reduction (T21)

T21 reproduces the released public capping intervention on `Qwen/Qwen3-32B` at revision `9216db5781bf21249d130ec9da846c4624c16137`, using BF16, no quantization, thinking disabled, and the released `layers_46:54-p0.25` setting across layers 46–53 (`results/t21/t21_final_report.json`, merged PR #45).

The public safety substitute is explicitly **adapted rather than an exact replay of Lu et al.'s persona-jailbreak evaluation**. On 100 frozen behavior-only items per condition, strict harmful compliance is **4/100 (4%) unsteered and 4/100 (4%) capped**. The paired reduction is **0.00**, with bootstrap 95% CI **[-0.04, 0.04]**. Refusal rises from **88% to 91%**. With only a 4% baseline harmful rate, the strict-harm endpoint is close to a floor, so this adapted set has limited room to demonstrate a reduction; the reported CI still allows modest movement in either direction.

The full public capability execution completes **6,224/6,224 generations** (`docs/T21_SOURCE_CAPPING_EXECUTION.md`; summary values in `results/t21/t21_full_capability_report.json`):

| benchmark | unsteered | source cap |
|---|---:|---:|
| IFEval strict prompt accuracy | 0.837338 | 0.835490 |
| MMLU-Pro accuracy | 0.707857 | 0.697857 |
| GSM8K accuracy | 0.960000 | 0.957000 |
| EQ-Bench | 82.30 | 82.04 |

The sum of relative performance reductions is **2.2619%**, but this cross-benchmark sum is descriptive rather than an uncertainty-qualified effect size. The committed aggregate-only uncertainty record gives conservative capped-minus-unsteered 95% intervals of **[-4.60, +4.23] pp** for IFEval, **[-4.38, +2.38] pp** for MMLU-Pro, and **[-2.08, +1.48] pp** for GSM8K. These are not paired intervals because the committed summaries do not contain discordance counts; no corresponding binomial interval applies to continuous EQ-Bench.

Because the exact source MMLU-Pro/GSM8K sampled membership and seeds are unavailable, this remains a Lu-sized public capability battery, not an exact item-for-item replay. T21 establishes a source-matched public intervention implementation and a bounded public evaluation result; it does not by itself establish causal specificity.

---

## 5. Human-primary T29 is directional but does not establish robust causal specificity (T24/T26/T27/T29)

T24 freezes the project-defined orthogonal-random and spherical-distance controls before matched-control outcomes are inspected. The four-condition Qwen population contains **400 rows, 100 per condition**, with the same 100 underlying evaluation items represented under `unsteered_baseline`, `assistant_cap`, `random_direction_cap`, and `sphere_proximity_cap`. The immutable population has SHA-256 `24a0c5a3699fd6b51a66d4be92bc054d14cb800db894c0162b65774c5667f06b` at HF revision `e29553fe75a7d89757258e7c6c38723de1960ca9`. Production automatic scoring completed 400/400 rows with zero final coverage failures.

T27 harmfulness is `MEASUREMENT_LIMITED`: the automatic judge passes harmful-vs-rest precision (**1.000**) and four-class κ (**0.717**), but misses the frozen IP-weighted recall (**0.452**) and four-class macro-F1 (**0.592**) criteria. The design-weighted T26 human subset is therefore primary, while the 400-row automatic panel is secondary sensitivity only.

On the human-primary strict outcome (`results/t29/t29_human_primary.json`), Assistant capping has an estimated harmful-compliance rate of **4.0%** and the matched random-direction cap **7.6923%**. The frozen Assistant-minus-random risk difference is **-3.6923 percentage points**, with stratified-bootstrap 95% CI **[-11.0769, 0.0000] pp**. The point estimate is in the preregistered direction, but the interval reaches zero. Human Assistant-minus-sphere and Assistant-minus-unsteered strict-harm differences are both **+1.0 pp**, so the human-primary result does not show a general reduction relative to the other controls. Inclusive Assistant-minus-random is **-2.6923 pp**, 95% CI **[-10.0769, +1.0000] pp**.

The exact 400-row production scorer artifact is retained as a paired automatic sensitivity (`results/t29/t29_automatic_sensitivity.json`). Automatic strict harmful-compliance rates are **3% Assistant, 2% random, 1% sphere, and 2% unsteered**. The paired Assistant-minus-random diagnostic is **+1.0 pp**, bootstrap 95% CI **[0.0, 3.0] pp**, exact two-sided McNemar **p = 1.0**. Assistant-minus-sphere is +2.0 pp, 95% CI [-2.0, 6.0] pp, p = 0.625. The sphere condition has a 2% automatic degeneration rate; the other three conditions have 0% automatic degeneration. These automatic numbers are secondary because the harmfulness instrument failed T27 and they do not override the human-primary estimate.

The two evidence sources therefore do not support a robust direction-specific harmfulness reduction. The human-primary Axis-vs-random estimate is favorable but imprecise, the full-corpus automatic sensitivity points slightly in the opposite direction, and the human-primary comparisons against sphere and unsteered do not show lower strict harm. We report T29 as a completed causal-specificity test with a **not-established** harmfulness-specificity conclusion rather than as an unfinished experiment or a positive causal result.

Identity remains separate. T27 mechanically returns `VALIDATED`, but the positive class contains only two human-positive items and both are adjudicator-dependent residual cases. We do not promote the full-corpus automatic identity result or infer that identity mediates harmfulness.

The DeepSeek additive-steering branch remains separate and should not be collapsed into the Qwen source-capping result. Any T28 claim must use its own held-out-role, signed-dose, and judge-validation authorities.

---

## 6. Limitations and claim boundaries

- **Prompt-conditioned rather than enactment-conditioned geometry.** C80 shows reproducible structure among roles the model was prompted to express; it does not show that every retained output successfully enacted the role.
- **High reliability is not unique-orientation evidence.** The T15/T16 statistic does not exceed the common-orientation sign-flip reference (`p_ge_observed = 0.3784`).
- **The role space is not effectively one-dimensional.** T17 gives PC1 variance explained 0.1932 and 18 components to 70% for the primary all-response role space.
- **Response-region direction and ordering can dissociate.** Answer ↔ reasoning Axis cosine is 0.5878 while same-role projection r is 0.9546; reasoning-region results are block-16 only.
- **Membership affects the estimated Axis.** On E80, score-3 and all-valid constructions have cosine 0.785438 and role-projection r 0.698940.
- **T21 is evaluation-limited.** Its 4%→4% strict-harm result comes from an adapted public substitute rather than the unavailable source persona-jailbreak evaluation.
- **T21 capability deltas are descriptive.** Aggregate uncertainty intervals are broad and not paired; the 2.2619% cross-benchmark sum is not a single inferential effect size.
- **Qwen harmfulness measurement is limited.** The 120-item human subset is primary because the automatic causal judge fails the frozen absolute-validation gate on IP-weighted recall and macro-F1. Full-corpus automatic harmfulness is secondary only.
- **Qwen identity validation is fragile.** The mechanical `VALIDATED` branch rests on two human-positive items, both adjudicator-dependent residual cases; it is not used as a strong full-corpus identity claim.
- **T29 causal specificity is not established.** The human-primary Axis-vs-random estimate is directionally favorable but reaches zero at the upper confidence bound; the paired automatic sensitivity does not corroborate it, and the human-primary sphere/unsteered comparisons do not show a general reduction.
- **Precision and deployment boundaries.** Representation and source-capping runs use BF16 rather than quantized inference. Provider-specific choices not specified by the source study are recorded as our reproducibility choices rather than attributed to Lu et al.

---

## 7. Reproducibility and provenance appendix

### Frozen configurations and immutable inputs

- Historical generation-time V3 config SHA-256: `73be73df4640c2d32bfbc8b6009741c0fadd8c466acb703d1f99df35d5c8cd79`.
- Active downstream V4 config SHA-256 used by the clean T15/T17 analyses: `f7e7dd9df8c7c72d5da08bf8201295dfc91f3208837d8cd0867054c393984e59`.
- E80 accepted HF snapshot: `[Author-A-HF]/persona-artifacts@d918c2a157eba4d6fb91899c6d08f90342949ab8`.
- T15 analysed bundle fetched at immutable HF revision `e1885e0865d1e6d7a39a9c2585b224b30e28b3f1`.
- T12 C80 manifest SHA-256s used by T15: C80-A `021524bf56416d7217baaae07b024d398e3fc5b869561015dff24b44abacb7e8`; C80-B `987571b3518b014296326fef853709df991a4e68c7ede3ac45a9832a9e09b58e`.

### E80/C80 execution boundary

E80 and C80 use different accepted decoding regimes (`results/audit/E80_C80_provenance_closure.json`). E80 uses ancestral sampling (`temperature=0.6`, `top_p=0.95`, `max_new_tokens=2560`, seed 0), while C80 uses greedy decoding (`temperature=0`, `do_sample=false`, seed 0) with a 2048→8192→16384 retry ladder. Therefore **C80-A vs C80-B is the decoding-matched confirmatory comparison**; E80-to-C80 pooling is not decoding-identical and must retain decoding-stratum provenance.

E80 is `ACCEPTED_WITH_RECORDED_GAP`: its generation-time model revision was not directly recorded. The later revision `6a6f4aa4197940add57724a7707d069478df56b1` is post-hoc supported, not proven, for E80. Each E80 role arm also contains **39 unclosed-`<think>` rows**; these are handled by the recorded prefilled-think segmentation deviation and do not silently enter final-answer judging.

C80-A has exactly **2** technical exclusions (`79331716c2cde644`, `885b75abc2900244`), leaving 21,998 downstream-eligible outputs. C80-B has no technical exclusions. Under the frozen attempt-selection rule, the lowest-retry valid attempt is selected; if no valid attempt exists, the highest-retry failed attempt is kept for provenance only and excluded downstream. C80-B contains 448 rollouts with more than one valid attempt; the rule resolves them deterministically, including **443** cases with different valid texts.

### Measurement provenance

The T07 role judge failed validation and is not used to define C80 membership. Its committed validation record is provisional and the historical judge deployment does not provide a recoverable immutable model revision, so no judge revision is invented in this draft. The later causal judge/scorer used for T26/T27/T29 is a separate measurement pipeline with its own production provenance.

For T29, the private scored 400-row artifact is bound by SHA-256 `d7f8feb34f8c2e8e8fff57764c95ff81b8128fbfa35d87c380322dce07565bd3`. Its `question_id` field is null, but the production UID has the frozen form `qwen_capping::<underlying-item>::<condition>`. The T29 automatic-sensitivity runner validates that contract and exact four-condition item membership before pairing. Only aggregate T29 results and figure/table source data are committed.

### Deviation records to audit with the manuscript

Current repository deviation records relevant to the method/provenance chain include:

- `docs/deviations/2026-08-05-t08-question-artifact-hash.md`
- `docs/deviations/2026-08-08-prefilled-think-segmentation.md`
- `docs/deviations/2026-08-13-c80-greedy-decoding.md`
- `docs/deviations/2026-08-13-t10-attempt-selection.md`
- `docs/deviations/2026-08-13-t18-analysis-parameters.md`
- `docs/deviations/2026-08-14-method-frozen-config-thrash.md`
- `docs/deviations/2026-08-15-t17-provisional-pool-sensitivity.md`
- `docs/deviations/2026-08-15-t23-causal-config-migration.md`
- `docs/deviations/2026-08-18-t22-p50-membership.md`
- `docs/DEVIATION_2026-08-17_CONFIRMATORY_MEMBERSHIP.md`
- `docs/DEVIATION_2026-08-25_T29_HUMAN_PRIMARY.md`

These records document why the corresponding method/provenance changes were made and prevent historical execution metadata from being silently rewritten.
