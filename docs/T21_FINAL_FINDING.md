# T21 final finding: source-model Qwen activation capping

**Status:** `COMPLETE — PARTIAL / EVALUATION-LIMITED`

T21 reproduces the released Qwen3-32B activation-capping intervention from Lu et al. using the pinned upstream commit and released capping configuration. The intervention is source-matched at the publicly specified model/configuration level: `Qwen/Qwen3-32B`, model revision `9216db…6137`, BF16, thinking disabled, and `layers_46:54-p0.25` (layers 46–53).

## Safety result

The public safety substitute contains 100 frozen JBB harmful behaviors evaluated under two conditions (200 generations total). It is explicitly **behavior-only and adapted**; it is not Lu et al.'s exact persona-jailbreak evaluation.

| condition | strict harmful | inclusive harmful | refusal | nonsensical |
|---|---:|---:|---:|---:|
| Unsteered | 4/100 (4%) | 4% | 88% | 0% |
| Source cap | 4/100 (4%) | 4% | 91% | 0% |

Paired strict-harm reduction (`unsteered - capped`) is **0.00**, with bootstrap 95% CI **[-0.04, 0.04]** (`n=100`, seed `210818`). The adapted set therefore does not reproduce the large safety reduction reported by Lu et al.; the result is evaluation-limited because the unsteered harmful baseline is already only 4% and the exact source 1,100 persona-jailbreak pair sample is not available in the public release.

## Capability result

The capability evaluation contains **6,224 generations** across the four benchmark families and sample sizes reported by Lu et al.

| benchmark | unsteered | capped | change |
|---|---:|---:|---:|
| IFEval strict prompt accuracy | 0.837338 | 0.835490 | -0.001848 |
| MMLU-Pro accuracy | 0.707857 | 0.697857 | -0.010000 |
| GSM8K accuracy | 0.960000 | 0.957000 | -0.003000 |
| EQ-Bench | 82.30 | 82.04 | -0.26 |

The sum of relative performance reductions across the four primary metrics is **2.2619%**, but this cross-benchmark sum is descriptive rather than an uncertainty-qualified effect size. From the committed aggregate counts, approximate 95% intervals for capped-minus-unsteered changes on the binary metrics are IFEval **[-4.60, +4.23] percentage points**, MMLU-Pro **[-4.38, +2.38] pp**, and GSM8K **[-2.08, +1.48] pp**. These are conservative aggregate-only Newcombe/Wilson intervals rather than paired intervals because the committed summaries do not contain discordance counts. EQ-Bench is a continuous score, so a binomial interval is not applicable and no valid paired interval can be reconstructed from the committed aggregate score alone. See `results/t21/t21_capability_uncertainty.json`. MMLU-Pro parseability is 99.71% unsteered and 99.64% capped; GSM8K and EQ-Bench are fully parseable.

## Judge provenance

The paper names the harmfulness judge as `deepseek-v3` and publishes the Appendix D.2.2 rubric/taxonomy. The source release does not specify the serving provider, exact serving revision, numerical precision/quantization, temperature, top-p, or maximum output tokens. This run therefore uses the original public `deepseek-ai/DeepSeek-V3` model through Novita as **our reproducibility choice**, with sampling arguments omitted rather than guessed. The final scorer uses the faithful Appendix D.2.2 transcription and the first-512-source-token rule.

## Claim boundary

T21 supports the following wording: **source-matched reproduction of the released Qwen capping intervention; source-max-matched DeepSeek-V3 judge; adapted public safety evaluation; sample-size-matched public capability battery.** It must not be called an exact Lu et al. replication, and T21 alone does not establish causal specificity against random/spherical controls.
