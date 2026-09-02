# Current artifact status

Baseline commit: `831a9bee48d20c4be6ee4f24d89029f5cfeee50a` (main)
Last updated: 30 August 2026

This file is a thin publication-facing index of which repository artifacts are current evidence for the paper. Detailed status, methods, and claim boundaries live with each task under `results/<task>/`; those task records take precedence if this index ever drifts.

## The one-line rule

`results/` and `design/` contain the active evidence/provenance chain. `replications/` is archived and superseded per `replications/README.md`; do not cite numbers from it as final paper evidence.

## `results/` — by task

| Task | Status | Notes |
|---|---|---|
| `t06` + `t07` | COMPLETE / MEASUREMENT-LIMITED | Human role-expression reference is frozen. The production role judge fails all four predeclared gates, so it does not define confirmatory C80 membership. PR #83 adds 10,000-replicate stratified bootstrap CIs without changing the frozen gate or measurement branch. See `results/t06/PROVISIONAL_STATUS.md`, `results/t07/judge_validation.json`, and `results/t07/judge_validation_with_uncertainty.json`. |
| `t12` | COMPLETE (infrastructure + accepted activation inputs) | Activation extraction/reconstruction and provenance consumed by T15/T17/T18. Use the task receipts/manifests rather than archived replication outputs. |
| `t14` | COMPLETE | Frozen 275-role retained/eligible set used by confirmatory geometry. See `results/t14/retained_roles.json`. |
| `t15` / `t16` | COMPLETE | Confirmatory C80-A/C80-B recovery is frozen. The 300 repeated frozen-question resplit sensitivity is also merged and reproduces the observed split while showing it is not an unusually favourable partition. See `results/t15/README.md`, `results/t15/confirmatory_reliability.json`, and `results/t15/t15_t16_resplit_results.json`. |
| `t17` | COMPLETE | C160 all-response/reasoning/answer response-region role-space analysis. Role ordering remains highly correlated across regions even where Axis directions differ. See `results/t17/PROVISIONAL_STATUS.md` and the frozen T17 source data. |
| `t18` | COMPLETE / `INADEQUATE` | Frozen doubly-grouped predictive adequacy test is reportable. Axis AUROC 0.678 vs full-dimensional linear 0.878; difference +0.200, simultaneous 95% CI [0.181, 0.221], above the 0.02 adequacy margin. This is predictive-summary evidence, not a causal or intrinsic-dimensionality claim. See `results/t18/README.md` and `results/t18/t18_report.json`. The response/reasoning/answer length nuisance baseline is also merged; use its task artifact for any manuscript statement. |
| `t21` | COMPLETE / FLOOR-LIMITED | Qwen source-capping reproduction and capability battery. Strict harmful compliance is 4% unsteered and 4% Assistant cap, so the safety comparison is severely floor/power limited. See `results/t21/README.md`. |
| `t24` | COMPLETE | Qwen Assistant/random/spherical numerical controls frozen with matched intervention incidence/engagement. This does not imply perturbation-magnitude or geometry matching. |
| `t26` / `t27` | COMPLETE / HARMFULNESS MEASUREMENT-LIMITED | Human validation and causal-judge gate are frozen. Harmfulness automatic scoring fails the absolute gate, so the human subset is primary; automatic full-panel harmfulness is secondary. Identity is separate and sparse. See `results/t27/`. |
| `t29` | COMPLETE | Qwen matched-control causal-specificity analysis is frozen. Human-primary Assistant-minus-random strict-harm risk difference is -3.69 pp with 95% CI [-11.08, 0.00] pp; the automatic 400-row sensitivity has the opposite sign (+1.0 pp). Robust direction-specific harmfulness reduction is not established. See `results/t29/README.md`. |
| `t20-q` | IN PROGRESS / PRIVATE-OUTPUT WORKSTREAM | Qwen inference-mode causal-control extension is not part of the frozen main evidence unless its own production/scoring artifacts are completed and reviewed. Raw generations remain private and must not be committed. |
| `t25` / `t28` | DROPPED FROM REVIEWER-FACING PAPER | The DeepSeek signed-steering branch is not part of the submission claim set. Do not retain result-shaped placeholders or cite T28 as completed evidence. Historical tooling may remain in the repository for provenance. |

## `design/` — frozen inputs and figure/table source data

`design/` holds frozen question blocks, causal questions, prompt/config artifacts, and figure/table source data. These are inputs/provenance rather than standalone results. Regenerate derived files through their corresponding tools and preserve their recorded hashes.

## `replications/` — archived, not evidence

`replications/assistant-axis`, `replications/assistant-axis-8b`, `replications/persona-vectors`, and `replications/engels-irreducibility` are historical/exploratory and superseded by the `results/` + `design/` pipeline. They are excluded from the anonymous review snapshot.

## Third-party source material

The four `data/lu_et_al/` JSON containers that embed verbatim text from the upstream Assistant Axis repository are retained in the working repository for provenance but are excluded from the anonymous/public-review snapshot while redistribution rights remain unresolved. `tools/fetch_lu_et_al_source.py` pins and reconstructs them from upstream; see `THIRD_PARTY_NOTICES.md` for the licensing boundary.
