# T18 — Dimensional adequacy and nonlinearity stress test

**Status:** pipeline prepared and validated. **The real run has not been made
and must not be made until the confirmatory geometry is frozen at G2.**

**Registry:** `analysis_registry.sensitivity[13]` in `configs/method_frozen.yaml`
— "grouped multidimensional linear and restrained nonlinear readouts".
**Frozen parameters:** `configs/t18_frozen.yaml` (see
`docs/deviations/2026-08-13-t18-analysis-parameters.md`).
**TODO board:** T18, owner [Reviewer], reviewer [Author B], 11–13 August, OPTIONAL,
CPU. This preparation hands over a validated pipeline; ownership of the run is
unchanged.

## 1. The question, and what an answer may claim

Does a **single direction** adequately summarise the default-Assistant versus
role-adopted distinction, on **unseen roles and unseen questions**?

This is diagnostic. Per the frozen YAML it does not define, re-sign, or
replace the Axis (`role_vectors_and_axis.PCA.descriptive_only`), and it must
not delay H0–H5. A win for a richer read-out is evidence about *summary
adequacy*, not evidence against the causal results, which are tested by
intervention rather than by prediction.

## 2. Design

**Target.** Binary: 1 = default-Assistant response, 0 = retained role-adopted
response. This is the distinction the Axis is built to capture and is signed
toward, so a one-dimensional score is a fair candidate for it.

**Rows.** Frozen primary slice, read from the config rather than hard-coded:
arm `USER_TRANSLATED_LU`, block 16, pool `ALL_RESPONSE_TOKENS`, blocks C80-A
and C80-B.

**Role eligibility is consumed, never recomputed.** The retained-role set comes
from the frozen T14/T15 `eligible_roles.json` artifact and its hash is bound
into the report. Deriving eligibility here from raw score-3 counts would admit
roles that fail the frozen per-block thresholds, and T18 would then diagnose a
different role set from the one T15/G2 describe. A role named in the artifact
with no rows in the run directory stops the run.

**Only the primary judge region is read.** T13 stores a `final_answer` row and
a paired `full_visible` sensitivity row for every generation. T18 selects
`final_answer` (`role_judge.primary_response_region`) explicitly; a repeated
primary measurement is an error, not a last-write-wins.

**Grouping.** Every fold holds out whole roles **and** whole questions at the
same time. Fold `(i, j)` tests rows whose role is in role-fold `i` *and* whose
question is in question-fold `j`, and trains on rows outside both. Rows
matching exactly one condition are used by neither side — that discard is the
price of testing on unseen roles and unseen questions simultaneously. 5 × 4
folds give complete out-of-fold coverage with each row tested exactly once.

Default responses carry no role, so each of the five conditions becomes the
synthetic group `__default__<index>`. Five role folds and five conditions is
not a coincidence: it lets every outer fold hold out exactly one condition and
still have both classes in its test set.

**Everything is fitted inside the fold** — the Axis, the PCA basis, the
centring and scaling statistics, and every hyper-parameter (chosen by a
doubly-grouped inner CV on training rows only).

## 3. Estimand

Two properties of the estimand are load-bearing and were both corrected during
review.

**Scores are never pooled across folds.** Outer folds are fitted independently,
so the same probability can mean different things in different folds and an
AUROC over pooled scores is not a well-defined quantity. The statistic is the
**unweighted mean of within-fold AUROC**, and every bootstrap replicate
recomputes that same statistic.
`test_fold_aggregation_does_not_compare_scores_across_folds` builds a case
where each fold ranks its own rows perfectly but the folds' probability bands
are offset: fold aggregation returns 1.000, pooling returns below 0.7.

**Rows are role-balanced.** The Axis equal-weights retained roles. Row-level
fitting and scoring would instead let a role with many retained responses carry
more influence than it carries in the geometry, so the read-outs would be
judged on a different estimand from the thing they assess. Weights equalise
each role's total contribution within its class, and equalise the two classes.
The linear candidates receive them as `sample_weight`; `MLPClassifier` accepts
neither sample nor class weights, so the same design is realised by
deterministic row repetition.

## 4. Candidates and frozen grids

| Family | What it is | Grid |
|---|---|---|
| `axis_1d` | Fold-built unit Assistant Axis, projection scored | none |
| `pca_linear` | L2 logistic on leading *k* training-fitted PCs | k ∈ {1,2,3,5,8,12,20,32} × C ∈ {0.001,0.01,0.1,1,10} |
| `full_linear` | L2 logistic on the full activation | C ∈ {0.001,0.01,0.1,1,10} |
| `small_nonlinear` | One hidden layer on *k* training-fitted PCs (L-BFGS) | k as above × hidden ∈ {4,8,16} × α ∈ {0.1,1,10} |

Features are standardised on training rows before the penalty is applied.
Without this, an L2 penalty punishes a low-variance component far more than a
high-variance one and `C` would mean something different for every *k*.

**Capacity rule (nonlinear only).** Trainable parameters `k·h + 2h + 1` must be
strictly fewer than the number of training roles in the fold. This **bounds
capacity relative to the grouping unit** — it is a capacity control, not a
proof that role-specific information cannot be represented. Settings over
budget are removed from the grid, and the budget shrinks with the fold. The
linear candidates are capacity-controlled by the penalty strength instead.

The nonlinear read-out uses L-BFGS, not Adam. On data this small the
stochastic solver's early-stopping heuristic occasionally halted on a
degenerate inverted solution (observed at AUROC 0.17 during validation), and
that heuristic would in any case have been a second, undeclared capacity
control.

**Every family that can determine a verdict is covered by the self-test.**
`test_every_verdict_family_is_covered_by_the_known_answer_worlds` fails if a
family is added to `families_that_may_determine_the_verdict` without a
synthetic world exercising it.

## 5. Inference, multiplicity, and the verdict rule

- **Resampling unit:** **roles** (`INFERENCE_CLUSTER = "role"`), 2000 cluster
  bootstrap replicates, clusters resampled within class.
- **Multiplicity:** a **simultaneous max-over-families bootstrap**. Each
  replicate takes `max_f (AUROC_f − AUROC_axis)` across all verdict families,
  so the interval already pays for having selected the best-performing
  alternative. Reporting per-family intervals and then acting on whichever
  looked best would let any one of three families trigger a verdict.
  Per-family marginal intervals are still reported, labelled as marginal.
- **Equivalence margin:** `adequacy_margin_auroc = 0.02`, frozen in
  `configs/t18_frozen.yaml` before any real activation was examined.

| Verdict | Condition |
|---|---|
| `INADEQUATE` | simultaneous lower bound **above** the margin |
| `ADEQUATE` | simultaneous upper bound **below** the margin |
| `INCONCLUSIVE` | the simultaneous interval straddles the margin |

The margin exists because the paired intervals are tight — a bare significance
test would report a 0.01 AUROC difference as a finding. `INCONCLUSIVE` is a
real possible outcome and must be reported as such, not rounded to `ADEQUATE`.

**Question-clustered intervals may not support a verdict.** They are computed
and stored as a diagnostic only, and `adequacy_verdict()` raises if handed one.
See §7.

## 6. Validation evidence

`python tools/run_t18_dimensional_adequacy.py --self-test` runs the pipeline
unchanged against synthetic worlds with known answers. All pass:

| World | Truth | Axis | Best alternative | Simultaneous CI | Verdict |
|---|---|---|---|---|---|
| `one_dimensional` | one direction suffices | 0.986 | `small_nonlinear` −0.0003 | [−0.0010, +0.0016] | `ADEQUATE` ✓ |
| `multidimensional_linear` | 5 linear directions needed | 0.866 | `pca_linear` +0.1176 | [+0.084, +0.143] | `INADEQUATE` ✓ |
| `radial_nonlinear` | only radius works | 0.560 | `small_nonlinear` +0.4402 | [+0.313, +0.602] | `INADEQUATE` ✓ |

The multidimensional world was constructed so the widest mean gap sits on the
*worst* signal-to-noise coordinate: the difference in means over-weights it and
reaches 0.866, while the optimal linear rule reaches ~0.98. That matches the
analytic prediction of 0.871.

In the radial world the nonlinear read-out wins outright while both linear
families sit near chance, so the pipeline distinguishes "needs more linear
dimensions" from "needs a nonlinearity" rather than lumping them together.

**Null world.** On `role_identity_only`, where the label is a deterministic
function of role identity and of nothing that transfers, the estimator is
unbiased for every family:

| Family | Mean over 6 null datasets |
|---|---|
| `axis_1d` | 0.476 |
| `pca_linear` | 0.515 |
| `full_linear` | 0.440 |
| `small_nonlinear` | 0.500 |

This is stated as a mean over independent null datasets, not as coverage of
one interval, deliberately: a 95% interval misses one time in twenty by
construction, and with only five default groups the per-dataset estimate
genuinely ranges from about 0.20 to 0.75. A single-dataset coverage check would
be a coin flip rather than a test.

The same data scored with a **row-random** split averages 0.873. Grouping is
what removes the leak, and both halves of that comparison are asserted in
`test_grouping_is_what_forces_chance_on_role_identity_data`.

## 7. Why roles and not questions are the resampling unit

Measured over 8 independent null datasets (a calibrated 95% interval should
cover 0.5 almost always):

| Cluster | Covered chance | Mean interval width |
|---|---|---|
| role | **8 / 8** | 0.515 |
| question | **1 / 8** | 0.028 |

Question-clustered intervals are badly anticonservative — they would have
declared a spurious effect in 7 of 8 pure-null datasets — because rows sharing
a role stay together inside every question cluster, so the dependence that
actually drives the estimate is never resampled. Locked in by
`test_question_clustering_is_anticonservative`, which fails if question
clustering ever becomes safe enough to justify switching to the tighter
interval.

## 8. Known limitations, to state in any write-up

1. **Only five default conditions.** The positive class has five groups, so
   marginal intervals are wide (≈0.5 in the synthetic setting) and the
   per-dataset estimate is volatile. The paired, simultaneous difference is
   the statistic to read: all families see the same conditions in a replicate,
   so that large shared variance component cancels.
2. **`INCONCLUSIVE` is likely if the true gain is small.** This design can
   separate a large gain from none; it cannot separate 0.02 from 0.04.
3. **The permutation diagnostic is descriptive.** Twenty group-level label
   permutations locate the null roughly and would expose a gross leak. No
   p-value is derived from it.
4. **The PCA here is not the descriptive role-space PCA.** T18's components are
   a predictive feature basis fitted on training *response* vectors inside each
   fold. `role_vectors_and_axis.PCA` is fitted on *role means* over the whole
   retained set. The component counts are not interchangeable and must not be
   quoted against each other or against Lu et al.'s 4 / 8 / 19.
5. **Prediction is not causation.** A richer read-out predicting better does not
   show that steering along more directions changes behaviour more.

## 9. The freeze gate

`--run-dir` refuses to start unless `docs/G2_GEOMETRY_FREEZE.json` exists with
`freeze_status: "FROZEN"`. T18 asks whether the frozen geometry is an adequate
summary; running it against a geometry that can still move would let the answer
influence the thing it describes, and would compete with T06/T07 for the same
artifacts. `--allow-unfrozen` exists for rehearsals only and stamps
`reportable: false` into the output.

## 10. Running it

Now, with no data and no freeze:

```bash
python tools/run_t18_dimensional_adequacy.py --self-test
```

After G2, with the frozen artifacts:

```bash
python tools/run_t18_dimensional_adequacy.py --run-dir runs/confirmatory
```

Expected `--run-dir` layout (all five required):

| File | Schema | Carries |
|---|---|---|
| `generation.jsonl` | `GENERATION_ROW` | arm, block, `role_id`, `question_id`, `default_condition_index`, validity |
| `judge.jsonl` | `JUDGE_ROW` + T13 | `role_score`, `judge_region` |
| `activations.jsonl` | `ACTIVATION_ROW` | `pool`, `block_index`, `vector_sha256` |
| `vectors.npz` | — | activation `row_id` → pooled vector |
| `eligible_roles.json` | T14/T15 | `eligible_roles_primary`, threshold, blocks |

Fail-closed before any fitting: vectors are re-hashed against their stored
`vector_sha256`; duplicate row IDs, duplicate primary judge measurements, mixed
config hashes, and mixed code commits are errors rather than last-write-wins;
and the slice must contain exactly the five frozen default conditions. Reports
stamp `source_git_sha`, `config_sha256`, `t18_config_sha256`, and the
eligibility artifact hash.

**Runtime note.** Validation runs in a few minutes on synthetic data. The real
slice is 4096-dimensional with ~250 roles; `full_linear` over 20 outer folds ×
12 inner folds is the expensive part. Budget CPU hours, not minutes, and
consider running the families separately.

## 11. Files

| Path | What |
|---|---|
| `configs/t18_frozen.yaml` | frozen grids, estimand, inference rule, margin |
| `docs/deviations/2026-08-13-t18-analysis-parameters.md` | why and when they were frozen |
| `src/t18_readouts.py` | splitting, candidates, metrics, bootstrap, verdict |
| `tools/run_t18_dimensional_adequacy.py` | runner, loader, freeze gate, self-test |
| `tests/test_t18_readouts.py` | 33 tests: the four worlds, leakage, capacity, estimand, multiplicity |
| `tests/test_t18_loader.py` | 18 tests: the loader against the repo's own schemas |
| `tests/test_t18_frozen.py` | 8 tests: no analysis parameter may drift from the frozen file |
