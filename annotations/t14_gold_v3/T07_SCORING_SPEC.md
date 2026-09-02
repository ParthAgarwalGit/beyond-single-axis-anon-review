# T07 — how these labels must be scored

Frozen before the labels exist, so the analysis cannot be tuned to them.

## 1. Gold labels come from adjudication, not intersection

Do **not** define gold as "rows where both labellers agree". Silently dropping
disagreements discards exactly the hard cases the 2/3 boundary is about and inflates
every downstream metric.

1. Report raw agreement, Cohen's kappa (4-class), and score-3-vs-other kappa on **all**
   300 items.
2. Report the full 4x4 confusion matrix between the two labellers.
3. Adjudicate every disagreement explicitly, without seeing the automatic score.
4. Gold = the adjudicated label. Record original labels and the adjudicated value.

## 2. Human validity disagreements are reported, not resolved by intersection

`human_validity` disagreements are themselves a finding about how degenerate the
outputs are. Report the validity confusion matrix, adjudicate, and state how many rows
changed judgeability status. Do not quietly restrict to rows both called `ok`.

## 3. Judge abstentions must be reported, never silently excluded

The production judge forces `lu_score = None` when the trace has no closing tag or the
answer region is empty, regardless of what the judge model emitted
(`production_judge_reference.py: parse_score`). These are **abstentions**, not zeros.

For every item where humans (adjudicated) marked `human_validity = ok`:

    judge_abstention_rate = |{human ok AND lu_score is None}| / |{human ok}|

Report this alongside precision/recall. An item that a human could judge but the judge
abstained on is a **judge coverage failure**, and it must appear in the reported
denominator rather than vanishing from the gate.

Report three populations separately, never merged:

| population | definition |
|---|---|
| **scored** | human ok AND lu_score is not None — precision/recall/F1/kappa computed here |
| **abstained** | human ok AND lu_score is None — the coverage failure above |
| **unjudgeable** | human (adjudicated) marked unjudgeable — reported, excluded from the role gate, and cross-tabbed against the machine validity flag |

The role-judge gate (score-3 precision >= 0.85, recall >= 0.85, macro-F1 >= 0.75,
kappa >= 0.70) is evaluated on the **scored** population, and the abstention rate is
reported next to it. A judge that abstains on many human-judgeable items has not
passed merely by scoring well on the rest.

## 4. Weighting

Selection was stratified SRS within arm x automatic-score strata, so inclusion
probability is exactly `want/len(stratum)` and `ip_weight` in the private key is its
exact inverse.

- **Score-3 precision** is unbiased with or without weights (it conditions on the
  judge's own label, which defines the strata).
- **Recall and macro-F1 must be ip-weighted** to be valid for the full E80 corpus;
  strata are deliberately disproportionate. Report unweighted values alongside, labelled
  as sample-only.

## 5. Report by arm

Report all metrics for the translated and wrapper arms separately as well as pooled. If
judge accuracy differs materially by arm, part of any cross-arm axis difference is judge
behaviour rather than model behaviour.

## 6. Sensitivity set

The 60-item full-response set measures whether seeing the reasoning trace
changes the human label. Report the paired label distribution on those items
(final-answer-only vs full-response, joined via the private key's `sensitivity_id`), and
whether any 2/3 flips occur. This is a sensitivity analysis, not the primary gate.
