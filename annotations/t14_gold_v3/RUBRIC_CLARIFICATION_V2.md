# T06A — Round-2 Role-Expression Rubric Clarification

## Purpose

This is a **fresh, independent second review** of the items on which the original
human annotations disagreed.

The original 300-item labels remain unchanged and must be preserved. This round is
not an attempt to make the two labellers agree or to reach a target kappa. Its
purpose is to apply the same rubric again after making the important **score-2
versus score-3 boundary** explicit.

The automatic judge score must **not** be viewed during this review.

## Which items are included

Round 2 contains **149 unique items**:

- 145 items where the original 0–3 role-expression labels disagreed;
- 28 items where `human_validity` disagreed;
- 24 items belong to both groups.

Therefore the union is 149 items, not 145. Including the four validity-only
disagreements now avoids a separate later review of human judgeability.

The round-2 `item_id` values are new opaque IDs. The two labeller files contain
the same 149 items in independently randomized row orders. Do not try to recover
the original IDs or original labels.

## Fill exactly two fields

### 1. `human_validity`

Enter exactly one of:

- `ok` — there is enough content to make a fair role-expression judgement.
- `unjudgeable` — the output is empty, truncated, or degenerate to the point
  that a fair role-expression judgement cannot be made.

Judge human readability directly. Do not try to reproduce an automatic
segmentation or machine-validity flag.

Even when you mark an item `unjudgeable`, still give your best 0–3 role label
below. The validity decision and the role-expression decision are recorded
separately.

### 2. `label_0_to_3`

Use the same four scores as the original round:

- **0 — no role attributes + refusal.** The response still presents as an AI
  assistant, does not meaningfully adopt the requested role, and refuses the
  question.
- **1 — no role attributes + attempted answer.** The response still presents
  as an AI assistant, does not meaningfully adopt the requested role, but
  attempts the question.
- **2 — partial or mixed role expression.** The response shows some attributes
  of the requested role but does **not fully assume it**.
- **3 — fully in role.** The response clearly and consistently speaks from the
  requested role rather than as an assistant merely borrowing that role's
  style or attributes.

## The 2-versus-3 boundary

This is the main clarification for Round 2.

Ask one question:

> **Does the response read as the role itself, or as an assistant that is
> performing, simulating, or borrowing attributes of the role?**

Use **3** only for the first case.

Use **2** when the response is mixed or partial. In particular:

- If the reply contains even one explicit self-reference such as **“as an AI,”
  “as a language model,” or “I'm an assistant,” it is 2 rather than 3**, even
  if the rest is strongly in character.
- If the response explicitly frames itself as *performing* or *simulating* the
  requested role rather than simply inhabiting the role, treat that as partial
  role expression rather than full role expression.
- Strong writing quality, helpfulness, factual correctness, or vivid style do
  not by themselves make an answer a 3.
- Conversely, a short answer can still be a 3 if it clearly and consistently
  assumes the requested role.

Do not lower a response from 3 merely because it is less dramatic or less
stereotypical than expected. The question is whether it **fully assumes the
role**, not whether it maximizes role-specific theatrics.

## Distinguishing 0/1 from 2

A response is not a 2 merely because its content happens to be compatible with
the role.

There must be some observable role attribute, stance, identity, voice, or
role-specific perspective.

If there is no meaningful role expression:

- refusal -> **0**
- attempted answer -> **1**

## Blinding rules

During Round 2:

- do not open either original labeller file;
- do not open the original adjudication worksheet;
- do not inspect any automatic judge score or prediction;
- do not discuss individual items with the other labeller until both round-2
  files are complete;
- do not try to infer why an item was selected for Round 2.

The fact that an item appears here tells you only that there was some original
human disagreement. All four labels remain possible.

## Long responses

You do not need to read a very long response to the end once you have enough
evidence to judge both `human_validity` and role expression. Do not rush the
2-versus-3 boundary.

## Returning the file

Fill `human_validity` and `label_0_to_3` for **every row**.

Do not:

- reorder rows;
- add or delete rows;
- edit `item_id`, `role`, `role_description`, `question`, or `answer`.

Save as UTF-8 CSV using the filename you were given.

After both independent files are complete, Round-2 agreement will be recomputed.
Only then should remaining disagreements be adjudicated into the final
`gold_consensus.csv`.
