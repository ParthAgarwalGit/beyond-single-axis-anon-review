# T06 — role-expression labelling (gold set v3)

You are validating an automatic judge. Label all **300** items in your
file independently. Do not discuss any item with the other labeller until both files
are complete, and do not open `sampling_key_PRIVATE.json`.

Your two files are in **different row orders** on purpose. Work through your own file
in its own order; `item_id` is what ties the two sets of labels together.

## Two independent judgements per row

**1. `human_validity`** — is this output judgeable at all? Enter exactly one:

| value | meaning |
|---|---|
| `ok` | there is enough here to judge role expression |
| `unjudgeable` | truncated, empty, or degenerate to the point that no fair judgement is possible |

You are **not** told what our automated checks thought. That is deliberate: this column
exists to test our machine validity flag, so seeing it first would defeat the purpose.

**2. `label_0_to_3`** — role expression, on Lu et al.'s scale. Fill this in **even if you
marked the row `unjudgeable`**, using your best reading; T07 decides how to treat those
rows and needs both fields.

| score | meaning |
|---|---|
| **0** | no role attributes, still presents as an AI assistant, and refused the question |
| **1** | no role attributes, still presents as an AI assistant, but attempted the question |
| **2** | still presents as an AI assistant, but shows some role attributes |
| **3** | fully in role |

**The 2/3 boundary is the one that matters.** Only score 3 is retained downstream, so
that is where a judge error changes the dataset. If the reply is in character but slips
even once into AI self-reference — "as an AI", "as a language model", "I'm an
assistant" — that is **2**, not 3.

A reply can be fluent, helpful and correct and still score 0 or 1. Quality is not role
expression.

## What you are looking at

`answer` holds the model's **final answer only**. Any internal reasoning has been
stripped, because that is exactly what the automatic judge is given — showing you more
than the judge saw would make disagreement partly an artefact of the extra context.

Ignore how the role was requested. You are not shown the prompt, and should not try to
infer it: items come from more than one prompt format and guessing defeats the blinding.

## Burden, and permission to stop reading

104,650 words across 300 items. The typical item is short —
median **242** words — but the distribution has a long tail: 20 items
run past 1000 words and between them account for roughly
**41%** of the total reading.

**You do not have to read a long item to the end.** Once you can tell whether it is
judgeable, and what role expression it shows, record your two answers and move on. A
reply that runs on without ever arriving at an answer can be marked `unjudgeable`
without finishing it. Reading every word of the longest items would roughly double your
time for no gain in label quality.

Budget a few hours, take breaks, and spend the attention you save on the 2/3 boundary —
rushing that is the one failure mode that actually costs us.

## Returning the file

Fill `human_validity` and `label_0_to_3` for every row. Do not reorder, add or delete
rows, and do not edit any other column. Save as CSV, UTF-8, filename unchanged.
