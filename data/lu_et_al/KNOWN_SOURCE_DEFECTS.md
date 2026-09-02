# Known defects in the Lu et al. source materials

Recorded at commit `a98961956072224eaf244eb289d6c01700b63795`. These are **not repaired**. T01 records the
source as it is; any correction is a decision of ours and belongs downstream.

A defect here means a property of the released materials that constrains what we can
claim. It does **not** mean the published method is deficient — where the paper
supplies something the repository omits, that is stated explicitly.

## D1 — 240 question IDs carry 239 unique texts

`ids [7, 227]` are byte-identical:

> How should society balance individual freedom with collective responsibility?

**Consequence.** The bank must never be described as "240 unique questions". Any block
containing both IDs double-weights one question, and a split placing them in different
halves inflates cross-half agreement.

**Required handling.** Keep both IDs in the same block, and report a sensitivity with
one removed.

## D2 — No question-category field

The schema is exactly `['id', 'question']`. There is no topic or category label.

**Consequence.** Category-balanced sampling cannot be attributed to Lu et al. Any
stratification is our construction and must be declared as an adaptation.

## D3 — Repository lacks the causal-evaluation dataset; the paper supplies the method

The released repository does not contain the exact 50-role membership list, the four
newly generated causal prompts per selected role, or a complete machine-readable
causal-evaluation configuration. 567 tracked text files were searched;
structural inspection of every JSON string-list found 0 proper
role-subset candidates.

This is a **repository** gap. The paper (`data/lu_et_al/paper/lu_et_al_2601.10387v1.pdf`) publishes the
selection procedure (p.32, D.1.1) and the five introspective questions in full
(p.32, D.1.2). Neither may be described as unpublished.

**Consequence.** Use the published questions exactly. Build any role list by applying
the published procedure to our target model(s), and label it an adaptation or an
approximation. Author the four prompts per role ourselves and label them ours.

## D4 — Paper–repository retention discrepancy

| source | minimum | scope |
|---|---|---|
| paper method (p.3, 2.1.2) | **10** | responses in **at least one of** {fully, somewhat} role-playing |
| released code default (`pipeline/4_vectors.py`) | **50** | `--min_count` CLI flag |
| our project threshold | **10** | valid **score-3** outputs per role |

The first two are independently true facts about different things — a published method
and a CLI script's generic default — and are not in contradiction; the repository does
not record which value produced the published vectors.

The third is **ours**. It is motivated by the paper's minimum-10 rule but is **stricter
and not directly inherited**. The published rule counts >=10 in *either* category, and
the paper states a retained role yields both a "fully X" and a "somewhat X" vector, so a
role retained on its *somewhat*-count alone could contribute a fully-role-playing vector
built from fewer than 10 score-3 responses. **The paper therefore does not establish
that every source role feeding the Axis had >=10 score-3 outputs**, and this package
attributes no such floor to the source.

**Consequence.** Neither published number may be labelled "Lu-equivalent" without citing
its source precisely, and our own threshold must be presented as a declared project
parameter — not as inherited — accompanied by a retention curve across alternative
minima as a predeclared sensitivity analysis.

## D5 — Selection procedure published; exact selected list unavailable

The paper (p.32, D.1.1) publishes the causal role-selection procedure: rank role
vectors by similarity to each model's own Assistant Axis, average that rank across the
three target models, take the top 50. **The procedure is published.** What is
unavailable is the resulting 50-name membership list, from both the repository and the
paper (~17 example names are recoverable from illustrative tables).

Because the ranking is defined relative to three specific models' own axes, no
model-independent "Lu 50-role list" exists even in principle.

**Consequence.** A DeepSeek-only ranking is a target-model-specific **adaptation** of
the published procedure. A DeepSeek+Llama averaged ranking is an **approximation** of
the published multi-model procedure. Describe whichever we use in those terms — never
as retrieval of Lu et al.'s exact list.

## Not a defect, but easy to misread

`data/roles/instructions/` holds **276** files for **275**
roles. The extra is `default`, the assistant baseline, which carries no `eval_prompt`
and no bespoke questions. An unrestricted glob over that directory treats the default
condition as a role and silently corrupts the contrast.
