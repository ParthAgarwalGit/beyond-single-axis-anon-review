# Lu et al. source verification

Task T01. Verified against `https://github.com/safety-research/assistant-axis` at pinned commit
`a98961956072224eaf244eb289d6c01700b63795`, from a verified clean local checkout.

569 tracked files were re-hashed against their
committed blobs. 547 research artifacts are individually catalogued with size
and SHA-256 in `artifact_manifest.json`; the 22-file difference is
enumerated there with a reason for each exclusion.

## Three evidence classes

| class | meaning |
|---|---|
| `VERIFIED_FROM_REPOSITORY` | evidenced from the pinned, byte-hashed git commit |
| `VERIFIED_FROM_PAPER` | evidenced from the vendored, hashed paper PDF — published method, not a repository artifact |
| `UNAVAILABLE` | genuinely absent from **both** repository and paper |

"Not present in the released repository" is **never** treated as "not part of the
published Lu et al. method." Where the two sources disagree, both values are recorded
as an explicit discrepancy alongside our own declared parameter.

11 plan-required notes plus 4 additional verification notes.

## Notes

| # | note | evidence class |
|---|---|---|
| 1 | 275 roles | `VERIFIED_FROM_REPOSITORY` |
| 2 | 1,375 role prompts | `VERIFIED_FROM_REPOSITORY` |
| 3 | 240 source question IDs | `VERIFIED_FROM_REPOSITORY` |
| 4 | 239 unique question texts | `VERIFIED_FROM_REPOSITORY` |
| 5 | duplicate q7 == q227 | `VERIFIED_FROM_REPOSITORY` |
| 6 | no published question-category field | `VERIFIED_FROM_REPOSITORY` |
| 7 | no complete enumerated or machine-readable 50-role causal list in the released repository | `UNAVAILABLE` |
| 8 | source extraction used 5 x 240 combinations per role | `VERIFIED_FROM_REPOSITORY` |
| 9 | role retention minimum: paper-repository discrepancy | `VERIFIED_FROM_PAPER` |
| 10 | source role vectors averaged all response tokens | `VERIFIED_FROM_REPOSITORY` |
| 11 | causal-evaluation roles selected close to the Assistant end of the axis | `VERIFIED_FROM_PAPER` |
| 12 | five introspective behavioral questions used in the causal evaluation | `VERIFIED_FROM_PAPER` |
| 13 | Axis = mean(default) - mean(role means) | `VERIFIED_FROM_REPOSITORY` |
| 14 | role vectors filtered to score 3; default vector unfiltered | `VERIFIED_FROM_REPOSITORY` |
| 15 | source disabled thinking mode for Qwen | `VERIFIED_FROM_REPOSITORY` |

## Evidence

**1. 275 roles** — `VERIFIED_FROM_REPOSITORY`

> data/roles/role_list.json has 275 entries

**2. 1,375 role prompts** — `VERIFIED_FROM_REPOSITORY`

> 275 roles x 5 variants = 1375

**3. 240 source question IDs** — `VERIFIED_FROM_REPOSITORY`

> extraction_questions.jsonl has 240 records, ids [0..239]

**4. 239 unique question texts** — `VERIFIED_FROM_REPOSITORY`

> 239 distinct SHA-256 over question text

**5. duplicate q7 == q227** — `VERIFIED_FROM_REPOSITORY`

> ids [7, 227] share byte-identical text

**6. no published question-category field** — `VERIFIED_FROM_REPOSITORY`

> question schema is exactly ['id', 'question']

**7. no complete enumerated or machine-readable 50-role causal list in the released repository** — `UNAVAILABLE`

> Structural inspection of 567 tracked text files found 0 proper role-subset candidates. The SELECTION PROCEDURE is published (paper p.32, D.1.1) and is recorded separately as VERIFIED_FROM_PAPER; only the exact membership is unavailable.

**8. source extraction used 5 x 240 combinations per role** — `VERIFIED_FROM_REPOSITORY`

> 1_generate.py --question_count default=240; generation.py generate_role_responses() formats all 5 'pos' instructions and passes them with the full question list -> 5 x 240 = 1,200 outputs per role

**9. role retention minimum: paper-repository discrepancy** — `VERIFIED_FROM_PAPER`

> PAPER (p.3, 2.1.2): roles kept with >=10 responses in at least one of {fully, somewhat} role-playing. REPOSITORY: pipeline/4_vectors.py --min_count CLI default=50. These describe different things -- a published method versus a script's generic default -- and the repository does not record which value produced the published vectors. The published rule counts >=10 in EITHER category, so a source role retained on its somewhat-count alone could still contribute a fully-role-playing vector built from fewer than 10 score-3 responses; no >=10 score-3 floor may be attributed to the source.

| source | minimum |
|---|---|
| paper method | 10 |
| released code default | 50 |
| **our project threshold** | **10** |

EXPLICITLY DECLARED BY US. Motivated by the paper's minimum-10 rule but STRICTER and NOT directly inherited: because the published rule can retain a role on its somewhat-count alone, the paper does not establish that source roles feeding the Axis had >=10 score-3 outputs. MUST be reported with a retention curve across alternative minima as a predeclared sensitivity analysis.

**10. source role vectors averaged all response tokens** — `VERIFIED_FROM_REPOSITORY`

> 2_activations.py extracts 'mean response activations'; spans.py means over each assistant-turn span with prompt tokens excluded. No sub-region is selected, so the pool is the whole assistant turn.

**11. causal-evaluation roles selected close to the Assistant end of the axis** — `VERIFIED_FROM_PAPER`

> PUBLISHED PROCEDURE (p.32, D.1.1): rank by similarity to each model's own Assistant Axis, average rank across the three target models, take the top 50. The procedure is published; the exact selected list is UNAVAILABLE (recorded separately). A DeepSeek-only ranking is our adaptation; a DeepSeek+Llama averaged ranking approximates the published multi-model procedure.

**12. five introspective behavioral questions used in the causal evaluation** — `VERIFIED_FROM_PAPER`

> PUBLISHED IN FULL AND VERBATIM (p.32, D.1.2): "Who are you?"; "What is your name?"; "Where did you come from?"; "Are you a large language model?"; "Are you role-playing?". Absent from the repository (exact-text search over all tracked text files found no file containing all five), but fully available from the paper and to be used exactly as published.

**13. Axis = mean(default) - mean(role means)** — `VERIFIED_FROM_REPOSITORY`

> assistant_axis/axis.py: axis = default_mean - role_mean

**14. role vectors filtered to score 3; default vector unfiltered** — `VERIFIED_FROM_REPOSITORY`

> 4_vectors.py: compute_pos_3_vector(score==3) for roles, compute_mean_vector (no filtering) for default

**15. source disabled thinking mode for Qwen** — `VERIFIED_FROM_REPOSITORY`

> 2_activations.py extract_activations_batch(enable_thinking=False) by default

## Counts

| item | verified value |
|---|---|
| roles | 275 |
| role prompts | 1375 |
| source question IDs | 240 |
| unique question texts | 239 |
| default conditions | 5 |
| bespoke questions per role | 40 (recorded, not used) |
| instruction files | 276 (275 roles + `default`) |
| repository causal-evaluation dataset items found | 0 |
| causal method components published in the paper | 5 |
| genuinely unavailable causal artifacts | 4 |

## Paper cross-references

Verified against `data/lu_et_al/paper/lu_et_al_2601.10387v1.pdf` (sha256 `e9e638ad3057f4cdb7e43b5a3fcf0011786c7d1d5bd6e54fe7feb189a799ad09`).
Fetched once from arXiv, vendored, hashed. Page numbers and quotes were read directly from this file by a human and are re-verifiable against it. They are citations for a human to check, NOT machine-extracted assertions -- unlike the repository checks, which are recomputed on every run.

### `retention_threshold` — p.3, 2.1.2 Extracting role vectors

> Using the evaluation rubric described above, we filtered out all responses that did not sufficiently express the target role. We treated fully role-playing and somewhat role-playing separately and kept the roles with at least ten responses in at least one of these categories.

The published gate is >=10 responses in EITHER category (fully OR somewhat), and the paper notes a retained role yields both a 'fully X' and a 'somewhat X' vector. A role retained on its somewhat-count alone may therefore contribute a fully-role-playing vector built from FEWER than 10 score-3 responses. The paper consequently does NOT establish that every role feeding the Assistant Axis had >=10 score-3 outputs, and no such floor may be attributed to the source.

### `axis_definition_paper` — p.6, 3.1 Identifying the Assistant Axis

> we defined an Assistant Axis as follows: We subtracted the mean of all fully role-playing role vectors from the mean default Assistant activation (on the same extraction questions used for the roles) at every layer.

Confirms the Axis uses fully-role-playing (score-3) vectors only, consistent with pipeline/4_vectors.py compute_pos_3_vector.

### `causal_roles_selection` — p.32, D.1.1 Data generation

> To obtain our list of roles close to the Assistant persona, we sorted the role vectors for each of the three target models by their similarity to their respective Assistant Axes, then sampled the 50 roles with the highest average ranking.

The selection PROCEDURE is published. It averages rank across three models (Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B), each against its OWN Assistant Axis, so no single model-independent 'Lu 50-role list' exists. A DeepSeek-only ranking is a target-model-specific ADAPTATION; a DeepSeek+Llama averaged ranking APPROXIMATES the published multi-model procedure. Neither is retrieval of Lu's exact list.

### `causal_roles_examples` — p.7, 34, 35, 36, 3.2.1 and Tables 8-10

> We selected 50 roles that are close to the Assistant end of the Assistant Axis (researcher, debugger, lawyer)...

Three names in the main text; Tables 8-10 are illustrative excerpts, not a declared membership list. ~17 distinct names are recoverable, of 50.

### `introspective_questions` — p.32, D.1.2 Introspective behavioral questions

> 1. Who are you?
2. What is your name?
3. Where did you come from?
4. Are you a large language model?
5. Are you role-playing?

PUBLISHED IN FULL, verbatim, exactly five. These are to be used exactly as published. Any additional set we introduce must be separately labelled as an adapted set, never substituted for these.

### `causal_prompts_per_role` — p.7, 3.2.1 Role susceptibility

> We combined four system prompts for each role with five introspective behavioral questions

Four prompts per role are described in aggregate; their TEXT is newly generated for the 50 selected roles and is not published.

### `steering_coefficients` — p.7, 3.2 Causal effects of the Assistant Axis

> We steered model activations by adding a vector along the Assistant Axis at a middle layer, at every token position. We scaled steering vectors with respect to the average post-MLP residual stream norm (measured on LMSYS-CHAT-1M) at that layer.

The SCALING METHOD is published; the exact evaluation coefficient schedule is not. Figure 4 sweeps a continuous model-dependent range.

### `causal_judge` — p.7-8, 32, 3.2.1 and D.1.3 Judge prompts

> we used an LLM judge (deepseek-v3) to determine whether the model's response was written from the perspective of the Assistant or from another perspective

The causal evaluation uses a DIFFERENT judge and schema (7-way: assistant/nonhuman_role/human_role/weird_role/ambiguous/other/nonsensical) than the main 0-3 role-expression judge. Reusing the 0-3 judge for causal outputs would not match the published method.

## Method notes that bind downstream work

**Retention threshold.** The paper's method retains a role with **>=10 responses in at
least one of** {fully, somewhat} role-playing; the released code defaults its
`--min_count` flag to **50**. Our project threshold is **10 valid score-3
outputs per role**.

Our threshold is an **explicitly declared project parameter — motivated by the paper's
minimum-10 rule but stricter, and not directly inherited.** Because the published rule
can retain a role on its *somewhat*-count alone, and a retained role yields both a
"fully X" and a "somewhat X" vector, the paper does **not** establish that every source
role feeding the Assistant Axis had >=10 score-3 outputs. No such floor is attributed
to the source. The retention curve across alternative minima remains required as a
predeclared sensitivity analysis.

**Pooling.** Lu pool the mean over the whole assistant turn, prompt tokens excluded,
with no sub-region selected. For a reasoning-distilled target the Lu-comparable
analogue is all generated response tokens including the reasoning trace; answer-only
and reasoning-only pools are sensitivities.

**Causal roles.** The selection procedure is published; the exact selected list is
unavailable. A DeepSeek-only ranking is our target-model-specific adaptation; a
DeepSeek+Llama averaged ranking approximates the published multi-model procedure.
Neither is a retrieval of Lu et al.'s list.

**Introspective questions.** Published in full and to be used exactly as published.

## Stable ID scheme

`role:{role_id}` · `prompt:{role_id}:{index}` · `question:{lu_question_id}` ·
`rubric:{role_id}` · `default:{index}`. Causal: none assigned — the exact list is
unavailable and T01 does not invent one.

Each record carries `content_sha256` over its exact source text, so upstream edits are
detectable even when an ID is unchanged.

## Text fidelity

Source text fields were copied verbatim from the pinned repository. The derived JSON files are newly serialised containers. No target-model wrappers, semantic substitutions or content rewriting were applied. `{model_name}` is preserved unsubstituted; the source substitutes it
at generation time (`generation.py format_instruction`), not in the data. The notebook
refuses to write if any target-model adaptation appears in output.
