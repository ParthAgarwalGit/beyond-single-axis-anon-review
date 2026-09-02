# Lu et al. Comparability Matrix

**Companion configuration:** `configs/method_frozen_v4.yaml`
**Method freeze:** `docs/METHOD_FREEZE_V4_13_AUG.md`
**Study label:** Assistant-Axis generalization in a reasoning-distilled model; source-anchored causal transfer study, not an exact replication

## Classification vocabulary

| Label | Meaning |
|---|---|
| `DIRECT_MATCH` | Preserved from the source method or released implementation. |
| `EXPLICIT_IMPLEMENTATION` | The source describes the operation; this project freezes an exact executable rule. |
| `REASONING_MODEL_ADAPTATION` | Needed because the target model's interface or visible response structure differs. |
| `RESOURCE_SAMPLING_DIFFERENCE` | Reduces the source factorial without selecting from observed outcomes. |
| `NEW_VALIDATION_CONTROL` | Added to test reliability, construct validity, or causal specificity. |
| `UNAVAILABLE_SOURCE_ARTIFACT` | The published procedure exists, but exact machine-readable material is unavailable. |
| `PROTOCOL_REFERENCE_CONTROL` | Added to separate target-model behavior from interface or family effects. |

## Matrix

| Component | Lu et al. | Frozen project primary | Classification | Reason and claim boundary | YAML key |
|---|---|---|---|---|---|
| Source revision | Released Assistant-Axis repository | Commit `a98961956072224eaf244eb289d6c01700b63795` | `DIRECT_MATCH` | All source-derived fields trace to one pinned revision. | `source_provenance.lu_pinned_commit` |
| Roles | 275 | 275 | `DIRECT_MATCH` | Full role inventory preserved. | `source_provenance.source_counts.roles` |
| Role prompts | Five per role, 1,375 total | Same five source prompts | `DIRECT_MATCH` | Text remains source-derived; target rendering is separate. | `source_provenance.source_counts.role_prompts` |
| Question IDs | 240 | All 240 IDs | `DIRECT_MATCH` | Source-ID completeness preserved. | `question_design.id_space` |
| Unique question texts | Source has duplicate IDs 7 and 227 | Report 239 unique texts; keep both IDs | `DIRECT_MATCH` plus `NEW_VALIDATION_CONTROL` | Never claim 240 unique questions. Duplicate is kept in one block and removed in sensitivity. | `question_design.duplicate_text_groups` |
| Question categories | No source category field | No category claim | `DIRECT_MATCH` | Length balancing is our construction, not a Lu category split. | `question_design.confirmatory_split_algorithm` |
| Extraction factorial | 275 × 5 × 240 = 330,000 outputs per model | 275 × 240 = 66,000 outputs per primary arm | `RESOURCE_SAMPLING_DIFFERENCE` | One-fifth factorial. Supports a balanced transfer audit, not exact rollout replication. | `prompt_assignment.project_method` |
| Prompt crossing | Every question under every prompt | One prompt per role-question pair | `RESOURCE_SAMPLING_DIFFERENCE` | Exact 16-per-index balance in each 80-ID block. | `prompt_assignment.rule` |
| E80 | Not a source concept | Existing inspected 80-ID exploratory block | `NEW_VALIDATION_CONTROL` | Exploratory data cannot serve as untouched confirmation. | `question_design.blocks.E80` |
| C80-A/C80-B | Not a source concept | Two untouched 80-ID confirmatory blocks | `NEW_VALIDATION_CONTROL` | Enables independent reconstruction without discarding source IDs. | `question_design.blocks` |
| Source prompt channel | Source target models use their source protocol | User-translated DeepSeek primary; matched system/user test | `REASONING_MODEL_ADAPTATION` | Channel transfer is measured rather than assumed. | `prompt_rendering.conditions` |
| User-translated arm | Not source-exact | Source instruction and question in one user message | `REASONING_MODEL_ADAPTATION` | Minimal semantic translation; primary DeepSeek arm. | `prompt_rendering.conditions.USER_TRANSLATED_LU` |
| Explicit wrapper | Not source method | Wrapper arm is sensitivity only | `NEW_VALIDATION_CONTROL` | May leak the score-3 criterion, so cannot define primary geometry. | `prompt_rendering.conditions.USER_EXPLICIT` |
| System channel | Source-style instruction placement | Same instruction in system, question in user | `PROTOCOL_REFERENCE_CONTROL` | Tests model × channel interaction; no assumed DeepSeek failure. | `prompt_rendering.conditions.SYSTEM_LU` |
| Chat serialization | Source model templates | Each model's frozen tokenizer chat template | `EXPLICIT_IMPLEMENTATION` | No manual special tokens; template and rendered hashes saved. | `prompt_rendering.render_function` |
| Default conditions | Five source default conditions | Same five conditions | `DIRECT_MATCH` | Empty control and four descriptions preserved. | `prompt_rendering.default_conditions` |
| Default channel | Source protocol | Match the role arm's channel | `REASONING_MODEL_ADAPTATION` | Prevents role/default contrast from conflating channel. | `prompt_rendering.default_rendering` |
| Default judging | Default vector unfiltered by role judge | Technical validity only; no role judge | `DIRECT_MATCH` | Preserves source asymmetry. | `role_vectors_and_axis.default_vector` |
| Default weighting | Source mean over default outputs | Mean within condition, then equal 0.2 condition weights | `EXPLICIT_IMPLEMENTATION` | Prevents failure/length imbalance across conditions. | `role_vectors_and_axis.default_vector.condition_weights` |
| Response start | Assistant turn span; prompt excluded | Exact stored prompt-token count | `EXPLICIT_IMPLEMENTATION` | Avoids text retokenization ambiguity. | `response_tokenization.response_start_index` |
| Primary token pool | Mean across all assistant-response tokens | `ALL_RESPONSE_TOKENS` | `DIRECT_MATCH` | Main Lu-comparable representation. | `activation_extraction.primary_pool` |
| Reasoning pool | Not separately analyzed | Prefix through final closing marker | `REASONING_MODEL_ADAPTATION` | Sensitivity for visible reasoning; not primary. | `response_segmentation` |
| Final-answer pool | Not separately analyzed | Suffix after final closing marker, or full direct answer | `REASONING_MODEL_ADAPTATION` | Sensitivity and primary role-judge region. | `response_segmentation` |
| Malformed reasoning markers | Not a source issue | All-response may remain usable; region-specific masks and role score abstain | `REASONING_MODEL_ADAPTATION` | Missing segmentation is never role score 0. | `response_segmentation.cases.malformed_reasoning` |
| Technical validity | Source filtering pipeline | Explicit validity enum separate from role expression | `NEW_VALIDATION_CONTROL` | Prevents technical failure from masquerading as persona failure. | `validity` |
| Role rubric | Per-role 0–3 rubric | Exact per-role 0–3 rubric | `DIRECT_MATCH` | Same ordinal construct. | `role_judge.rubric` |
| Judge response region | Source response field | Final answer under frozen segmentation | `REASONING_MODEL_ADAPTATION` | Reasoning model exposes separable visible regions; full response is sensitivity. | `role_judge.primary_response_region` |
| Judge validation | Source study uses automatic evaluation | Two-human gold set and frozen gates | `NEW_VALIDATION_CONTROL` | Geometry is not interpreted until measurement validity passes. | `role_judge.validation_gates` |
| Source role retention | Paper: ≥10 in either fully or somewhat category; released CLI default: 50 | ≥10 valid score-3 outputs per 80-ID block | `EXPLICIT_IMPLEMENTATION` | Our score-3 threshold is a declared project choice, not attributed as Lu's exact Axis gate. | `role_vectors_and_axis.project_role_vector_threshold_per_80_id_block` |
| Retention sensitivities | Not frozen in source | Thresholds 5 and 15 | `NEW_VALIDATION_CONTROL` | Predeclared stability curve; no post-result threshold lowering. | `role_vectors_and_axis.predeclared_threshold_sensitivities_per_80_id_block` |
| Role vector | Mean of fully role-playing response activations | Mean of valid score-3 response vectors | `DIRECT_MATCH` | Output validity is added but role class is preserved. | `role_vectors_and_axis.role_vector` |
| Role weighting | Equal weighting of role vectors | Equal weighting | `DIRECT_MATCH` | Roles with more retained outputs do not dominate the Axis. | `role_vectors_and_axis.role_weighting_in_axis` |
| Activation site | Middle-layer post-MLP/block-output residual | `model.model.layers[L]` block output | `DIRECT_MATCH` plus `EXPLICIT_IMPLEMENTATION` | Exact tensor and hidden-state mapping are frozen. | `activation_extraction.hook` |
| Middle layer | Middle layer in each source model | Block 16 output; block 15 sensitivity | `REASONING_MODEL_ADAPTATION` | Exact 32-block convention and off-by-one check. | `activation_extraction.middle_layer` |
| Assistant Axis | Default mean minus mean retained-role vector | Same formula and orientation | `DIRECT_MATCH` | Positive direction is toward default Assistant. | `role_vectors_and_axis.axis_formula` |
| Block-specific defaults | Not a source reliability design | Independent default means for C80-A and C80-B | `NEW_VALIDATION_CONTROL` | Avoids shared-default inflation in split reliability. | `confirmatory_geometry.independent_default_means` |
| PCA | Descriptive role-space analysis | Descriptive, centred across roles | `DIRECT_MATCH` | PC1 never replaces the contrast-defined Axis. | `role_vectors_and_axis.PCA` |
| Independent reliability | Not the source's main test | Cross-axis role-projection correlation | `NEW_VALIDATION_CONTROL` | Tests whether role ordering reconstructs on untouched questions. | `confirmatory_geometry.primary_statistic` |
| Axis cosine | Reported geometric similarity | Secondary reliability statistic | `NEW_VALIDATION_CONTROL` | Raw cosine can be inflated by a shared default component. | `confirmatory_geometry.secondary_statistics` |
| Bootstrap and influence | Not source-defining | 2,000 role bootstraps and LORO | `NEW_VALIDATION_CONTROL` | Quantifies uncertainty and role concentration. | `confirmatory_geometry.uncertainty` |
| Role-correspondence null | Not source-defining | 5,000 permutations | `NEW_VALIDATION_CONTROL` | Destroys same-role matching while preserving marginals. | `confirmatory_geometry.nulls.role_correspondence_permutation` |
| Common-orientation null | Not source-defining | 5,000 role-specific sign flips | `NEW_VALIDATION_CONTROL` | Tests whether role contrasts share an Assistant orientation. | `confirmatory_geometry.nulls.common_orientation_sign_flip` |
| Plain role shuffle | Can leave a grand mean unchanged | Forbidden as sole global-Axis null | `NEW_VALIDATION_CONTROL` | Prevents an invalid near-identity null. | `confirmatory_geometry.nulls.plain_role_name_shuffle_for_global_axis_forbidden` |
| Causal-role procedure | Rank by each source model's Assistant Axis, average ranks, take 50 | DeepSeek-only E80 ranking, top 50 | `REASONING_MODEL_ADAPTATION` | Target-model-specific version of published procedure. | `causal_role_selection` |
| Exact source P50 membership | Not fully published | Unavailable | `UNAVAILABLE_SOURCE_ARTIFACT` | Never describe project P50 as retrieved from Lu et al. | `causal_role_selection.exact_Lu_P50_list_available` |
| Four causal prompts per role | Newly generated source prompts not released in full | Project-authored frozen prompts | `UNAVAILABLE_SOURCE_ARTIFACT` | Must be labelled project adaptations. | Causal prompt manifest, frozen later |
| Five introspective questions | Published verbatim | Optional small diagnostic | `DIRECT_MATCH` | Primary security causal run instead uses frozen harmful questions. | Causal design manifest, frozen later |
| Main causal questions | Source includes persona/jailbreak evaluations | Five harmful-behaviour questions per role | `REASONING_MODEL_ADAPTATION` | Harmfulness is application outcome; identity is separate mechanism outcome. | `steering.main_output_count` |
| Held-out causal Axis | Source exact leakage handling not our design | P50 excluded from P50 Axis | `NEW_VALIDATION_CONTROL` | Prevents evaluated roles from defining their own intervention. | `causal_role_selection.P50_excluded_from_final_causal_axis` |
| Steering direction | Add/subtract Assistant direction | Same signed orientation | `DIRECT_MATCH` | Positive is toward default Assistant. | `steering.formula` |
| Steering timing | Source intervention at residual stream | Generated response positions only | `REASONING_MODEL_ADAPTATION` | Prompt tokens remain unmodified; first sampled token is therefore unsteered. | `steering.timing` |
| Residual scaling | Source scales relative to typical residual norm | Mean per-token block-16 default residual norm | `EXPLICIT_IMPLEMENTATION` | Makes coefficient dimensionless and reproducible. | `steering.residual_norm_scaling` |
| Coefficient schedule | Exact source schedule unavailable | Symmetric `−a, 0, +a`; `a` frozen on development data | `UNAVAILABLE_SOURCE_ARTIFACT` plus `EXPLICIT_IMPLEMENTATION` | No tuning on causal outcomes. | `steering.nonzero_coefficient` |
| Random direction | Not the source causal specificity control | Orthogonal unit Gaussian direction | `NEW_VALIDATION_CONTROL` | Distinguishes semantic direction effects from generic perturbation. | `steering.random_control` |
| Zero condition | Could be represented twice conceptually | One shared zero | `EXPLICIT_IMPLEMENTATION` | Avoids 1,000 duplicate generations. | `steering.unique_conditions` |
| Causal workload | Source evaluation differs by task | 5,000 unique outputs | `RESOURCE_SAMPLING_DIFFERENCE` | Exact project factorial is reported. | `steering.main_output_count` |
| Harmfulness outcome | Source persona jailbreak behavior | Strict harmful compliance primary; inclusive secondary | `REASONING_MODEL_ADAPTATION` | Safety application, not proof of a deployable defense. | `causal_outcomes` |
| Identity outcome | Source visible identity categories | Seven-way identity plus binary non-Assistant rate | `REASONING_MODEL_ADAPTATION` | Identity and harmfulness are analyzed jointly but not conflated. | `causal_outcomes.identity_labels` |
| Causal judge region | Source task-specific | Full visible completion primary; answer-only sensitivity | `REASONING_MODEL_ADAPTATION` | Harmful material anywhere visible counts. | `causal_outcomes.primary_judge_region` |
| Causal specificity model | No matched random slope test | Shared-zero two-slope Axis-versus-random contrast | `NEW_VALIDATION_CONTROL` | Directly tests direction specificity without duplicating zero. | `analysis_registry.primary` |
| Protocol-reference model | Source uses three larger targets | Llama-3.1-8B-Instruct | `PROTOCOL_REFERENCE_CONTROL` | Separates DeepSeek/interface effects from broader transfer. | `models.protocol_reference` |
| Cross-model vector comparison | Source compares source models | Compare metrics and role rankings, not raw vector cosine | `PROTOCOL_REFERENCE_CONTROL` | Hidden coordinate systems are not assumed aligned across independently trained models. | Cross-model analysis plan |
| One-dimensional adequacy | Axis emphasized | Axis vs grouped linear and restrained nonlinear readouts | `NEW_VALIDATION_CONTROL` | A useful Axis need not exhaust persona geometry. | `analysis_registry.sensitivity` |

## Permitted description

> We implement a source-anchored transfer audit that preserves the full role and question-ID inventories,
> the source role rubric, whole-response pooling, equal role weighting, and the default-minus-role Axis,
> while introducing a balanced incomplete prompt assignment, reasoning-model segmentation, independent
> confirmatory blocks, human measurement validation, leakage controls, and an orthogonal random-direction
> causal comparison.

## Prohibited descriptions

Do not write:

- “exact replication”;
- “240 unique questions”;
- “Lu's published 50-role list”;
- “Lu required ten score-3 outputs per role”;
- “answer-only Lu replication”;
- “system prompts fail on DeepSeek” before the matched result;
- “one universal persona direction”;
- “jailbreak defense” without validated harmfulness and random-control evidence.

## 13 August 2026 V4 amendments

| Component | Lu et al. | Frozen project primary | Classification | Reason and claim boundary | YAML key |
|---|---|---|---|---|---|
| Prompt-prefilled reasoning open | Not a source issue for the target models used there | DeepSeek prompt already opens `<think>`; generated output may begin inside reasoning | `REASONING_MODEL_ADAPTATION` | Corrects segmentation labels without changing model behavior. | `response_segmentation.prompt_prefills_opening_marker` |
| C80 attempt selection | Project implementation detail | Lowest-retry technically valid attempt per semantic rollout | `EXPLICIT_IMPLEMENTATION` | Prevents later retries from silently replacing an earlier completed output. | `validity.attempt_selection` |
| E80 versus C80 decoding | Not a source split | C80-A/B share greedy decoding; E80 used accepted E80/T09 sampling settings | `EXPLICIT_IMPLEMENTATION` | C80-A/B is decoding-matched confirmation; E80-to-C80 is not a pure block-only contrast. | `executed_C80_generation_provenance` |
| Response-region comparison | Source pools response representation | All-response remains Lu-comparable primary; all-response/reasoning/final-answer is a main paper comparison | `REASONING_MODEL_ADAPTATION` | Tests reasoning-model region dependence without redefining the Axis. | `activation_extraction.paper_level_main_response_region_comparison` |
| Exact PCA verification | Source uses PCA descriptively | Exact variance/alignment/retention/centering record required | `EXPLICIT_IMPLEMENTATION` | Makes dimensionality comparison reproducible; 70% remains descriptive. | `role_vectors_and_axis.PCA.verification_record_required` |
| Fixed post-steering basis | Not a source causal requirement | Primary movement measured in one frozen pre-steering PCA basis | `NEW_VALIDATION_CONTROL` | Prevents condition-specific refits from changing the coordinate system. | `post_intervention_representation_analysis` |
| Causal activation capture | Not a source requirement for this analysis | Save causal activations needed for fixed-basis movement analysis | `NEW_VALIDATION_CONTROL` | Measures behavioral and representation changes on the same causal outputs. | `steering.causal_activation_capture` |
