# T29 Qwen causal-scoring production run

## Scope

This records completion of the automatic causal-judge scoring stage for the frozen 400-row Qwen matched-control population. It is a scoring/provenance record only. It does not establish T27 judge validity or the final T29 causal-specificity claim.

T26 remains owned by [Author A]. This PR deliberately does not draw the human-validation sample, create blinded worksheets, adjudicate labels, or run T27/T29 inference.

## Production authority

The run used repository source commit:

`5b60b8e7bd32e8422b0886d27c04c52fb61e29f5`

That commit contains the post-PR #62 global Novita pacing/backoff fix and the post-PR #63 `max_retries=0` guard that disables hidden SDK retries. The production checkout was detached at this SHA and clean before scoring.

## Frozen input

The exact input population was retrieved from the immutable Hugging Face dataset revision:

- repo: `[Author-B-HF]/[anonymized-repository-name]-artifacts`
- revision: `e29553fe75a7d89757258e7c6c38723de1960ca9`
- path: `t29-qwen-matched-controls/qwen_capping_400_generation_population.jsonl`
- SHA-256: `24a0c5a3699fd6b51a66d4be92bc054d14cb800db894c0162b65774c5667f06b`

The production notebook re-verified 400 unique rows, 100 canonical `item_id`s, and exactly 100 rows in each frozen condition:

- `unsteered_baseline`
- `assistant_cap`
- `random_direction_cap`
- `sphere_proximity_cap`

It also verified exact four-condition item pairing, question/completion availability, zero technical-error rows, prompt/protocol invariants across conditions, and the expected 200 reused T21 + 200 T24 matched-control provenance split.

## Judge and transport

Scientific judge:

`deepseek-ai/DeepSeek-V3`

Novita provider route:

`deepseek/deepseek_v3`

Endpoint:

`https://api.novita.ai/v3/openai`

Transport controls active for every hosted attempt, including canaries and retries:

- 7.0 s minimum request start-to-start interval;
- 15.0 s backoff after HTTP 429 / rate-limit errors;
- 2.0 s generic request-error backoff;
- SDK automatic retries disabled with `max_retries=0`;
- scorer-level retry budget unchanged.

These are transport/provenance controls only. The frozen judge identity, prompt, schema, temperature, T26 sampling rule, T27 gates, and T29 estimand were not changed.

## Production result

Run window:

- start: `2026-08-23T09:21:41.960309+00:00`
- finish: `2026-08-23T10:23:04.359653+00:00`

Final status:

- `production = true`
- `drawable_by_t26 = true`
- scored rows: 400
- coverage failures: 0
- request-error attempts: 11
- rate-limit attempts: 11

All eleven 429 attempts were recovered inside the explicit scorer-controlled retry/backoff path. None became a final coverage failure.

Per condition, all 100 rows received usable automatic labels and each condition had zero coverage failures.

## Canary result

The frozen 50-item canary was usable at both ends of the run.

- start SHA-256: `6e48843fbd6222cde4afb17ee5416933654aad902a39bfcf34e3fe07b856b9f8`
- end SHA-256: `6e48843fbd6222cde4afb17ee5416933654aad902a39bfcf34e3fe07b856b9f8`
- drift detected: no
- changed canary items: 0 / 50

Thus the scorer passed the repository's production/drawability and start/end drift checks for this invocation window.

## Private and committed artifacts

The item-level scored JSONL and raw judge responses remain private and must not be committed because item-level automatic labels joined to condition would unblind the T26 sample.

Their exact hashes are recorded for downstream binding:

- scored artifact SHA-256: `d7f8feb34f8c2e8e8fff57764c95ff81b8128fbfa35d87c380322dce07565bd3`
- raw-response artifact SHA-256: `130bff99ab2888c0989154d5fbf69970e77ceb7d6f1d508b9704d58745cfa2ed`
- scoring-report SHA-256: `e45bbae7dee21a49e32309576b5ba17b369a380a6fc22b4697d272b8f797ed0b`

The aggregate production receipt is committed at:

`results/t29/QWEN_CAUSAL_SCORING_PRODUCTION_RECEIPT.json`

## Handoff boundary

This scoring stage is complete. No rerun is indicated.

The next Qwen causal step belongs to T26/T27:

1. [Author A] draws the frozen 120-item Qwen sample, 30 per condition, from this exact scored artifact;
2. the condition/automatic-label key remains private;
3. [Author-B-GitHub] and [Reviewer] label independently while blinded;
4. [Author A] runs T27 after both label files and adjudication are complete;
5. T29 full-corpus automatic inference is permitted only for outcomes whose T27 branch is `VALIDATED`.

## Claim boundary

This run establishes that the frozen 400-row Qwen population was successfully scored in one production invocation window by the frozen causal judge and is mechanically drawable by T26. It does not show that the judge is valid against human labels, and it does not establish Assistant-Axis causal specificity.
