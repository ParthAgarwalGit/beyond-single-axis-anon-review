# Prefilled `<think>` breaks reasoning segmentation

## Date
2026-08-08

## Issue

`configs/method_frozen.yaml` `response_segmentation` classifies a completion by
counting `<think>` / `</think>` markers **in the model output** and requires a
balanced pair (`balanced_reasoning`). It treats an orphan `</think>` (a close
with no open) as `malformed_reasoning`, which is not eligible for role judging.

DeepSeek-R1-Distill's chat template **prefills the opening `<think>`** into the
prompt via `add_generation_prompt=True` — every rendered prompt ends with
`…<｜Assistant｜><think>`. The model therefore generates *inside* the reasoning
block, and a well-formed output is `reasoning</think>answer`: **zero `<think>`,
one `</think>`.**

Consequence, observed on the real C80-B run (22,000 outputs): **100% of
otherwise-valid outputs were labelled `malformed_reasoning`** → 0 role-judgeable
outputs. The raw generations were correct; only the case-classification was
wrong. The method's own `reasoning_mask` definition ("response start through the
end of the final `</think>`") already assumes the prefilled open, so the
segmentation *code* was inconsistent with the frozen mask spec.

Secondary: because `<think>`/`</think>` are single tokens (128013 / 128014), a
token-based boundary is both authoritative (per `response_tokenization`) and
robust to detokenize→retokenize round-trip differences — which is what produced
the vLLM-path `token_alignment_failure` rows.

## Correction

`src/generation.py`:
- `segment_response` / `classify_validity` gain `prefilled_open` and
  `close_token_id`. When `prefilled_open` is set, the response is treated as
  starting inside the reasoning block: reasoning = response start → end of the
  final `</think>` (located by `close_token_id`), answer = tokens after it.
  No output `<think>` and one `</think>` → `balanced_reasoning`; no `</think>` →
  unclosed → `malformed_reasoning`; empty answer → `empty_answer_after_close`.
- Generation workers detect the prefilled open from the rendered prompt
  (`rendered.rstrip().endswith("<think>")`) and pass the single-token
  `</think>` id, so no runtime override of a frozen field is required.

Default behaviour (`prefilled_open=False`) is unchanged; all prior tests pass.
New tests: `tests/test_segmentation_prefilled.py`.

## Scientific impact

No question IDs, prompts, model outputs, or activations changed. This corrects
the **segmentation labelling** only. Re-segmenting the existing C80-B run with
the fix recovers **21,769 / 22,000 rollouts (98.95%)** as role-judgeable
(`balanced_reasoning`); the remaining 231 are `truncated` at the 2048 token cap
and require regeneration at a larger budget, not a segmentation change.

## Follow-up for T02 owner

`method_frozen.yaml response_segmentation` should explicitly declare
`prompt_prefills_opening_marker: true` for the primary model so the behaviour is
frozen in the spec rather than only auto-detected. This deviation records the
code correction ahead of that config amendment.
