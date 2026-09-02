# Novita routing correction

Date: 2026-08-23

The frozen T27 judge identity remains `deepseek-ai/DeepSeek-V3`. The first production-scoring attempt passed that Hugging Face-style identifier directly to Novita and received `MODEL_NOT_FOUND` on every request, so it produced no usable automatic labels.

A one-request provider probe confirmed that Novita serves the intended model family under routing identifier `deepseek/deepseek_v3`.

This patch therefore keeps the scientific judge identity unchanged and uses `deepseek/deepseek_v3` only as the provider routing identifier. The scoring report records both `judge_model` and `provider_model_id`.

The patch also closes the failure mode exposed by the first attempt: a run with zero usable production labels, or an unusable start/end canary, is non-production and non-drawable. T26 independently rejects legacy reports in which every production row is a coverage failure.

## Partial-coverage policy

The canary usability check is intentionally a total-outage detector rather than a new post-hoc quality threshold. A canary is considered usable when it contains at least one row with both harmfulness and identity parsed successfully.

Partial production coverage failure does not by itself make a run non-production. The frozen T26 workflow preserves such failures explicitly in the `PARSE_FAILURE` auto-harmfulness stratum, and rare strata are included in full for human validation rather than silently discarded. Coverage-failure counts and request-error attempts are still recorded in the scoring report for auditability.

This choice avoids inventing a new partial-coverage cutoff after observing the failed run. If the corrected production run shows material partial degradation, that degradation must be reported and carried through T26/T27 validation rather than filtered away.

No prompt, schema, decoding temperature, validation threshold, sampling rule, or causal estimand is changed. The corrected scorer must still pass T26/T27 before downstream inference.
