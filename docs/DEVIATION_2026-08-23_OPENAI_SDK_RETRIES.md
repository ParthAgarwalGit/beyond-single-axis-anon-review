# 2026-08-23 transport correction: disable implicit OpenAI SDK retries

## Problem found before the next Qwen production rerun

PR #62 made the repository scorer pace each explicit hosted request attempt with a 7.0 second minimum start-to-start interval and retained exactly two scorer-level attempts per item. During the pre-production audit after merge, we found one remaining transport ambiguity: the OpenAI Python client was constructed without an explicit `max_retries` value.

The scorer already implements its own retry loop, pacing, backoff, raw-attempt retention, and attempt counting. Allowing the client library to retry internally could therefore create hosted HTTP attempts that are not separately paced or visible to the scorer. That would weaken both the provider-rate-limit fix and the intended maximum of two scorer-controlled attempts per item.

## Correction

The Novita client is now constructed with:

```python
OpenAI(..., max_retries=0)
```

This makes the scorer's existing two-attempt loop the only retry authority. Every hosted attempt therefore passes through the shared request pacer, is eligible for the frozen 429/generic backoff, and is represented in the raw audit record.

The scoring report now records `transport.sdk_automatic_retries_disabled = true`, and a regression test fails if client construction no longer passes `max_retries=0`.

## Scientific boundary

This is transport/provenance hardening only. It does not change:

- scientific judge identity (`deepseek-ai/DeepSeek-V3`);
- Novita provider route (`deepseek/deepseek_v3`);
- prompt or output schema;
- temperature;
- the scorer-level maximum of two attempts per item;
- T26 sampling;
- T27 thresholds or branch logic;
- canary drift rules;
- the T29 causal estimand.

Neither failed earlier Qwen scoring run becomes usable as a result of this correction. The next production run must use a fresh clean checkout containing this fix and a new output directory.
