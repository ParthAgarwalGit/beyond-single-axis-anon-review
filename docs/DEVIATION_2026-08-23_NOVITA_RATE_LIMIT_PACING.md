# Novita request-rate pacing correction

Date: 2026-08-23

The corrected-routing Qwen causal-scoring run after PR #60 reached the intended Novita DeepSeek-V3 route, but Novita enforced an explicit transport quota of `10 requests per minute`. The scorer sent requests back-to-back and waited only two seconds after exceptions, producing 129/400 production coverage failures and 338 request-error attempts. The failures were distributed almost uniformly across the four Qwen conditions (32, 32, 33, 32).

The start/end canary changed on 25/50 items, but the diagnostic decomposition showed no parsed-label disagreements. The changes were entirely availability transitions (`PARSED -> FAIL` or `FAIL -> PARSED`) caused by HTTP 429 `RATE_LIMIT_EXCEEDED` errors. The run therefore remains non-production and non-drawable, and it must not feed T26/T27.

## Operational correction

The scorer now enforces a minimum 7-second start-to-start interval across every hosted judge attempt, including start canary, production rows, retries, and end canary. This corresponds to about 8.6 requests per minute, below the observed 10 requests/minute provider ceiling. A 429 rate-limit exception receives an additional 15-second backoff before the existing single retry; other request exceptions retain the existing 2-second backoff.

The scoring report records the observed provider limit, request interval, rate-limit backoff, generic error backoff, and the number of rate-limit error attempts.

## Method boundary

This is a provider-transport correction only. It does not change:

- scientific judge identity (`deepseek-ai/DeepSeek-V3`),
- Novita provider route (`deepseek/deepseek_v3`),
- frozen judge prompt or schema,
- temperature or maximum output,
- number of per-row scoring attempts,
- T26 sampling rule,
- T27 validation thresholds,
- branch-selection logic,
- causal estimand,
- canary drift gate.

The canary gate is deliberately not weakened. After pacing is applied, any start/end parsed-label disagreement or other canary mismatch still makes the run non-production/non-drawable under the existing scorer logic.

## Rerun requirement

After this patch is reviewed and merged, Qwen causal scoring must be rerun from a fresh clean checkout of the new `main` into a new output directory. Neither the earlier `MODEL_NOT_FOUND` run nor the 429-limited run may be reused as production evidence.
