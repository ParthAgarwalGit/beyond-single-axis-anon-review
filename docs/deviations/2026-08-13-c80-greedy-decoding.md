# C80 confirmatory generation used greedy decoding

## Date
2026-08-13

## Recorded settings
Executed C80-A and C80-B translated-arm generations were decoding-matched to
each other using greedy generation:
- `temperature = 0.0`
- `do_sample = false`
- `seed = 0`

The accepted existing E80/T09 settings used for E80 were:
- `temperature = 0.6`
- `top_p = 0.95`
- `max_new_tokens = 2560`

Therefore C80 is not decoding-identical to E80.

## Scientific handling
1. C80-A versus C80-B is the primary untouched confirmatory comparison because
   the two C80 halves share the same executed decoding regime.
2. E80 remains exploratory/development evidence.
3. Any full-240 analysis combining E80 and C80 must retain decoding-stratum
   provenance and must not be described as if all 240 IDs used identical
   generation settings.
4. No C80 outputs are regenerated solely to erase this historical difference.
5. Causal generation settings are frozen separately before the causal run.

## Retry-budget provenance
C80-A used the recorded escalation 2048 -> 8192 -> 16384.

The accumulated C80-B raw artifact contains multiple retry passes. Its counter
extends through 4. In the 15 histories with retry >= 3, retry 3 records 2048
tokens and retry 4 records 8192. Token budget must therefore be read from row
provenance rather than inferred from retry index.

## Claim boundary
C80-A/C80-B supports a decoding-matched confirmatory comparison. E80-to-C80 is
not a pure question-block comparison because decoding differs.
