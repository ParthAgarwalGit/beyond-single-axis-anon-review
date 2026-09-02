# T29 Qwen matched-control execution

## Scope

This records completion of the two Qwen3-32B matched control conditions required for the source-model causal-specificity branch. It is an execution/provenance record only; it does not contain causal-judge outcomes and does not establish the final T29 causal claim.

The completed four-condition population is:

- `unsteered_baseline` — 100 rows, reused exactly from the recovered T21 source run;
- `assistant_cap` — 100 rows, reused exactly from the recovered T21 source run;
- `random_direction_cap` — 100 newly generated rows using the frozen orthogonal random direction;
- `sphere_proximity_cap` — 100 newly generated rows using the frozen spherical-distance control.

Total: 400 unique rows, with the same 100 evaluation items represented in every condition.

## Frozen generation protocol

- Model: `Qwen/Qwen3-32B`
- Model/tokenizer revision: `9216db5781bf21249d130ec9da846c4624c16137`
- Precision: BF16
- Thinking mode: disabled
- Layers: 46–53 inclusive
- Sampling: `do_sample=True`
- Temperature: `0.7`
- Top-p: `0.9`
- `max_new_tokens=512`
- Retry budget: 2
- Selection seed: `210816`
- Item seed: `210816 * 100000 + manifest_index`, held fixed across conditions
- Checkpointing: atomic per-item Drive shard

This preserves the original T21 production generation regime rather than substituting the T24 calibration decoding settings.

## Frozen provenance bindings

- T21 frozen evaluation manifest SHA-256: `d611e904f3b5d47c71e5ab1f3fa1a84ead6cfd5d94dba337f5c3b71aafef7793`
- Recovered T21 baseline-generation SHA-256: `6e3158622e8467199c8845b7bacd8cc70f99173587e51b5a7fc63bfb989634a4`
- T24 canonical numerical-controls NPZ SHA-256: `adf1c43cd2715676169351f69fced45d0046e70c5a74f52baa2dc5f2aa243291`
- T24 freeze SHA-256: `08fb4bb67b366c81ddca33835fb97e130db49efb56a4cece64c0cd63dc6a99c4`
- Released Assistant-Axis SHA-256: `a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211`
- Source capping-config SHA-256: `6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24`
- T24 freeze merge commit used by the matched-control notebook: `20915157c906ab320765563f449d2655b7a9026c`

## Exact execution receipt

The generated receipt records:

- created UTC: `2026-08-22T14:39:07.619851+00:00`
- status: `MATCHED_CONTROLS_COMPLETE_SCORING_READY`
- new matched controls: 200 rows, SHA-256 `66ef447ef035a35eb0fb9502949dc3412fe2e540de0e20abba246cd71283f582`
- complete four-condition population: 400 rows, SHA-256 `24a0c5a3699fd6b51a66d4be92bc054d14cb800db894c0162b65774c5667f06b`
- execution receipt SHA-256: `8539aa603e35f1ffb373ceff0e6f5435a9c36187e0757385b0aa8ad79ee2ecac`
- causal-scorer dry-run: passed

These hashes were recorded directly from the produced execution artifacts and identify the exact generation bytes.

## Immutable artifact archive

The generation artifacts are no longer Drive-only. The 400-row population, 200-row new-control file, and execution receipt were archived together in the Hugging Face dataset `[Author-B-HF]/[anonymized-repository-name]-artifacts` under `t29-qwen-matched-controls/`.

Immutable revision:

`e29553fe75a7d89757258e7c6c38723de1960ca9`

The archive was downloaded again from that exact revision and verified byte-for-byte against the local execution artifacts:

- `t29-qwen-matched-controls/qwen_capping_400_generation_population.jsonl`
  - SHA-256: `24a0c5a3699fd6b51a66d4be92bc054d14cb800db894c0162b65774c5667f06b`
  - round-trip verified: yes
- `t29-qwen-matched-controls/t29_qwen_new_matched_controls_200.jsonl`
  - SHA-256: `66ef447ef035a35eb0fb9502949dc3412fe2e540de0e20abba246cd71283f582`
  - round-trip verified: yes
- `t29-qwen-matched-controls/T29_QWEN_MATCHED_CONTROL_EXECUTION_RECEIPT.json`
  - SHA-256: `8539aa603e35f1ffb373ceff0e6f5435a9c36187e0757385b0aa8ad79ee2ecac`
  - round-trip verified: yes

The archive verification was completed at `2026-08-23T04:57:30.030849+00:00`. Its repo-side receipt is `results/t29/T29_QWEN_MATCHED_CONTROL_HF_ARCHIVE_RECEIPT.json`, SHA-256 `a03b56a84b03b2a96bdd0566b735e7089cf22fd4bb0af48cd886f3eac93b97e5`.

The original Drive paths remain recorded as execution locations, but the immutable Hugging Face revision is the durable retrieval authority for downstream scoring and validation.

The production notebook ends with `MATCHED_CONTROLS_COMPLETE_SCORING_READY` only after the 400-row panel satisfies the 400-total / 100-per-condition production contract and the causal scorer dry-run passes without judge API calls.

## Claim boundary

This closes matched-control generation and artifact durability only. It does not establish causal specificity. Causal scoring, the T26 blinded draw, two independent human labels, the T27 validation gate, and final T29 inference remain separate downstream stages.
