# T07 reproducibility record

This note freezes what can be reproduced from the repository today and explicitly records the one private artifact that is still external.

## Public, committed artifacts

- Pull request: **#37** (`t06-t07/measurement-metrics`).
- Scorer: `tools/run_t07_judge_validation.py`.
- Canonical reviewed scorer commit: `07bf0d34af5f8132fe1550ae29111be6c475aa76`.
- SHA-256 of the scorer at that commit: `314d0faf07d4c36ce66a61a9fed9fc00d2f087e4e331c083e88f1f091d7df5d7`.
- Final human consensus: `results/t06/gold_consensus.csv`.
- SHA-256 of final `gold_consensus.csv`: `b67d10e683d888948e42762b9b35829ed10c736d818b0bebcfab632ca66cfc44`.
- Committed T07 aggregate: `results/t07/judge_validation.json`.
- SHA-256 of the committed aggregate: `5394754a92da31957f4204bce3a0ac429404397ca2ece91162f7c6c41f8c012d`.

The committed aggregate records `source_git_sha=49b7464878c55a4760536596827af18c10a3c728`, `source_git_dirty=true`, and `working_tree_diff_sha256=be63bdec67d376aa0b3f71e6131241654c9376cf904387c789ea6d3a32e49c3f`. Because that historical run came from a dirty working tree, the source commit alone must not be described as a byte-for-byte reconstruction of the exact executed working tree. The current reviewed scorer is therefore pinned independently by commit and SHA-256 above.

## Private sampling key

The required private artifact is:

`annotations/t14_gold_v3/sampling_key_PRIVATE.json`

It contains de-blinding fields required for the data-bearing T07 calculation, including archived automatic scores, arm membership, inverse-probability weights, and sensitivity mappings. It must **never** be committed to the repository or exposed to annotators.

The repository already ignores both `sampling_key_PRIVATE.json` and the broader `*_PRIVATE.json` pattern. A clean repository checkout was also checked and the private key was absent, as intended.

At the time of this record, the original private file itself was not available through the connected repository or File Library. Therefore its SHA-256 cannot be truthfully filled in yet, and this repository does **not** claim that the private artifact has been secured in a team-controlled store. The matching field in `results/t07/provenance_manifest.json` is deliberately `null` rather than invented.

When the original file is recovered, preserve the exact bytes in a team-controlled private location and compute:

```bash
sha256sum annotations/t14_gold_v3/sampling_key_PRIVATE.json
```

Then update only the `private_sampling_key.sha256`, `hash_status`, and `secure_team_storage_status` fields in `results/t07/provenance_manifest.json`. Do not regenerate a replacement key and describe it as the historical original.

## Exact T07 reproduction command

Run from the repository root with the original private key restored locally at the ignored path above:

```bash
python tools/run_t07_judge_validation.py run \
  --consensus results/t06/gold_consensus.csv \
  --key annotations/t14_gold_v3/sampling_key_PRIVATE.json \
  --sensitivity annotations/t14_gold_v3/sensitivity_labeller_1.csv \
  --out results/t07
```

Before accepting a regenerated aggregate, verify the final gold hash and private-key hash against the provenance manifest and compare the regenerated T07 metrics with the committed aggregate.

## Historical judge-model boundary

The archived production reference identifies the judge model as `Qwen/Qwen2.5-32B-Instruct-AWQ`, but an exact historical Hugging Face model revision was not recorded in the recoverable artifacts. We therefore describe T07 as **validation of archived production judge scores against adjudicated human gold**. We do not claim that T07 is a fresh bit-for-bit rerun of a revision-pinned judge model, and we do not substitute a current Hub revision for the missing historical one.
