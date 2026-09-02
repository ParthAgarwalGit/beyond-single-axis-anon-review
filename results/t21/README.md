# T21 results

Canonical artifacts for the T21 source-model Qwen activation-capping reproduction.

- `t21_final_report.json`: single closure-level result used for paper integration.
- `source_capping_baseline_report_source_max_v5_novita.json`: safety-side reconciliation report.
- `t21_full_capability_report.json` and `t21_full_capability_summary.csv`: full 6,224-generation capability battery.
- `t21_capability_uncertainty.json`: aggregate-only uncertainty sidecar for capability deltas and the limits of reconstructing paired/continuous-score uncertainty from committed summaries.
- `t21_safety_table_source_max_v5_novita.csv`: safety table.
- `t21_source_harm_judge_provenance_source_max_v5_novita.json`: source-max judge provenance and public-source boundary.
- `ARTIFACT_MANIFEST.json`: pointers/counts/hashes for reviewed and external artifacts.
- `T21_CLOSURE_VALIDATION.json`: static closure attestation; it records reviewed checks and does not recompute metrics from the external raw JSONLs.

Large JSONL generation/judgment checkpoints remain on persistent storage and are not committed.
