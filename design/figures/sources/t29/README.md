# T29 causal figure/table sources

These files are aggregate-only sources for the completed Qwen T29 causal-specificity result.

- `t29_human_primary_condition_rates.csv`: design-weighted T26 human rates. This is the primary harmfulness evidence under T27 `MEASUREMENT_LIMITED`.
- `t29_human_primary_contrasts.csv`: frozen human-primary and secondary/context contrasts.
- `t29_automatic_sensitivity_condition_rates.csv`: full 400-row automatic-label condition rates, including refusal and degeneration diagnostics.
- `t29_automatic_sensitivity_contrasts.csv`: paired full-corpus automatic-label contrasts. These are secondary sensitivity evidence only.
- `t29_primary_vs_automatic_axis_random.csv`: compact source for a figure/table that juxtaposes the primary human Axis-vs-random contrast with the secondary automatic sensitivity without conflating their evidential status.
- `t29_frozen_engagement_matching.csv`: outcome-blind T24 calibration diagnostic showing that the primary random and spherical controls match the measured source-cap engagement rate at every intervention layer. It records intervention incidence, not intervention magnitude or geometric equivalence.

The preferred T29 figure should foreground the human-primary estimate and label the automatic full-corpus result as secondary. Do not plot the automatic result as an equally validated replicate. The T27 harmfulness gate failed on IP-weighted recall and macro-F1, so automatic harmfulness cannot become primary evidence.

The private scored 400-row JSONL, item IDs, completions, judge rationales, and item-level labels joined to condition are not stored here.
