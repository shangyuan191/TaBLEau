# Stage-aggregated summary for regression target distributions

This short report summarizes the stage-level aggregation of per-model ranks across regression target distributions.

Source data
- Aggregated CSV: `analysis/regression_target_distributions_analysis/regression_stage_aggregated_by_model_avg_ratios.csv` (rows = GNN-injection stage; columns = distribution averages; each cell is the mean rank across models for that stage & distribution).
- Original pivot: `analysis/regression_target_distributions_analysis/regression_avg_rank_pivot.csv`
- Per-dataset ranks: `analysis/regression_target_distributions_analysis/regression_ranks_per_dataset.csv`

Method (brief)
- For each (model, stage) we averaged the two split ratios (if both present) to produce one per-model-stage vector of average ranks across distributions.
- Then, for each stage we averaged those per-model-stage values across models to get a stage × distribution matrix (this gives equal weight to each model).
- Ranks are "lower is better" (these are average ranks computed earlier from per-dataset rankings).

Aggregated table (stage × distribution means)

| stage | model_count | approx_normal (11) | constant (3) | heavy_tailed (9) | highly_skewed (12) | moderately_skewed (9) | multimodal (14) |
|---|---:|---:|---:|---:|---:|---:|---:|
| columnwise | 10 | 49.0727 | 44.6500 | 49.6667 | 52.1417 | 50.4000 | 49.8571 |
| decoding | 10 | 57.2591 | 54.4000 | 56.3944 | 56.9000 | 53.3167 | 59.4464 |
| encoding | 10 | 46.9955 | 45.7167 | 49.8167 | 52.3083 | 48.1222 | 49.8357 |
| materialize | 10 | 49.3364 | 54.2500 | 47.9389 | 45.9917 | 50.4000 | 46.3214 |
| start | 10 | 49.8000 | 53.3500 | 48.6833 | 45.1167 | 50.1944 | 47.0179 |

(Values are mean ranks across models; parentheses after distribution names indicate number of datasets in that bucket.)

Key findings
- No single GNN injection stage dominates all distribution types. Different stages are better for different target distributions:
  - `encoding` is best for `approx_normal` and `moderately_skewed`.
  - `materialize` is best for `heavy_tailed` and `multimodal`.
  - `columnwise` is best for `constant`.
  - `start` is best for `highly_skewed`.
  - `decoding` is the worst performer on average across these distributions (it is not best for any distribution and has the largest mean ranks).
- The spread (best vs worst stage) per distribution ranges from ~5.2 to ~13.1 mean-rank units, showing stage choice can materially affect average rank on some distributions (e.g., `multimodal` spread ≈ 13.1).

Interpretation notes and caveats
- This is an aggregate, model-weighted-equally summary (each model contributes one averaged entry per stage). It hides within-model variability: some models may benefit strongly from a stage while others do not.
- Averaging the two split ratios gives equal weight to fully-trained and few-shot settings. If you prefer to treat fully- and few-shot differently, consider alternative aggregation strategies (e.g., keep separate or weight by dataset count).
- The ranking values are summary ranks — they do not directly translate to metric units (RMSE, etc.). Statistical tests should be used to claim significance of differences.

Recommended next steps
- Visualize: create a grouped bar chart or heatmap of the table above (stage on y, distribution on x) with error bars from per-model values.
- Significance testing: for each distribution, run pairwise paired tests (e.g., Wilcoxon signed-rank or permutation tests) across models to compare stages (e.g., `encoding` vs `columnwise`).
- Per-model diagnostics: produce a table showing, for each model, the best stage per distribution and the gain vs second-best.

Files created in this analysis
- `analysis/regression_target_distributions_analysis/regression_stage_aggregated_by_model_avg_ratios.csv` (stage × distribution aggregated means)

How to reproduce (quick)

A short script was used to compute the aggregation from `regression_avg_rank_pivot.csv`. To reproduce quickly you can run a small Python snippet similar to the one used in this analysis (reading the pivot, averaging ratios per model+stage, then averaging across models per stage).

If you want, I can now:
- Produce the grouped bar charts / heatmap and save to `analysis/figures/`.
- Run paired permutation/Wilcoxon tests and attach a table of p-values and effect sizes.
- Produce a per-model report showing which stage is best for each model & distribution.

Tell me which of the above you want next and I'll implement it and attach results to this folder.
