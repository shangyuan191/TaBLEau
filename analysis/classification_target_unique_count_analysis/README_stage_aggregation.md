# Stage-aggregated summary by target-value count (classification)

This report summarizes how GNN-injection stages perform across classification datasets grouped by target-value count (n_unique).

Source files
- Aggregated CSV: `analysis/classification_target_unique_count_analysis/stage_aggregated_by_nunique_avg_ratios.csv` (rows = GNN-injection stage; columns = n_unique buckets; values = mean rank across models, lower is better).
- Per-dataset rankings: `analysis/classification_target_unique_count_analysis/rankings_per_dataset.csv`.

Method (brief)
- For each (model, stage) we took all available split ratios (fully-trained and few-shot) and averaged their per-dataset ranks within the same n_unique bucket, producing one per-model-stage value per n_unique. 
- Then for each stage we averaged those per-model-stage values across models to obtain a stage × n_unique mean-rank table. Each model contributes at most one value per stage per n_unique.
- Ranks are "lower is better" (these are ranking positions computed from per-dataset performance comparisons).

Aggregated table (stage × n_unique means)

| stage | n_unique=2 | n_unique=3 | n_unique=4 | n_unique=100 |
|---|---:|---:|---:|---:|
| columnwise | 50.4055 | 42.7000 | 41.2500 | 44.0500 |
| decoding | 53.2055 | 63.5000 | 60.2000 | 64.9000 |
| encoding | 49.2473 | 43.0000 | 40.6000 | 42.1000 |
| materialize | 48.2218 | 51.1500 | 55.6000 | 50.5000 |
| start | 46.5673 | 51.1500 | 54.7500 | 50.8500 |

(Values are mean ranks across models; lower = better.)

Key findings

- Per-bucket best stage (lowest mean rank):
  - n_unique = 2: `start` (mean rank ≈ 46.57)
  - n_unique = 3: `columnwise` (mean rank = 42.70)
  - n_unique = 4: `encoding` (mean rank = 40.60)
  - n_unique = 100: `encoding` (mean rank = 42.10)

- `decoding` is consistently the worst-performing stage across all n_unique buckets in this aggregation (largest mean ranks). 

- `encoding` performs well for larger/complex label spaces (n_unique 4 and 100). `columnwise` has an advantage in the n_unique=3 bucket. `start` shows the best mean for n_unique=2. `materialize` and `start` tend to be middling-to-poor for larger n_unique in this summary.

Interpretation and caveats

- This is a model‑level aggregated summary: each model contributes one averaged value per (stage, n_unique). Differences can be driven by a few models; the aggregation hides per-model variability.

- We averaged fully-trained and few-shot ratios in the same per-model-stage value. If you prefer to weigh these settings differently (for example, give more weight to fully-trained results), rerun with a different aggregation.

- Mean rank differences should be tested for statistical significance before claiming robust effects. The current table indicates practical differences (for example, difference between encoding and decoding can be large), but p-values / effect sizes are needed to be confident.

Suggested next steps

- Visualize: create boxplots (per-model values) or grouped bar charts with error bars for each n_unique to show variance across models.
- Statistical testing: for each n_unique, run paired tests (Wilcoxon signed-rank or permutation tests) comparing pairs of stages (e.g., `encoding` vs `columnwise`) using per-model values.
- Per-model diagnostics: output a table listing, for each model, which stage is best for each n_unique and the gain vs second best (to see whether aggregated effects are driven by many models or a few).

Files created
- `analysis/classification_target_unique_count_analysis/stage_aggregated_by_nunique_avg_ratios.csv`

If you want, I can now:
- Produce the grouped bar charts / boxplots and save PNGs to `analysis/classification_target_unique_count_analysis/figures/`.
- Run paired permutation/Wilcoxon tests and add a p-value table to this report.
- Produce a per-model table (best stage per model per n_unique) and attach it.

Tell me which follow-up you prefer and I'll implement it next.
