# Cross-model GNN injection aggregation — consolidated observations
This report summarizes and synthesizes the cross-model aggregation tables in `gnn_injection_crossmodel_aggregation.md`.

## Quick global summary

| Injection stage | total wins (sum over all categories & refs) | total opportunities | overall % |
|---|---:|---:|---:|
| columnwise | 245 | 800 | 30.6% |
| decoding | 208 | 800 | 26.0% |
| encoding | 223 | 800 | 27.9% |
| materialize | 197 | 800 | 24.6% |
| none | 135 | 800 | 16.9% |
| start | 198 | 800 | 24.8% |

## Best stage per reference (aggregated across categories)

For each reference comparator (e.g., `beats few-shot-non-gnn`), this table shows which injection stage produced the most wins summed across all categories.

| Reference | Best stage | wins | opportunities | pct |
|---|---|---:|---:|---:|
| beats few-shot GNN | columnwise | 44 | 100 | 44.0% |
| beats few-shot tabpfn | encoding | 14 | 100 | 14.0% |
| beats few-shot trees | columnwise | 48 | 100 | 48.0% |
| beats few-shot-non-gnn | decoding | 16 | 100 | 16.0% |
| beats full GNN | columnwise | 40 | 100 | 40.0% |
| beats full tabpfn | columnwise | 25 | 100 | 25.0% |
| beats full trees | columnwise | 44 | 100 | 44.0% |
| beats full-non-gnn | columnwise | 19 | 100 | 19.0% |

## Per-category highlights

### large_datasets+binclass+numerical (6 datasets) (10 models parsed)
- Best stage by aggregated wins: **columnwise** (12/80 = 15.0% of opportunities)

| stage | wins | opportunities | pct |
|---|---:|---:|---:|
| columnwise | 12 | 80 | 15.0% |
| encoding | 7 | 80 | 8.8% |
| decoding | 6 | 80 | 7.5% |
| none | 6 | 80 | 7.5% |
| start | 3 | 80 | 3.8% |
| materialize | 3 | 80 | 3.8% |

### large_datasets+multiclass+numerical (3 datasets) (10 models parsed)
- Best stage by aggregated wins: **encoding** (8/80 = 10.0% of opportunities)

| stage | wins | opportunities | pct |
|---|---:|---:|---:|
| encoding | 8 | 80 | 10.0% |
| columnwise | 6 | 80 | 7.5% |
| none | 5 | 80 | 6.2% |
| start | 1 | 80 | 1.2% |
| materialize | 0 | 80 | 0.0% |
| decoding | 0 | 80 | 0.0% |

### large_datasets+regression+categorical (1 dataset) (10 models parsed)
- Best stage by aggregated wins: **decoding** (44/80 = 55.0% of opportunities)

| stage | wins | opportunities | pct |
|---|---:|---:|---:|
| decoding | 44 | 80 | 55.0% |
| materialize | 38 | 80 | 47.5% |
| encoding | 37 | 80 | 46.2% |
| columnwise | 33 | 80 | 41.2% |
| start | 31 | 80 | 38.8% |
| none | 17 | 80 | 21.2% |

### large_datasets+regression+numerical (10 datasets) (10 models parsed)
- Best stage by aggregated wins: **columnwise** (44/80 = 55.0% of opportunities)

| stage | wins | opportunities | pct |
|---|---:|---:|---:|
| columnwise | 44 | 80 | 55.0% |
| start | 41 | 80 | 51.2% |
| materialize | 41 | 80 | 51.2% |
| encoding | 38 | 80 | 47.5% |
| decoding | 35 | 80 | 43.8% |
| none | 18 | 80 | 22.5% |

### small_datasets+binclass+balanced (14 datasets) (10 models parsed)
- Best stage by aggregated wins: **columnwise** (5/80 = 6.2% of opportunities)

| stage | wins | opportunities | pct |
|---|---:|---:|---:|
| columnwise | 5 | 80 | 6.2% |
| none | 3 | 80 | 3.8% |
| encoding | 2 | 80 | 2.5% |
| decoding | 1 | 80 | 1.2% |
| start | 0 | 80 | 0.0% |
| materialize | 0 | 80 | 0.0% |

### small_datasets+binclass+categorical (7 datasets) (10 models parsed)
- Best stage by aggregated wins: **start** (7/80 = 8.8% of opportunities)

| stage | wins | opportunities | pct |
|---|---:|---:|---:|
| start | 7 | 80 | 8.8% |
| encoding | 7 | 80 | 8.8% |
| columnwise | 7 | 80 | 8.8% |
| none | 7 | 80 | 8.8% |
| materialize | 1 | 80 | 1.2% |
| decoding | 0 | 80 | 0.0% |

### small_datasets+binclass+numerical (28 datasets) (10 models parsed)
- Best stage by aggregated wins: **columnwise** (11/80 = 13.8% of opportunities)

| stage | wins | opportunities | pct |
|---|---:|---:|---:|
| columnwise | 11 | 80 | 13.8% |
| encoding | 7 | 80 | 8.8% |
| start | 4 | 80 | 5.0% |
| none | 4 | 80 | 5.0% |
| materialize | 2 | 80 | 2.5% |
| decoding | 2 | 80 | 2.5% |

### small_datasets+regression+balanced (6 datasets) (10 models parsed)
- Best stage by aggregated wins: **encoding** (40/80 = 50.0% of opportunities)

| stage | wins | opportunities | pct |
|---|---:|---:|---:|
| encoding | 40 | 80 | 50.0% |
| decoding | 40 | 80 | 50.0% |
| columnwise | 39 | 80 | 48.8% |
| materialize | 36 | 80 | 45.0% |
| start | 34 | 80 | 42.5% |
| none | 24 | 80 | 30.0% |

### small_datasets+regression+categorical (5 datasets) (10 models parsed)
- Best stage by aggregated wins: **columnwise** (43/80 = 53.8% of opportunities)

| stage | wins | opportunities | pct |
|---|---:|---:|---:|
| columnwise | 43 | 80 | 53.8% |
| start | 41 | 80 | 51.2% |
| encoding | 41 | 80 | 51.2% |
| materialize | 40 | 80 | 50.0% |
| decoding | 38 | 80 | 47.5% |
| none | 27 | 80 | 33.8% |

### small_datasets+regression+numerical (36 datasets) (10 models parsed)
- Best stage by aggregated wins: **columnwise** (45/80 = 56.2% of opportunities)

| stage | wins | opportunities | pct |
|---|---:|---:|---:|
| columnwise | 45 | 80 | 56.2% |
| decoding | 42 | 80 | 52.5% |
| start | 36 | 80 | 45.0% |
| materialize | 36 | 80 | 45.0% |
| encoding | 36 | 80 | 45.0% |
| none | 24 | 80 | 30.0% |