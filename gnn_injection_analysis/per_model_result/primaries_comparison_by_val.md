# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 59.61 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 60.74 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 94.82 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 94.93 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 101.68 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 104.04 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 78.89 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 80.74 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 100.49 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 101.89 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 109.22 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 110.24 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 71.08 | Yes | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 72.50 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 84.41 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 91.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 102.43 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 107.49 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 49.02 | No | No | 2/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| materialize | 49.82 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 53.78 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 54.24 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 56.87 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 93.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 85.34 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 95.98 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 97.28 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 98.59 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 102.38 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 102.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 67.96 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 75.44 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 77.23 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 79.14 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 83.99 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 89.91 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 62.06 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 62.12 | Yes (tie) | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 71.44 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 72.20 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 110.80 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 113.08 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 83.25 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 83.90 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 84.36 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 85.63 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 88.54 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 95.09 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 53.58 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 56.22 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 57.38 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 68.55 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 70.39 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 100.05 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 79.17 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 79.22 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 79.72 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 82.13 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 87.62 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 88.60 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

