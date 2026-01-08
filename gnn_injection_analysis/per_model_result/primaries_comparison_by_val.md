# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 59.26 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 60.29 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 95.11 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 95.43 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 102.01 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 104.37 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 79.12 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 80.98 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 101.23 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 102.02 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 109.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 110.51 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 71.05 | Yes | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 72.36 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 84.64 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 91.66 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 102.92 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 107.91 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 49.15 | No | No | 2/3 | 0/3 | 1/2 | 1/2 | Yes (tie) | No |
| materialize | 50.03 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 53.87 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 54.37 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 57.00 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 93.47 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 85.66 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 96.16 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 97.48 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 98.66 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 102.52 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 102.52 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 67.72 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 75.53 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 77.35 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 79.38 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 84.09 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 89.93 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 62.41 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 62.66 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 65.09 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 65.14 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 99.38 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 105.80 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 83.00 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 83.48 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 86.09 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 87.68 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 98.66 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 102.59 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 58.01 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 59.39 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 60.94 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 64.82 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 70.41 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 90.37 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 79.22 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 79.28 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 79.77 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 82.12 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 87.71 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 88.57 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

