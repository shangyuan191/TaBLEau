# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 60.39 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 61.58 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 95.93 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 95.97 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 102.61 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 104.96 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 80.10 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 81.97 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 101.47 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 102.70 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 110.02 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 110.90 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 66.90 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 67.03 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 67.43 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 68.38 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 74.12 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 87.21 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 49.79 | No | No | 2/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| materialize | 50.60 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 54.65 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 55.07 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 57.79 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 94.64 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 86.66 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 96.96 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 98.29 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 99.59 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 103.43 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 103.59 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 68.79 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 76.50 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 78.21 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 79.94 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 84.97 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 90.82 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 62.87 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 62.91 | Yes (tie) | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 72.68 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 73.34 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 111.49 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 113.70 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 84.20 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 84.94 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 85.35 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 86.62 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 89.56 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 96.07 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 54.25 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 56.76 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 58.03 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 69.48 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 71.38 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 101.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 80.29 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 80.29 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 80.83 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 83.32 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 88.93 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 89.96 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

