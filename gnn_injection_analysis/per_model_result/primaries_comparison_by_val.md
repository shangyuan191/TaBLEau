# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 61.14 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 62.33 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 97.21 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 97.26 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 103.78 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 106.13 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| start | 68.07 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 69.11 | Yes | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 76.18 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 76.24 | Yes (tie) | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 78.45 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 107.39 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 67.87 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 68.01 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 68.58 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 69.45 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 75.52 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 88.21 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 50.63 | No | No | 2/3 | 0/3 | 1/2 | 1/2 | Yes (tie) | No |
| materialize | 51.40 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 55.69 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 56.01 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 58.85 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 96.09 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 88.11 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 98.08 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 99.44 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 100.84 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 104.64 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 104.82 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 69.88 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 77.79 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 79.48 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 80.90 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 86.18 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 92.16 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 63.83 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 63.84 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 74.03 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 74.59 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 112.36 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 114.59 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 85.47 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 86.19 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 86.47 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 87.76 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 91.09 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 97.29 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 55.02 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 57.51 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 58.88 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 70.65 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 72.59 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 102.22 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 81.59 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 81.63 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 82.14 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 84.60 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 90.37 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 91.45 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

