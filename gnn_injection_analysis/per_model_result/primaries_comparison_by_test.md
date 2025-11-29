# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 63.21 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 64.47 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 67.29 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 74.89 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 81.93 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 86.54 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| start | 81.94 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 84.72 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 90.62 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 91.55 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 94.53 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 110.28 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 74.22 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 76.60 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 80.37 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 80.44 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 80.53 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 96.28 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 66.96 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 68.21 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 69.69 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 77.84 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 78.14 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 97.78 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 83.53 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 93.53 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 94.31 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 95.45 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 99.59 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 100.08 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 68.92 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 74.43 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 77.74 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 78.53 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 84.59 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 88.27 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 69.26 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 70.47 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 77.74 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 79.63 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 112.10 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 112.91 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 85.45 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 89.43 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 89.82 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 91.01 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 91.66 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 97.03 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 54.92 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 56.16 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 57.53 | No | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 68.34 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 72.44 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 99.53 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 75.01 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 75.68 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 75.72 | Yes (tie) | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 80.70 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 85.06 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 86.96 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

