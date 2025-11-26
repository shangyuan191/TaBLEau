# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 57.97 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| columnwise | 59.20 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| encoding | 60.96 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| start | 69.54 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| materialize | 74.34 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| decoding | 79.28 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes (tie) | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| start | 68.95 | Yes | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| materialize | 69.86 | Yes | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| columnwise | 76.81 | Yes | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| none | 76.87 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| encoding | 79.17 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes (tie) | No |
| decoding | 107.91 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 68.52 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| encoding | 68.66 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| start | 69.46 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| materialize | 70.33 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| columnwise | 76.34 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| decoding | 88.87 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 51.16 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | Yes |
| materialize | 51.96 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | Yes |
| start | 56.43 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | Yes |
| columnwise | 56.63 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | Yes |
| encoding | 59.46 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| decoding | 96.93 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 88.97 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 98.78 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 100.20 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 101.53 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 105.40 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 105.58 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 70.60 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| start | 78.68 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| materialize | 80.40 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 81.51 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 86.91 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 92.74 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 64.49 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| none | 64.51 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| start | 74.94 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| materialize | 75.44 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| encoding | 112.54 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 114.59 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 86.21 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 86.97 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 87.31 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 88.50 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 91.99 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 98.06 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 55.55 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | Yes |
| encoding | 58.06 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| none | 59.44 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| start | 71.62 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| materialize | 73.50 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| decoding | 102.83 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 82.40 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 82.46 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 82.86 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 85.47 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 91.27 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 92.28 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

