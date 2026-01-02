# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 63.71 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 64.24 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 97.34 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 100.07 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 100.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 105.37 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 86.55 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 87.59 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 104.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 105.49 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 111.80 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 112.26 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 72.32 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 73.68 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 85.97 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 91.11 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 99.72 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 106.81 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 64.78 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 66.21 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 67.64 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 75.02 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 75.53 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 94.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 80.89 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 91.19 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 91.92 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 93.26 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 97.15 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 97.48 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 67.27 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 72.10 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 75.41 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 77.01 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 82.16 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 85.97 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 66.73 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 67.83 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 74.56 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 76.73 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 109.75 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 110.43 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 82.92 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 86.93 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 87.25 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 88.58 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 88.73 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 93.95 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 54.03 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 55.09 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 56.33 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 66.63 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 70.47 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 96.99 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 72.51 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 73.13 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 73.18 | Yes (tie) | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 78.12 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 82.10 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 84.13 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

