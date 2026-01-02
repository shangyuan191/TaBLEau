# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 63.60 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 63.73 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 96.83 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 99.56 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 99.82 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 104.86 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 85.89 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 87.02 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 103.66 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 104.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 110.96 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 111.51 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 72.28 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 73.66 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 85.94 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 91.09 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 99.70 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 106.78 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 64.95 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 66.38 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 67.82 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 75.17 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 75.70 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 94.61 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 81.06 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 91.26 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 91.99 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 93.41 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 97.23 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 97.55 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 67.49 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 72.28 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 75.57 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 77.19 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 82.24 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 86.04 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 66.70 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 67.80 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 74.51 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 76.70 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 109.72 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 110.41 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 83.09 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 87.09 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 87.41 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 88.73 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 88.89 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 94.11 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 53.92 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 54.92 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 56.13 | No | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 66.44 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 70.30 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 96.51 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 72.10 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 72.73 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 72.78 | Yes (tie) | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 77.80 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 81.70 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 83.71 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

