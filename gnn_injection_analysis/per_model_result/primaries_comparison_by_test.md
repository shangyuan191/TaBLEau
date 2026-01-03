# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 63.74 | Yes | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 63.77 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 97.21 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 99.96 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 100.21 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 105.28 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 86.19 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 87.23 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 104.03 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 105.11 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 111.30 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 111.95 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 72.47 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 73.75 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 86.26 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 91.47 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 100.04 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 107.12 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 65.16 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 66.58 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 68.12 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 75.57 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 76.06 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 94.88 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 81.37 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 91.47 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 92.28 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 93.73 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 97.49 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 97.78 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 67.42 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 72.49 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 75.86 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 77.40 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 82.55 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 86.41 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 66.31 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 66.68 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 67.74 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 74.54 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 103.28 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 103.41 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 83.47 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 87.44 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 87.75 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 88.95 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 89.19 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 94.25 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 53.89 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 54.89 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 56.06 | No | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 66.59 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 70.43 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 96.93 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 72.46 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 73.11 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 73.17 | Yes (tie) | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 78.10 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 82.17 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 84.17 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

