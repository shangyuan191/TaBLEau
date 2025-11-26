# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 62.69 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| none | 63.97 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| encoding | 66.78 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| start | 74.34 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | Yes | No |
| materialize | 81.44 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 86.28 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| start | 81.47 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 84.20 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 90.20 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 91.12 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 94.11 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 109.97 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 73.69 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | Yes | No |
| encoding | 76.05 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | Yes | No |
| start | 79.84 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 79.93 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 80.03 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 96.15 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 66.40 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| materialize | 67.67 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| start | 69.14 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| columnwise | 77.28 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 77.58 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 97.31 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 83.05 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 93.19 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 93.99 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 95.11 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 99.31 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 99.79 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 68.46 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| start | 73.91 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | Yes | No |
| materialize | 77.20 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 78.01 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 84.16 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 87.84 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 68.71 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| decoding | 70.22 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| start | 77.21 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 79.09 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 111.78 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 112.52 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 84.90 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 88.91 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 89.30 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 90.54 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 91.14 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 96.52 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 54.38 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| encoding | 55.60 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| none | 57.00 | No | No | 2/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| start | 67.81 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| materialize | 71.88 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| decoding | 99.20 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 74.47 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | Yes | No |
| none | 75.11 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | Yes | No |
| columnwise | 75.19 | Yes (tie) | No | 0/3 | 0/3 | 1/2 | 0/2 | Yes | No |
| decoding | 80.16 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 84.52 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 86.41 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

