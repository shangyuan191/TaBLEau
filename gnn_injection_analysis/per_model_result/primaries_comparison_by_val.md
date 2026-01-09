# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 57.12 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 58.32 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 93.52 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 94.05 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 100.22 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 102.32 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 77.03 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 78.83 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 99.42 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 99.90 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 107.83 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 109.58 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 68.91 | Yes | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 70.44 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 82.73 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 89.58 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 100.84 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 106.91 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 84.53 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 94.30 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 95.11 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 101.42 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 102.04 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 103.06 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 83.68 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 95.03 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 96.41 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 97.41 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 101.49 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 101.57 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 65.62 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 73.15 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 75.16 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 77.27 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 82.29 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 88.16 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 60.22 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 60.31 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 62.79 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 63.19 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 97.25 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 104.03 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 80.78 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 81.43 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 84.06 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 85.92 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 96.62 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 100.69 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 56.11 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 57.46 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 59.18 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 63.03 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 68.82 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 89.15 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 77.64 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 77.66 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 78.17 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 80.19 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 86.08 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 86.99 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

