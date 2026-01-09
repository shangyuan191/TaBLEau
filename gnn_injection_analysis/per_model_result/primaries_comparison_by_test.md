# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 62.03 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 62.43 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 95.89 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 98.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 99.84 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 103.66 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 84.09 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 85.16 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 102.03 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 103.01 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 110.22 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 110.54 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 70.63 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 71.85 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 83.93 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 89.14 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 97.69 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 106.32 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 82.14 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 92.87 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 93.71 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 100.69 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 103.58 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 107.66 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 78.96 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 90.53 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 91.22 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 92.60 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 96.54 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 96.85 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 65.36 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 69.69 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 73.22 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 75.15 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 79.59 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 83.59 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 63.33 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 64.59 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 66.01 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 69.56 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 102.52 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 104.45 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 83.30 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 86.06 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none | 86.08 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 86.32 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 98.28 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 100.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 56.38 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 57.51 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 59.05 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 64.77 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 68.45 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 99.72 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 70.74 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 71.47 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 71.49 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 76.42 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 80.22 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 82.31 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

