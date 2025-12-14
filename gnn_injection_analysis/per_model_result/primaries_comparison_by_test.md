# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 64.00 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 64.48 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 98.19 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 101.06 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 101.35 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 106.19 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| start | 80.23 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 83.10 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 89.08 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 90.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 92.91 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 109.06 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 72.87 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 75.37 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 78.62 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 78.79 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 79.12 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 94.86 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 65.56 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 66.95 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 68.51 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 76.08 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 76.59 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 95.87 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 81.91 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 91.89 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 92.69 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 94.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 98.03 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 98.47 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 67.86 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 72.90 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 76.32 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 77.38 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 83.04 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 86.78 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 67.87 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 69.05 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 75.94 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 78.06 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 110.87 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 111.73 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 84.69 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 88.49 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 88.86 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 90.16 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 90.68 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 96.03 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 54.01 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 55.16 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 56.46 | No | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 67.10 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 71.05 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 98.06 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 73.95 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 74.57 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 74.61 | Yes (tie) | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 79.72 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 83.95 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 85.81 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

