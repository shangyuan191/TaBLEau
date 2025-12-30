# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 63.76 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 64.54 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 97.91 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 100.77 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 101.03 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 105.92 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 87.22 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 88.20 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 105.05 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 106.03 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 112.17 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 112.69 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 72.74 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 75.05 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 78.34 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 78.37 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 78.76 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 94.83 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 65.43 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 66.72 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 68.11 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 75.74 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 76.22 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 95.37 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 81.47 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 91.73 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 92.47 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 93.77 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 97.74 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 98.14 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 67.49 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 72.62 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 75.82 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 77.15 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 82.71 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 86.53 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 67.66 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 69.15 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 75.66 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 77.85 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 110.79 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 111.59 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 83.60 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 87.39 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 87.76 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 89.03 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 89.35 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 94.60 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 53.91 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 55.04 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 56.39 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 66.80 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 70.76 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 97.68 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 73.02 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 73.62 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 73.72 | Yes (tie) | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 78.67 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 82.85 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 84.82 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

