# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 59.48 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 60.47 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 95.04 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 95.37 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 102.03 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 104.36 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 79.18 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 81.01 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 101.11 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 102.10 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 109.47 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 110.34 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 71.16 | Yes | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 72.57 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 84.81 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 91.65 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 102.87 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 107.78 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 49.03 | No | No | 2/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| materialize | 49.86 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 53.66 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 54.21 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 56.85 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 93.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 85.64 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 95.94 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 97.30 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 98.58 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 102.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 102.37 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 67.81 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 75.53 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 77.34 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 79.33 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 84.17 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 90.16 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 62.53 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 62.68 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 65.05 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 65.06 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 99.32 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 105.89 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 82.89 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 83.47 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 86.04 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 87.66 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 98.55 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 102.57 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 53.39 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 56.16 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 57.34 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 68.90 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 70.56 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 100.26 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 79.34 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 79.42 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 79.82 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 82.22 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 87.82 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 88.69 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

