# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 59.65 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 60.69 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 95.26 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 95.65 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 102.23 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 104.59 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 79.30 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 81.12 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 101.13 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 102.41 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 109.72 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 110.60 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 71.21 | Yes | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 72.78 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 84.95 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 91.96 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 102.89 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 107.91 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 49.30 | No | No | 2/3 | 0/3 | 1/2 | 1/2 | Yes | No |
| materialize | 50.03 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 54.03 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 54.53 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 57.11 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 93.84 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 85.87 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 96.40 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 97.73 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 99.04 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 102.68 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 102.85 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 68.08 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 75.89 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 77.68 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 79.62 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 84.46 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 90.57 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 61.40 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 63.90 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 64.16 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 64.78 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 98.04 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 101.81 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 83.73 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 84.46 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 84.89 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 85.97 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 88.99 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 95.53 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 53.59 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 56.31 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 57.53 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 69.08 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 70.77 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 100.47 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 79.81 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 79.93 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 80.27 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 82.68 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 88.38 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 89.10 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

