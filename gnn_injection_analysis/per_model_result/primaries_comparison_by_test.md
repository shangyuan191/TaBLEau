# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 61.20 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 61.22 | Yes (tie) | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 93.65 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 95.81 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 98.06 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 101.05 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 82.46 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 83.27 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 99.78 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 100.91 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 108.18 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 109.40 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 69.48 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 70.66 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 82.13 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 87.37 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 95.38 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 105.36 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 80.47 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 90.75 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 91.26 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 98.08 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 101.45 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 105.60 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 87.28 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 90.66 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 90.66 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 97.95 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 115.28 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 117.02 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 59.11 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 61.10 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 62.05 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 96.60 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 97.22 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 97.60 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 62.13 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 63.80 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 64.75 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 68.66 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 100.53 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 102.49 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 81.43 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 84.60 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none | 84.77 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 84.86 | Yes (tie) | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 95.65 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 98.25 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 55.48 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 56.79 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 58.19 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 63.65 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 67.42 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 97.53 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 68.74 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 70.22 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 71.21 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 94.16 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 100.18 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 100.51 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

