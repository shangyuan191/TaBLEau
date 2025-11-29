# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 58.51 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 59.70 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 61.44 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 70.05 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 74.85 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 79.63 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| start | 69.49 | Yes | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 70.37 | Yes | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 77.37 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 77.39 | Yes (tie) | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 79.73 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 108.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 69.03 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 69.22 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 69.99 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 70.86 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 76.89 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 89.14 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 51.55 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 52.34 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 56.80 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 57.06 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 59.85 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 97.48 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 89.52 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 99.15 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 100.56 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 101.91 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 105.70 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 105.93 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 71.08 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 79.22 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 80.98 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 82.03 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 87.42 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 93.28 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 64.78 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 65.09 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 75.52 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 76.03 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 113.01 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 115.17 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 86.80 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 87.55 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 87.88 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 89.07 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 92.54 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 98.59 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 56.09 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 58.59 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 59.97 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 72.17 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 74.02 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 103.16 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 82.97 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 83.02 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 83.44 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 86.03 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 91.85 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 92.89 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

