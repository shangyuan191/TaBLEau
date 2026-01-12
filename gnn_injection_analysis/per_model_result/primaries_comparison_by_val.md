# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 56.56 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 57.62 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 92.04 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 92.81 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 98.19 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 100.39 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 75.74 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 77.47 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 97.40 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 97.91 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 106.20 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 108.25 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 67.97 | Yes | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 69.56 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 81.06 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 88.12 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 98.91 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 105.72 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 82.83 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 92.28 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 93.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 99.18 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 99.96 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 100.95 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 92.90 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 96.59 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 96.65 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 97.46 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 115.84 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 119.31 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 59.11 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 59.96 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 60.73 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 91.07 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 101.13 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 101.34 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 59.45 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 59.69 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 61.91 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 62.82 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 95.55 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 102.11 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 79.70 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 80.32 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 82.77 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 84.41 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 94.26 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 98.61 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 55.74 | Yes | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 57.04 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 58.91 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 62.60 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 68.35 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 87.84 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 75.71 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 76.71 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 76.88 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 94.86 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 105.92 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 106.72 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

