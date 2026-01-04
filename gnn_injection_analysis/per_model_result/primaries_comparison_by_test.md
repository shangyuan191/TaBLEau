# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 63.60 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 63.77 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 97.43 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 99.98 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 100.25 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 105.22 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 85.99 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 87.04 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 103.81 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 104.92 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 111.06 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 111.78 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 72.34 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 73.56 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 86.08 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 91.17 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 99.78 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 106.96 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 64.92 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 66.53 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 67.93 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 75.33 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 75.62 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 94.65 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 80.99 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 91.26 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 92.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 93.52 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 97.25 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 97.54 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 67.09 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 71.81 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 75.26 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 76.87 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 81.71 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 85.61 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 64.99 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 66.38 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 67.80 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 70.97 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 104.18 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 105.91 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 85.05 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 87.76 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 87.86 | Yes (tie) | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 87.95 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 100.34 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 102.63 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 54.03 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 55.03 | Yes | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| none | 56.20 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 66.73 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 70.55 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 96.74 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 72.57 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 73.19 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 73.26 | Yes (tie) | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 78.03 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 81.97 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 84.09 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

