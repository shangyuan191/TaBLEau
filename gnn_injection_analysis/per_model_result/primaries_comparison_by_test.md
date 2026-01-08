# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 63.35 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 63.73 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 97.53 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 99.97 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 100.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 105.34 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 85.72 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 86.72 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 103.54 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 104.72 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 111.09 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 111.52 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 72.04 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 73.23 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 85.67 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 90.92 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 99.65 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 106.95 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 64.78 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 66.41 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 67.75 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 75.28 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 75.49 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 94.46 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 80.86 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 91.22 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 91.97 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 93.37 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 97.23 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 97.47 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 66.82 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 71.53 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 74.99 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 76.70 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 81.38 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 85.32 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 64.86 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 66.23 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 67.76 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 70.86 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 104.03 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 105.90 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 84.82 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 87.59 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 87.62 | Yes (tie) | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 87.93 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 100.17 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 102.43 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 57.66 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 58.75 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 60.44 | No | No | 1/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 66.07 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 69.66 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 100.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 72.49 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 73.25 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 73.26 | Yes (tie) | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 78.21 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 82.02 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 83.99 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

