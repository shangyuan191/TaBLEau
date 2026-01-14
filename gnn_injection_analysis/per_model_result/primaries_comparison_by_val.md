# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 60.23 | Yes | No | 1/3 | 0/3 | 6/2 | 4/2 | No | No |
| none | 61.31 | No | No | 1/3 | 0/3 | 6/2 | 4/2 | No | No |
| encoding | 98.74 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 99.34 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 105.04 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 107.66 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 81.01 | No | No | 0/3 | 0/3 | 4/2 | 2/2 | No | No |
| columnwise | 82.88 | No | No | 0/3 | 0/3 | 4/2 | 1/2 | No | No |
| encoding | 104.39 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 104.95 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 114.01 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 116.09 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 72.59 | Yes | No | 0/3 | 0/3 | 4/2 | 3/2 | No | No |
| none | 74.26 | No | No | 0/3 | 0/3 | 4/2 | 3/2 | No | No |
| encoding | 86.84 | No | No | 0/3 | 0/3 | 4/2 | 1/2 | No | No |
| materialize | 94.53 | No | No | 0/3 | 0/3 | 2/2 | 1/2 | No | No |
| start | 106.04 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 113.30 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 88.53 | No | No | 0/3 | 0/3 | 4/2 | 1/2 | No | No |
| start | 98.78 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 99.98 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 106.58 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 107.18 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 108.36 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 99.39 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 103.53 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 103.59 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 104.34 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 124.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 127.93 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 63.06 | No | No | 1/3 | 0/3 | 5/2 | 4/2 | No | No |
| encoding | 64.05 | No | No | 1/3 | 0/3 | 5/2 | 4/2 | No | No |
| columnwise | 64.90 | No | No | 1/3 | 0/3 | 5/2 | 4/2 | No | No |
| decoding | 97.49 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 108.43 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 108.63 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 63.34 | No | No | 1/3 | 0/3 | 5/2 | 4/2 | No | No |
| encoding | 63.51 | No | No | 1/3 | 0/3 | 5/2 | 4/2 | No | No |
| decoding | 66.04 | No | No | 1/3 | 0/3 | 5/2 | 4/2 | No | No |
| columnwise | 66.82 | No | No | 1/3 | 0/3 | 5/2 | 4/2 | No | No |
| materialize | 102.38 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 109.53 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 85.10 | No | No | 0/3 | 0/3 | 4/2 | 1/2 | No | No |
| encoding | 85.73 | No | No | 0/3 | 0/3 | 4/2 | 1/2 | No | No |
| columnwise | 88.41 | No | No | 0/3 | 0/3 | 4/2 | 1/2 | No | No |
| decoding | 90.12 | No | No | 0/3 | 0/3 | 3/2 | 1/2 | No | No |
| start | 100.97 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 105.80 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 59.20 | Yes | No | 1/3 | 0/3 | 6/2 | 4/2 | No | No |
| none | 60.67 | No | No | 1/3 | 0/3 | 6/2 | 4/2 | No | No |
| columnwise | 62.67 | No | No | 1/3 | 0/3 | 5/2 | 4/2 | No | No |
| start | 66.66 | No | No | 1/3 | 0/3 | 5/2 | 4/2 | No | No |
| materialize | 72.80 | No | No | 0/3 | 0/3 | 4/2 | 3/2 | No | No |
| decoding | 94.06 | No | No | 0/3 | 0/3 | 2/2 | 1/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| encoding | 81.03 | Yes | No | 0/3 | 0/3 | 4/2 | 2/2 | No | No |
| none | 82.09 | No | No | 0/3 | 0/3 | 4/2 | 1/2 | No | No |
| columnwise | 82.19 | Yes (tie) | No | 0/3 | 0/3 | 4/2 | 1/2 | No | No |
| decoding | 101.61 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| materialize | 113.69 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 114.56 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

