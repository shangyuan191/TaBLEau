# Primaries comparison (based on test metrics)

(eps for ties = 0.10)


## Primary: excelformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| columnwise | 65.35 | Yes | No | 0/3 | 0/3 | 5/2 | 3/2 | No | No |
| none | 65.41 | No | No | 0/3 | 0/3 | 5/2 | 3/2 | No | No |
| encoding | 100.61 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 102.75 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 105.14 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 108.66 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: fttransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 88.55 | No | No | 0/3 | 0/3 | 4/2 | 1/2 | No | No |
| columnwise | 89.62 | No | No | 0/3 | 0/3 | 3/2 | 1/2 | No | No |
| encoding | 107.26 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 108.24 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 116.23 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 117.47 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: resnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 74.44 | No | No | 0/3 | 0/3 | 4/2 | 2/2 | No | No |
| columnwise | 75.64 | No | No | 0/3 | 0/3 | 4/2 | 2/2 | No | No |
| encoding | 88.34 | No | No | 0/3 | 0/3 | 4/2 | 1/2 | No | No |
| materialize | 93.81 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 102.40 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 113.12 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: scarf


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 86.16 | No | No | 0/3 | 0/3 | 4/2 | 1/2 | No | No |
| materialize | 97.22 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 97.95 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 105.52 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise | 108.93 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 113.46 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: subtab


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 93.68 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| columnwise | 97.37 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 97.38 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 105.15 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 123.87 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 125.54 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabm


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 62.96 | No | No | 1/3 | 0/3 | 6/2 | 4/2 | No | No |
| encoding | 64.99 | No | No | 0/3 | 0/3 | 5/2 | 3/2 | No | No |
| columnwise | 66.28 | No | No | 0/3 | 0/3 | 5/2 | 3/2 | No | No |
| start | 103.73 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 104.26 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 104.93 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: tabnet


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 66.18 | No | No | 0/3 | 0/3 | 5/2 | 3/2 | No | No |
| encoding | 68.18 | No | No | 0/3 | 0/3 | 5/2 | 3/2 | No | No |
| decoding | 69.22 | No | No | 0/3 | 0/3 | 5/2 | 3/2 | No | No |
| columnwise | 73.33 | No | No | 0/3 | 0/3 | 5/2 | 2/2 | No | No |
| start | 108.03 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 110.04 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |


## Primary: tabtransformer


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| decoding | 87.16 | Yes | No | 0/3 | 0/3 | 4/2 | 1/2 | No | No |
| encoding | 90.62 | Yes | No | 0/3 | 0/3 | 2/2 | 1/2 | No | No |
| none | 90.72 | No | No | 0/3 | 0/3 | 2/2 | 1/2 | No | No |
| columnwise | 91.00 | No | No | 0/3 | 0/3 | 2/2 | 1/2 | No | No |
| materialize | 102.69 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 105.44 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: trompt


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 58.89 | No | No | 1/3 | 0/3 | 6/2 | 4/2 | No | No |
| encoding | 60.17 | No | No | 1/3 | 0/3 | 6/2 | 4/2 | No | No |
| columnwise | 61.99 | No | No | 1/3 | 0/3 | 6/2 | 4/2 | No | No |
| start | 67.80 | No | No | 0/3 | 0/3 | 5/2 | 3/2 | No | No |
| materialize | 71.73 | No | No | 0/3 | 0/3 | 5/2 | 3/2 | No | No |
| decoding | 104.81 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |


## Primary: vime


| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 7) | beats full GNN (out of 7) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| none | 73.49 | No | No | 0/3 | 0/3 | 5/2 | 2/2 | No | No |
| encoding | 75.06 | No | No | 0/3 | 0/3 | 4/2 | 2/2 | No | No |
| columnwise | 76.11 | No | No | 0/3 | 0/3 | 4/2 | 2/2 | No | No |
| decoding | 101.09 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 107.55 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 108.12 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

