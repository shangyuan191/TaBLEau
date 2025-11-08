# Cross-model GNN injection aggregation (10 models)

This report counts, for each dataset category and each GNN injection stage, how many of the parsed models (out of the models that include that category) had the injection "beat" each reference group.

## Category: large_datasets+binclass+numerical (6 datasets) (10 models parsed)

| Injection | beats few-shot-non-gnn | beats full-non-gnn | beats few-shot trees | beats full trees | beats few-shot GNN | beats full GNN | beats few-shot tabpfn | beats full tabpfn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 2/10, 20% | 1/10, 10% | 0/10, 0% | 0/10, 0% |
| materialize | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 2/10, 20% | 1/10, 10% | 0/10, 0% | 0/10, 0% |
| encoding | 1/10, 10% | 0/10, 0% | 1/10, 10% | 0/10, 0% | 3/10, 30% | 2/10, 20% | 0/10, 0% | 0/10, 0% |
| columnwise | 3/10, 30% | 1/10, 10% | 1/10, 10% | 0/10, 0% | 4/10, 40% | 3/10, 30% | 0/10, 0% | 0/10, 0% |
| decoding | 1/10, 10% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 3/10, 30% | 2/10, 20% | 0/10, 0% | 0/10, 0% |
| none | 0/10, 0% | 0/10, 0% | 1/10, 10% | 0/10, 0% | 3/10, 30% | 2/10, 20% | 0/10, 0% | 0/10, 0% |


## Category: large_datasets+multiclass+numerical (3 datasets) (10 models parsed)

| Injection | beats few-shot-non-gnn | beats full-non-gnn | beats few-shot trees | beats full trees | beats few-shot GNN | beats full GNN | beats few-shot tabpfn | beats full tabpfn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 1/10, 10% | 0/10, 0% | 0/10, 0% | 0/10, 0% |
| materialize | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% |
| encoding | 2/10, 20% | 0/10, 0% | 1/10, 10% | 0/10, 0% | 3/10, 30% | 1/10, 10% | 1/10, 10% | 0/10, 0% |
| columnwise | 1/10, 10% | 0/10, 0% | 1/10, 10% | 0/10, 0% | 3/10, 30% | 1/10, 10% | 0/10, 0% | 0/10, 0% |
| decoding | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% |
| none | 0/10, 0% | 0/10, 0% | 1/10, 10% | 0/10, 0% | 3/10, 30% | 1/10, 10% | 0/10, 0% | 0/10, 0% |


## Category: large_datasets+regression+categorical (1 dataset) (10 models parsed)

| Injection | beats few-shot-non-gnn | beats full-non-gnn | beats few-shot trees | beats full trees | beats few-shot GNN | beats full GNN | beats few-shot tabpfn | beats full tabpfn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start | 2/10, 20% | 2/10, 20% | 8/10, 80% | 8/10, 80% | 4/10, 40% | 4/10, 40% | 1/10, 10% | 2/10, 20% |
| materialize | 3/10, 30% | 3/10, 30% | 9/10, 90% | 9/10, 90% | 6/10, 60% | 5/10, 50% | 1/10, 10% | 2/10, 20% |
| encoding | 2/10, 20% | 3/10, 30% | 8/10, 80% | 8/10, 80% | 6/10, 60% | 5/10, 50% | 2/10, 20% | 3/10, 30% |
| columnwise | 1/10, 10% | 3/10, 30% | 7/10, 70% | 7/10, 70% | 4/10, 40% | 4/10, 40% | 3/10, 30% | 4/10, 40% |
| decoding | 4/10, 40% | 4/10, 40% | 8/10, 80% | 8/10, 80% | 8/10, 80% | 7/10, 70% | 2/10, 20% | 3/10, 30% |
| none | 0/10, 0% | 3/10, 30% | 5/10, 50% | 5/10, 50% | 1/10, 10% | 1/10, 10% | 0/10, 0% | 2/10, 20% |


## Category: large_datasets+regression+numerical (10 datasets) (10 models parsed)

| Injection | beats few-shot-non-gnn | beats full-non-gnn | beats few-shot trees | beats full trees | beats few-shot GNN | beats full GNN | beats few-shot tabpfn | beats full tabpfn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start | 3/10, 30% | 3/10, 30% | 9/10, 90% | 9/10, 90% | 8/10, 80% | 8/10, 80% | 0/10, 0% | 1/10, 10% |
| materialize | 3/10, 30% | 3/10, 30% | 9/10, 90% | 9/10, 90% | 8/10, 80% | 8/10, 80% | 0/10, 0% | 1/10, 10% |
| encoding | 2/10, 20% | 3/10, 30% | 9/10, 90% | 9/10, 90% | 6/10, 60% | 7/10, 70% | 1/10, 10% | 1/10, 10% |
| columnwise | 2/10, 20% | 4/10, 40% | 10/10, 100% | 10/10, 100% | 8/10, 80% | 9/10, 90% | 0/10, 0% | 1/10, 10% |
| decoding | 2/10, 20% | 3/10, 30% | 7/10, 70% | 8/10, 80% | 7/10, 70% | 8/10, 80% | 0/10, 0% | 0/10, 0% |
| none | 0/10, 0% | 3/10, 30% | 6/10, 60% | 4/10, 40% | 2/10, 20% | 3/10, 30% | 0/10, 0% | 0/10, 0% |


## Category: small_datasets+binclass+balanced (14 datasets) (10 models parsed)

| Injection | beats few-shot-non-gnn | beats full-non-gnn | beats few-shot trees | beats full trees | beats few-shot GNN | beats full GNN | beats few-shot tabpfn | beats full tabpfn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% |
| materialize | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% |
| encoding | 1/10, 10% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 1/10, 10% | 0/10, 0% | 0/10, 0% | 0/10, 0% |
| columnwise | 0/10, 0% | 0/10, 0% | 2/10, 20% | 0/10, 0% | 2/10, 20% | 1/10, 10% | 0/10, 0% | 0/10, 0% |
| decoding | 1/10, 10% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% |
| none | 0/10, 0% | 0/10, 0% | 1/10, 10% | 0/10, 0% | 1/10, 10% | 1/10, 10% | 0/10, 0% | 0/10, 0% |


## Category: small_datasets+binclass+categorical (7 datasets) (10 models parsed)

| Injection | beats few-shot-non-gnn | beats full-non-gnn | beats few-shot trees | beats full trees | beats few-shot GNN | beats full GNN | beats few-shot tabpfn | beats full tabpfn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start | 1/10, 10% | 0/10, 0% | 1/10, 10% | 1/10, 10% | 3/10, 30% | 1/10, 10% | 0/10, 0% | 0/10, 0% |
| materialize | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 1/10, 10% | 0/10, 0% | 0/10, 0% | 0/10, 0% |
| encoding | 0/10, 0% | 1/10, 10% | 1/10, 10% | 1/10, 10% | 1/10, 10% | 1/10, 10% | 1/10, 10% | 1/10, 10% |
| columnwise | 1/10, 10% | 0/10, 0% | 1/10, 10% | 1/10, 10% | 1/10, 10% | 1/10, 10% | 1/10, 10% | 1/10, 10% |
| decoding | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% |
| none | 0/10, 0% | 1/10, 10% | 1/10, 10% | 1/10, 10% | 1/10, 10% | 1/10, 10% | 1/10, 10% | 1/10, 10% |


## Category: small_datasets+binclass+numerical (28 datasets) (10 models parsed)

| Injection | beats few-shot-non-gnn | beats full-non-gnn | beats few-shot trees | beats full trees | beats few-shot GNN | beats full GNN | beats few-shot tabpfn | beats full tabpfn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start | 0/10, 0% | 0/10, 0% | 2/10, 20% | 0/10, 0% | 2/10, 20% | 0/10, 0% | 0/10, 0% | 0/10, 0% |
| materialize | 0/10, 0% | 0/10, 0% | 1/10, 10% | 0/10, 0% | 1/10, 10% | 0/10, 0% | 0/10, 0% | 0/10, 0% |
| encoding | 1/10, 10% | 0/10, 0% | 2/10, 20% | 0/10, 0% | 2/10, 20% | 1/10, 10% | 1/10, 10% | 0/10, 0% |
| columnwise | 3/10, 30% | 0/10, 0% | 3/10, 30% | 0/10, 0% | 3/10, 30% | 2/10, 20% | 0/10, 0% | 0/10, 0% |
| decoding | 1/10, 10% | 0/10, 0% | 1/10, 10% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% | 0/10, 0% |
| none | 0/10, 0% | 0/10, 0% | 2/10, 20% | 0/10, 0% | 2/10, 20% | 0/10, 0% | 0/10, 0% | 0/10, 0% |


## Category: small_datasets+regression+balanced (6 datasets) (10 models parsed)

| Injection | beats few-shot-non-gnn | beats full-non-gnn | beats few-shot trees | beats full trees | beats few-shot GNN | beats full GNN | beats few-shot tabpfn | beats full tabpfn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start | 2/10, 20% | 3/10, 30% | 6/10, 60% | 6/10, 60% | 4/10, 40% | 4/10, 40% | 2/10, 20% | 7/10, 70% |
| materialize | 3/10, 30% | 3/10, 30% | 6/10, 60% | 6/10, 60% | 4/10, 40% | 4/10, 40% | 3/10, 30% | 7/10, 70% |
| encoding | 0/10, 0% | 3/10, 30% | 8/10, 80% | 8/10, 80% | 6/10, 60% | 6/10, 60% | 3/10, 30% | 6/10, 60% |
| columnwise | 0/10, 0% | 3/10, 30% | 7/10, 70% | 7/10, 70% | 7/10, 70% | 7/10, 70% | 3/10, 30% | 5/10, 50% |
| decoding | 3/10, 30% | 3/10, 30% | 7/10, 70% | 7/10, 70% | 4/10, 40% | 4/10, 40% | 5/10, 50% | 7/10, 70% |
| none | 0/10, 0% | 3/10, 30% | 5/10, 50% | 5/10, 50% | 3/10, 30% | 3/10, 30% | 1/10, 10% | 4/10, 40% |


## Category: small_datasets+regression+categorical (5 datasets) (10 models parsed)

| Injection | beats few-shot-non-gnn | beats full-non-gnn | beats few-shot trees | beats full trees | beats few-shot GNN | beats full GNN | beats few-shot tabpfn | beats full tabpfn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start | 1/10, 10% | 3/10, 30% | 8/10, 80% | 8/10, 80% | 5/10, 50% | 5/10, 50% | 3/10, 30% | 8/10, 80% |
| materialize | 2/10, 20% | 3/10, 30% | 9/10, 90% | 9/10, 90% | 4/10, 40% | 4/10, 40% | 2/10, 20% | 7/10, 70% |
| encoding | 0/10, 0% | 3/10, 30% | 7/10, 70% | 8/10, 80% | 6/10, 60% | 6/10, 60% | 4/10, 40% | 7/10, 70% |
| columnwise | 1/10, 10% | 4/10, 40% | 8/10, 80% | 9/10, 90% | 5/10, 50% | 5/10, 50% | 3/10, 30% | 8/10, 80% |
| decoding | 1/10, 10% | 3/10, 30% | 7/10, 70% | 7/10, 70% | 6/10, 60% | 6/10, 60% | 2/10, 20% | 6/10, 60% |
| none | 0/10, 0% | 3/10, 30% | 5/10, 50% | 5/10, 50% | 3/10, 30% | 3/10, 30% | 2/10, 20% | 6/10, 60% |


## Category: small_datasets+regression+numerical (36 datasets) (10 models parsed)

| Injection | beats few-shot-non-gnn | beats full-non-gnn | beats few-shot trees | beats full trees | beats few-shot GNN | beats full GNN | beats few-shot tabpfn | beats full tabpfn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| start | 3/10, 30% | 3/10, 30% | 8/10, 80% | 8/10, 80% | 4/10, 40% | 4/10, 40% | 1/10, 10% | 5/10, 50% |
| materialize | 2/10, 20% | 3/10, 30% | 9/10, 90% | 9/10, 90% | 4/10, 40% | 4/10, 40% | 1/10, 10% | 4/10, 40% |
| encoding | 0/10, 0% | 3/10, 30% | 8/10, 80% | 9/10, 90% | 5/10, 50% | 5/10, 50% | 1/10, 10% | 5/10, 50% |
| columnwise | 2/10, 20% | 4/10, 40% | 8/10, 80% | 10/10, 100% | 7/10, 70% | 7/10, 70% | 1/10, 10% | 6/10, 60% |
| decoding | 3/10, 30% | 3/10, 30% | 8/10, 80% | 8/10, 80% | 6/10, 60% | 6/10, 60% | 2/10, 20% | 6/10, 60% |
| none | 0/10, 0% | 3/10, 30% | 6/10, 60% | 6/10, 60% | 3/10, 30% | 3/10, 30% | 0/10, 0% | 3/10, 30% |

