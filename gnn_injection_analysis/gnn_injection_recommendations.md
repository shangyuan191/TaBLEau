# GNN injection recommendations (cross-model)
Generated from gnn_injection_crossmodel_aggregation.md

## large_datasets+binclass+numerical (10 models)
- No injection consistently beats reference groups across models; prefer full-sample baselines or trees when possible.

## large_datasets+multiclass+numerical (10 models)
- No injection consistently beats reference groups across models; prefer full-sample baselines or trees when possible.

## large_datasets+regression+categorical (10 models)
- start: beats few-shot trees in 8/10 models; beats full trees in 8/10 models.
- materialize: beats few-shot trees in 9/10 models; beats full trees in 9/10 models.
- encoding: beats few-shot trees in 8/10 models; beats full trees in 8/10 models.
- columnwise: beats few-shot trees in 7/10 models; beats full trees in 7/10 models.
- decoding: beats few-shot trees in 8/10 models; beats full trees in 8/10 models.
- none: beats few-shot trees in 5/10 models; beats full trees in 5/10 models.

## large_datasets+regression+numerical (10 models)
- start: beats few-shot trees in 9/10 models; beats full trees in 9/10 models.
- materialize: beats few-shot trees in 9/10 models; beats full trees in 9/10 models.
- encoding: beats few-shot trees in 9/10 models; beats full trees in 9/10 models.
- columnwise: beats few-shot trees in 10/10 models; beats full trees in 10/10 models.
- decoding: beats few-shot trees in 7/10 models; beats full trees in 8/10 models.
- none: beats few-shot trees in 6/10 models.

## small_datasets+binclass+balanced (10 models)
- No injection consistently beats reference groups across models; prefer full-sample baselines or trees when possible.

## small_datasets+binclass+categorical (10 models)
- No injection consistently beats reference groups across models; prefer full-sample baselines or trees when possible.

## small_datasets+binclass+numerical (10 models)
- No injection consistently beats reference groups across models; prefer full-sample baselines or trees when possible.

## small_datasets+regression+balanced (10 models)
- decoding: beats few-shot trees in 7/10 models; beats full trees in 7/10 models; beats few-shot tabpfn in 5/10 models; beats full tabpfn in 7/10 models.
- start: beats few-shot trees in 6/10 models; beats full trees in 6/10 models; beats full tabpfn in 7/10 models.
- materialize: beats few-shot trees in 6/10 models; beats full trees in 6/10 models; beats full tabpfn in 7/10 models.
- encoding: beats few-shot trees in 8/10 models; beats full trees in 8/10 models; beats full tabpfn in 6/10 models.
- columnwise: beats few-shot trees in 7/10 models; beats full trees in 7/10 models; beats full tabpfn in 5/10 models.
- none: beats few-shot trees in 5/10 models; beats full trees in 5/10 models.

## small_datasets+regression+categorical (10 models)
- start: beats few-shot trees in 8/10 models; beats full trees in 8/10 models; beats full tabpfn in 8/10 models.
- materialize: beats few-shot trees in 9/10 models; beats full trees in 9/10 models; beats full tabpfn in 7/10 models.
- encoding: beats few-shot trees in 7/10 models; beats full trees in 8/10 models; beats full tabpfn in 7/10 models.
- columnwise: beats few-shot trees in 8/10 models; beats full trees in 9/10 models; beats full tabpfn in 8/10 models.
- decoding: beats few-shot trees in 7/10 models; beats full trees in 7/10 models; beats full tabpfn in 6/10 models.
- none: beats few-shot trees in 5/10 models; beats full trees in 5/10 models; beats full tabpfn in 6/10 models.

## small_datasets+regression+numerical (10 models)
- start: beats few-shot trees in 8/10 models; beats full trees in 8/10 models; beats full tabpfn in 5/10 models.
- encoding: beats few-shot trees in 8/10 models; beats full trees in 9/10 models; beats full tabpfn in 5/10 models.
- columnwise: beats few-shot trees in 8/10 models; beats full trees in 10/10 models; beats full tabpfn in 6/10 models.
- decoding: beats few-shot trees in 8/10 models; beats full trees in 8/10 models; beats full tabpfn in 6/10 models.
- materialize: beats few-shot trees in 9/10 models; beats full trees in 9/10 models.
- none: beats few-shot trees in 6/10 models; beats full trees in 6/10 models.
