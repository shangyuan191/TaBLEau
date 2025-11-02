GNN增強效果分析 - 模型: TABTRANSFORMER

說明：
- 比較 tabtransformer 模型的原始表現與GNN增強變體
- 包含5個參考模型（t2gformer, tabpfn, xgboost, catboost, lightgbm）
- 共17個競爭者：
  * tabtransformer的7個配置：大訓練集baseline(1) + 小訓練集6變體(6)
  * 5個參考模型的兩種ratio：每個模型2個配置(10)
- 排名越小表示表現越好


* #### 分類: large_datasets+binclass+numerical (包含 6 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.83 | 6 |
| 2 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.67 | 6 |
| 3 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.33 | 6 |
| 4 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.33 | 6 |
| 5 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.17 | 6 |
| 6 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.00 | 6 |
| 7 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.67 | 6 |
| 8 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.83 | 6 |
| 9 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.50 | 6 |
| 10 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.33 | 6 |
| 11 | tabtransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.50 | 6 |
| 12 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 12.83 | 6 |
| 13 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.17 | 6 |
| 14 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.00 | 6 |
| 15 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.33 | 6 |
| 16 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 15.83 | 6 |
| 17 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 16.67 | 6 |
| 18 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 16.83 | 6 |
| 19 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 17.17 | 6 |
* #### 分類: large_datasets+multiclass+numerical (包含 3 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.33 | 3 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.67 | 3 |
| 3 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.33 | 3 |
| 4 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.00 | 3 |
| 5 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.33 | 3 |
| 6 | tabtransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.67 | 3 |
| 7 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.33 | 3 |
| 8 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.67 | 3 |
| 9 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.67 | 3 |
| 10 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.00 | 3 |
| 11 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.67 | 3 |
| 12 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.00 | 3 |
| 13 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 14.00 | 3 |
| 14 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 14.33 | 3 |
| 15 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.33 | 3 |
| 16 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.67 | 3 |
| 17 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 17.67 | 3 |
| 18 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 18.00 | 3 |
| 19 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 18.33 | 3 |
* #### 分類: large_datasets+regression+categorical (包含 1 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 1.00 | 1 |
| 2 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 2.00 | 1 |
| 3 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.00 | 1 |
| 4 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 4.00 | 1 |
| 5 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 5.00 | 1 |
| 6 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 6.00 | 1 |
| 7 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.00 | 1 |
| 8 | tabtransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.00 | 1 |
| 9 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.00 | 1 |
| 10 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.00 | 1 |
| 11 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 11.00 | 1 |
| 12 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.00 | 1 |
| 13 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 13.00 | 1 |
| 14 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.00 | 1 |
| 15 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.00 | 1 |
| 16 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 16.00 | 1 |
| 17 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.00 | 1 |
| 18 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 18.00 | 1 |
| 19 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 19.00 | 1 |
* #### 分類: large_datasets+regression+numerical (包含 10 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 1.60 | 10 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.10 | 10 |
| 3 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.90 | 10 |
| 4 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.80 | 10 |
| 5 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 6.40 | 10 |
| 6 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 7.00 | 10 |
| 7 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 7.00 | 10 |
| 8 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 7.10 | 10 |
| 9 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 7.80 | 10 |
| 10 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.90 | 10 |
| 11 | tabtransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.80 | 10 |
| 12 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.60 | 10 |
| 13 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.00 | 10 |
| 14 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.20 | 10 |
| 15 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.40 | 10 |
| 16 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.70 | 10 |
| 17 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.90 | 10 |
| 18 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.10 | 10 |
| 19 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.70 | 10 |
* #### 分類: small_datasets+binclass+balanced (包含 14 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.93 | 14 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.86 | 14 |
| 3 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.86 | 14 |
| 4 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.93 | 14 |
| 5 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.00 | 14 |
| 6 | tabtransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.14 | 14 |
| 7 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.00 | 14 |
| 8 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.43 | 14 |
| 9 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.57 | 14 |
| 10 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.57 | 14 |
| 11 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.21 | 14 |
| 12 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.50 | 14 |
| 13 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.79 | 14 |
| 14 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 12.71 | 14 |
| 15 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 12.93 | 14 |
| 16 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.93 | 14 |
| 17 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 14.86 | 14 |
| 18 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 14.86 | 14 |
| 19 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 14.93 | 14 |
* #### 分類: small_datasets+binclass+categorical (包含 7 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.29 | 7 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.14 | 7 |
| 3 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 6.14 | 7 |
| 4 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.14 | 7 |
| 5 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.14 | 7 |
| 6 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.43 | 7 |
| 7 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.57 | 7 |
| 8 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.86 | 7 |
| 9 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.14 | 7 |
| 10 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.57 | 7 |
| 11 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.43 | 7 |
| 12 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.86 | 7 |
| 13 | tabtransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.29 | 7 |
| 14 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.43 | 7 |
| 15 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 14.00 | 7 |
| 16 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 14.86 | 7 |
| 17 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 15.14 | 7 |
| 18 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 15.29 | 7 |
| 19 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 16.29 | 7 |
* #### 分類: small_datasets+binclass+numerical (包含 28 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.11 | 28 |
| 2 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.61 | 28 |
| 3 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.57 | 28 |
| 4 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.89 | 28 |
| 5 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.46 | 28 |
| 6 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.00 | 28 |
| 7 | tabtransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.04 | 28 |
| 8 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.36 | 28 |
| 9 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.86 | 28 |
| 10 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.96 | 28 |
| 11 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.04 | 28 |
| 12 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.14 | 28 |
| 13 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 12.36 | 28 |
| 14 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.39 | 28 |
| 15 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 14.14 | 28 |
| 16 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.21 | 28 |
| 17 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 14.79 | 28 |
| 18 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 14.82 | 28 |
| 19 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 16.25 | 28 |
* #### 分類: small_datasets+regression+balanced (包含 6 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.17 | 6 |
| 2 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 5.50 | 6 |
| 3 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 5.83 | 6 |
| 4 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 6.00 | 6 |
| 5 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.50 | 6 |
| 6 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 8.17 | 6 |
| 7 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.67 | 6 |
| 8 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.50 | 6 |
| 9 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 9.67 | 6 |
| 10 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.83 | 6 |
| 11 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.67 | 6 |
| 12 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.50 | 6 |
| 13 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 11.50 | 6 |
| 14 | tabtransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.67 | 6 |
| 15 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.00 | 6 |
| 16 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.17 | 6 |
| 17 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.17 | 6 |
| 18 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 16.50 | 6 |
| 19 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.00 | 6 |
* #### 分類: small_datasets+regression+categorical (包含 5 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.80 | 5 |
| 2 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.80 | 5 |
| 3 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 5.00 | 5 |
| 4 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 6.20 | 5 |
| 5 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 6.20 | 5 |
| 6 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 7.20 | 5 |
| 7 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 7.40 | 5 |
| 8 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 7.60 | 5 |
| 9 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.80 | 5 |
| 10 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 8.20 | 5 |
| 11 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.40 | 5 |
| 12 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.80 | 5 |
| 13 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.00 | 5 |
| 14 | tabtransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 12.40 | 5 |
| 15 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.40 | 5 |
| 16 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.60 | 5 |
| 17 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.60 | 5 |
| 18 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.20 | 5 |
| 19 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.40 | 5 |
* #### 分類: small_datasets+regression+numerical (包含 36 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.56 | 36 |
| 2 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 5.17 | 36 |
| 3 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 6.44 | 36 |
| 4 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 6.97 | 36 |
| 5 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 7.11 | 36 |
| 6 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.19 | 36 |
| 7 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 7.50 | 36 |
| 8 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.50 | 36 |
| 9 | tabtransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 7.78 | 36 |
| 10 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.22 | 36 |
| 11 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.17 | 36 |
| 12 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.81 | 36 |
| 13 | tabtransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.94 | 36 |
| 14 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.69 | 36 |
| 15 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.78 | 36 |
| 16 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.08 | 36 |
| 17 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.14 | 36 |
| 18 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.14 | 36 |
| 19 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.81 | 36 |
