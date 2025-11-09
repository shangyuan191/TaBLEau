GNN增強效果分析 - 模型: FTTRANSFORMER

說明：
- 比較 fttransformer 模型的原始表現與GNN增強變體
- 包含5個參考模型（t2gformer, tabpfn, xgboost, catboost, lightgbm）
- 共17個競爭者：
  * fttransformer的7個配置：大訓練集baseline(1) + 小訓練集6變體(6)
  * 5個參考模型的兩種ratio：每個模型2個配置(10)
- 排名越小表示表現越好


* #### 分類: large_datasets+binclass+numerical (包含 6 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.83 | 6 |
| 2 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.83 | 6 |
| 3 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.67 | 6 |
| 4 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.67 | 6 |
| 5 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.83 | 6 |
| 6 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.50 | 6 |
| 7 | fttransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.50 | 6 |
| 8 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.67 | 6 |
| 9 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.67 | 6 |
| 10 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.67 | 6 |
| 11 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.67 | 6 |
| 12 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.50 | 6 |
| 13 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 13.50 | 6 |
| 14 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 13.83 | 6 |
| 15 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 14.83 | 6 |
| 16 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 15.17 | 6 |
| 17 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.50 | 6 |
| 18 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 16.17 | 6 |
| 19 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 18.00 | 6 |
* #### 分類: large_datasets+multiclass+numerical (包含 3 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.33 | 3 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.00 | 3 |
| 3 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.67 | 3 |
| 4 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.00 | 3 |
| 5 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.33 | 3 |
| 6 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.33 | 3 |
| 7 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.33 | 3 |
| 8 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.33 | 3 |
| 9 | fttransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.67 | 3 |
| 10 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.67 | 3 |
| 11 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.33 | 3 |
| 12 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.00 | 3 |
| 13 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.67 | 3 |
| 14 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 14.67 | 3 |
| 15 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.00 | 3 |
| 16 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 15.00 | 3 |
| 17 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 15.67 | 3 |
| 18 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 16.67 | 3 |
| 19 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 18.33 | 3 |
* #### 分類: large_datasets+regression+categorical (包含 1 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 1.00 | 1 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.00 | 1 |
| 3 | fttransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.00 | 1 |
| 4 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.00 | 1 |
| 5 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.00 | 1 |
| 6 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 6.00 | 1 |
| 7 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 7.00 | 1 |
| 8 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 8.00 | 1 |
| 9 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 9.00 | 1 |
| 10 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.00 | 1 |
| 11 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.00 | 1 |
| 12 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 12.00 | 1 |
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
| 1 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 2.10 | 10 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.60 | 10 |
| 3 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.30 | 10 |
| 4 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.90 | 10 |
| 5 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 6.00 | 10 |
| 6 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 6.90 | 10 |
| 7 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 7.10 | 10 |
| 8 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 7.60 | 10 |
| 9 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 8.40 | 10 |
| 10 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.70 | 10 |
| 11 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.00 | 10 |
| 12 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.60 | 10 |
| 13 | fttransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 12.80 | 10 |
| 14 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.30 | 10 |
| 15 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.50 | 10 |
| 16 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.80 | 10 |
| 17 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.70 | 10 |
| 18 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.10 | 10 |
| 19 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.60 | 10 |
* #### 分類: small_datasets+binclass+balanced (包含 14 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.00 | 14 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.64 | 14 |
| 3 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.71 | 14 |
| 4 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.57 | 14 |
| 5 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.00 | 14 |
| 6 | fttransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.00 | 14 |
| 7 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.93 | 14 |
| 8 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.29 | 14 |
| 9 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.00 | 14 |
| 10 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.43 | 14 |
| 11 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.57 | 14 |
| 12 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.64 | 14 |
| 13 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.00 | 14 |
| 14 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 12.93 | 14 |
| 15 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 13.21 | 14 |
| 16 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.64 | 14 |
| 17 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 14.79 | 14 |
| 18 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 15.64 | 14 |
| 19 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 17.00 | 14 |
* #### 分類: small_datasets+binclass+categorical (包含 7 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.00 | 7 |
| 2 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 6.29 | 7 |
| 3 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 6.71 | 7 |
| 4 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.29 | 7 |
| 5 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.43 | 7 |
| 6 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.57 | 7 |
| 7 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.86 | 7 |
| 8 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.43 | 7 |
| 9 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.71 | 7 |
| 10 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.29 | 7 |
| 11 | fttransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.29 | 7 |
| 12 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.43 | 7 |
| 13 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.57 | 7 |
| 14 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 10.57 | 7 |
| 15 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 11.29 | 7 |
| 16 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 14.57 | 7 |
| 17 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.43 | 7 |
| 18 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 15.57 | 7 |
| 19 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 18.71 | 7 |
* #### 分類: small_datasets+binclass+numerical (包含 28 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.21 | 28 |
| 2 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.04 | 28 |
| 3 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.68 | 28 |
| 4 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.11 | 28 |
| 5 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.68 | 28 |
| 6 | fttransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.71 | 28 |
| 7 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.18 | 28 |
| 8 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.75 | 28 |
| 9 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.04 | 28 |
| 10 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.36 | 28 |
| 11 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.46 | 28 |
| 12 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.46 | 28 |
| 13 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.57 | 28 |
| 14 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 12.82 | 28 |
| 15 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 13.14 | 28 |
| 16 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.39 | 28 |
| 17 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 14.54 | 28 |
| 18 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 14.64 | 28 |
| 19 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 17.21 | 28 |
* #### 分類: small_datasets+regression+balanced (包含 6 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 3.67 | 6 |
| 2 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 3.83 | 6 |
| 3 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.33 | 6 |
| 4 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 5.67 | 6 |
| 5 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 6.50 | 6 |
| 6 | fttransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.83 | 6 |
| 7 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.50 | 6 |
| 8 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 9.50 | 6 |
| 9 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 10.00 | 6 |
| 10 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.33 | 6 |
| 11 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.50 | 6 |
| 12 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.17 | 6 |
| 13 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 12.00 | 6 |
| 14 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 12.33 | 6 |
| 15 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.33 | 6 |
| 16 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.67 | 6 |
| 17 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.33 | 6 |
| 18 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 16.50 | 6 |
| 19 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.00 | 6 |
* #### 分類: small_datasets+regression+categorical (包含 5 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 3.80 | 5 |
| 2 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.00 | 5 |
| 3 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.60 | 5 |
| 4 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 5.00 | 5 |
| 5 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 6.00 | 5 |
| 6 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 6.40 | 5 |
| 7 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.60 | 5 |
| 8 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 7.20 | 5 |
| 9 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.20 | 5 |
| 10 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.60 | 5 |
| 11 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.00 | 5 |
| 12 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.60 | 5 |
| 13 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.20 | 5 |
| 14 | fttransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 13.60 | 5 |
| 15 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.40 | 5 |
| 16 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.40 | 5 |
| 17 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.60 | 5 |
| 18 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.20 | 5 |
| 19 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.60 | 5 |
* #### 分類: small_datasets+regression+numerical (包含 36 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.47 | 36 |
| 2 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 4.58 | 36 |
| 3 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 5.36 | 36 |
| 4 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 6.78 | 36 |
| 5 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 7.14 | 36 |
| 6 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.36 | 36 |
| 7 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.78 | 36 |
| 8 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 8.56 | 36 |
| 9 | fttransformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 8.75 | 36 |
| 10 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.89 | 36 |
| 11 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.61 | 36 |
| 12 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.11 | 36 |
| 13 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.28 | 36 |
| 14 | fttransformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 12.69 | 36 |
| 15 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.64 | 36 |
| 16 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.83 | 36 |
| 17 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.25 | 36 |
| 18 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.33 | 36 |
| 19 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.58 | 36 |
