GNN增強效果分析 - 模型: TABM

說明：
- 比較 tabm 模型的原始表現與GNN增強變體
- 包含5個參考模型（t2gformer, tabpfn, xgboost, catboost, lightgbm）
- 共17個競爭者：
  * tabm的7個配置：大訓練集baseline(1) + 小訓練集6變體(6)
  * 5個參考模型的兩種ratio：每個模型2個配置(10)
- 排名越小表示表現越好


* #### 分類: large_datasets+binclass+numerical (包含 6 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.83 | 6 |
| 2 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.67 | 6 |
| 3 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.33 | 6 |
| 4 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.33 | 6 |
| 5 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.17 | 6 |
| 6 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 6.83 | 6 |
| 7 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.67 | 6 |
| 8 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.67 | 6 |
| 9 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.33 | 6 |
| 10 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.17 | 6 |
| 11 | tabm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 12.00 | 6 |
| 12 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.67 | 6 |
| 13 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 13.17 | 6 |
| 14 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.67 | 6 |
| 15 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 14.67 | 6 |
| 16 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 16.50 | 6 |
| 17 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 16.67 | 6 |
| 18 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 17.17 | 6 |
| 19 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 17.50 | 6 |
* #### 分類: large_datasets+multiclass+numerical (包含 3 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.33 | 3 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.33 | 3 |
| 3 | tabm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.67 | 3 |
| 4 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.33 | 3 |
| 5 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.33 | 3 |
| 6 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.00 | 3 |
| 7 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.00 | 3 |
| 8 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.00 | 3 |
| 9 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.00 | 3 |
| 10 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.00 | 3 |
| 11 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.67 | 3 |
| 12 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.33 | 3 |
| 13 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.67 | 3 |
| 14 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 15.00 | 3 |
| 15 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 15.33 | 3 |
| 16 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 16.00 | 3 |
| 17 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 16.33 | 3 |
| 18 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 16.67 | 3 |
| 19 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 19.00 | 3 |
* #### 分類: large_datasets+regression+categorical (包含 1 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 1.00 | 1 |
| 2 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 2.00 | 1 |
| 3 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 3.00 | 1 |
| 4 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.00 | 1 |
| 5 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 5.00 | 1 |
| 6 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.00 | 1 |
| 7 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 7.00 | 1 |
| 8 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.00 | 1 |
| 9 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.00 | 1 |
| 10 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 10.00 | 1 |
| 11 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.00 | 1 |
| 12 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 12.00 | 1 |
| 13 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.00 | 1 |
| 14 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.00 | 1 |
| 15 | tabm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.00 | 1 |
| 16 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 16.00 | 1 |
| 17 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.00 | 1 |
| 18 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 18.00 | 1 |
| 19 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 19.00 | 1 |
* #### 分類: large_datasets+regression+numerical (包含 10 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 1.70 | 10 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.80 | 10 |
| 3 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.60 | 10 |
| 4 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.10 | 10 |
| 5 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 6.90 | 10 |
| 6 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 6.90 | 10 |
| 7 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 7.30 | 10 |
| 8 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 7.40 | 10 |
| 9 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 8.90 | 10 |
| 10 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.50 | 10 |
| 11 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.10 | 10 |
| 12 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.10 | 10 |
| 13 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.50 | 10 |
| 14 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.60 | 10 |
| 15 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.10 | 10 |
| 16 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.90 | 10 |
| 17 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.40 | 10 |
| 18 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.00 | 10 |
| 19 | tabm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.20 | 10 |
* #### 分類: small_datasets+binclass+balanced (包含 14 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.00 | 14 |
| 2 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.07 | 14 |
| 3 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.21 | 14 |
| 4 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.79 | 14 |
| 5 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.50 | 14 |
| 6 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.79 | 14 |
| 7 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.64 | 14 |
| 8 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.50 | 14 |
| 9 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.64 | 14 |
| 10 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.00 | 14 |
| 11 | tabm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.57 | 14 |
| 12 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.79 | 14 |
| 13 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 12.00 | 14 |
| 14 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.07 | 14 |
| 15 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 12.50 | 14 |
| 16 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.57 | 14 |
| 17 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 13.21 | 14 |
| 18 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 13.57 | 14 |
| 19 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 14.57 | 14 |
* #### 分類: small_datasets+binclass+categorical (包含 7 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.71 | 7 |
| 2 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.57 | 7 |
| 3 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.86 | 7 |
| 4 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.14 | 7 |
| 5 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.29 | 7 |
| 6 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.71 | 7 |
| 7 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.00 | 7 |
| 8 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.00 | 7 |
| 9 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.14 | 7 |
| 10 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.43 | 7 |
| 11 | tabm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.57 | 7 |
| 12 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.86 | 7 |
| 13 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.14 | 7 |
| 14 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.29 | 7 |
| 15 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 10.86 | 7 |
| 16 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 13.29 | 7 |
| 17 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 13.43 | 7 |
| 18 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 14.29 | 7 |
| 19 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 14.43 | 7 |
* #### 分類: small_datasets+binclass+numerical (包含 28 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.07 | 28 |
| 2 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.14 | 28 |
| 3 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.57 | 28 |
| 4 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.79 | 28 |
| 5 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.39 | 28 |
| 6 | tabm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.39 | 28 |
| 7 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.43 | 28 |
| 8 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.68 | 28 |
| 9 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.29 | 28 |
| 10 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.36 | 28 |
| 11 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.86 | 28 |
| 12 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.00 | 28 |
| 13 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 12.71 | 28 |
| 14 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.71 | 28 |
| 15 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.82 | 28 |
| 16 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 13.75 | 28 |
| 17 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 14.04 | 28 |
| 18 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 14.61 | 28 |
| 19 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 15.39 | 28 |
* #### 分類: small_datasets+regression+balanced (包含 6 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.83 | 6 |
| 2 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 6.83 | 6 |
| 3 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 7.33 | 6 |
| 4 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.50 | 6 |
| 5 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 8.17 | 6 |
| 6 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.50 | 6 |
| 7 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.67 | 6 |
| 8 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.33 | 6 |
| 9 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 9.67 | 6 |
| 10 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.83 | 6 |
| 11 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 10.17 | 6 |
| 12 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.83 | 6 |
| 13 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.83 | 6 |
| 14 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.00 | 6 |
| 15 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.17 | 6 |
| 16 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.17 | 6 |
| 17 | tabm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 12.33 | 6 |
| 18 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.50 | 6 |
| 19 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.33 | 6 |
* #### 分類: small_datasets+regression+categorical (包含 5 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.20 | 5 |
| 2 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 4.00 | 5 |
| 3 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 4.20 | 5 |
| 4 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.80 | 5 |
| 5 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.60 | 5 |
| 6 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 6.20 | 5 |
| 7 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.80 | 5 |
| 8 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 8.80 | 5 |
| 9 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.80 | 5 |
| 10 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.20 | 5 |
| 11 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.60 | 5 |
| 12 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.60 | 5 |
| 13 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 10.80 | 5 |
| 14 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.00 | 5 |
| 15 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.20 | 5 |
| 16 | tabm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.40 | 5 |
| 17 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.40 | 5 |
| 18 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.00 | 5 |
| 19 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.40 | 5 |
* #### 分類: small_datasets+regression+numerical (包含 36 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.97 | 36 |
| 2 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.67 | 36 |
| 3 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 5.28 | 36 |
| 4 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 5.75 | 36 |
| 5 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.64 | 36 |
| 6 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 8.06 | 36 |
| 7 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 8.42 | 36 |
| 8 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.06 | 36 |
| 9 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.11 | 36 |
| 10 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 9.28 | 36 |
| 11 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.39 | 36 |
| 12 | tabm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.42 | 36 |
| 13 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.42 | 36 |
| 14 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.19 | 36 |
| 15 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.39 | 36 |
| 16 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.72 | 36 |
| 17 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.86 | 36 |
| 18 | tabm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.17 | 36 |
| 19 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.22 | 36 |
