GNN增強效果分析 - 模型: TABNET

說明：
- 比較 tabnet 模型的原始表現與GNN增強變體
- 包含5個參考模型（t2gformer, tabpfn, xgboost, catboost, lightgbm）
- 共17個競爭者：
  * tabnet的7個配置：大訓練集baseline(1) + 小訓練集6變體(6)
  * 5個參考模型的兩種ratio：每個模型2個配置(10)
- 排名越小表示表現越好


* #### 分類: large_datasets+binclass+numerical (包含 6 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.00 | 6 |
| 2 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.83 | 6 |
| 3 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.67 | 6 |
| 4 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.83 | 6 |
| 5 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.67 | 6 |
| 6 | tabnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.17 | 6 |
| 7 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.67 | 6 |
| 8 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.50 | 6 |
| 9 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.83 | 6 |
| 10 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.17 | 6 |
| 11 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.00 | 6 |
| 12 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 12.83 | 6 |
| 13 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.17 | 6 |
| 14 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 14.33 | 6 |
| 15 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 14.50 | 6 |
| 16 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.33 | 6 |
| 17 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 16.50 | 6 |
| 18 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 18.33 | 6 |
| 19 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 18.67 | 6 |
* #### 分類: large_datasets+multiclass+numerical (包含 3 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.33 | 3 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.67 | 3 |
| 3 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.33 | 3 |
| 4 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.67 | 3 |
| 5 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.33 | 3 |
| 6 | tabnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.67 | 3 |
| 7 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.67 | 3 |
| 8 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.67 | 3 |
| 9 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.67 | 3 |
| 10 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.00 | 3 |
| 11 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.33 | 3 |
| 12 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.67 | 3 |
| 13 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.33 | 3 |
| 14 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.00 | 3 |
| 15 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 14.67 | 3 |
| 16 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 16.00 | 3 |
| 17 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 17.00 | 3 |
| 18 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 18.00 | 3 |
| 19 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 19.00 | 3 |
* #### 分類: large_datasets+regression+categorical (包含 1 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 1.00 | 1 |
| 2 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 2.00 | 1 |
| 3 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 3.00 | 1 |
| 4 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.00 | 1 |
| 5 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 5.00 | 1 |
| 6 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.00 | 1 |
| 7 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 7.00 | 1 |
| 8 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.00 | 1 |
| 9 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.00 | 1 |
| 10 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 10.00 | 1 |
| 11 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.00 | 1 |
| 12 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 12.00 | 1 |
| 13 | tabnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 13.00 | 1 |
| 14 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.00 | 1 |
| 15 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.00 | 1 |
| 16 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 16.00 | 1 |
| 17 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.00 | 1 |
| 18 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 18.00 | 1 |
| 19 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 19.00 | 1 |
* #### 分類: large_datasets+regression+numerical (包含 10 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 2.00 | 10 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.60 | 10 |
| 3 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 2.80 | 10 |
| 4 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 3.00 | 10 |
| 5 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 5.30 | 10 |
| 6 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.60 | 10 |
| 7 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 7.80 | 10 |
| 8 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 7.80 | 10 |
| 9 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.20 | 10 |
| 10 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.80 | 10 |
| 11 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.00 | 10 |
| 12 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.20 | 10 |
| 13 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.40 | 10 |
| 14 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.20 | 10 |
| 15 | tabnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.40 | 10 |
| 16 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.50 | 10 |
| 17 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.10 | 10 |
| 18 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 16.50 | 10 |
| 19 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.80 | 10 |
* #### 分類: small_datasets+binclass+balanced (包含 14 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.07 | 14 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.93 | 14 |
| 3 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.00 | 14 |
| 4 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.86 | 14 |
| 5 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.21 | 14 |
| 6 | tabnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.50 | 14 |
| 7 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.79 | 14 |
| 8 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.43 | 14 |
| 9 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.57 | 14 |
| 10 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.71 | 14 |
| 11 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.00 | 14 |
| 12 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.36 | 14 |
| 13 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 11.36 | 14 |
| 14 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.50 | 14 |
| 15 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 11.93 | 14 |
| 16 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.79 | 14 |
| 17 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 13.29 | 14 |
| 18 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 17.29 | 14 |
| 19 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 17.43 | 14 |
* #### 分類: small_datasets+binclass+categorical (包含 7 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.43 | 7 |
| 2 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 6.00 | 7 |
| 3 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.43 | 7 |
| 4 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.00 | 7 |
| 5 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.14 | 7 |
| 6 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.29 | 7 |
| 7 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.57 | 7 |
| 8 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.71 | 7 |
| 9 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.14 | 7 |
| 10 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.57 | 7 |
| 11 | tabnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.86 | 7 |
| 12 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.43 | 7 |
| 13 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.86 | 7 |
| 14 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.00 | 7 |
| 15 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 13.86 | 7 |
| 16 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 14.43 | 7 |
| 17 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 14.71 | 7 |
| 18 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 17.71 | 7 |
| 19 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 18.86 | 7 |
* #### 分類: small_datasets+binclass+numerical (包含 28 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.29 | 28 |
| 2 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.14 | 28 |
| 3 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.64 | 28 |
| 4 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.00 | 28 |
| 5 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.57 | 28 |
| 6 | tabnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.32 | 28 |
| 7 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.46 | 28 |
| 8 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.14 | 28 |
| 9 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.54 | 28 |
| 10 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.61 | 28 |
| 11 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.79 | 28 |
| 12 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.32 | 28 |
| 13 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 11.36 | 28 |
| 14 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 12.71 | 28 |
| 15 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 12.75 | 28 |
| 16 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.93 | 28 |
| 17 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.46 | 28 |
| 18 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 17.25 | 28 |
| 19 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 17.71 | 28 |
* #### 分類: small_datasets+regression+balanced (包含 6 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.33 | 6 |
| 2 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 6.83 | 6 |
| 3 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 7.17 | 6 |
| 4 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.50 | 6 |
| 5 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 8.67 | 6 |
| 6 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 8.67 | 6 |
| 7 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 8.83 | 6 |
| 8 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.17 | 6 |
| 9 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.50 | 6 |
| 10 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 10.33 | 6 |
| 11 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.33 | 6 |
| 12 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.50 | 6 |
| 13 | tabnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.67 | 6 |
| 14 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.67 | 6 |
| 15 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.83 | 6 |
| 16 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.17 | 6 |
| 17 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.33 | 6 |
| 18 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 16.00 | 6 |
| 19 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.50 | 6 |
* #### 分類: small_datasets+regression+categorical (包含 5 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.60 | 5 |
| 2 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 4.20 | 5 |
| 3 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.40 | 5 |
| 4 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.40 | 5 |
| 5 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 6.20 | 5 |
| 6 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 6.60 | 5 |
| 7 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.20 | 5 |
| 8 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 7.40 | 5 |
| 9 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 8.80 | 5 |
| 10 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.20 | 5 |
| 11 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.40 | 5 |
| 12 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.00 | 5 |
| 13 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.20 | 5 |
| 14 | tabnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.20 | 5 |
| 15 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.20 | 5 |
| 16 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.60 | 5 |
| 17 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.80 | 5 |
| 18 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.00 | 5 |
| 19 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.60 | 5 |
* #### 分類: small_datasets+regression+numerical (包含 36 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.44 | 36 |
| 2 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.67 | 36 |
| 3 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 5.47 | 36 |
| 4 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 5.78 | 36 |
| 5 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 7.44 | 36 |
| 6 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.61 | 36 |
| 7 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 8.33 | 36 |
| 8 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 8.69 | 36 |
| 9 | tabnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.83 | 36 |
| 10 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.92 | 36 |
| 11 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.33 | 36 |
| 12 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.89 | 36 |
| 13 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.94 | 36 |
| 14 | tabnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 13.78 | 36 |
| 15 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.47 | 36 |
| 16 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.72 | 36 |
| 17 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.97 | 36 |
| 18 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.14 | 36 |
| 19 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.56 | 36 |
