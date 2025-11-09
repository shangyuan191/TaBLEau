GNN增強效果分析 - 模型: TROMPT

說明：
- 比較 trompt 模型的原始表現與GNN增強變體
- 包含5個參考模型（t2gformer, tabpfn, xgboost, catboost, lightgbm）
- 共17個競爭者：
  * trompt的7個配置：大訓練集baseline(1) + 小訓練集6變體(6)
  * 5個參考模型的兩種ratio：每個模型2個配置(10)
- 排名越小表示表現越好


* #### 分類: large_datasets+binclass+numerical (包含 6 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.83 | 6 |
| 2 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.83 | 6 |
| 3 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.67 | 6 |
| 4 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.83 | 6 |
| 5 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.17 | 6 |
| 6 | trompt<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.33 | 6 |
| 7 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.17 | 6 |
| 8 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.67 | 6 |
| 9 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 9.83 | 6 |
| 10 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.83 | 6 |
| 11 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 10.50 | 6 |
| 12 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.67 | 6 |
| 13 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.83 | 6 |
| 14 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.17 | 6 |
| 15 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 14.17 | 6 |
| 16 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 15.67 | 6 |
| 17 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.50 | 6 |
| 18 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 18.00 | 6 |
| 19 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 18.33 | 6 |
* #### 分類: large_datasets+multiclass+numerical (包含 3 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.33 | 3 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.33 | 3 |
| 3 | trompt<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.33 | 3 |
| 4 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.00 | 3 |
| 5 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.33 | 3 |
| 6 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.67 | 3 |
| 7 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.33 | 3 |
| 8 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 9.67 | 3 |
| 9 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.33 | 3 |
| 10 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.33 | 3 |
| 11 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.00 | 3 |
| 12 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.67 | 3 |
| 13 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 12.33 | 3 |
| 14 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 12.33 | 3 |
| 15 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.33 | 3 |
| 16 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 16.67 | 3 |
| 17 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 16.67 | 3 |
| 18 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 17.33 | 3 |
| 19 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 19.00 | 3 |
* #### 分類: large_datasets+regression+categorical (包含 1 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 1.00 | 1 |
| 2 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 2.00 | 1 |
| 3 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.00 | 1 |
| 4 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.00 | 1 |
| 5 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.00 | 1 |
| 6 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 6.00 | 1 |
| 7 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.00 | 1 |
| 8 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.00 | 1 |
| 9 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 9.00 | 1 |
| 10 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.00 | 1 |
| 11 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.00 | 1 |
| 12 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 12.00 | 1 |
| 13 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.00 | 1 |
| 14 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 14.00 | 1 |
| 15 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.00 | 1 |
| 16 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.00 | 1 |
| 17 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 17.00 | 1 |
| 18 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 18.00 | 1 |
| 19 | trompt<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 19.00 | 1 |
* #### 分類: large_datasets+regression+numerical (包含 10 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 1.80 | 10 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.90 | 10 |
| 3 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 3.20 | 10 |
| 4 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.00 | 10 |
| 5 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.10 | 10 |
| 6 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 6.90 | 10 |
| 7 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 7.10 | 10 |
| 8 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 10.40 | 10 |
| 9 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 10.70 | 10 |
| 10 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.80 | 10 |
| 11 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.80 | 10 |
| 12 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.90 | 10 |
| 13 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.30 | 10 |
| 14 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.30 | 10 |
| 15 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.00 | 10 |
| 16 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.90 | 10 |
| 17 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.10 | 10 |
| 18 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.20 | 10 |
| 19 | trompt<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.60 | 10 |
* #### 分類: small_datasets+binclass+balanced (包含 14 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.07 | 14 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.07 | 14 |
| 3 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.07 | 14 |
| 4 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.14 | 14 |
| 5 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.14 | 14 |
| 6 | trompt<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.29 | 14 |
| 7 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.71 | 14 |
| 8 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.14 | 14 |
| 9 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.86 | 14 |
| 10 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.29 | 14 |
| 11 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.43 | 14 |
| 12 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.79 | 14 |
| 13 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.79 | 14 |
| 14 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 11.79 | 14 |
| 15 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 12.71 | 14 |
| 16 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.86 | 14 |
| 17 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 13.29 | 14 |
| 18 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 13.50 | 14 |
| 19 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 15.07 | 14 |
* #### 分類: small_datasets+binclass+categorical (包含 7 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.00 | 7 |
| 2 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.43 | 7 |
| 3 | trompt<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.00 | 7 |
| 4 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 8.00 | 7 |
| 5 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.00 | 7 |
| 6 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 8.43 | 7 |
| 7 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.57 | 7 |
| 8 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.57 | 7 |
| 9 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.71 | 7 |
| 10 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.29 | 7 |
| 11 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.14 | 7 |
| 12 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.14 | 7 |
| 13 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.43 | 7 |
| 14 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.00 | 7 |
| 15 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.14 | 7 |
| 16 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 12.00 | 7 |
| 17 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.43 | 7 |
| 18 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 15.00 | 7 |
| 19 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 15.71 | 7 |
* #### 分類: small_datasets+binclass+numerical (包含 28 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.50 | 28 |
| 2 | trompt<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.14 | 28 |
| 3 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.61 | 28 |
| 4 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.93 | 28 |
| 5 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.21 | 28 |
| 6 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.89 | 28 |
| 7 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 10.00 | 28 |
| 8 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.29 | 28 |
| 9 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.61 | 28 |
| 10 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.14 | 28 |
| 11 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 11.43 | 28 |
| 12 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.64 | 28 |
| 13 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.96 | 28 |
| 14 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.39 | 28 |
| 15 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 13.00 | 28 |
| 16 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 13.36 | 28 |
| 17 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.18 | 28 |
| 18 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.71 | 28 |
| 19 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 17.00 | 28 |
* #### 分類: small_datasets+regression+balanced (包含 6 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 2.50 | 6 |
| 2 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 5.33 | 6 |
| 3 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.67 | 6 |
| 4 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 7.83 | 6 |
| 5 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.67 | 6 |
| 6 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.00 | 6 |
| 7 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 9.17 | 6 |
| 8 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.50 | 6 |
| 9 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.83 | 6 |
| 10 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 10.00 | 6 |
| 11 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.17 | 6 |
| 12 | trompt<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.83 | 6 |
| 13 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.17 | 6 |
| 14 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.33 | 6 |
| 15 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.33 | 6 |
| 16 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.00 | 6 |
| 17 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 12.50 | 6 |
| 18 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.33 | 6 |
| 19 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.83 | 6 |
* #### 分類: small_datasets+regression+categorical (包含 5 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.20 | 5 |
| 2 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.60 | 5 |
| 3 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 4.80 | 5 |
| 4 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 5.40 | 5 |
| 5 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.60 | 5 |
| 6 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 6.00 | 5 |
| 7 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.20 | 5 |
| 8 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.20 | 5 |
| 9 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.60 | 5 |
| 10 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.20 | 5 |
| 11 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.80 | 5 |
| 12 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 10.20 | 5 |
| 13 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 10.40 | 5 |
| 14 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.00 | 5 |
| 15 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.00 | 5 |
| 16 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.20 | 5 |
| 17 | trompt<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.40 | 5 |
| 18 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.80 | 5 |
| 19 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.40 | 5 |
* #### 分類: small_datasets+regression+numerical (包含 36 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.50 | 36 |
| 2 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 3.89 | 36 |
| 3 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.39 | 36 |
| 4 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.28 | 36 |
| 5 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 8.17 | 36 |
| 6 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.44 | 36 |
| 7 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.67 | 36 |
| 8 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.72 | 36 |
| 9 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 8.72 | 36 |
| 10 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.14 | 36 |
| 11 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 9.36 | 36 |
| 12 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.19 | 36 |
| 13 | trompt<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 10.50 | 36 |
| 14 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.06 | 36 |
| 15 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.42 | 36 |
| 16 | trompt<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.67 | 36 |
| 17 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.89 | 36 |
| 18 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.97 | 36 |
| 19 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.03 | 36 |
