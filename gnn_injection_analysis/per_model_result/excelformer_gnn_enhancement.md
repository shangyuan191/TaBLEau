GNN增強效果分析 - 模型: EXCELFORMER

說明：
- 比較 excelformer 模型的原始表現與GNN增強變體
- 包含5個參考模型（t2gformer, tabpfn, xgboost, catboost, lightgbm）
- 共17個競爭者：
  * excelformer的7個配置：大訓練集baseline(1) + 小訓練集6變體(6)
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
| 6 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.17 | 6 |
| 7 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.17 | 6 |
| 8 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.33 | 6 |
| 9 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.83 | 6 |
| 10 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.83 | 6 |
| 11 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 13.00 | 6 |
| 12 | excelformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 13.00 | 6 |
| 13 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.17 | 6 |
| 14 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 13.17 | 6 |
| 15 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 13.83 | 6 |
| 16 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.17 | 6 |
| 17 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 16.33 | 6 |
| 18 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 16.67 | 6 |
| 19 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 17.00 | 6 |
* #### 分類: large_datasets+multiclass+numerical (包含 3 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.33 | 3 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.00 | 3 |
| 3 | excelformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.67 | 3 |
| 4 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.00 | 3 |
| 5 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.33 | 3 |
| 6 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.67 | 3 |
| 7 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.67 | 3 |
| 8 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.67 | 3 |
| 9 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.33 | 3 |
| 10 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.00 | 3 |
| 11 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.67 | 3 |
| 12 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 12.33 | 3 |
| 13 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.67 | 3 |
| 14 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 13.67 | 3 |
| 15 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 14.00 | 3 |
| 16 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.33 | 3 |
| 17 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 17.00 | 3 |
| 18 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 18.00 | 3 |
| 19 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 18.67 | 3 |
* #### 分類: large_datasets+regression+categorical (包含 1 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 1.00 | 1 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.00 | 1 |
| 3 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.00 | 1 |
| 4 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.00 | 1 |
| 5 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 5.00 | 1 |
| 6 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 6.00 | 1 |
| 7 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.00 | 1 |
| 8 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 8.00 | 1 |
| 9 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.00 | 1 |
| 10 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.00 | 1 |
| 11 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 11.00 | 1 |
| 12 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.00 | 1 |
| 13 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.00 | 1 |
| 14 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.00 | 1 |
| 15 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.00 | 1 |
| 16 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.00 | 1 |
| 17 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 17.00 | 1 |
| 18 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 18.00 | 1 |
| 19 | excelformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 19.00 | 1 |
* #### 分類: large_datasets+regression+numerical (包含 10 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 1.60 | 10 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.60 | 10 |
| 3 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.30 | 10 |
| 4 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.70 | 10 |
| 5 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 5.70 | 10 |
| 6 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 6.80 | 10 |
| 7 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.30 | 10 |
| 8 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.70 | 10 |
| 9 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.80 | 10 |
| 10 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.00 | 10 |
| 11 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 11.00 | 10 |
| 12 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.10 | 10 |
| 13 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 11.10 | 10 |
| 14 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 13.80 | 10 |
| 15 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.40 | 10 |
| 16 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.60 | 10 |
| 17 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.80 | 10 |
| 18 | excelformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.90 | 10 |
| 19 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.80 | 10 |
* #### 分類: small_datasets+binclass+balanced (包含 14 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.07 | 14 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.71 | 14 |
| 3 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.86 | 14 |
| 4 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.00 | 14 |
| 5 | excelformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.21 | 14 |
| 6 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.21 | 14 |
| 7 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.50 | 14 |
| 8 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.86 | 14 |
| 9 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.86 | 14 |
| 10 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.86 | 14 |
| 11 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.43 | 14 |
| 12 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.57 | 14 |
| 13 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.93 | 14 |
| 14 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.71 | 14 |
| 15 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 13.14 | 14 |
| 16 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 13.71 | 14 |
| 17 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 13.79 | 14 |
| 18 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 14.36 | 14 |
| 19 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 15.21 | 14 |
* #### 分類: small_datasets+binclass+categorical (包含 7 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.43 | 7 |
| 2 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 6.86 | 7 |
| 3 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.14 | 7 |
| 4 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.71 | 7 |
| 5 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.86 | 7 |
| 6 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.29 | 7 |
| 7 | excelformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.29 | 7 |
| 8 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.57 | 7 |
| 9 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.86 | 7 |
| 10 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.86 | 7 |
| 11 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 8.86 | 7 |
| 12 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.57 | 7 |
| 13 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.00 | 7 |
| 14 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.14 | 7 |
| 15 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 10.86 | 7 |
| 16 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.71 | 7 |
| 17 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 12.71 | 7 |
| 18 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 18.00 | 7 |
| 19 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 18.29 | 7 |
* #### 分類: small_datasets+binclass+numerical (包含 28 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.29 | 28 |
| 2 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.18 | 28 |
| 3 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.68 | 28 |
| 4 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.11 | 28 |
| 5 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.68 | 28 |
| 6 | excelformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.07 | 28 |
| 7 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.71 | 28 |
| 8 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.18 | 28 |
| 9 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.46 | 28 |
| 10 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.79 | 28 |
| 11 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.61 | 28 |
| 12 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 12.18 | 28 |
| 13 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.32 | 28 |
| 14 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 13.07 | 28 |
| 15 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 13.18 | 28 |
| 16 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.36 | 28 |
| 17 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.36 | 28 |
| 18 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 14.25 | 28 |
| 19 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 15.54 | 28 |
* #### 分類: small_datasets+regression+balanced (包含 6 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.17 | 6 |
| 2 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 6.67 | 6 |
| 3 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 7.33 | 6 |
| 4 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.50 | 6 |
| 5 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.50 | 6 |
| 6 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 8.50 | 6 |
| 7 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.00 | 6 |
| 8 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 9.33 | 6 |
| 9 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.33 | 6 |
| 10 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.67 | 6 |
| 11 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 10.67 | 6 |
| 12 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.83 | 6 |
| 13 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.33 | 6 |
| 14 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.50 | 6 |
| 15 | excelformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.50 | 6 |
| 16 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.67 | 6 |
| 17 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.00 | 6 |
| 18 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.50 | 6 |
| 19 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.00 | 6 |
* #### 分類: small_datasets+regression+categorical (包含 5 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.80 | 5 |
| 2 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 4.80 | 5 |
| 3 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 5.40 | 5 |
| 4 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.80 | 5 |
| 5 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 6.00 | 5 |
| 6 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.40 | 5 |
| 7 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 7.80 | 5 |
| 8 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 7.80 | 5 |
| 9 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 8.00 | 5 |
| 10 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.20 | 5 |
| 11 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.00 | 5 |
| 12 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.80 | 5 |
| 13 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.20 | 5 |
| 14 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.60 | 5 |
| 15 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.00 | 5 |
| 16 | excelformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.20 | 5 |
| 17 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.80 | 5 |
| 18 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.20 | 5 |
| 19 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.20 | 5 |
* #### 分類: small_datasets+regression+numerical (包含 36 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 3.61 | 36 |
| 2 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.69 | 36 |
| 3 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 5.92 | 36 |
| 4 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.33 | 36 |
| 5 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 7.56 | 36 |
| 6 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 7.86 | 36 |
| 7 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.28 | 36 |
| 8 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.33 | 36 |
| 9 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.47 | 36 |
| 10 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.75 | 36 |
| 11 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.75 | 36 |
| 12 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 9.94 | 36 |
| 13 | excelformer<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 10.36 | 36 |
| 14 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.44 | 36 |
| 15 | excelformer<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.42 | 36 |
| 16 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.47 | 36 |
| 17 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.92 | 36 |
| 18 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.11 | 36 |
| 19 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.78 | 36 |
