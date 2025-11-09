GNN增強效果分析 - 模型: RESNET

說明：
- 比較 resnet 模型的原始表現與GNN增強變體
- 包含5個參考模型（t2gformer, tabpfn, xgboost, catboost, lightgbm）
- 共17個競爭者：
  * resnet的7個配置：大訓練集baseline(1) + 小訓練集6變體(6)
  * 5個參考模型的兩種ratio：每個模型2個配置(10)
- 排名越小表示表現越好


* #### 分類: large_datasets+binclass+numerical (包含 6 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.17 | 6 |
| 2 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.17 | 6 |
| 3 | resnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.67 | 6 |
| 4 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.67 | 6 |
| 5 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.83 | 6 |
| 6 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.83 | 6 |
| 7 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.33 | 6 |
| 8 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.17 | 6 |
| 9 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.67 | 6 |
| 10 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.00 | 6 |
| 11 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.00 | 6 |
| 12 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 11.83 | 6 |
| 13 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 12.00 | 6 |
| 14 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.33 | 6 |
| 15 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 14.67 | 6 |
| 16 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 16.17 | 6 |
| 17 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 16.50 | 6 |
| 18 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.50 | 6 |
| 19 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 18.50 | 6 |
* #### 分類: large_datasets+multiclass+numerical (包含 3 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 1.33 | 3 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.00 | 3 |
| 3 | resnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.67 | 3 |
| 4 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.67 | 3 |
| 5 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.33 | 3 |
| 6 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.00 | 3 |
| 7 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.67 | 3 |
| 8 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.67 | 3 |
| 9 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.33 | 3 |
| 10 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.33 | 3 |
| 11 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.33 | 3 |
| 12 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.67 | 3 |
| 13 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 12.33 | 3 |
| 14 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.33 | 3 |
| 15 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 14.67 | 3 |
| 16 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 16.00 | 3 |
| 17 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 16.33 | 3 |
| 18 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 18.00 | 3 |
| 19 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 18.33 | 3 |
* #### 分類: large_datasets+regression+categorical (包含 1 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 1.00 | 1 |
| 2 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 2.00 | 1 |
| 3 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.00 | 1 |
| 4 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.00 | 1 |
| 5 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.00 | 1 |
| 6 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 6.00 | 1 |
| 7 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 7.00 | 1 |
| 8 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.00 | 1 |
| 9 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.00 | 1 |
| 10 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 10.00 | 1 |
| 11 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.00 | 1 |
| 12 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 12.00 | 1 |
| 13 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.00 | 1 |
| 14 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.00 | 1 |
| 15 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.00 | 1 |
| 16 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.00 | 1 |
| 17 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 17.00 | 1 |
| 18 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 18.00 | 1 |
| 19 | resnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 19.00 | 1 |
* #### 分類: large_datasets+regression+numerical (包含 10 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 2.30 | 10 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 2.30 | 10 |
| 3 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.00 | 10 |
| 4 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.50 | 10 |
| 5 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 5.70 | 10 |
| 6 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 6.20 | 10 |
| 7 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 6.80 | 10 |
| 8 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 9.20 | 10 |
| 9 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.50 | 10 |
| 10 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 10.70 | 10 |
| 11 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.80 | 10 |
| 12 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.90 | 10 |
| 13 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.40 | 10 |
| 14 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.60 | 10 |
| 15 | resnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.60 | 10 |
| 16 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.10 | 10 |
| 17 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.30 | 10 |
| 18 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.80 | 10 |
| 19 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.30 | 10 |
* #### 分類: small_datasets+binclass+balanced (包含 14 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.14 | 14 |
| 2 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.79 | 14 |
| 3 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.79 | 14 |
| 4 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.86 | 14 |
| 5 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.36 | 14 |
| 6 | resnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.79 | 14 |
| 7 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.86 | 14 |
| 8 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.57 | 14 |
| 9 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.57 | 14 |
| 10 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.00 | 14 |
| 11 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.14 | 14 |
| 12 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.43 | 14 |
| 13 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.43 | 14 |
| 14 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 12.21 | 14 |
| 15 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 13.29 | 14 |
| 16 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 13.71 | 14 |
| 17 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 13.71 | 14 |
| 18 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.07 | 14 |
| 19 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 15.29 | 14 |
* #### 分類: small_datasets+binclass+categorical (包含 7 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 5.86 | 7 |
| 2 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.14 | 7 |
| 3 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 6.71 | 7 |
| 4 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.86 | 7 |
| 5 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.86 | 7 |
| 6 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.00 | 7 |
| 7 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.29 | 7 |
| 8 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.71 | 7 |
| 9 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.14 | 7 |
| 10 | resnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.29 | 7 |
| 11 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.14 | 7 |
| 12 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.14 | 7 |
| 13 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.29 | 7 |
| 14 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 11.71 | 7 |
| 15 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.71 | 7 |
| 16 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 13.57 | 7 |
| 17 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 14.00 | 7 |
| 18 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 15.29 | 7 |
| 19 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 15.29 | 7 |
* #### 分類: small_datasets+binclass+numerical (包含 28 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 3.50 | 28 |
| 2 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.04 | 28 |
| 3 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 4.75 | 28 |
| 4 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.21 | 28 |
| 5 | resnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.25 | 28 |
| 6 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 5.68 | 28 |
| 7 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.18 | 28 |
| 8 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.71 | 28 |
| 9 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 10.18 | 28 |
| 10 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.07 | 28 |
| 11 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.11 | 28 |
| 12 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.96 | 28 |
| 13 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 13.29 | 28 |
| 14 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 13.43 | 28 |
| 15 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.43 | 28 |
| 16 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.50 | 28 |
| 17 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 14.11 | 28 |
| 18 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 14.57 | 28 |
| 19 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 15.04 | 28 |
* #### 分類: small_datasets+regression+balanced (包含 6 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 4.33 | 6 |
| 2 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.83 | 6 |
| 3 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 6.33 | 6 |
| 4 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 7.00 | 6 |
| 5 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 7.17 | 6 |
| 6 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.50 | 6 |
| 7 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 8.50 | 6 |
| 8 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.17 | 6 |
| 9 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.50 | 6 |
| 10 | resnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.50 | 6 |
| 11 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 10.67 | 6 |
| 12 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 10.83 | 6 |
| 13 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.50 | 6 |
| 14 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 11.83 | 6 |
| 15 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.17 | 6 |
| 16 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 12.33 | 6 |
| 17 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 13.33 | 6 |
| 18 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.33 | 6 |
| 19 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.17 | 6 |
* #### 分類: small_datasets+regression+categorical (包含 5 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 3.00 | 5 |
| 2 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.40 | 5 |
| 3 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 4.60 | 5 |
| 4 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 6.00 | 5 |
| 5 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 6.20 | 5 |
| 6 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 7.20 | 5 |
| 7 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 8.00 | 5 |
| 8 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.20 | 5 |
| 9 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.20 | 5 |
| 10 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 8.40 | 5 |
| 11 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 9.20 | 5 |
| 12 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.00 | 5 |
| 13 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.60 | 5 |
| 14 | resnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 12.80 | 5 |
| 15 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.00 | 5 |
| 16 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 15.40 | 5 |
| 17 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.60 | 5 |
| 18 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.60 | 5 |
| 19 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 17.60 | 5 |
* #### 分類: small_datasets+regression+numerical (包含 36 個資料集)

| 排名 | 競爭者 | 平均排名 | 資料集數 |
|---:|---|---:|---:|
| 1 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=decoding) | 3.39 | 36 |
| 2 | tabgnn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 4.67 | 36 |
| 3 | tabpfn<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 5.36 | 36 |
| 4 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=columnwise) | 6.69 | 36 |
| 5 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=materialize) | 7.44 | 36 |
| 6 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=start) | 7.58 | 36 |
| 7 | tabpfn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 7.67 | 36 |
| 8 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=encoding) | 8.31 | 36 |
| 9 | resnet<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 8.39 | 36 |
| 10 | tabgnn<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 8.92 | 36 |
| 11 | xgboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.61 | 36 |
| 12 | lightgbm<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 9.92 | 36 |
| 13 | catboost<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 11.31 | 36 |
| 14 | t2g-former<br>(ratio=0.05/0.15/0.8, gnn_stage=none) | 14.42 | 36 |
| 15 | lightgbm<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.67 | 36 |
| 16 | resnet<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 14.75 | 36 |
| 17 | catboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.31 | 36 |
| 18 | xgboost<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 15.31 | 36 |
| 19 | t2g-former<br>(ratio=0.8/0.15/0.05, gnn_stage=none) | 16.31 | 36 |
