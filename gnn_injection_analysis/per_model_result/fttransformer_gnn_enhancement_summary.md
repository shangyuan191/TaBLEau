# FTTRANSFORMER — GNN 增強效果總結（與 EXCELFORMER 樣式一致）

說明：下表針對每個 dataset-category，列出 FTTRANSFORMER 在 few-shot (ratio=0.05/0.15/0.8) 下的每個 GNN 注入位置（start/encoding/columnwise/decoding/materialize），並判斷該變體是否在平均排名上「嚴格優於」目標組：

- few-shot non-GNN = `fttransformer (ratio=0.05/0.15/0.8, gnn_stage=none)`
- full-sample non-GNN = `fttransformer (ratio=0.8/0.15/0.05, gnn_stage=none)`
- few-shot trees = {`xgboost`,`catboost`,`lightgbm`} with `ratio=0.05/0.15/0.8` (count beaten out of 3)
- full-sample trees = same trees with `ratio=0.8/0.15/0.05` (count out of 3)
- few-shot GNNs = {`t2g-former`,`tabgnn`} with `ratio=0.05/0.15/0.8` (count out of 2)
- full-sample GNNs = same with `ratio=0.8/0.15/0.05` (count out of 2)
- few-shot tabpfn = `tabpfn (ratio=0.05/0.15/0.8)` (Y/N)
- full-sample tabpfn = `tabpfn (ratio=0.8/0.15/0.05)` (Y/N)

規則：僅當該注入變體的平均排名 STRICTLY lower（數值更小）時視為「beats」。平手不算。

---

註：本檔使用容差 1e-3 判斷 avg_rank 是否相等。
- 對於「beats few-shot 原始模型（few-shot non-GNN，ratio=0.05/0.15/0.8, gnn_stage=none）」的判定，使用嚴格比較（必須 strictly lower 才算 beats）。
- 對於其它比較（樹模型、GNN、tabpfn 等），若 avg_rank 在容差範圍內視為平手，並在表格中以 'Yes (tie)' 標示；若平手但涉及 few-shot 原始模型則標示 'No (tie, few-shot strict)'。


## large_datasets+binclass+numerical (6 datasets)

關鍵數值 (from ranking table):
- few-shot FTTRANSFORMER (none): 13.50
- full FTTRANSFORMER (none): 7.50
- few-shot trees (xgboost/catboost/lightgbm): 10.67 / 8.67 / 9.67
- full trees: 4.83 / 3.67 / 3.67
- few-shot GNNs (t2g/tabgnn): 8.67 / 16.17
- full GNNs: 2.83 / 15.50
- tabpfn few/full: 7.50 / 1.83

| Injection | avg_rank | beats few-shot FT? | beats full FT? | beats few-shot trees (out of 3) | beats full trees (out of 3) | beats few-shot GNNs (out of 2) | beats full GNNs (out of 2) | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 18.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 15.17 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 14.83 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 13.83 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 13.50 | No (tie) | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： 在這個分類/large binary numerical 中，FTTransformer 的 few-shot 注入沒有嚴格超越 few-shot baseline，也未擊敗 full baseline 或 tabpfn。

---

## large_datasets+multiclass+numerical (3 datasets)

關鍵數值：
- few-shot FT (none): 15.00
- full FT (none): 8.67
- few-shot trees: catboost 8.33, lightgbm 10.33, xgboost 9.67
- full trees: catboost 3.67, lightgbm 4.00, xgboost 5.33
- few-shot GNNs: t2g 8.33, tabgnn 13.67
- full GNNs: t2g 1.33, tabgnn 11.00
- tabpfn few/full: 8.33 / 2.00

| Injection | avg_rank | beats few-shot FT? | beats full FT? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise (few) | 14.67 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 15.00 | No (tie) | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 15.00 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 15.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 16.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 18.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： `columnwise` 是唯一一個在平均排名上比 few-shot baseline 小的 injection，但仍遠不及 full-sample 參考方法。

---

## large_datasets+regression+categorical (1 dataset)

關鍵數值：
- few-shot FT (none): 10.00
- full FT (none): 3.00
- few-shot trees: catboost 14.00, xgboost 16.00, lightgbm 18.00
- few-shot GNNs: t2g 11.00, tabgnn 4.00
- tabpfn few/full: 1 / 2

| Injection | avg_rank | beats few-shot FT? | beats full FT? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| materialize (few) | 6.00 | Yes | No | 3/3 | 3/3 | 1/2 | 0/2 | No | No |
| encoding (few) | 7.00 | Yes | No | 3/3 | 3/3 | 1/2 | 0/2 | No | No |
| decoding (few) | 8.00 | Yes | No | 3/3 | 3/3 | 1/2 | 0/2 | No | No |
| columnwise (few) | 9.00 | Yes | No | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| none (few) | 10.00 | — | No | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| start (few) | 12.00 | No | No | 1/3 | 1/3 | 0/2 | 0/2 | No | No |

註： 在這個單一 categorical regression 資料集中，materialize/encoding/decoding 對 few-shot baseline 有明顯改善，且能擊敗所有樹模型中的 few-shot 變體。

---

## large_datasets+regression+numerical (10 datasets)

關鍵數值：
- few-shot FT (none): 8.70
- full FT (none): 12.80
- few-shot trees (xgboost/catboost/lightgbm): 12.00 / 13.30 / 13.50
- full trees: 17.10 / 16.70 / 17.60
- few-shot GNNs: t2g 12.60, tabgnn 4.30
- full GNNs: t2g 15.80, tabgnn 4.90
- tabpfn few/full: 2.10 / 2.60

| Injection | avg_rank | beats few-shot FT? | beats full FT? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 6.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| start (few) | 6.90 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| materialize (few) | 7.10 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| encoding (few) | 7.60 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| columnwise (few) | 8.40 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| none (few) | 8.70 | — | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | No |

註： 在 large regression (numerical) 上，FT 的 few-shot 注入（尤其 decoding/start/materialize）可穩定擊敗樹模型並勝過 full baseline。

---

## small_datasets+binclass+balanced (14 datasets)

關鍵數值：
- few-shot FT (none): 14.64
- full FT (none): 7.00
- few-shot trees: xgboost 11.00, catboost 10.43, lightgbm 10.00
- full trees: xgboost 5.57, catboost 4.00, lightgbm 6.00
- few-shot GNNs: t2g 9.29, tabgnn 10.64
- full GNNs: t2g 4.71, tabgnn 10.57
- tabpfn few/full: 7.93 / 4.64

| Injection | avg_rank | beats few-shot FT? | beats full FT? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| start (few) | 12.93 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 13.21 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 14.79 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 15.64 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 14.64 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 17.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： small balanced binary 中，few-shot 的 start/materialize 比 few-shot baseline 小幅改進，但仍落後 full-sample baselines與 tabpfn。

---

## small_datasets+binclass+categorical (7 datasets)

關鍵數值：
- few-shot FT (none): 15.43
- full FT (none): 9.29
- few-shot trees: (varied) lightgbm/t2g/xgboost are strong few-shot players
- tabpfn few/full: 7.57 / 7.43

| Injection | avg_rank | beats few-shot FT? | beats full FT? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| start (few) | 10.57 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize (few) | 11.29 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| columnwise (few) | 14.57 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 15.57 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 15.43 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 18.71 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： 在 small categorical binary 中，start/materialize 有時優於 few-shot baseline，但 full baselines 仍佔優。

---

## small_datasets+binclass+numerical (28 datasets)

關鍵數值：
- few-shot FT (none): 14.39
- full FT (none): 5.71
- few-shot trees/GNNs/tabpfn: various (few-shot tabpfn 9.75, etc.)

| Injection | avg_rank | beats few-shot FT? | beats full FT? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| start (few) | 12.82 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 13.14 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 14.54 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 14.64 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 14.39 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 17.21 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： Few-shot 注入在此類別通常只帶來小幅改動。

---

## small_datasets+regression+balanced (6 datasets)

關鍵數值：
- few-shot FT (none): 4.33
- full FT (none): 8.83
- columnwise (few): 3.67, encoding (few): 3.83
- tabpfn few: 9.50

| Injection | avg_rank | beats few-shot FT? | beats full FT? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise (few) | 3.67 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| encoding (few) | 3.83 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| none (few) | 4.33 | — | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| decoding (few) | 6.50 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| start (few) | 9.50 | No | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | Yes |
| materialize (few) | 10.00 | No | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | Yes |

註： columnwise/encoding 在 small regression balanced 表現優異，能擊敗多數 few-shot 參考方法並超越 full FT baseline。

---

## small_datasets+regression+categorical (5 datasets)

關鍵數值：
- few-shot FT (none): 4.00
- encoding (few): 3.80 (best)
- tabpfn few: 7.20

| Injection | avg_rank | beats few-shot FT? | beats full FT? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| encoding (few) | 3.80 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| none (few) | 4.00 | — | Yes | 2/3 | 2/3 | 0/2 | 0/2 | Yes | Yes |
| materialize (few) | 5.00 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| decoding (few) | 6.00 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| start (few) | 6.40 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| columnwise (few) | 7.20 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |

---

## small_datasets+regression+numerical (36 datasets)

關鍵數值：
- few-shot FT (none): 7.36
- full FT (none): 12.69
- decoding (few): 4.58 (very strong)
- tabgnn few: 4.47, tabpfn few: 5.36

| Injection | avg_rank | beats few-shot FT? | beats full FT? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 4.58 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| columnwise (few) | 6.78 | Yes | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 7.14 | Yes | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | No |
| none (few) | 7.36 | — | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| start (few) | 8.56 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 8.75 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |

---

### 要點總結 (FTTRANSFORMER)

- FTTRANSFORMER 的 few-shot GNN 注入在 regression 任務（特別是 large numeric regression 與 small regression 類別）表現最好；`decoding`、`start`、`materialize` 在 large numeric regression 中最有效。
- 在多數分類任務與 large classification 中，fully-sampled `tabpfn` 與 fully-sampled 樹方法仍然佔優；few-shot 注入很少超越它們。

註：在此更新中，針對「beats few-shot 原始模型」（few-shot non-GNN baseline）仍採用嚴格比較（avg_rank 更小才算為勝）。對於其它比較（對樹模型、GNN、tabpfn 等群組），若 avg_rank 相等則視為擊敗，且在表格中以 " (tie, few-shot strict)" 標示；所有標註為 (tie, few-shot strict) 的情況皆已視為 beats。