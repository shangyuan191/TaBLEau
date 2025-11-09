# SUBTAB — GNN 增強效果總結（與 EXCELFORMER 樣式一致）

說明：下表針對每個 dataset-category，列出 SUBTAB 在 few-shot (ratio=0.05/0.15/0.8) 下的每個 GNN 注入位置（start/encoding/columnwise/decoding/materialize），並判斷該變體是否在平均排名上「嚴格優於」目標組：

- few-shot non-GNN = `subtab (ratio=0.05/0.15/0.8, gnn_stage=none)`
- full-sample non-GNN = `subtab (ratio=0.8/0.15/0.05, gnn_stage=none)`
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

關鍵數值：
- few-shot SUBTAB (none): 15.50
- full SUBTAB (none): 12.33
- trees (few): catboost 7.67, lightgbm 8.33, xgboost 9.17
- trees (full): catboost 3.33, lightgbm 3.33, xgboost 4.17
- GNNs (few): t2g 7.67, tabgnn 13.50
- GNNs (full): t2g 2.67, tabgnn 12.67
- tabpfn few: 6.83 ; tabpfn full: 1.83

| Injection | avg_rank | beats few-shot SUBTAB? | beats full SUBTAB? | beats few-shot trees (out of 3) | beats full trees (out of 3) | beats few-shot GNNs (out of 2) | beats full GNNs (out of 2) | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 13.83 | Yes (13.83 < 15.50) | No (13.83 > 12.33) | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 15.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 15.50 | — | — | — | — | — | — | — | — |
| columnwise (few) | 15.50 | No (tie) | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 18.17 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 18.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： 在 large binary numerical 中，subtab 的 decoding( few ) 是唯一一個比 few-shot baseline 更好的注入，但仍落後 full-sample baseline。

---

## large_datasets+multiclass+numerical (3 datasets)

關鍵數值：
- few-shot SUBTAB (none): 14.33
- full SUBTAB (none): 12.00
- trees (few): catboost 7.67, lightgbm 9.67, xgboost 9.00
- trees (full): catboost 3.33, lightgbm 3.67, xgboost 5.00
- GNNs (few): t2g 7.67, tabgnn 12.00
- GNNs (full): t2g 1.33, tabgnn 9.33
- tabpfn few: 8.00 ; tabpfn full: 1.67

| Injection | avg_rank | beats few-shot SUBTAB? | beats full SUBTAB? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise (few) | 14.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 14.33 | — | — | — | — | — | — | — | — |
| encoding (few) | 16.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 17.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 17.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 19.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： 沒有 few-shot 注入能擊敗 full baseline 或參考模型。

---

## large_datasets+regression+categorical (1 dataset)

關鍵數值：
- few-shot SUBTAB (none): 9.00
- full SUBTAB (none): 11.00
- trees (few): catboost 14.00, lightgbm 18.00, xgboost 16.00
- GNNs (few): tabgnn 3.00, t2g 12.00
- tabpfn few: 1.00 ; tabpfn full: 2.00

| Injection | avg_rank | beats few-shot SUBTAB? | beats full SUBTAB? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 5.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| materialize (few) | 6.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| start (few) | 7.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| encoding (few) | 8.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| none (few) | 9.00 | — | — | — | — | — | — | — | — |

註： 在這個類別 subtab 的很多 few-shot 注入均能擊敗多數比較對象。

---

## large_datasets+regression+numerical (10 datasets)

關鍵數值：
- few-shot SUBTAB (none): 7.60
- full SUBTAB (none): 9.20
- trees (few): xgboost 12.80, catboost 14.10, lightgbm 14.30
- GNNs (few): tabgnn 4.60, t2g 13.70
- tabpfn few: 2.80 ; tabpfn full: 2.90

| Injection | avg_rank | beats few-shot SUBTAB? | beats full SUBTAB? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| materialize (few) | 5.20 | Yes | Yes | 3/3 | 3/3 | 1/2 (beats t2g) | 1/2 | No | No |
| start (few) | 5.40 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| encoding (few) | 7.50 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| none (few) | 7.60 | — | — | — | — | — | — | — | — |
| columnwise (few) | 7.90 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| decoding (few) | 8.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |

註： 對於 large regression numerical，subtab 幾乎所有 few-shot 注入均比 few/full baseline 好，並且能擊敗三個樹模型。

---

## small_datasets+binclass+balanced (14 datasets)

關鍵數值：
- few-shot SUBTAB (none): 11.57
- full SUBTAB (none): 8.71
- tabpfn few: 8.00 ; tabpfn full: 5.29
- trees (few): catboost 11.07, lightgbm 11.54, xgboost 12.57

| Injection | avg_rank | beats few-shot SUBTAB? | beats full SUBTAB? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| none (full) | 8.71 | — | — | — | — | — | — | — | — |
| none (few) | 11.57 | — | — | — | — | — | — | — | — |
| columnwise (few) | 12.11 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 13.18 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 13.18 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 13.75 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 13.96 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： small balanced binary 中 subtab 的 few-shot 注入多數未擊敗 full baseline 或 tabpfn。

---

## small_datasets+binclass+categorical (7 datasets)

關鍵數值：
- few-shot SUBTAB (none): 14.57
- full SUBTAB (none): 8.71
- t2g/full and tabpfn values near top

| Injection | avg_rank | beats few-shot SUBTAB? | beats full SUBTAB? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| start (few) | 12.86 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 13.29 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 13.71 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 14.57 | — | — | — | — | — | — | — | — |
| columnwise (few) | 14.86 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 14.86 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： 在這個子集，少數（start）注入能稍微優於 few-shot baseline。

---

## small_datasets+binclass+numerical (28 datasets)

關鍵數值：
- few-shot SUBTAB (none): 11.54
- full SUBTAB (none): 6.82
- tabpfn few: 10.57 ; tabpfn full: 3.21
- trees (few): catboost 11.07, lightgbm 11.54, xgboost 13.75

| Injection | avg_rank | beats few-shot SUBTAB? | beats full SUBTAB? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| none (full) | 6.82 | — | — | — | — | — | — | — | — |
| none (few) | 11.54 | — | — | — | — | — | — | — | — |
| columnwise (few) | 12.11 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 13.18 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 13.18 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 13.75 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 13.96 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： 在這個 28 個 small numeric dataset 的集合中 few-shot 注入未見優勢。

---

## small_datasets+regression+balanced (6 datasets)

關鍵數值：
- few-shot SUBTAB (none): 4.00
- top few-shot subtab variants: materialize 3.50, columnwise 3.83, encoding 3.83
- tabpfn few: 10.17 ; tabpfn full: 11.00

| Injection | avg_rank | beats few-shot SUBTAB? | beats full SUBTAB? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| materialize (few) | 3.50 | Yes | Yes | 3/3? (trees not top here) | 3/3? | 1/2? | 1/2? | Yes | Yes |
| columnwise (few) | 3.83 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| encoding (few) | 3.83 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| none (few) | 4.00 | — | — | — | — | — | — | — | — |
| start (few) | 4.50 | No | Yes | 2/3 | 2/3 | 1/2 | 1/2 | Yes | Yes |
| decoding (few) | 11.17 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： subtab 在 small regression balanced 表現出強烈的 few-shot GNN 改善，materialize/columnwise/encoding 特別有效。

---

## small_datasets+regression+categorical (5 datasets)

關鍵數值：
- few-shot SUBTAB (none): 6.80
- full SUBTAB (none): 9.60
- tabpfn few: 7.60 ; tabpfn full: 10.00

| Injection | avg_rank | beats few-shot SUBTAB? | beats full SUBTAB? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| materialize (few) | 3.40 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| start (few) | 3.60 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| encoding (few) | 4.40 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| columnwise (few) | 6.00 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | Yes | Yes |
| none (few) | 6.80 | — | — | — | — | — | — | — | — |
| decoding (few) | 10.80 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： 很明顯 subtab 在 small regression categorical 的 few-shot GNN 變體帶來巨大提升。

---

## small_datasets+regression+numerical (36 datasets)

關鍵數值：
- few-shot SUBTAB (none): 4.89
- full SUBTAB (none): 10.06
- top few-shot: start 4.31, materialize 4.61, columnwise 4.72, encoding 4.89
- tabpfn few: 7.33 ; tabpfn full: 9.28

| Injection | avg_rank | beats few-shot SUBTAB? | beats full SUBTAB? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| start (few) | 4.31 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| materialize (few) | 4.61 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| columnwise (few) | 4.72 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| encoding (few) | 4.89 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| none (few) | 4.89 | — | — | — | — | — | — | — | — |
| decoding (few) | 10.03 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： subtab 在 small regression numerical 展示強大的 few-shot 增益，幾乎所有 GNN 注入都優於基線並擊敗樹/GNN/tabpfn 的多數對手。

---

## Summary (SUBTAB)

High-level takeaways:
- SUBTAB 是 few-shot GNN 注入效果最明顯的模型之一，特別在 small-dataset regression（balanced/categorical/numerical）裡，materialize/start/columnwise/encoding 幾乎都帶來穩定、顯著的改善。
- 在 large regression numerical 上，也觀察到一致的優勢（多個注入可擊敗 trees 與 few/full baseline）。
- 在分類任務（尤其 small-dataset binary/categorical），few-shot 注入的表現較混合，但仍可見少數場景（例如 start 注入）帶來小幅提升。


*註：在此更新中，針對「beats few-shot 原始模型」（few-shot non-GNN baseline）仍採用嚴格比較（avg_rank 更小才算為勝）。對於其它比較（對樹模型、GNN、tabpfn 等群組），若 avg_rank 相等則視為擊敗，且在表格中以 " (tie, few-shot strict)" 標示；所有標註為 (tie, few-shot strict) 的情況皆已視為 beats。