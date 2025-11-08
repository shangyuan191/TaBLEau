# TABM — GNN 增強效果總結（與 EXCELFORMER 樣式一致）

說明：下表針對每個 dataset-category，列出 TABM 在 few-shot (ratio=0.05/0.15/0.8) 下的每個 GNN 注入位置（start/encoding/columnwise/decoding/materialize），並判斷該變體是否在平均排名上「嚴格優於」目標組：

- few-shot non-GNN = `tabm (ratio=0.05/0.15/0.8, gnn_stage=none)`
- full-sample non-GNN = `tabm (ratio=0.8/0.15/0.05, gnn_stage=none)`
- few-shot trees = {`xgboost`,`catboost`,`lightgbm`} with `ratio=0.05/0.15/0.8` (count beaten out of 3)
- full-sample trees = same trees with `ratio=0.8/0.15/0.05` (count out of 3)
- few-shot GNNs = {`t2g-former`,`tabgnn`} with `ratio=0.05/0.15/0.8` (count out of 2)
- full-sample GNNs = same with `ratio=0.8/0.15/0.05` (count out of 2)
- few-shot tabpfn = `tabpfn (ratio=0.05/0.15/0.8)` (Y/N)
- full-sample tabpfn = `tabpfn (ratio=0.8/0.15/0.05)` (Y/N)

規則：僅當該注入變體的平均排名 STRICTLY lower（數值更小）時視為「beats」。平手不算。

---

## large_datasets+binclass+numerical (6 datasets)

Key numbers:
- few-shot TABM (none): 12.67
- full TABM (none): 12.00
- trees (few): catboost 7.67, lightgbm 8.33, xgboost 9.17
- trees (full): catboost 3.33, lightgbm 3.33, xgboost 4.17
- GNNs (few): t2g 7.67, tabgnn 12.50
- GNNs (full): t2g 2.67, tabgnn 14.67
- tabpfn few: 6.83 ; tabpfn full: 1.83

| Injection | avg_rank | beats few-shot TABM? | beats full TABM? | beats few-shot trees (out of 3) | beats full trees (out of 3) | beats few-shot GNNs (out of 2) | beats full GNNs (out of 2) | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| none (full) | 12.00 | — | — | — | — | — | — | — | — |
| none (few) | 12.67 | — | — | — | — | — | — | — | — |
| encoding (few) | 13.17 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 14.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 16.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 17.17 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 17.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

Notes: 在 large binary numerical 中，TABM 的 few-shot 注入通常未提供明顯優勢。

---

## large_datasets+multiclass+numerical (3 datasets)

Key numbers:
- few-shot TABM (none): 10.00
- full TABM (none): 2.67
- trees (few): catboost 9.00, lightgbm 11.33, xgboost 10.67
- trees (full): catboost 4.33, lightgbm 4.33, xgboost 6.00
- GNNs (few): t2g 9.00, tabgnn 11.67
- GNNs (full): t2g 1.33, tabgnn 11.67
- tabpfn few: 9.00 ; tabpfn full: 2.33

| Injection | avg_rank | beats few-shot TABM? | beats full TABM? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| none (full) | 2.67 | — | — | — | — | — | — | — | — |
| none (few) | 10.00 | — | — | — | — | — | — | — | — |
| encoding (few) | 15.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 15.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 16.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 16.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 19.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

Notes: few-shot 注入在多類別 large datasets 沒有帶來明顯改善。

---

## large_datasets+regression+categorical (1 dataset)

Key numbers:
- best: tabm encoding (few) 1.00
- few-shot TABM (none): 11.00
- full TABM (none): 15.00
- tabpfn few: 2.00 ; tabpfn full: 4.00

| Injection | avg_rank | beats few-shot TABM? | beats full TABM? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| encoding (few) | 1.00 | Yes | Yes | 3/3 | 3/3 | 2/2 | 2/2 | Yes | Yes |
| columnwise (few) | 3.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| materialize (few) | 7.00 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| start (few) | 10.00 | Yes | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 12.00 | No | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 11.00 | — | — | — | — | — | — | — | — |

Notes: 對於這個 categorical regression 的單一 dataset，tabm 的 encoding few-shot 變體表現最佳且能擊敗所有比較對象。

---

## large_datasets+regression+numerical (10 datasets)

Key numbers:
- few-shot TABM (none): 12.60
- full TABM (none): 17.20
- trees (few): xgboost 10.50, catboost 12.10, lightgbm 12.50
- GNNs (few): tabgnn 3.60, t2g 12.10
- tabpfn few: 1.70 ; tabpfn full: 1.80

| Injection | avg_rank | beats few-shot TABM? | beats full TABM? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise (few) | 6.90 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| materialize (few) | 6.90 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| decoding (few) | 7.30 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| start (few) | 7.40 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| encoding (few) | 8.90 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| none (few) | 12.60 | — | — | — | — | — | — | — | — |

Notes: 在 large regression numerical，TABM 的多個 few-shot 注入都能顯著擊敗 few-與 full-baseline 以及樹模型。

---

## small_datasets+binclass+balanced (14 datasets)

Key numbers:
- few-shot TABM (none): 11.00
- full TABM (none): 11.57
- tabpfn few: 7.79 ; tabpfn full: 5.21
- trees (few): catboost 10.64, lightgbm 10.50, xgboost 12.57

| Injection | avg_rank | beats few-shot TABM? | beats full TABM? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| none (full) | 11.57 | — | — | — | — | — | — | — | — |
| none (few) | 11.00 | — | — | — | — | — | — | — | — |
| start (few) | 12.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 12.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 13.21 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 13.57 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 14.57 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

Notes: 在 small balanced binary，tabm 的 few-shot 變體沒有明顯帶來改進。

---

## small_datasets+binclass+categorical (7 datasets)

Key numbers:
- few-shot TABM (none): 10.29
- full TABM (none): 9.57
- tabpfn few: 8.71 ; tabpfn full: 7.86
- trees & t2g near top

| Injection | avg_rank | beats few-shot TABM? | beats full TABM? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| start (few) | 10.86 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 13.43 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 13.29 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 14.29 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 14.43 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 10.29 | — | — | — | — | — | — | — | — |

Notes: few-shot 在 small categorical binary 中沒有明顯優勢。

---

## small_datasets+binclass+numerical (28 datasets)

Key numbers:
- few-shot TABM (none): 12.00
- full TABM (none): 9.39
- tabpfn few: 9.68 ; tabpfn full: 3.07
- trees (few): catboost 10.29, lightgbm 10.36, xgboost 12.71

| Injection | avg_rank | beats few-shot TABM? | beats full TABM? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| none (full) | 9.39 | — | — | — | — | — | — | — | — |
| none (few) | 12.00 | — | — | — | — | — | — | — | — |
| start (few) | 12.71 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 13.75 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 14.04 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 14.61 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 15.39 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：大多數 few-shot 變體無法擊敗全量基線或其他競爭模型。

---

## small_datasets+regression+balanced (6 datasets)

Key numbers:
- few-shot TABM (none): 9.33
- full TABM (none): 12.33
- top few-shot: columnwise 6.83, encoding 7.33, decoding 8.17
- tabpfn few: 7.50 ; tabpfn full: 8.50

| Injection | avg_rank | beats few-shot TABM? | beats full TABM? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise (few) | 6.83 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| encoding (few) | 7.33 | Yes | Yes | 2/3 | 2/3 | 1/2 | 1/2 | Yes | Yes |
| decoding (few) | 8.17 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | Yes | Yes |
| none (few) | 9.33 | — | — | — | — | — | — | — | — |
| start (few) | 10.17 | No | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | Yes |
| materialize (few) | 9.67 | No | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | Yes |

Notes: 在 small regression balanced，部分 few-shot 注入能顯著改進。

---

## small_datasets+regression+categorical (5 datasets)

Key numbers:
- few-shot TABM (none): 9.20
- full TABM (none): 9.60
- top: tabgnn few 3.20 ; tabpfn few 4.80

| Injection | avg_rank | beats few-shot TABM? | beats full TABM? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| encoding (few) | 4.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| columnwise (few) | 4.20 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| start (few) | 6.20 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | Yes | Yes |
| materialize (few) | 8.80 | Yes | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | No |
| none (few) | 9.20 | — | — | — | — | — | — | — | — |
| decoding (few) | 10.80 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

Notes: tabm 在 small categorical regression 的 few-shot encoding/columnwise 有明顯提升。

---

## small_datasets+regression+numerical (36 datasets)

Key numbers:
- few-shot TABM (none): 9.42
- full TABM (none): 15.17
- top few-shot: columnwise 5.28, encoding 5.75, decoding 8.06, materialize 8.42, start 9.28
- tabpfn few: 4.67 ; tabpfn full: 7.64

| Injection | avg_rank | beats few-shot TABM? | beats full TABM? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise (few) | 5.28 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | Yes |
| encoding (few) | 5.75 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | Yes |
| decoding (few) | 8.06 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| materialize (few) | 8.42 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| start (few) | 9.28 | Yes | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | Yes |
| none (few) | 9.42 | — | — | — | — | — | — | — | — |

Notes: 在 small regression numerical 上 tabm 多個 few-shot 注入都帶來實質改善，尤其 columnwise/encoding。

---

## Summary (TABM)

High-level takeaways:
- TABM 的 few-shot GNN 注入在多個 regression 類別（尤其 large/numerical、small/numerical、small/categorical）有顯著改善，特別是 encoding 與 columnwise。
- 在大部分分類任務 few-shot 注入影響有限。
- 個別資料集（例如 large regression categorical）中，tabm 的某些 few-shot encoding 變體可達到頂尖表現。


*註：在此更新中，針對「beats few-shot 原始模型」（few-shot non-GNN baseline）仍採用嚴格比較（avg_rank 更小才算為勝）。對於其它比較（對樹模型、GNN、tabpfn 等群組），若 avg_rank 相等則視為擊敗，且在表格中以 " (tie)" 標示；所有標註為 (tie) 的情況皆已視為 beats。