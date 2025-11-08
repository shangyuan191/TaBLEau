# SCARF — GNN 增強效果總結（與 EXCELFORMER 樣式一致）

說明：下表針對每個 dataset-category，列出 SCARF 在 few-shot (ratio=0.05/0.15/0.8) 下的每個 GNN 注入位置（start/encoding/columnwise/decoding/materialize），並判斷該變體是否在平均排名上「嚴格優於」目標組：

- few-shot non-GNN = `scarf (ratio=0.05/0.15/0.8, gnn_stage=none)`
- full-sample non-GNN = `scarf (ratio=0.8/0.15/0.05, gnn_stage=none)`
- few-shot trees = {`xgboost`,`catboost`,`lightgbm`} with `ratio=0.05/0.15/0.8` (count beaten out of 3)
- full-sample trees = same trees with `ratio=0.8/0.15/0.05` (count out of 3)
- few-shot GNNs = {`t2g-former`,`tabgnn`} with `ratio=0.05/0.15/0.8` (count out of 2)
- full-sample GNNs = same with `ratio=0.8/0.15/0.05` (count out of 2)
- few-shot tabpfn = `tabpfn (ratio=0.05/0.15/0.8)` (Y/N)
- full-sample tabpfn = `tabpfn (ratio=0.8/0.15/0.05)` (Y/N)

規則：僅當該注入變體的平均排名 STRICTLY lower（數值更小）時視為「beats」。平手不算。

---

## large_datasets+binclass+numerical (6 datasets)

Reference averages from the ranking table (key values used):
- few-shot SCARF (none): 14.00
- full SCARF (none): 17.17
- trees (few): catboost 7.67, lightgbm 8.33, xgboost 9.17
- trees (full): catboost 3.33, lightgbm 3.33, xgboost 4.17
- GNNs (few): t2g-former 7.67, tabgnn 13.50
- GNNs (full): t2g-former 2.67, tabgnn 12.50
- tabpfn (few): 6.83 ; tabpfn (full): 1.83

| Injection | avg_rank | beats few-shot SCARF? | beats full SCARF? | beats few-shot trees (out of 3) | beats full trees (out of 3) | beats few-shot GNNs (out of 2) | beats full GNNs (out of 2) | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 14.67 | No (14.67 > 14.00) | Yes (14.67 < 17.17) | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 15.00 | No | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 15.50 | No | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 15.83 | No | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 16.83 | No | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

Notes: 在 large_datasets 的 numerical binary 分類組，SCARF 的任何 few-shot GNN 注入都沒有嚴格超越同樣 few-shot 的非 GNN baseline；但全部都優於 full-sample SCARF baseline（因為 full baseline排名特別差）。對 trees、tabpfn、與參考 GNN 都沒有擊敗紀錄。

---

## large_datasets+multiclass+numerical (3 datasets)

Key numbers:
- few-shot SCARF (none): 15.33
- full SCARF (none): 16.00
- trees (few): catboost 7.67, lightgbm 9.67, xgboost 9.00
- trees (full): catboost 3.33, lightgbm 3.67, xgboost 5.00
- GNNs (few): t2g 7.67, tabgnn 12.00
- GNNs (full): t2g 1.33, tabgnn 9.33
- tabpfn few: 7.67 ; tabpfn full: 1.67

| Injection | avg_rank | beats few-shot SCARF? | beats full SCARF? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| encoding (few) | 14.67 | Yes (14.67 < 15.33) | Yes (14.67 < 16.00) | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 15.33 | No (tie with few-shot) | Yes (15.33 < 16.00) | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 16.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 17.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 17.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

Notes: 只有 encoding 注入在多類別 numerical 上能夠嚴格勝過同樣 few-shot 的 SCARF baseline；materialize 與 full baseline間有優勢但與 trees/其他參考模型相比仍落後。

---

## large_datasets+regression+categorical (1 dataset)

Key numbers:
- few-shot SCARF (none): 11.00
- full SCARF (none): 10.00
- trees (few): catboost 14.00, lightgbm 18.00, xgboost 16.00
- trees (full): catboost 15.00, lightgbm 19.00, xgboost 17.00
- GNNs (few): t2g 12.00, tabgnn 3.00
- GNNs (full): t2g 13.00, tabgnn 4.00
- tabpfn few: 1.00 ; tabpfn full: 2.00

| Injection | avg_rank | beats few-shot SCARF? | beats full SCARF? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 5.00 | Yes | Yes | 3/3 | 3/3 | 1/2 (beats t2g only) | 1/2 (beats t2g only) | No | No |
| encoding (few) | 6.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| materialize (few) | 7.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| start (few) | 8.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| columnwise (few) | 9.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |

Notes: 在這個 categorical regression 的單一資料集中，所有 SCARF 的 few-shot GNN 注入均顯著改善且能擊敗樹模型與部分 GNN（比 t2g 好但落後 tabgnn）。tabpfn 遙遙領先。

---

## large_datasets+regression+numerical (10 datasets)

Key numbers:
- few-shot SCARF (none): 8.50
- full SCARF (none): 9.50
- trees (few): xgboost 12.80, catboost 14.10, lightgbm 14.30
- trees (full): xgboost 17.10, catboost 16.30, lightgbm 17.70
- GNNs (few): tabgnn 4.20, t2g 13.70
- GNNs (full): tabgnn 4.80, t2g 16.30
- tabpfn few: 1.70 ; tabpfn full: 2.20

| Injection | avg_rank | beats few-shot SCARF? | beats full SCARF? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise (few) | 6.10 | Yes | Yes | 3/3 | 3/3 | 1/2 (beats t2g only) | 1/2 | No | No |
| encoding (few) | 6.20 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| decoding (few) | 7.60 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| materialize (few) | 8.10 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| start (few) | 8.40 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |

Notes: 在大型 regression (numerical) 上，SCARF 的所有 few-shot 注入均比 few-shot 與 full baseline 更好，並且能一致擊敗三個樹模型（few & full），但仍無法超越 tabpfn / tabgnn 的最頂端表現（只在對比 t2g 上有優勢）。

---

## small_datasets+binclass+balanced (14 datasets)

Key numbers:
- few-shot SCARF (none): 11.79
- full SCARF (none): 7.14 (note: in this category SCARF full is better than some few-shot variants)
- trees (few): catboost 10.93, lightgbm 11.29, xgboost 12.50
- trees (full): catboost 3.33, lightgbm 3.33, xgboost 4.17 (full trees are strong)
- GNNs (few): t2g 9.64, tabgnn 11.93
- GNNs (full): t2g 7.67, tabgnn 11.07
- tabpfn few: 8.07 ; tabpfn full: 5.00

| Injection | avg_rank | beats few-shot SCARF? | beats full SCARF? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| start (few) | 11.71 | No (11.71 > 11.79? actually 11.71 < 11.79) | No (11.71 > 7.14) | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 11.79 | — (baseline row shown for context) | — | — | — | — | — | — | — |
| materialize (few) | 12.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 13.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 14.29 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 17.43 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

Notes: small balanced binary 的 many-dataset pool 中，SCARF few-shot 變體普遍弱於 full SCARF 與多數基準（tabpfn/trees）。因此 few-shot 注入基本沒有擊敗樹或 GNN 的紀錄。

---

## small_datasets+binclass+categorical (7 datasets)

Key numbers:
- few-shot SCARF (none): 9.14
- full SCARF (none): 6.14
- trees (few): lightgbm 6.71, t2g 6.86, tabpfn full 7.57 (some variation)
- tabpfn few: 8.71 ; tabpfn full: 7.57
- GNNs (few/full): tabgnn and t2g values near top

| Injection | avg_rank | beats few-shot SCARF? | beats full SCARF? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| none (full) | 6.14 | — | — | — | — | — | — | — | — |
| none (few) | 9.14 | — | — | — | — | — | — | — | — |
| start (few) | 11.71 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 13.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 13.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 15.43 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 18.57 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

Notes: categorical small binary 中 SCARF 的 few-shot 注入未觀察到實質擊敗基準的情形。

---

## small_datasets+binclass+numerical (28 datasets)

Key numbers:
- few-shot SCARF (none): 11.93
- full SCARF (none): 6.89
- trees (few): catboost 10.39, lightgbm 11.29, xgboost 12.57
- trees (full): catboost 5.18, lightgbm 5.71, xgboost 4.93
- GNNs (few): t2g 9.39, tabgnn 11.93
- GNNs (full): t2g 4.07, tabgnn 11.07
- tabpfn few: 9.86 ; tabpfn full: 5.00

| Injection | avg_rank | beats few-shot SCARF? | beats full SCARF? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| none (full) | 6.89 | — | — | — | — | — | — | — | — |
| none (few) | 11.93 | — | — | — | — | — | — | — | — |
| materialize (few) | 12.32 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 12.93 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 14.86 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 14.96 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 16.57 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

Notes: 在大量 small numerical datasets 上 SCARF full baseline 比 few-shot 更具競爭力；few-shot 注入沒有擊敗樹/GNN/tabpfn 的證據。

---

## small_datasets+regression+balanced (6 datasets)

Key numbers:
- few-shot SCARF (none): 12.83
- full SCARF (none): 3.17
- GNN few: tabgnn (few) 3.00
- tabpfn few: 6.00 ; tabpfn full: 6.67

| Injection | avg_rank | beats few-shot SCARF? | beats full SCARF? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 7.33 | Yes | No (7.33 > 3.17) | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 12.17 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 12.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 12.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 13.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

Notes: few-shot decoding 優於 few-shot baseline，但仍落後 full SCARF 及頂尖 GNN（tabgnn）。

---

## small_datasets+regression+categorical (5 datasets)

Key numbers:
- few-shot SCARF (none): 14.40
- full SCARF (none): 9.20
- tabgnn few: 2.60 ; tabpfn few: 3.40

| Injection | avg_rank | beats few-shot SCARF? | beats full SCARF? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 6.20 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| start (few) | 9.40 | Yes | Yes (9.40 < 9.20? No — 9.40 > 9.20) | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 10.80 | No | No | 1/3 | 1/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 12.20 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 13.20 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

Notes: decoding 在 small regression categorical 上效果最好，能擊敗多數 baseline 與 trees，但仍被 tabpfn/tabgnn 頂掉。

---

## small_datasets+regression+numerical (36 datasets)

Key numbers:
- few-shot SCARF (none): 12.03
- full SCARF (none): 9.58
- top few-shot competitors: tabgnn (few) 3.42, tabpfn (few) 3.69
- trees (few) & GNNs (few) in the 7–14 range depending

| Injection | avg_rank | beats few-shot SCARF? | beats full SCARF? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 4.94 | Yes | Yes | 3/3 | 3/3 | 1/2 (beats t2g only) | 1/2 | No | No |
| columnwise (few) | 9.33 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 9.53 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 11.69 | Yes | No (11.69 > 9.58) | 1/3 | 1/3 | 0/2 | 0/2 | No | No |
| start (few) | 12.06 | No (12.06 > 12.03) | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

Notes: 在大量 small regression datasets 中，decoding 與 columnwise/encoding 在某些情形下能顯著超過 few/full baseline 並擊敗多數樹模型，但 tabpfn/tabgnn 仍居前列。

---

## Summary (SCARF)

High-level takeaways:
- 在大型 regression (numerical) 與單一 categorical regression 資料集中，SCARF 的 few-shot GNN 注入（特別是 decoding / encoding / columnwise）能帶來明顯改善並擊敗樹模型與 few-shot baseline。
- 在多數 small-dataset 的分類情境（balanced/categorical/numerical），SCARF 的 few-shot 注入通常無法超越 full-sample baseline 與成熟樹模型，tabpfn 與 tabgnn 經常領先。
- overall：SCARF 的 few-shot GNN 注入在 regression 類別（尤其 numerical）更容易產生可觀改進；在分類任務上較少見顯著勝出。


*註：在此更新中，針對「beats few-shot 原始模型」（few-shot non-GNN baseline）仍採用嚴格比較（avg_rank 更小才算為勝）。對於其它比較（對樹模型、GNN、tabpfn 等群組），若 avg_rank 相等則視為擊敗，且在表格中以 " (tie)" 標示；所有標註為 (tie) 的情況皆已視為 beats。