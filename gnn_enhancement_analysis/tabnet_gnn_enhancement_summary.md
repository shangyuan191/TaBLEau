# TABNET — GNN 增強效果總結（與 EXCELFORMER 樣式一致）

說明：本檔基於 `tabnet_gnn_enhancement.md` 的 avg_rank 值，對每個 dataset-category 列出 TabNet 在 few-shot (ratio=0.05/0.15/0.8) 下各注入位置的平均排名，並判斷是否嚴格優於目標群組（strictly lower 才算 beats）。

---

註：本檔使用容差 1e-3 判斷 avg_rank 是否相等。
- 對於「beats few-shot 原始模型（few-shot non-GNN，ratio=0.05/0.15/0.8, gnn_stage=none）」的判定，使用嚴格比較（必須 strictly lower 才算 beats）。
- 對於其它比較（樹模型、GNN、tabpfn 等），若 avg_rank 在容差範圍內視為平手，並在表格中以 'Yes (tie)' 標示；若平手但涉及 few-shot 原始模型則標示 'No (tie, few-shot strict)'。


## large_datasets+binclass+numerical (6 datasets)

關鍵數值：
- few-shot tabnet (none): 13.17
- full tabnet (none): 5.17
- few-shot trees: catboost 8.50, lightgbm 9.17, xgboost 10.00
- few-shot GNNs: t2g 8.83, tabgnn 16.50
- full GNNs: t2g 2.83, tabgnn 15.33
- tabpfn few/full: 7.67 / 2.00

| Injection | avg_rank | beats few-shot tabnet? | beats full tabnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 12.83 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 14.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 14.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 13.17 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 18.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 18.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： 在 large binary numerical 中，TabNet 的 few-shot 注入與 few-shot baseline 相差不大；full-sample 和 tabpfn 仍佔優。

---

## large_datasets+multiclass+numerical (3 datasets)

關鍵數值：
- few-shot tabnet (none): 14.00
- full tabnet (none): 5.67
- few-shot trees/GNNs: catboost few 8.67, t2g few 8.67, tabpfn few 8.67
- tabpfn full 1.67
- tabnet few-shot injections (avg): none 14.00, decoding 14.67, start 16.00, materialize 17.00, encoding 18.00, columnwise 19.00

| Injection | avg_rank | beats few-shot tabnet? | beats full tabnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| none (few) | 14.00 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 14.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 16.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 17.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 18.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 19.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： 在這個 multiclass small set 中，few-shot 注入未見改善。

---

## large_datasets+regression+categorical (1 dataset)

關鍵數值：
- tabnet few-shot encoding: 1.00 (best)
- tabpfn few/full: 2.00 / 4.00
- tabnet few columnwise: 3.00, materialize: 7.00, start: 10.00, none: 11.00, decoding: 12.00
- trees and GNNs: tabgnn few 5.00, t2g few 8.00

| Injection | avg_rank | beats few-shot tabnet? | beats full tabnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| encoding (few) | 1.00 | Yes | Yes | 3/3 | 3/3 | 2/2 | 2/2 | Yes | Yes |
| columnwise (few) | 3.00 | Yes | Yes | 2/3 | 2/3 | 1/2 | 1/2 | Yes | Yes |
| materialize (few) | 7.00 | Yes | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | Yes |
| start (few) | 10.00 | Yes | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | Yes |
| none (few) | 11.00 | — | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | Yes |
| decoding (few) | 12.00 | No | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | Yes |

註： encoding 是此 dataset 的最佳 few-shot 注入，甚至擊敗 tabpfn。

---

## large_datasets+regression+numerical (10 datasets)

關鍵數值：
- tabnet few-shot encoding: 2.00
- tabpfn few/full: 2.80 / 2.60
- tabnet few columnwise: 3.00, start/materialize: 7.80, none 10.20, decoding 16.50
- few-shot trees: xgboost few 10.80, catboost few 12.00, lightgbm few 12.40

| Injection | avg_rank | beats few-shot tabnet? | beats full tabnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| encoding (few) | 2.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| columnwise (few) | 3.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | Yes |
| start (few) | 7.80 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| materialize (few) | 7.80 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| none (few) | 10.20 | — | No | 1/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 16.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： tabnet 在 large numeric regression 的 few-shot encoding/columnwise 表現很好，能擊敗多數 few-shot 樹方法，encoding 甚至超越 tabpfn。

---

## small_datasets+binclass+balanced (14 datasets)

關鍵數值：
- tabnet few-shot none: 12.79
- tabnet few-shot decoding: 11.36, start 11.93, materialize 13.29, columnwise 17.29, encoding 17.43
- full tabnet none: 7.50
- few-shot trees: catboost 10.57, xgboost 11.36, lightgbm 11.00
- tabpfn few/full: 7.79 / 4.93

| Injection | avg_rank | beats few-shot tabnet? | beats full tabnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 11.36 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 11.93 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 12.79 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 13.29 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 17.29 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 17.43 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： small balanced binary 中 few-shot decoding/start 能小幅超越 few-shot baseline，但仍落後於 full-sample 參考。

---

## small_datasets+binclass+categorical (7 datasets)

關鍵數值：
- tabnet few-shot none: 11.00
- top few-shot competitors: t2g few 7.00, lightgbm few 6.00, tabpfn few 7.29
- tabnet few-shot injections: none 11.00, materialize 13.86, decoding 14.43, start 14.71, encoding 17.71, columnwise 18.86

| Injection | avg_rank | beats few-shot tabnet? | beats full tabnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| none (few) | 11.00 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 13.86 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 14.43 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 14.71 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 17.71 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 18.86 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： small categorical binary 中 TabNet few-shot 注入通常無法超越基線或強樹方法。

---

## small_datasets+binclass+numerical (28 datasets)

關鍵數值：
- tabnet few-shot none: 11.32
- tabpfn few: 10.14, t2g few: 9.46
- tabnet few-shot injections: decoding 11.36, materialize 12.71, start 12.75, encoding 17.25, columnwise 17.71

| Injection | avg_rank | beats few-shot tabnet? | beats full tabnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 11.36 | No (slightly worse than none=11.32) | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 12.75 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 12.71 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 11.32 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 17.25 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 17.71 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： 在 small numeric binary 分類中，TabNet 的 few-shot 注入並不總能顯著超越 few-shot baseline 或競爭方法。

---

## small_datasets+regression+balanced (6 datasets)

關鍵數值：
- tabnet few-shot columnwise: 8.83, encoding:7.17, decoding:8.67, materialize:8.67, start:10.33, none:10.83
- tabpfn few/full: 6.83 / 8.50; tabgnn few: 3.33

| Injection | avg_rank | beats few-shot tabnet? | beats full tabnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| encoding (few) | 7.17 | Yes | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| columnwise (few) | 8.83 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| decoding (few) | 8.67 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| materialize (few) | 8.67 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| start (few) | 10.33 | No | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | Yes |
| none (few) | 10.83 | — | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | Yes |

註： 在 small regression balanced 中，encoding 是 few-shot 中較強的注入，能擊敗 full-sample 大多數基線。

---

## small_datasets+regression+categorical (5 datasets)

關鍵數值：
- tabnet few-shot decoding: 4.20, columnwise:6.20, encoding:6.60, materialize:7.40, start:8.80, none:9.40
- tabpfn few: 4.40, tabgnn few: 3.60

| Injection | avg_rank | beats few-shot tabnet? | beats full tabnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 4.20 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| columnwise (few) | 6.20 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| encoding (few) | 6.60 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| materialize (few) | 7.40 | Yes | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | Yes |
| start (few) | 8.80 | Yes | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | Yes |
| none (few) | 9.40 | — | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | Yes |

註： decoding 在 small categorical regression 中表現良好且可擊敗 tabpfn/few trees。

---

## small_datasets+regression+numerical (36 datasets)

關鍵數值：
- tabnet few-shot columnwise: 5.47, encoding:5.78, decoding:7.44, start:8.33, materialize:8.69, none:8.83
- tabgnn few: 4.44, tabpfn few: 4.67

| Injection | avg_rank | beats few-shot tabnet? | beats full tabnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise (few) | 5.47 | Yes | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| encoding (few) | 5.78 | Yes | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| decoding (few) | 7.44 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| start (few) | 8.33 | Yes | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | Yes |
| materialize (few) | 8.69 | Yes | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | Yes |
| none (few) | 8.83 | — | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | Yes |

註： TabNet 在 small numeric regression 中的 columnwise/encoding 表現很穩定，勝過大部分樹模型，雖然 tabgnn/tabpfn few-shot 仍更強。

---

### 要點總結 (TABNET)

- TabNet 的 few-shot GNN 注入在 regression 任務（尤其 large numeric regression 與 small regression 類別）有明顯幫助；encoding/columnwise/decoding 通常是最佳注入位置。
- 在大多數 classification 類別，tabpfn (full) 或 fully-sampled trees 仍領先。

註：在此更新中，針對「beats few-shot 原始模型」（few-shot non-GNN baseline）仍採用嚴格比較（avg_rank 更小才算為勝）。對於其它比較（對樹模型、GNN、tabpfn 等群組），若 avg_rank 相等則視為擊敗，且在表格中以 " (tie, few-shot strict)" 標示；所有標註為 (tie, few-shot strict) 的情況皆已視為 beats。