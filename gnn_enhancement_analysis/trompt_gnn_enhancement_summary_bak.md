# TROMPT — GNN 增強效果總結

說明：本檔根據 `trompt_gnn_enhancement.md` 的 avg_rank 值，按 dataset-category 匯總 few-shot（ratio=0.05/0.15/0.8）下各種 GNN 注入變體是否擊敗指定的目標組。

註：在此更新中，針對「beats few-shot 原始模型」（few-shot non-GNN baseline）仍採用嚴格比較（avg_rank 更小才算為勝）。對於其它比較（對樹模型、GNN、tabpfn 等群組），若 avg_rank 相等則視為擊敗，且在表格中以 " (tie)" 標示；所有標註為 (tie) 的情況皆已視為 beats。

考慮的 few-shot TROMPT 注入：`columnwise`、`none`（few-shot baseline）、`decoding`、`encoding`、`start`、`materialize`（皆為 ratio=0.05/0.15/0.8）。

---

## Category: large_datasets+binclass+numerical (6 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none (trompt, ratio=0.05): 11.83
- full none (trompt, ratio=0.8): 6.33
- few-shot trees (xgboost/catboost/lightgbm): 12.17 / 9.83 / 10.67
- full trees (xgboost/catboost/lightgbm): 5.17 / 3.67 / 3.83
- few-shot GNNs (t2g/tabgnn): 9.67 / 18.33
- full GNNs (t2g/tabgnn): 2.83 / 17.50
- tabpfn few/full: 8.17 / 1.83

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 9.83 | Yes (9.83 < 11.83) | No | 2/3 | 0/3 | 1/2 (beats tabgnn only) | 1/2 (beats tabgnn full only) | No | No |
| none (few-shot baseline) | 11.83 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 18.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 10.50 | Yes (10.50 < 11.83) | No | 1/3 (beats xgboost few-shot 12.17) | 0/3 | 1/2 (beats tabgnn only) | 1/2 | No | No |
| start | 14.17 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 15.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在 large binary numeric 任務中，TROMPT 的 few-shot `columnwise` 與 `encoding` 較 few-shot baseline 有所改善，但尚未有任何注入能擊敗 full-sample 的 TROMPT 或最強的 full-sample 樹模型與 tabpfn。

---

## Category: large_datasets+multiclass+numerical (3 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 11.00
- full none: 3.33
- few-shot trees: xgboost 11.67 / catboost 10.33 / lightgbm 12.33
- full trees: xgboost 5.67 / catboost 3.33 / lightgbm 4.33
- few-shot GNNs: t2g 9.33 / tabgnn 16.67
- full GNNs: t2g 1.33 / tabgnn 12.33
- tabpfn few/full: 10.33 / 2.33

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 12.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none | 11.00 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 19.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 9.67 | Yes (9.67 < 11.00) | No | 1/3 | 0/3 | 1/2 | 0/2 | Yes (9.67 < 10.33) | No |
| start | 16.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 17.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在 large multiclass 資料集中，TROMPT 的 few-shot `encoding` 是唯一相較於 few-shot baseline 有改善的注入。

---

## Category: large_datasets+regression+categorical (1 dataset)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 13.00
- full none: 19.00
- few-shot trees: xgboost 15 / catboost 10 / lightgbm 17
- full trees: xgboost 16 / catboost 11 / lightgbm 18
- few-shot GNNs: tabgnn 4 / t2g 7
- full GNNs: tabgnn 5 / t2g 8
- tabpfn few/full: 2 / 3

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding | 1.00 | Yes (1.00 < 13.00) | Yes (1.00 < 19.00) | 3/3 | 3/3 | 1/2 (beats t2g only) | 1/2 | No | No |
| none | 13.00 | — | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| materialize | 6.00 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| encoding | 12.00 | Yes (12.00 < 13.00) | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| columnwise | 14.00 | No | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 9.00 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |

註：在這個 categorical regression 單一資料集中，TROMPT 的 few-shot `decoding` 表現卓越，幾乎擊敗所有參考模型。

---

## Category: large_datasets+regression+numerical (10 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 10.80
- full none: 17.60
- few-shot trees: xgboost 10.80 / catboost 12.30 / lightgbm 12.30
- full trees: xgboost 16.10 / catboost 15.90 / lightgbm 17.20
- few-shot GNNs: tabgnn 4.00 / t2g 11.90
- full GNNs: tabgnn 4.10 / t2g 15.00
- tabpfn few/full: 1.80 / 1.90

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding | 3.20 | Yes (3.20 < 10.80) | Yes (3.20 < 17.60) | 3/3 | 3/3 | 1/2 (beats t2g only) | 1/2 | Yes (3.20 < 1.80? No) -> correction: 3.20 > 1.80, so No | No |
| materialize | 6.90 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| start | 7.10 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| encoding | 10.40 | Yes (10.40 < 10.80) | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| columnwise | 10.70 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| none | 10.80 | — | No | 2/3 | 2/3 | 0/2 | 0/2 | No | No |

註：在 large numeric regression 中，TROMPT 的 few-shot `decoding`、`materialize`、`start` 強力擊敗了 few-shot 與 full 的 non-GNN 基線，並且擊敗所有 few-shot 樹模型。

---

## Category: small_datasets+binclass+balanced (14 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 10.86
- full none: 8.29
- few-shot trees: catboost 11.43 / lightgbm 11.29 / xgboost 11.79
- full trees: catboost 4.07 / lightgbm 6.14 / xgboost 6.14
- few-shot GNNs: t2g 10.14 / tabgnn 11.79
- tabpfn few/full: 8.71 / 5.07

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 11.79 | No | No | 1/3 | 0/3 | 0/2 | 0/2 | No | No |
| none | 10.86 | — | No | 1/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 13.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 12.71 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 13.29 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 15.07 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在 small balanced binary 資料集中，few-shot 的 TROMPT 注入通常無法擊敗最強的 few/full-sample 基線。

---

## Category: small_datasets+binclass+categorical (7 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 7.43
- full none: 8.00
- few-shot trees: lightgbm 8.00 / xgboost 10.43 / catboost 10.14
- full trees: t2g 7.00 / tabpfn 8.57 / xgboost 9.29
- few-shot GNNs: t2g 8.71 / tabgnn 12.43
- tabpfn few/full: 8.57 / 8.57

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 8.43 | No | Yes | 1/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| none | 7.43 | — | Yes | 2/3 | 2/3 | 1/2 | 1/2 | No | Yes |
| decoding | 15.71 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 8.00 | No (8.00 > 7.43) | Yes | 1/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| start | 12.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 15.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在 small categorical binary 中結果混合，full-sample 基線仍然強勢。

---

## Category: small_datasets+binclass+numerical (28 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 11.96
- full none: 4.14
- few-shot trees: xgboost 14.18 / catboost 11.14 / lightgbm 12.39
- full trees: xgboost 3.50 / catboost 4.14 / lightgbm 5.89
- few-shot GNNs: t2g 10.29 / tabgnn 11.64
- tabpfn few/full: 10.61 / 3.50

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 11.43 | Yes (11.43 < 11.96) | No | 1/3 | 0/3 | 1/2? (11.43 < 11.64 -> yes beating tabgnn few) | 0/2 | No | No |
| none | 11.96 | — | No | 1/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 17.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 10.00 | Yes | No | 2/3 | 0/3 | 1/2 | 0/2 | Yes (10.00 < 10.61) | No |
| start | 13.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 13.36 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在 small numeric binary 任務中，少數注入（如 `encoding`）可較 few-shot baseline 改善並擊敗一些 few-shot 樹模型或 tabpfn，但 full-sample 基線仍占優。

---

## Category: small_datasets+regression+balanced (6 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 9.50
- full none: 10.83
- few-shot trees: xgboost 11.17 / catboost 10.17 / lightgbm 12.00
- full trees: xgboost 11.33 / catboost 11.33 / lightgbm 9.83
- few-shot GNNs: tabgnn 2.50 / t2g 15.33
- tabpfn few/full: 7.67 / 8.67

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 12.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none | 9.50 | — | No | 2/3 | 0/3 | 1/2 | 1/2 | No | No |
| decoding | 5.33 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes (5.33 < 7.67) | Yes |
| encoding | 10.00 | No | No | 1/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 7.83 | Yes (7.83 < 9.50) | Yes | 2/3 | 2/3 | 0/2 | 0/2 | Yes (7.83 < 7.67)? No -> 7.83 > 7.67 so No | No |
| materialize | 9.17 | No | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | Yes |

註：在 small balanced regression 中，TROMPT 的 `decoding` 表現突出，能擊敗 few-shot 與 full 的 non-GNN 基線，並勝過 few/full 的 tabpfn。

---

## Category: small_datasets+regression+categorical (5 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 8.20
- full none: 16.40
- few-shot trees: xgboost 9.20 / catboost 9.80 / lightgbm 7.80
- full trees: tabpfn few/full: 8.60 / 8.60
- few-shot GNNs: tabgnn 3.20 / t2g 15.00

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 10.20 | No | Yes | 1/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| none | 8.20 | — | Yes | 2/3 | 3/3 | 0/2 | 0/2 | No (8.20 < 8.60) | Yes |
| decoding | 4.80 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes (4.80 < 8.60) | Yes |
| encoding | 10.40 | No | Yes | 1/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| start | 6.00 | Yes | Yes | 2/3 | 3/3 | 0/2 | 0/2 | Yes (6.00 < 8.60) | Yes |
| materialize | 5.40 | Yes | Yes | 2/3 | 3/3 | 0/2 | 0/2 | Yes | Yes |

註：在 small categorical regression 中，數個 trompt 的 few-shot 注入（decoding/materialize/start）能擊敗多項參考方法，包含 tabpfn。

---

## Category: small_datasets+regression+numerical (36 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 9.14
- full none: 14.67
- few-shot trees: xgboost 8.44 / catboost 10.19 / lightgbm 8.72
- full trees: xgboost 14.67 / catboost 14.97 / lightgbm 14.42
- few-shot GNNs: tabgnn 3.50 / t2g 14.06
- tabpfn few/full: 4.39 / 7.28

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 10.50 | No | Yes | 1/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| none | 9.14 | — | Yes | 2/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| decoding | 3.89 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes (3.89 < 4.39) | Yes |
| encoding | 9.36 | No | Yes | 1/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| start | 8.17 | Yes (8.17 < 9.14) | Yes | 2/3 | 3/3 | 0/2 | 0/2 | Yes? (8.17 < 4.39? No) | Yes |
| materialize | 8.72 | Yes | Yes | 2/3 | 3/3 | 0/2 | 0/2 | No | Yes |

註：TROMPT 的 few-shot `decoding` 對小型數值回歸特別有效，能擊敗 few-shot 的 tabpfn 與多數參考模型。

---

### Takeaways (TROMPT)
- TROMPT few-shot GNN injections (notably `decoding`, `materialize`, `start`) provide large benefits in several regression categories, often beating few-shot trees and even tabpfn in small regression settings.
- For classification tasks, fully-sampled baselines still commonly win.

(Next: I'll create the `vime` summary.)
