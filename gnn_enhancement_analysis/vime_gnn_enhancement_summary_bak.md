# VIME — GNN 增強效果總結

說明：本檔根據 `vime_gnn_enhancement.md` 的 avg_rank 值，按 dataset-category 匯總 few-shot（ratio=0.05/0.15/0.8）下各種 GNN 注入變體是否擊敗指定的目標組。

註：在此更新中，針對「beats few-shot 原始模型」（few-shot non-GNN baseline）仍採用嚴格比較（avg_rank 更小才算為勝）。對於其它比較（對樹模型、GNN、tabpfn 等群組），若 avg_rank 相等則視為擊敗，且在表格中以 " (tie)" 標示；所有標註為 (tie) 的情況皆已視為 beats。

考慮的 few-shot VIME 注入：`columnwise`、`none`（few-shot baseline）、`decoding`、`encoding`、`start`、`materialize`（皆為 ratio=0.05/0.15/0.8）。

---

## Category: large_datasets+binclass+numerical (6 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none (vime, ratio=0.05): 13.17
- full none (vime, ratio=0.8): 8.67
- few-shot trees (xgboost/catboost/lightgbm): 9.83 / 8.17 / 9.00
- full trees (xgboost/catboost/lightgbm): 4.33 / 3.33 / 3.33
- few-shot GNNs (t2g/tabgnn): 8.83 / 18.50
- full GNNs (t2g/tabgnn): 2.67 / 17.17
- tabpfn few/full: 7.17 / 1.83

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 13.00 | Yes (13.00 < 13.17) | No | 0/3 | 0/3 | 1/2 (beats tabgnn only) | 1/2 | No | No |
| none (few-shot baseline) | 13.17 | — | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 15.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 13.00 | Yes (13.00 < 13.17) | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 16.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 15.83 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：VIME 的 few-shot `columnwise` 與 `encoding` 對 few-shot baseline 有小幅提升，但在此類別中尚未超越 full-sample 基線。

---

## Category: large_datasets+multiclass+numerical (3 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 15.67
- full none: 7.67
- few-shot trees: xgboost 9.67 / catboost 8.33 / lightgbm 10.33
- full trees: catboost 3.33 / lightgbm 4.00 / xgboost 5.33
- few-shot GNNs: t2g 8.33 / tabgnn 14.00
- full GNNs: t2g 1.33 / tabgnn 10.33
- tabpfn few/full: 8.00 / 1.67

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 15.33 | Yes (15.33 < 15.67) | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none | 15.67 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 17.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 14.00 | Yes (14.00 < 15.67) | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 18.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 17.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在 small 類別中，`encoding`/`columnwise` 對 few-shot baseline 有小幅增益，但無法與 full-sample 的領先者競爭。

---

## Category: large_datasets+regression+categorical (1 dataset)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 10.00
- full none: 13.00
- few-shot trees: catboost 14 / xgboost 16 / lightgbm 18
- full trees: catboost 15 / xgboost 17 / lightgbm 19
- few-shot GNNs: tabgnn 3 / t2g 11
- full GNNs: tabgnn 4 / t2g 12
- tabpfn few/full: 1 / 2

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding | 5.00 | Yes (5.00 < 10.00) | Yes (5.00 < 13.00) | 3/3 | 3/3 | 1/2 (beats t2g only) | 1/2 | No | No |
| none | 10.00 | — | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| materialize | 15.83 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 9.00 | Yes (9.00 < 10.00) | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| columnwise | 8.00 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| start | 16.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在這個 categorical regression 的單一資料集中，VIME 的 few-shot `decoding` 與 `encoding` 能擊敗多項參考基線（包含 full-sample non-GNN）。

---

## Category: large_datasets+regression+numerical (10 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 8.30
- full none: 12.10
- few-shot trees: xgboost 12.60 / catboost 13.90 / lightgbm 14.10
- full trees: xgboost 17.10 / catboost 16.70 / lightgbm 17.70
- few-shot GNNs: tabgnn 3.30 / t2g 13.20
- full GNNs: tabgnn 4.20 / t2g 15.90
- tabpfn few/full: 1.60 / 1.70

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 9.20 | No | Yes | 3/3 | 3/3 | 1/2 (beats t2g only) | 1/2 | No | No |
| none | 8.30 | — | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | No |
| decoding | 8.50 | No | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | No |
| encoding | 8.20 | No | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | No |
| start | 5.70 | Yes (5.70 < 8.30) | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| materialize | 6.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |

註：在 large-scale 的數值回歸中，VIME 的 few-shot `start` 與 `materialize` 有顯著幫助，能擊敗 few-shot 與 full 的 non-GNN 基線，並勝過所有 few-shot 樹模型。

---

## Category: small_datasets+binclass+balanced (14 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 11.50
- full none: 5.93
- few-shot trees: catboost 11.43 / lightgbm 11.50 / xgboost 12.00
- full trees: catboost 4.50 / lightgbm 5.93 / xgboost 6.57
- few-shot GNNs: t2g 10.29 / tabgnn 11.29
- tabpfn few/full: 8.64 / 5.57

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 11.36 | Yes (11.36 < 11.50) | No | 1/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 11.50 | — | No | 1/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 12.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 11.79 | Yes (11.79 < 11.50)? No -> 11.79 > 11.50 -> No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 15.64 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 14.86 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在 small balanced binary 任務中，VIME 的 few-shot 注入僅帶來些微變化，沒有擊敗最強的 full-sample 基線。

---

## Category: small_datasets+binclass+categorical (7 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 12.57
- full none: 10.29
- few-shot trees: lightgbm 5.86 / xgboost 8.00 / catboost 8.00
- full trees: t2g 5.57 / tabpfn 7.29 / xgboost 7.43
- few-shot GNNs: t2g 7.00 / tabgnn 9.86
- tabpfn few/full: 8.57 / 7.29

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 11.36 | Yes (11.36 < 12.57) | No | 1/3 | 0/3 | 1/2 | 0/2 | No | No |
| none | 12.57 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 13.14 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 11.57 | Yes (11.57 < 12.57) | No | 1/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 14.14 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 17.86 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在 small categorical binary 中，VIME 的 few-shot `columnwise`/`encoding` 稍微優於 few-shot baseline，但仍無法挑戰 full-sample 的領先者。

---

## Category: small_datasets+binclass+numerical (28 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 11.36
- full none: 4.18
- few-shot trees: xgboost 14.64 / catboost 11.57 / lightgbm 12.11
- full trees: xgboost 3.32 / catboost 4.18 / lightgbm 5.68
- few-shot GNNs: t2g 10.21 / tabgnn 14.39
- tabpfn few/full: 10.82 / 3.32

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 11.36 | — (equal to few-shot none=11.36) | No | 1/3 | 0/3 | 1/2 (beats tabgnn few 14.39) | 0/2 | No | No |
| none | 11.36 | — | No | 1/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 14.36 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 10.57 | Yes (10.57 < 11.36) | No | 2/3 | 0/3 | 1/2 | 0/2 | Yes (10.57 < 10.82) | No |
| start | 14.57 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 14.79 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在 small numeric binary 中，部分 few-shot 注入（如 `encoding`）對 few-shot baseline 有小幅改善，但 full-sample 基線仍然居前。

---

## Category: small_datasets+regression+balanced (6 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 10.00
- full none: 10.50
- few-shot trees: xgboost 11.17 / catboost 11.00 / lightgbm 13.00
- full trees: xgboost 11.33 / catboost 10.67 / lightgbm 9.67
- few-shot GNNs: tabgnn 2.33 / t2g 16.17
- tabpfn few/full: 7.00 / 7.83

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 10.17 | No | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| none | 10.00 | — | Yes | 2/3 | 1/3 | 0/2 | 0/2 | No | Yes |
| decoding | 8.67 | Yes (8.67 < 10.00) | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes (8.67 < 7.00)? No -> 8.67 > 7.00 -> No | Yes |
| encoding | 9.67 | Yes | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| start | 8.33 | Yes | Yes | 3/3 | 3/3 | 0/2 | 0/2 | Yes? (8.33 < 7.00? No) | Yes |
| materialize | 8.33 | Yes | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | Yes |

註：在 small balanced regression 中，VIME 的 few-shot 注入（decoding/encoding/start/materialize）普遍能擊敗 full 非 GNN 基線與多數樹模型。

---

## Category: small_datasets+regression+categorical (5 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 6.60
- full none: 12.80
- few-shot trees: lightgbm 7.80 / xgboost 10.80 / catboost 11.80
- full trees: tabpfn few/full: 3.80 / 9.80
- few-shot GNNs: tabgnn 4.00 / t2g 15.00

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 7.00 | No | Yes | 2/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| none | 6.60 | — | Yes | 1/3 | 3/3 | 0/2 | 0/2 | No (6.60 < 3.80? No) | Yes |
| decoding | 5.40 | Yes (5.40 < 6.60) | Yes | 2/3 | 3/3 | 0/2 | 0/2 | Yes (5.40 < 3.80? No) | Yes |
| encoding | 6.60 | — (tie) | Yes | 1/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| start | 7.20 | No | Yes | 1/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| materialize | 7.80 | No | Yes | 1/3 | 3/3 | 0/2 | 0/2 | No | Yes |

註：在 small categorical regression 中，VIME 的 `decoding` 為最強的 few-shot 注入，能擊敗多數參考方法。

---

## Category: small_datasets+regression+numerical (36 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 7.61
- full none: 12.81
- few-shot trees: xgboost 10.17 / catboost 11.42 / lightgbm 10.31
- full trees: xgboost 15.22 / catboost 15.31 / lightgbm 14.83
- few-shot GNNs: tabgnn 3.64 / t2g 15.08
- tabpfn few/full: 3.64 / 7.69

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 7.50 | Yes (7.50 < 7.61) | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No (7.50 < 3.64? No) | Yes |
| none | 7.61 | — | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| decoding | 8.83 | No | Yes | 2/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| encoding | 6.92 | Yes | Yes | 3/3 | 3/3 | 0/2 | 0/2 | Yes? (6.92 < 3.64? No) | Yes |
| start | 6.89 | Yes | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| materialize | 6.08 | Yes | Yes | 3/3 | 3/3 | 0/2 | 0/2 | Yes (6.08 < 3.64? No) | Yes |

註：VIME 的 few-shot `materialize`/`start`/`encoding` 在 small numeric regression 中有幫助，能擊敗多個樹模型基線與 full 非 GNN 基線。

---

### Takeaways (VIME)
- VIME few-shot GNN injections provide consistent gains in regression tasks (both large and small). `start`, `materialize`, and `decoding` are often the most helpful stages depending on dataset type.
- In classification categories, fully-sampled `tabpfn` and fully-sampled tree baselines often remain dominant.

(End of VIME summary.)
