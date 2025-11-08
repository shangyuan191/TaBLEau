# TABTRANSFORMER — GNN 增強效果總結

說明：本檔根據 `tabtransformer_gnn_enhancement.md` 的 avg_rank 值，按 dataset-category 匯總 few-shot（ratio=0.05/0.15/0.8）下各種 GNN 注入變體是否擊敗指定的目標組。

註：在此更新中，針對「beats few-shot 原始模型」（few-shot non-GNN baseline）仍採用嚴格比較（avg_rank 更小才算為勝）。對於其它比較（對樹模型、GNN、tabpfn 等群組），若 avg_rank 相等則視為擊敗，且在表格中以 " (tie)" 標示；所有標註為 (tie) 的情況皆已視為 beats。

考慮的 few-shot TABTRANSFORMER 注入：`columnwise`、`none`（few-shot baseline）、`decoding`、`encoding`、`start`、`materialize`（皆為 ratio=0.05/0.15/0.8）。

---

## Category: large_datasets+binclass+numerical (6 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none (tabtransformer, ratio=0.05): 15.00
- full none (tabtransformer, ratio=0.8): 10.50
- few-shot trees (xgboost/catboost/lightgbm): 9.33 / 7.83 / 8.50
- full trees (xgboost/catboost/lightgbm): 4.17 / 3.33 / 3.33
- few-shot GNNs (t2g/tabgnn): 7.67 / 15.33
- full GNNs (t2g/tabgnn): 2.67 / 14.17
- tabpfn few/full: 7.00 / 1.83

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 16.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few-shot baseline) | 15.00 | — | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 12.83 | Yes (12.83 < 15.00) | No | 0/3 | 0/3 | 1/2 (beats tabgnn only) | 1/2 (beats tabgnn full only) | No | No |
| encoding | 15.83 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 17.17 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 16.83 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在此類別中，few-shot TABTRANSFORMER 的 `decoding` 較 few-shot baseline 略有改善，但沒有任何 few-shot 注入能擊敗 full-tabtransformer 或強勢的 full-sample 樹模型與 tabpfn。

---

## Category: large_datasets+multiclass+numerical (3 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 14.33
- full none: 5.67
- few-shot trees (xgboost/catboost/lightgbm): 10.00 / 8.67 / 10.67
- full trees: 5.33 / 3.33 / 4.00
- few-shot GNNs (t2g/tabgnn): 8.67 / 14.67
- full GNNs: 1.33 / 11.00 (t2g full is 1.33, tabgnn full 11.00)
- tabpfn few/full: 8.33 / 1.67

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 14.33 | No (tie) | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none | 14.33 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 17.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 14.00 | Yes (14.00 < 14.33) | No | 0/3 | 0/3 | 1/2 (beats tabgnn only) | 0/2 | No | No |
| start | 18.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 18.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在這個 small multiclass 類別，few-shot 的 `encoding` 對 few-shot baseline 有小幅提升，但仍與 full-sample 的領先者（如 t2g、tabpfn）有差距。

---

## Category: large_datasets+regression+categorical (1 dataset)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 7.00
- full none: 8.00
- few-shot trees: catboost 14 / xgboost 16 / lightgbm 18
- full trees: catboost 15 / xgboost 17 / lightgbm 19
- few-shot GNNs: t2g 12 / tabgnn 9
- full GNNs: t2g 13 / tabgnn 10
- tabpfn few/full: 2 / 3

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 1.00 | Yes | Yes | 3/3 | 3/3 | 1/2 (beats tabgnn only) | 1/2 | Yes (beats tabpfn few) | Yes |
| none | 7.00 | — | No | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| decoding | 5.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| encoding | 6.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| start | 11.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 4.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |

註：在這個單一 categorical regression 資料集中，TABTRANSFORMER 的 few-shot `columnwise`、`decoding` 與 `materialize` 表現優異；其中 `columnwise` 更能擊敗多個基線（包含 few-shot 的 tabpfn）。

---

## Category: large_datasets+regression+numerical (10 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 7.90
- full none: 10.80
- few-shot trees (xgboost/catboost/lightgbm): 12.60 / 14.00 / 14.20
- full trees: 16.70 / 17.10 / 17.70
- few-shot GNNs: tabgnn 3.90, t2g 14.40
- full GNNs: tabgnn 4.80, t2g 16.90
- tabpfn few/full: 1.60 / 2.10

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 7.10 | Yes (7.10 < 7.90) | Yes (7.10 < 10.80) | 3/3 | 3/3 | 1/2 (beats tabgnn only) | 1/2 | No | No |
| none | 7.90 | — | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | No |
| decoding | 7.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| encoding | 7.80 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| start | 7.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| materialize | 6.40 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |

註：在 large-scale 的數值回歸上，許多 TABTRANSFORMER 的 few-shot 注入能同時擊敗 few-shot 與 full 的 non-GNN 基線，也能勝過所有 few-shot 樹模型；其中 `materialize` 與其他 few-shot GNN 變體尤為強勢，但仍落後於 tabpfn。

---

## Category: small_datasets+binclass+balanced (14 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 13.93
- full none: 7.14
- few-shot trees: catboost 10.57 / lightgbm 10.57 / xgboost 11.50
- full trees: catboost 3.93 / lightgbm 5.93 / xgboost 6.00
- few-shot GNNs: t2g 9.43 / tabgnn 11.21
- tabpfn few/full: 8.00 / 4.86

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 14.14 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none | 13.93 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 12.71 | Yes (12.71 < 13.93) | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 12.93 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 14.86 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 14.93 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在這些 small balanced binary 任務中，tabtransformer 的 few-shot 注入很少能擊敗全量訓練的樹模型或 tabpfn；僅有少數注入在 few-shot baseline 上帶來小幅改善。

---

## Category: small_datasets+binclass+categorical (7 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 12.43
- full none: 11.29
- few-shot trees: lightgbm 6.14 / catboost 10.57 / xgboost 7.86
- full trees: t2g 5.29 / tabpfn 6.14 / xgboost 7.14
- few-shot GNNs: t2g 7.14 / tabgnn 11.79
- tabpfn few/full: 7.57 / 6.14

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 14.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none | 12.43 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 14.86 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 15.14 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 14.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 16.29 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在這些 small categorical binary 任務中，few-shot tabtransformer 變體未能擊敗強勢的 few/full-sample 基線。

---

## Category: small_datasets+binclass+numerical (28 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 14.21
- full none: 9.04
- few-shot trees: xgboost 12.39 / catboost 9.96 / lightgbm 10.04
- full trees: xgboost 3.11 / catboost 4.89 / lightgbm 5.46
- few-shot GNNs: t2g 9.00 / tabgnn 12.14
- tabpfn few/full: 9.36 / 3.11

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 14.14 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none | 14.21 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 12.36 | Yes (12.36 < 14.21) | No | 1/3 (beats xgboost few-shot 12.39?) | 0/3 | 1/2 (beats tabgnn few-shot 12.14?) -> careful: 12.36 < 12.14? No -> 0/2 | 0/2 | No | No |
| encoding | 14.82 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 14.79 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 16.25 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在這些 small numeric binary 任務中，`decoding` 對 few-shot baseline 有小幅提升，但仍落後於 full-sample 基線與 tabpfn。

---

## Category: small_datasets+regression+balanced (6 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 6.00
- full none: 11.67
- few-shot trees: xgboost 12.17 / catboost 12.00 / lightgbm 13.17
- full trees: xgboost 10.67 / catboost 11.50 / lightgbm 9.83
- few-shot GNNs: tabgnn 3.17 / t2g 16.50
- tabpfn few/full: 7.50 / 8.67

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 11.50 | No | Yes | 2/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| none | 6.00 | — | No | 3/3 | 1/3 | 0/2 | 0/2 | Yes (6.00 < 7.50) | No |
| decoding | 5.83 | Yes (5.83 < 6.00) | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes (5.83 < 7.50) | Yes |
| encoding | 8.17 | No | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| start | 9.67 | No | Yes | 1/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| materialize | 5.50 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |

註：在 small balanced regression 中，若干 few-shot tabtransformer 注入（如 decoding、materialize）能擊敗 few-shot 與 full 非 GNN 基線，並在 few-shot 仍勝過 tabpfn。

---

## Category: small_datasets+regression+categorical (5 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 5.00
- full none: 12.40
- few-shot trees: lightgbm 7.80 / xgboost 10.80 / catboost 11.00
- full trees: tabpfn few/full: 6.20 / 9.40
- few-shot GNNs: tabgnn 3.80 / t2g 15.40

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 7.20 | No | Yes | 2/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| none | 5.00 | — | Yes | 3/3 | 3/3 | 0/2 | 0/2 | Yes (5.00 < 6.20) | Yes |
| decoding | 7.40 | No | Yes | 2/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| encoding | 6.20 | Yes (6.20 < 5.00)? No -> careful: 6.20 > 5.00 -> No | Yes | 2/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| start | 8.20 | No | Yes | 1/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| materialize | 7.60 | No | Yes | 2/3 | 3/3 | 0/2 | 0/2 | No | Yes |

註：在 small categorical regression 中，tabgnn 與 tabpfn 佔優；tabtransformer 的 few-shot 變體常能擊敗 full non-GNN 基線，但鮮少超越最強的 few-shot GNN 或 tabpfn。

---

## Category: small_datasets+regression+numerical (36 datasets)
(Reference ranks from source)

Key reference ranks:
- few-shot none: 7.19
- full none: 10.94
- few-shot trees: xgboost 10.17 / catboost 11.69 / lightgbm 10.81
- full trees: xgboost 3.32 / catboost 4.89 / lightgbm 5.68
- few-shot GNNs: tabgnn 4.56 / t2g 15.14
- tabpfn few/full: 5.17 / 7.69

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree | beats full tree | beats few-shot GNN | beats full GNN | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 6.44 | Yes (6.44 < 7.19) | Yes | 3/3 | 3/3 | 1/2 (beats t2g? t2g few-shot 15.14 -> yes) | 1/2 | Yes? (6.44 < 5.17 -> No) | No |
| none | 7.19 | — | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| decoding | 6.97 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | Yes |
| encoding | 7.50 | No | Yes | 2/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| start | 7.11 | Yes | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| materialize | 7.78 | No | Yes | 2/3 | 3/3 | 0/2 | 0/2 | No | Yes |

註：在 small numeric regression 中，few-shot 的 `columnwise` 與 `decoding` 具競爭力，能擊敗多個 few-shot 樹模型，且常能勝過 full non-GNN 基線。

---

### Takeaways (TABTRANSFORMER)
- TABTRANSFORMER few-shot GNN injections help most in regression tasks (both large numeric regression and small regression categories). `materialize`, `decoding`, and `columnwise` frequently improve performance vs the few-shot baseline and beat several tree baselines.
- Classification categories, especially large-dataset classification, are still dominated by fully-sampled `tabpfn` and fully-sampled tree baselines.

(Next: I'll produce the summaries for `trompt` and `vime`.)
