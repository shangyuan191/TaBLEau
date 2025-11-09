# RESNET — GNN 增強效果總結（與 EXCELFORMER 樣式一致）

說明：下表使用 resnet 排名檔中的 avg_rank 值，對每個 dataset-category 列出 few-shot 注入（start/encoding/columnwise/decoding/materialize）的平均排名，並判斷它們是否嚴格優於下列目標組（嚴格較小的 avg_rank 才算 beats）：few-shot non-GNN、full non-GNN、few-shot trees (xgboost/catboost/lightgbm)、full trees、few-shot GNNs (t2g-former/tabgnn)、full GNNs、tabpfn few/full。

---

註：本檔使用容差 1e-3 判斷 avg_rank 是否相等。
- 對於「beats few-shot 原始模型（few-shot non-GNN，ratio=0.05/0.15/0.8, gnn_stage=none）」的判定，使用嚴格比較（必須 strictly lower 才算 beats）。
- 對於其它比較（樹模型、GNN、tabpfn 等），若 avg_rank 在容差範圍內視為平手，並在表格中以 'Yes (tie)' 標示；若平手但涉及 few-shot 原始模型則標示 'No (tie, few-shot strict)'。


## large_datasets+binclass+numerical (6 datasets)

關鍵數值 (from ranking):
- few-shot resnet (none): 13.33
- full resnet (none): 3.67
- few-shot trees (xgboost/catboost/lightgbm): 11.00 / 9.17 / 10.00
- full trees: 4.83 / 3.67 / 3.83
- few-shot GNNs (t2g/tabgnn): 9.67 / 18.50
- full GNNs: 3.17 / 17.50
- tabpfn few/full: 8.33 / 2.17

| Injection | avg_rank | beats few-shot resnet? | beats full resnet? | beats few-shot trees (out of 3) | beats full trees (out of 3) | beats few-shot GNNs (out of 2) | beats full GNNs (out of 2) | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise (few) | 11.83 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding (few) | 12.00 | Yes | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize (few) | 16.17 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 16.50 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 14.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 13.33 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： 幾個注入（columnwise/encoding）比 few-shot baseline 略好，但都遠不及 full-sample baselines與 tabpfn。

---

## large_datasets+multiclass+numerical (3 datasets)

關鍵數值：
- few-shot resnet (none): 13.33
- full resnet (none): 3.67
- few-shot trees: catboost 8.67, xgboost 10.33, lightgbm 11.33
- few-shot GNNs: t2g 8.67, tabgnn 16.00
- tabpfn few/full: 9.33 / 2.00
- resnet few-shot injections: encoding 12.33, none 13.33, columnwise 14.67, decoding 16.33, start 18.00, materialize 18.33

| Injection | avg_rank | beats few-shot resnet? | beats full resnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| encoding (few) | 12.33 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 13.33 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 14.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 16.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 18.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 18.33 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： encoding 輕微優於 few-shot baseline，但依然落後於 full-sample 方法。

---

## large_datasets+regression+categorical (1 dataset)

關鍵數值：
- resnet few-shot decoding: 1.00 (best)
- resnet few-shot materialize: 6.00, start: 7.00, columnwise: 10.00, none: 11.00, encoding: 12.00
- tabpfn few/full: 2.00 / 3.00
- few-shot trees: catboost 13.00, xgboost 15.00, lightgbm 17.00
- few-shot GNNs: t2g 8.00, tabgnn 4.00

| Injection | avg_rank | beats few-shot resnet? | beats full resnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 1.00 | Yes | Yes | 3/3 | 3/3 | 2/2 | 2/2 | Yes | Yes |
| materialize (few) | 6.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| start (few) | 7.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| columnwise (few) | 10.00 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| none (few) | 11.00 | — | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| encoding (few) | 12.00 | No | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | Yes |

註： 在這個單一 categorical regression 資料集中，resnet 的 decoding 幾乎無可匹敵，並且超越了 tabpfn 與其它所有競爭者。

---

## large_datasets+regression+numerical (10 datasets)

關鍵數值：
- resnet few-shot (none): 10.80
- resnet full (none): 14.60
- few-shot resnet injections: decoding 5.70, materialize 6.20, start 6.80, columnwise 9.20, encoding 10.70
- few-shot trees: xgboost 10.50, catboost 12.40, lightgbm 12.60
- few-shot GNNs: t2g 11.90, tabgnn 4.00
- tabpfn few/full: 2.30 / 2.30

| Injection | avg_rank | beats few-shot resnet? | beats full resnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 5.70 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| materialize (few) | 6.20 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| start (few) | 6.80 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| columnwise (few) | 9.20 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 10.70 | Yes | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | No |
| none (few) | 10.80 | — | No (full is 14.60) | 1/3 | 0/3 | 0/2 | 0/2 | No | No |

註： resnet 的 few-shot 注入在 large numeric regression 中表現出色（decoding/materialize/start 均能擊敗樹模型與 full baseline）。

---

## small_datasets+binclass+balanced (14 datasets)

關鍵數值：
- few-shot resnet (none): 15.07
- full resnet (none): 7.79
- resnet few-shot injections: start 12.21, materialize 13.29, columnwise 13.71, encoding 13.71, decoding 15.29
- few-shot trees: catboost 11.14, xgboost 11.43, lightgbm 10.57
- tabpfn few/full: 7.86 / 4.79

| Injection | avg_rank | beats few-shot resnet? | beats full resnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| start (few) | 12.21 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 13.29 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 13.71 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 13.71 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 15.29 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 15.07 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： few-shot 注入會比 few-shot baseline 小幅改善，但仍明顯落後 full-sample baselines及 top trees。

---

## small_datasets+binclass+categorical (7 datasets)

關鍵數值：
- few-shot resnet (none): 11.71
- full resnet (none): 9.29
- resnet few-shot injections: materialize 11.71, none 11.71, columnwise 13.57, start 14.00, encoding 15.29, decoding 15.29
- few-shot trees: lightgbm 5.86, t2g 6.71, xgboost 8.00
- tabpfn few/full: 7.86 / 7.86

| Injection | avg_rank | beats few-shot resnet? | beats full resnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| materialize (few) | 11.71 | No (tie) | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 11.71 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 13.57 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start (few) | 14.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 15.29 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 15.29 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： small categorical binary 上樹方法與 t2g/tabpfn 佔優，resnet 的 few-shot 注入難以超越它們。

---

## small_datasets+binclass+numerical (28 datasets)

關鍵數值：
- few-shot resnet (none): 13.43
- full resnet (none): 5.25
- resnet few-shot injections: start 13.29, materialize 13.43, none 13.43, decoding 14.11, encoding 14.57, columnwise 15.04
- few-shot trees/GNNs/tabpfn: t2g few 9.18, tabpfn few 9.71, catboost few 10.18

| Injection | avg_rank | beats few-shot resnet? | beats full resnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| start (few) | 13.29 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize (few) | 13.43 | No (tie) | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none (few) | 13.43 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding (few) | 14.11 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding (few) | 14.57 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise (few) | 15.04 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註： 在多數 small numeric binary 資料集中，few-shot 注入與 few-shot baseline 相近；full-sample 仍獲得優勢。

---

## small_datasets+regression+balanced (6 datasets)

關鍵數值：
- resnet few-shot decoding: 4.33
- resnet few-shot start: 6.33, none:7.00, encoding:7.17, materialize:8.50, columnwise:10.83
- full resnet none: 10.50
- tabpfn few/full: 8.50 / 9.17

| Injection | avg_rank | beats few-shot resnet? | beats full resnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 4.33 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| start (few) | 6.33 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | Yes | Yes |
| none (few) | 7.00 | — | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | Yes |
| encoding (few) | 7.17 | Yes | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | Yes |
| materialize (few) | 8.50 | No | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | Yes |
| columnwise (few) | 10.83 | No | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | Yes |

註： decoding 在 small regression balanced 中表現最佳，能擊敗 few-shot 與 full baseline，多數樹方法也被超越。

---

## small_datasets+regression+categorical (5 datasets)

關鍵數值：
- resnet few-shot decoding: 3.00; start 4.60; encoding 7.20; materialize 8.00; none 8.20; columnwise 8.40
- full resnet none: 12.80
- tabpfn few/full: 6.00 / 9.20

| Injection | avg_rank | beats few-shot resnet? | beats full resnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 3.00 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| start (few) | 4.60 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | Yes | Yes |
| encoding (few) | 7.20 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| materialize (few) | 8.00 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| none (few) | 8.20 | — | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| columnwise (few) | 8.40 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |

註： decoding/start 在 small categorical regression 中都優於大多數基線。

---

## small_datasets+regression+numerical (36 datasets)

關鍵數值：
- resnet few-shot decoding: 3.39 (top)
- resnet few-shot columnwise: 6.69, materialize: 7.44, start:7.58, encoding:8.31, none:8.39
- full resnet none: 14.75
- tabpfn few/full: 5.36 / 7.67

| Injection | avg_rank | beats few-shot resnet? | beats full resnet? | beats few-shot trees | beats full trees | beats few-shot GNNs | beats full GNNs | beats few tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding (few) | 3.39 | Yes | Yes | 3/3 | 3/3 | 2/2 | 2/2 | Yes | Yes |
| columnwise (few) | 6.69 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | Yes |
| materialize (few) | 7.44 | Yes | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| start (few) | 7.58 | Yes | Yes | 3/3 | 3/3 | 0/2 | 0/2 | No | Yes |
| encoding (few) | 8.31 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |
| none (few) | 8.39 | — | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | Yes |

註： resnet 在 small regression numerical 中的 few-shot decoding 表現非常突出，能擊敗 tabgnn/tabpfn 及大多數樹方法。

---

### 要點總結 (RESNET)

- RESNET 的 few-shot GNN 注入在各類 regression 資料集中（特別是 small/large numeric regression 與 categorical regression）能帶來明顯提升，decoding 與 materialize/start 是常見的贏家。
- 在多數 classification 任務（尤其 large/mid classification）中，fully-sampled 樹方法與 tabpfn 仍然是最強的 baseline，few-shot 注入通常無法超越。

註：在此更新中，針對「beats few-shot 原始模型」（few-shot non-GNN baseline）仍採用嚴格比較（avg_rank 更小才算為勝）。對於其它比較（對樹模型、GNN、tabpfn 等群組），若 avg_rank 相等則視為擊敗，且在表格中以 " (tie, few-shot strict)" 標示；所有標註為 (tie, few-shot strict) 的情況皆已視為 beats。