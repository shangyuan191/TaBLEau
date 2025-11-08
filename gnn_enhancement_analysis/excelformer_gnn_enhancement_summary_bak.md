# EXCELFORMER — GNN 增強效果總結

說明：本檔按 dataset-category 匯總 few-shot EXCELFORMER（ratio=0.05/0.15/0.8）各種 GNN 注入變體是否在平均排名（avg_rank）上擊敗指定的目標組。"擊敗" 在此預設為比對方的 avg_rank 更小（數值更小表示更好）。

目標組說明：
- few-shot non-GNN（少樣本非 GNN 基線）：`excelformer(ratio=0.05/0.15/0.8, gnn_stage=none)`
- full-sample non-GNN（全量樣本非 GNN 基線）：`excelformer(ratio=0.8/0.15/0.05, gnn_stage=none)`
- few-shot 樹模型（xgboost/catboost/lightgbm，ratio=0.05/0.15/0.8）
- full-sample 樹模型（xgboost/catboost/lightgbm，ratio=0.8/0.15/0.05）
- few-shot GNN（t2g-former, tabgnn，ratio=0.05/0.15/0.8）
- full-sample GNN（t2g-former, tabgnn，ratio=0.8/0.15/0.05）
- few-shot tabpfn: `tabpfn` ratio `0.05/0.15/0.8`
- full-sample tabpfn: `tabpfn` ratio `0.8/0.15/0.05`

考慮的 few-shot EXCELFORMER 注入位置：
- `columnwise`（ratio=0.05）
- `none`（few-shot baseline，ratio=0.05）
- `decoding`（ratio=0.05）
- `encoding`（ratio=0.05）
- `start`（ratio=0.05）
- `materialize`（ratio=0.05）

---

## Category: large_datasets+binclass+numerical (6 datasets)
(Reference ranks from source)

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 13.00 | Yes (13.00 < 13.17) | Yes (tie) | 0/3 | 0/3 | 1/2 (beats tabgnn only) | 1/2 (beats tabgnn only; t2g-former much better) | No | No |
| none (few-shot baseline) | 13.17 | — (baseline) | No | 0/3 | 0/3 | 1/2 (beats tabgnn only) | 1/2 | No | No |
| decoding | 13.17 | No (tie with baseline) | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| encoding | 13.83 | No | No | 0/3 | 0/3 | 1/2 | 1/2 | No | No |
| start | 16.33 | No | No | 0/3 | 0/3 | 1/2 | 0/2 (does not beat tabgnn full) | No | No |
| materialize | 16.67 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |

註：在此類別中，表現最好的為全量訓練的 `tabpfn` 與全量訓練的 `t2g-former`；few-shot EXCELFORMER 變體未能擊敗強大的樹模型或全量基線。在少數情況下，它們僅能擊敗較弱的 `tabgnn`（few/full）。

---

## Category: large_datasets+multiclass+numerical (3 datasets)

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 14.00 | No | No | 0/3 | 0/3 | 1/2 (beats tabgnn only) | 0/2 | No | No |
| none | 12.67 | — | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| decoding | 18.67 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 13.67 | No | No | 0/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 17.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 18.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在此類別中，全量訓練的 EXCELFORMER 表現相對較佳；few-shot 的注入未能超越在此類別中領先的全量基線或樹/gnn/全量 tabpfn 的表現。

---

## Category: large_datasets+regression+categorical (1 dataset)

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding | 5.00 | Yes (5.00 < 12.00) | Yes (5.00 < 19.00) | 3/3 | 3/3 | 1/2 (beats t2g only) | 1/2 (beats t2g only vs full tabgnn) | No | No |
| start | 8.00 | Yes | Yes | 3/3?* | 3/3* | 0/2 | 0/2 | No | No |
| materialize | 11.00 | Yes | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| none | 12.00 | — | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |
| columnwise | 17.00 | No | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 18.00 | No | Yes | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

*Explanation: tree-based few-shot ranks are catboost 9 / xgboost 13 / lightgbm 15 — decoding (5) and start (8) beat all three; materialize and none beat xgboost/lightgbm but not catboost.

註：在這個單一的分類回歸資料集中，某些 few-shot EXCELFORMER 注入（尤其 `decoding`、`start`、`materialize`）確實能明顯擊敗多數參考方法（包含全量訓練的 EXCELFORMER）。

---

## Category: large_datasets+regression+numerical (10 datasets)

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| start | 5.70 | Yes (5.70 < 11.00) | Yes (5.70 < 15.90) | 3/3 | 3/3 | 1/2 (beats t2g only) | 1/2 | No | No |
| materialize | 6.80 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| none | 11.00 | — | Yes | 1/3 (beats lightgbm only) | 3/3 | 0/2 | 1/2 | No | No |
| encoding | 11.00 | No (tie with none) | Yes | 1/3 | 3/3 | 0/2 | 1/2 | No | No |
| columnwise | 11.10 | No | Yes | 0/3 (ties/worse) | 3/3 | 0/2 | 1/2 | No | No |
| decoding | 13.80 | No | Yes | 0/3 | 3/3 | 0/2 | 1/2 | No | No |

註：在大型數值型回歸中，`start` 與 `materialize` few-shot 變體表現最佳——它們能擊敗 few-shot 樹模型與全量樹基線，並擊敗兩個 GNN 基線中的較弱者（t2g/former 的優劣順序視 ratio 而定）。然而，它們仍無法超越 tabpfn。

---

## Category: small_datasets+binclass+balanced (14 datasets)

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| none | 12.71 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| columnwise | 13.14 | Yes (13.14 < 12.71?) NO -> actually 13.14 > 12.71 -> No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| start | 13.71 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 13.79 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 14.36 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 15.21 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在小型平衡二分類資料集中，EXCELFORMER 的 few-shot 注入未能超越主要基線或樹/GNN 參考模型；全量訓練的 EXCELFORMER 與全量樹模型 / tabpfn 仍佔主導地位。

---

## Category: small_datasets+binclass+categorical (7 datasets)

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| start | 8.86 | Yes (8.86 < 12.71) | No (8.86 > 8.29) | 0/3 | 1/3 | 1/2 | 1/2 | No | No |
| columnwise | 10.86 | Yes | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| none | 12.71 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| encoding | 12.71 | — | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 18.00 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| materialize | 18.29 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：`start` 是少數會在某些情況改善 few-shot baseline 並勝過部分參考設置（例如部分 GNN/樹模型的全量配置）的注入，但它仍無法擊敗全量表現最好的模型。

---

## Category: small_datasets+binclass+numerical (28 datasets)

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 12.18 | Yes (12.18 < 12.32) | No | 1/3 (beats xgboost few-shot only) | 0/3 | 1/2 (beats tabgnn few-shot only) | 0/2 | No | No |
| none | 12.32 | — | No | 1/3 | 0/3 | 1/2 | 0/2 | No | No |
| start | 13.07 | No | No | 1/3 | 0/3 | 1/2 | 0/2 | No | No |
| encoding | 13.18 | No | No | 1/3 | 0/3 | 1/2 | 0/2 | No | No |
| materialize | 14.25 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |
| decoding | 15.54 | No | No | 0/3 | 0/3 | 0/2 | 0/2 | No | No |

註：在許多小型數值二分類資料集中，few-shot EXCELFORMER 注入很少擊敗表現最佳的 few-shot 樹模型或 tabpfn；`columnwise` 與 few-shot baseline 偶爾能擊敗最弱的 few-shot 樹模型（例如某些分割下的 xgboost）及較弱的 tabgnn few-shot。

---

## Category: small_datasets+regression+balanced (6 datasets)

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| decoding | 6.67 | Yes (6.67 < 9.33) | Yes (6.67 < 11.50) | 3/3 | 3/3 | 1/2 | 2/2 | Yes (beats tabpfn few-shot 7.50) | Yes (beats tabpfn full 8.50) |
| materialize | 7.33 | Yes | Yes | 3/3 | 3/3 | 1/2 | 2/2 | Yes | Yes |
| start | 8.50 | Yes (8.50 < 9.33) | Yes | 3/3 | 3/3 | 1/2 | 2/2 | No (8.50 > 7.50) | Yes (tie) |
| encoding | 9.33 | — (equal to baseline none=9.33) | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| none | 9.33 | — | Yes | 3/3 | 3/3 | 0/2 | 1/2 | No | No |
| columnwise | 10.67 | No | Yes | 2/3 | 2/3 | 0/2 | 0/2 | No | No |

註：在此類別（小型平衡回歸）中，部分 few-shot EXCELFORMER 注入（特別是 `decoding` 與 `materialize`）明顯超越樹模型基線及 tabpfn。

---

## Category: small_datasets+regression+categorical (5 datasets)

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | beats few-shot tree (out of 3) | beats full tree (out of 3) | beats few-shot GNN (out of 2) | beats full GNN (out of 2) | beats few-shot tabpfn? | beats full tabpfn? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| columnwise | 4.80 | Yes (4.80 < 7.40) | Yes (4.80 < 15.20) | 3/3 | 3/3 | 1/2 (beats t2g only) | 1/2 | No | No |
| materialize | 5.40 | Yes | Yes | 3/3 | 3/3 | 1/2 | 1/2 | No | No |
| none | 7.40 | — | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | No |
| decoding | 7.80 | Yes | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | No |
| start | 7.80 | Yes | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | No |
| encoding | 8.00 | Yes | Yes | 1/3 | 1/3 | 0/2 | 0/2 | No | No |

註：在小型分類回歸任務中，數個 few-shot EXCELFORMER 注入（尤其 `columnwise`、`materialize` 等）勝過多數參考方法。

---

### 要點總結（EXCELFORMER）
- 在某些情況下，EXCELFORMER 的 few-shot GNN 注入確實能帶來幫助，尤其中的回歸任務（包含 small 與 large）以及部分小型分類的 categorical 回歸案例。`decoding` 與 `materialize`（在部分 large numeric regression 中 `start` 亦有顯著表現）是最常見的有益注入。
- 在很多分類類別（尤其是 large binary numerical 與多數 small binary）中，full-sample 的 `tabpfn` 與樹模型的全量基線仍然佔優，few-shot EXCELFORMER 變體很少超越它們。
- 在與分組參考比較時，few-shot EXCELFORMER 注入比較常擊敗那些 full-sample 的樹基線（因為 few-shot 設定下這些變體較為合適），並在少數情況下擊敗較弱的 GNN 基線（通常是 `tabgnn`），但很少能超越 `t2g-former` 或 full-sample 的 `tabpfn`。

---

檔案由自動化摘要產生（來源：`excelformer_gnn_enhancement.txt`）。下一步可採取相同格式處理其餘 9 個模型檔案，並彙整整體 GNN 注入效果報告。
