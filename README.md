# TaBLEau
> TaBLEau (Tabular Benchmark Learning Evaluation and Analysis Union)
## HackMD
https://hackmd.io/@shangyuan191/B1EeV1PVC
## Introduction
TaBLEau is a comprehensive benchmarking and evaluation toolkit for deep learning models on tabular datasets. This repository standardizes and integrates multiple state-of-the-art models from various research papers, offering a unified framework to seamlessly test and compare their performance across a diverse collection of tabular datasets. With support for both classification and regression tasks, TaBLEau enables researchers to efficiently benchmark, analyze, and extend models for tabular data analysis under consistent input-output configurations, promoting fair and reproducible comparisons.


## My Dataset
### GoogleDrive
https://drive.google.com/file/d/1EBOpdZe6o7TkMZ5E50qbca6bN74Ns_gV/view?usp=sharing
### HuggingFace
* **small + binclass**
https://huggingface.co/datasets/SkylerChuang/small_binclass_datasets
* **small + regression**
https://huggingface.co/datasets/SkylerChuang/small_regression_datasets
* **large + binclass**
https://huggingface.co/datasets/SkylerChuang/large_binclass_datasets
* **large + multiclass**
https://huggingface.co/datasets/SkylerChuang/large_multiclass_datasets
* **large + regression**
https://huggingface.co/datasets/SkylerChuang/large_regression_datasets

## ref
* **ExcelFormer HuggingFace**
https://huggingface.co/datasets/jyansir/excelformer
* **pytorch-frame**
https://github.com/pyg-team/pytorch-frame
* **TabPFN**
https://github.com/automl/TabPFN


## GNN 注入策略管線（中文說明）
- **共同基底**: `encoder` 將資料編碼為 `x:[B, F, C]`；`convs` 在欄位維度做多頭注意力以進行列間互動，維持 `x:[B, F, C]`；`decoder` 產生最終輸出 `out:[B, out_channels]`。輔助組件包含 `column_embed:[F, C]`（欄位位置編碼）、`pool_query:[C]`（注意力池化）、`DGM_d`（動態圖，輸入 `[1, B, C]`，輸出 `(X_dgm:[B, C], edge_index:[2, E])`）、`SimpleGCN`（圖卷積，`[B, Cin] → [B, Cout]`）、`fusion_alpha_param`（殘差融合門）等。

- **gnn_stage==none**:
	- Start: `start_fn`（不變更）
	- Materialize: `materialize_fn` 生成 TensorFrame/DataLoader，處理類別為數值與互信息排序
	- Encoding: `encoder` → `x:[B, F, C]`
	- Columnwise: `convs`（多層）→ `x:[B, F, C]`
	- Decoding: `decoder` → `out:[B, out_channels]`
	- 目的: 純 ExcelFormer 流程，無 GNN 注入。

- **gnn_stage==start**（離線特徵預注入）:
	- `gnn_after_start_fn` 對原始 DataFrame 進行：
		- `input_proj`: `[N, F, 1] → [N, F, D]`
		- `attn_in`（自注意力）: `[N, F, D] → [N, F, D]`
		- 注意力池化: `[N, F, D] → [N, D]`（row-level 向量）
		- `DGM_d`: `[1, N, D] → ([N, D], edge_index)`（樣本為節點的動態圖）
		- `GCN`: `[N, D] → [N, G]`；`pred_head`: `[N, G] → [N, out_dim]`（監督訓練）
		- 重建欄位尺度: `gcn_to_attn:[N, G] → [N, 1, D]`、`attn_out:[N, F, D] → [N, F, D]`、`out_proj:[N, F, D] → [N, F]`
	- 產出新 DataFrame（保留 F 欄位），再走 `materialize_fn` 與核心 ExcelFormer（Encoding → Columnwise → Decoding）。

- **gnn_stage==materialize**（物化後離線注入）:
	- `materialize_fn` 後執行 `gnn_after_materialize_fn`（與 start 同構，對 TensorFrame 轉回 DataFrame 處理）
	- 重建後再包裝成 Dataset/TensorFrame、DataLoader，核心 ExcelFormer（Encoding → Columnwise → Decoding）不變。

- **gnn_stage==encoding**（聯訓：編碼後、卷積前注入）:
	- Tokens: `x + column_embed` → `[B, F, C]`
	- `self_attn`: `[B, F, C] → [B, F, C]`
	- 注意力池化: `[B, F, C] → [B, C]`
	- `DGM_d`: `[1, B, C] → ([B, C], edge_index)`
	- `GCN`: `[B, C] → [B, C]`
	- Self-Attn 解碼: `gcn_to_attn:[B, C] → [B, 1, C]`、`self_attn_out:[B, F, C]`
	- 殘差融合: `x = x + sigmoid(fusion_alpha_param) * tokens_out` → `[B, F, C]`
	- 之後 `convs`（Columnwise）與 `decoder` 如常。

- **gnn_stage==columnwise**（聯訓：卷積後注入）:
	- 在完成 `convs` 後，執行與 encoding 同構的 Self-Attn + DGM + GCN + Self-Attn 解碼 + 殘差融合，維持 `x:[B, F, C]`，再進入 `decoder`。

- **gnn_stage==decoding**（聯訓：以 GNN 取代解碼器）:
	- 在 `convs` 之後：`x + column_embed` → `self_attn:[B, F, C]` → 注意力池化 `[B, C]`
	- `DGM_d` 建圖 → `GCN` 直接作為解碼器輸出 `out:[B, out_channels]`
	- 完全繞過原 `decoder`，以圖上的 row-level 表徵直接預測。


