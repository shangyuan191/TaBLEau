# TaBLEau + SAGE 全面說明檔（供 AI 使用）

> 目的：為 AI 代理提供對 TaBLEau 框架與 SAGE 研究的完整上下文，涵蓋資料、模型、流程、GNN 插入策略、實驗設計與後續消融計畫，方便自動化運行、分析與繪圖。

## -1. 論文題目（Paper / Thesis Title）

- English: **Where to Graph-ify Tabular Deep Learning: Finding the Sweet Spot with Actionable Insights**
- 中文：**圖化表格深度學習: 找出圖神經網路嵌入階段甜蜜點與資料導向可落地洞察**

---

## 0. 研究背景與動機

### 0.1 表格資料學習的挑戰
隨著表格資料（tabular data）在真實世界應用中日益普及，近年來眾多深度學習模型相繼被提出以提升其預測效能。然而，不同類型模型在資料結構與表徵學習上各有所長：
- **神經網路模型**：擅長學習複雜的非線性關係，但往往需要大量標註數據。
- **樹狀模型（如 XGBoost、CatBoost、LightGBM）**：在中小型資料集上表現優異，對特徵工程要求較低。
- **預訓練模型（如 TabPFN）**：利用大規模預訓練知識，能在極少樣本下快速適應。

這些模型雖各具優勢，但如何在多種架構中有效結合圖神經網路（Graph Neural Network, GNN）來捕捉資料的結構關係與相似性，成為一項關鍵研究課題。

### 0.2 SAGE 研究框架
本研究提出 **SAGE（Stage-Aware Graph Enhancement）**，一個針對「在表格深度學習中何時/何處引入圖結構與圖神經網路」的系統性分析框架。

**核心研究問題**：
1. GNN 能否在表格學習中帶來結構感知與小數據增益？
2. 在模型的哪個階段插入 GNN 最有效？
3. GNN 的增益是否隨資料型態（大小、任務類型、特徵類型）而異？
4. 在少樣本（few-shot）與充足樣本（fully-supervised）情境下，GNN 的作用有何差異？

**研究方法**：
- 於各類 SOTA 表格模型的不同階段（編碼層、特徵抽取層、輸出層）插入 GNN。
- 透過 116 個異質資料集與雙重切分設定（few-shot vs fully-supervised）進行廣泛實驗。
- 對比多種基線模型（樹模型、原生 GNN 模型、預訓練模型）。

**預期發現**：
- GNN 的引入能在低標註資料情境下有效提升模型的泛化與結構感知能力。
- 其效益隨模型架構與資料型態而異。
- 為 GNN 在少樣本表格學習中的應用提供實證分析與設計指引。

---

## 1. TaBLEau 框架總覽

### 1.1 框架定位
**TaBLEau (Tabular Benchmark Learning Evaluation and Analysis Union)** 是一個統一的表格數據深度學習基準測試框架，專為系統性評估與比較多種 SOTA 模型而設計。

**核心特色**：
- **統一介面**：所有模型遵循相同的輸入輸出格式與訓練流程。
- **多樣數據**：涵蓋 116 個資料集，按規模、任務、特徵類型分類。
- **一致環境**：標準化的資料切分、評估指標與超參數設定。
- **可擴展性**：支援動態添加新模型與新資料集。

### 1.2 專案目標
- 提供跨模型的公平比較基準。
- 系統性分析 GNN 插入對表格模型的影響。
- 揭示不同模型在不同資料型態下的優劣勢。
- 為表格深度學習研究提供可重現的實驗環境。

---

## 2. 資料集詳細說明

### 2.1 資料集位置與結構
- **根目錄**：`/home/skyler/ModelComparison/TaBLEau/datasets`
- **總數**：116 個表格資料集（CSV 格式）
- **組織方式**：按三個維度進行分類

**維度一：資料集規模（Dataset Size）**
- `small_datasets/`：小型資料集（row count 較少）
- `large_datasets/`：大型資料集（row count 較多）

**維度二：任務類型（Task Type）**
- `binclass/`：二分類任務
- `multiclass/`：多分類任務
- `regression/`：迴歸任務

**維度三：特徵類型占比（Feature Type Proportion）**
- `numerical/`：數值型欄位占多數
- `categorical/`：類別型欄位占多數
- `balanced/`：數值與類別型欄位比例相近

### 2.2 典型路徑示例
```
datasets/
├── small_datasets/
│   ├── binclass/
│   │   ├── numerical/       # 小型二分類數值型資料集
│   │   ├── categorical/     # 小型二分類類別型資料集
│   │   └── balanced/        # 小型二分類混合型資料集
│   ├── multiclass/
│   │   ├── numerical/
│   │   ├── categorical/
│   │   └── balanced/
│   └── regression/
│       ├── numerical/
│       ├── categorical/
│       └── balanced/
└── large_datasets/
    ├── binclass/
    │   ├── numerical/       # 大型二分類數值型資料集
    │   ├── categorical/
    │   └── balanced/
    ├── multiclass/
    │   ├── numerical/
    │   ├── categorical/
    │   └── balanced/
    └── regression/
        ├── numerical/
        ├── categorical/
        └── balanced/
```

### 2.3 標準資料切分
本研究使用兩種標準切分策略，以對比少樣本與充足樣本情境：

**Few-Shot 設定（少樣本學習）**
- Train : Val : Test = 0.05 : 0.15 : 0.80
- 目的：模擬標註數據稀缺的真實場景
- 用途：評估模型在低資源情境下的泛化能力

**Fully-Supervised 設定（充足監督學習）**
- Train : Val : Test = 0.80 : 0.15 : 0.05
- 目的：提供充足訓練數據的對照組
- 用途：評估模型在資源充足時的上限表現

### 2.4 資料特性
- **欄位類型**：數值型（連續/離散）、類別型（需編碼處理）
- **欄位數量**：從數個到數十個不等，差異較大
- **樣本數量**：small（數百到數千）、large（數千到數萬）
- **任務難度**：涵蓋簡單到複雜的多種預測任務
- **資料處理**：由 `DatasetLoader` 自動處理缺失值、編碼、標準化等

---

## 3. 模型族群詳細分類

### 3.1 可拆分模型（10 個，支援 GNN 插入）

這 10 個模型是本研究的核心實驗對象，它們都可以按照統一的五階段流水線進行拆分，並在各階段插入 GNN。

**模型列表**：
1. **ExcelFormer** - 基於混合注意力機制的表格模型
2. **FT-Transformer** - Feature Tokenizer + Transformer
3. **ResNet** - 殘差網路用於表格數據
4. **TabNet** - 序列注意力機制與特徵選擇
5. **TabTransformer** - 類別特徵的 Transformer 編碼
6. **Trompt** - 提示學習用於表格數據
7. **SCARF** - 自監督對比學習框架
8. **SubTab** - 子空間表徵學習
9. **VIME** - 變分互信息最大化編碼器
10. **TabM** - 多模態表格學習

**程式位置**：
- PyTorch Frame 家族（6 個）：`/home/skyler/ModelComparison/TaBLEau/models/pytorch_frame/`
  - excelformer.py, fttransformer.py, resnet.py, tabnet.py, tabtransformer.py, trompt.py
- Custom 家族（4 個）：`/home/skyler/ModelComparison/TaBLEau/models/custom/`
  - scarf.py, subtab.py, vime.py, tabm.py

**階段映射文件**：
每個模型都有對應的映射說明文件，描述其原始架構如何對應到五階段流水線：
- `models/pytorch_frame/excelformer_pytorch_frame_mapping.md`
- `models/pytorch_frame/fttransformer_pytorch_frame_mapping.md`
- `models/pytorch_frame/resnet_pytorch_frame_mapping.md`
- `models/pytorch_frame/tabnet_pytorch_frame_mapping.md`
- `models/pytorch_frame/tabtransformer_pytorch_frame_mapping.md`
- `models/pytorch_frame/trompt_pytorch_frame_mapping.md`
- `models/custom/scarf_pytorch_frame_mapping.md`
- `models/custom/subtab_pytorch_frame_mapping.md`
- `models/custom/vime_pytorch_frame_mapping.md`
- `models/custom/tabm_pytorch_frame_mapping.md`

**拆分策略**：
根據 `/home/skyler/ModelComparison/TaBLEau/Paper/PytorchFrame.pdf` 提出的策略，將這 10 個模型的架構統一拆分為：
- **materialize**（物化階段）
- **encoding**（編碼階段）
- **columnwise**（列間交互階段）
- **decoding**（解碼階段）

為了實作一致性，額外新增 **start** 作為 dummy stage（不對資料做任何處理，僅作為「在整個 pipeline 最前面插入 GNN」的標記點）。

### 3.2 參考基線模型（11 個，不可拆分）

這些模型作為對照組，不進行階段拆分與 GNN 插入，僅用於性能比較。

**位置**：`/home/skyler/ModelComparison/TaBLEau/models/comparison/`

**Tree-Based 模型（3 個）**：
1. **XGBoost** (xgboost.py) - 極端梯度提升
2. **CatBoost** (catboost.py) - 類別特徵優化的梯度提升
3. **LightGBM** (lightgbm.py) - 輕量級梯度提升機

**自含式（Self-contained）GNN 模型（7 個）**：
1. **TabGNN** (tabgnn.py) - 基於圖神經網路的表格學習
2. **T2G-Former** (t2gformer.py) - Table-to-Graph Transformer
3. **DGM** (dgm.py) - Differentiable Graph Module，動態圖結構學習
4. **LAN-GNN** (lan_gnn.py) - Learning Adaptive Neighborhoods（自適應鄰接學習）的表格預測改寫版
5. **IDGL-GNN** (idgl_gnn.py) - Iterative Deep Graph Learning（可學習圖構建）風格的表格預測 baseline
6. **GLCN** (glcn.py) - Graph Learning Convolutional Network（TF1；可學習稀疏圖結構）的表格預測 baseline
7. **LDS-GNN** (lds_gnn.py) - Learning Discrete Structures（圖結構作為可學習超參數）的表格預測 baseline

**Graph domain / interaction type（重要，避免混淆）**：
- **Row-graph baseline（row-level message passing / graph learning）**：TabGNN、DGM、LAN-GNN、IDGL-GNN、GLCN、LDS-GNN
  - 節點是「樣本（row）」；`columnwise` 類比的是在樣本圖上做 message passing。
- **Feature/token-graph baseline（feature interaction）**：T2G-Former
  - 節點是「欄位/feature tokens」；`columnwise` 類比的是欄位交互（feature interaction），而非樣本間傳遞。

**實驗狀態（重要）**：
- 以上 7 個 self-contained GNN baseline 已在 116 個資料集上完成兩種切分（0.05/0.15/0.80 與 0.80/0.15/0.05）的 runs。
- 匯總結果檔位於：`/home/skyler/ModelComparison/TaBLEau/summary_results/`

**重要區分**：
- 這裡的「自含式 GNN（self-contained）」指的是模型本身的主幹就依賴圖結構/訊息傳遞來完成預測（例如 TabGNN/T2G-Former/DGM/LAN-GNN）。
- 這與「GNN 插入」是完全不同的概念：
  - **GNN 插入**：在現有非 GNN 模型（如 ExcelFormer）中刻意插入 GNN 模組進行聯合訓練。
  - **自含式 GNN 模型**：模型本身就是基於 GNN 設計（如 TabGNN），用於驗證「從頭設計的 GNN 模型」vs「插入 GNN 的傳統模型」的性能差異。

**新增 self-contained GNN baselines 的研究目的（SAGE 對照語境）**：
在 SAGE 第一階段（10 個可拆分模型 × 多階段注入）之外，額外引入 DGM 與 LAN-GNN 等「自含式 GNN baseline」主要有三個目的：
1. **Few-shot 全局對照**：比較「自含式 GNN」與「10 個可拆分模型（含/不含注入）」在 few-shot 下的整體強弱，避免只在同一骨幹內做相對比較。
2. **對齊 GNN 作用位置（stage mapping）**：將 self-contained 模型內部的 GNN 訊息傳遞，對應回 PyTorch-Frame 五階段（start/materialize/encoding/columnwise/decoding），再與 SAGE 注入結果的強勢 stage 排序進行對照。
3. **檢驗『位置對齊』假說**：當某個注入 stage（例如 columnwise）在 SAGE 中最有效時，檢查「把 GNN 放在等效位置」的可拆分 SOTA 變體，是否能在公平條件下超越 self-contained GNN baseline。

（詳細的 DGM/LAN-GNN stage 對應與資料管線分析，見 [DGM_GNN_Stage_Analysis.md](DGM_GNN_Stage_Analysis.md) 與 [LAN_GNN_Architecture_Analysis.md](LAN_GNN_Architecture_Analysis.md)）

**DGM 模型特色**：
- **動態圖學習**：不依賴預定義圖結構，而是在訓練過程中自動學習最佳圖連接
- **可微分 k-NN 採樣**：使用 Gumbel softmax 進行可微分的近鄰選擇
- **階段對應**：在 PyTorch Frame 五階段中，DGM 的 GNN 相當於：
  - **encoding 階段**：embed_f 將表格特徵編碼到圖空間
  - **columnwise 階段**：DGM_d 動態構建圖 + GNN 層進行節點信息傳遞
- **技術依賴**：使用 KeOps 進行高效的大規模 k-NN 計算（實現包含降級方案）

**預訓練模型（1 個）**：
1. **TabPFN** (tabpfn.py) - 表格數據的預訓練基礎模型
   - 利用大規模預訓練知識
   - 在極少樣本下能快速適應
   - 作為強基線對照

### 3.3 模型總覽表

| 類別 | 數量 | 模型名稱 | 支援 GNN 插入 | 用途 |
|------|------|---------|--------------|------|
| PyTorch Frame | 6 | ExcelFormer, FT-Transformer, ResNet, TabNet, TabTransformer, Trompt | ✓ | 主實驗對象 |
| Custom | 4 | SCARF, SubTab, VIME, TabM | ✓ | 主實驗對象 |
| Tree-Based | 3 | XGBoost, CatBoost, LightGBM | ✗ | 基線對照 |
| GNN-Based | 7 | TabGNN, T2G-Former, DGM, LAN-GNN, IDGL-GNN, GLCN, LDS-GNN | ✗ | 基線對照 |
| Pretrained | 1 | TabPFN | ✗ | 基線對照 |
| **總計** | **21** | - | 10 可拆分 / 11 不可拆分 | - |

---

## 4. 五階段統一流水線詳解

### 4.1 階段設計理念
為了在不同架構的模型間進行公平的 GNN 插入比較，我們參考 PyTorch Frame 論文（`/home/skyler/ModelComparison/TaBLEau/Paper/PytorchFrame.pdf`）的設計理念，將表格深度學習模型抽象為統一的五階段流水線。

### 4.2 五階段詳細說明

#### Stage 0: **start**（起始點，Dummy Stage）
- **功能**：不對資料做任何處理
- **存在目的**：作為「在整個模型 pipeline 最前面插入 GNN」的標記點
- **實作意義**：
  - 當 GNN 插在 start 階段後 = 插在 materialize 階段前
  - 允許在資料進入模型主幹之前進行圖結構增強
- **資料流**：原始 DataFrame → (可選 GNN) → 進入 materialize

#### Stage 1: **materialize**（資料物化階段）
- **功能**：
  - 將原始 DataFrame 轉換為 TensorFrame 或 DataLoader
  - 處理類別型特徵（one-hot encoding, embedding 等）
  - 進行互信息排序（mutual information ranking）
  - 資料標準化與正規化
- **輸出**：結構化的 tensor 表示，準備進入神經網路
- **GNN 插入時機**：在物化後、編碼前

#### Stage 2: **encoding**（編碼階段）
- **功能**：
  - 將每個欄位編碼為 token 向量
  - 添加位置編碼（column embeddings）
  - 生成初始表徵
- **輸出**：tokens `x: [Batch, Features, Channels]`
- **GNN 插入時機**：在 token 生成後、列間交互前
- **典型實作**：
  ```python
  x = encoder(tensor_frame)  # [B, F, C]
  x = x + column_embed       # 加入欄位位置編碼
  # 可在此插入 GNN
  ```

#### Stage 3: **columnwise**（列間交互階段）
- **功能**：
  - 在欄位（features）維度進行交互學習
  - 通常使用多頭注意力機制（multi-head attention）
  - 捕捉不同欄位之間的關係
- **輸出**：經過交互後的 tokens，維持 `[B, F, C]` 形狀
- **GNN 插入時機**：在列間交互完成後、解碼前
- **典型實作**：
  ```python
  for conv_layer in convs:
      x = conv_layer(x)  # 多層注意力
  # 可在此插入 GNN
  ```

#### Stage 4: **decoding**（解碼階段）
- **功能**：
  - 將特徵表徵解碼為最終預測
  - 通常包含池化、全連接層
  - 輸出 logits 或預測值
- **輸出**：`out: [Batch, output_channels]`
- **GNN 插入時機**：用 GNN 完全取代原始 decoder
- **典型實作**：
  ```python
  out = decoder(x)  # [B, out_channels]
  # 或用 GNN 取代整個 decoder
  ```

### 4.3 階段流程圖
```
原始 DataFrame
    ↓
[start] ← GNN 插入點 0（在最前面）
    ↓
[materialize] 物化為 TensorFrame
    ↓ ← GNN 插入點 1（物化後）
[encoding] 生成 tokens [B,F,C]
    ↓ ← GNN 插入點 2（編碼後）
[columnwise] 列間交互 [B,F,C]
    ↓ ← GNN 插入點 3（交互後）
[decoding] 輸出預測 [B,O]
    ↓ ← GNN 插入點 4（取代 decoder）
最終預測結果
```

### 4.4 不同階段的 GNN 作用機制
- **start/materialize**：離線式特徵增強（資料預處理）
- **encoding**：token-level 的圖結構學習（聯合訓練）
- **columnwise**：高階特徵的圖交互（聯合訓練）
- **decoding**：圖表徵直接用於預測（端到端）

---

## 5. GNN 插入策略完整說明

### 5.1 GNN 架構設計（gnn_injection.py）

**核心組件**：
1. **DGM_d（動態圖生成模組）**
   - 功能：根據輸入特徵動態構建圖結構
   - 輸入：節點特徵 `[N, D]` 或 `[B, D]`
   - 輸出：圖結構 `(node_features, edge_index)`
   - 構圖策略：k-近鄰（k-NN）或完全連接

2. **SimpleGCN（圖卷積網路）**
   - 功能：在構建的圖上進行消息傳遞
   - 架構：多層 GCNConv 堆疊
   - 輸入：`[N, Cin]`
   - 輸出：`[N, Cout]`
   - 可配置：層數、隱藏維度、dropout

3. **殘差融合門（Residual Fusion Gate）**
   - 功能：將 GNN 輸出與原始特徵融合
   - 機制：`output = original + sigmoid(alpha) * gnn_output`
   - 可學習參數：fusion_alpha_param
   - 例外：decoding 階段不使用殘差（直接取代）

4. **輔助模組**
   - `column_embed`：欄位位置編碼 `[F, C]`
   - `pool_query`：注意力池化查詢向量 `[C]`
   - `input_proj/output_proj`：維度投影層
   - `attn_in/attn_out`：自注意力編解碼

### 5.1.1 注入風格統一與關鍵觀察（論文敘事用）

在 TaBLEau 的 10 個可拆分模型中，歷史上曾同時存在兩種主要的 GNN injection 實作風格（詳見 [SAGE_10Models_GNN_Stage_Rationale.md](SAGE_10Models_GNN_Stage_Rationale.md)）：

- **Dynamic-Graph + Attention（DGM + Self-Attn）風格**：先在 token/欄位表徵上做自注意力與 pooling 產生 row-level 表徵，再於 row-level 透過 **DGM 動態建圖**學習圖結構，並以 GCN 更新，最後將圖訊息透過 attention decode + 殘差門控回寫到 token 表徵。這個風格的關鍵不在於「屬於哪個 backbone」，而在於 **動態建圖（DGM）+ 注意力式的 pooling/回寫**所形成的端到端管線。
- **Static kNN Graph（Simple kNN+GCN）風格**：更偏向以 **靜態 kNN** 在某個表示空間上構圖，接著用 GCN 做特徵轉換（離線或 batch-level mixing），或把 GNN 當作最後 head；通常缺少「回寫 token」這種與 backbone 內部 token 幾何強耦合、且以注意力機制調整維度/融合的設計。

本研究的最終目標是將 10 個模型的 GNN injection 統一為 **Dynamic-Graph + Attention（DGM + Self-Attn）**風格，以降低「實作差異」對實驗解釋的干擾，使觀察更聚焦於「插入位置（stage）× 模型本質」本身。

另外，為了在論文中更清楚地討論「建圖策略」本身的影響（而不是混入其他工程差異），我們也保留了兩種風格各自對應的 git branch，使得後續可以在控制其他條件近似不變的前提下，比較 **DGM 動態建圖** vs **靜態 kNN 建圖**是否會帶來可重現的差異與額外發現。

同時，我們也強調一個在實驗中會明確呈現的觀察：**將 GNN 插入既有 SOTA 表格模型並不保證帶來增益**。其成敗高度依賴模型本身的表徵學習目標與幾何特性；例如：

- **對比式自監督（contrastive）模型（如 SCARF 類）**通常需要強烈的 instance discrimination（讓不同樣本可分），而 GNN 的鄰域聚合傾向把相鄰節點表徵拉近，可能與對比目標產生張力，導致性能下降或不穩定。
- **重建/去噪（reconstruction/denoising）導向的目標**在某些情況下會受益於局部平滑，但也可能因過度混合而抹除細節訊號，使得重建誤差型的學習目標變得更困難。

因此，SAGE 的分析不僅比較「是否插入 GNN」，也會特別關注：當模型的 pretext/learning objective 與 message passing 的 inductive bias 不相容時，GNN injection 可能反而帶來負效應。

### 5.2 六種 GNN 插入策略詳解

#### 策略 0: **none**（無插入，作為基線）
```
Pipeline: materialize → encoding → columnwise → decoding
```
- **目的**：提供不含 GNN 的性能基線
- **用途**：計算 GNN 帶來的性能增益（gain）

#### 策略 1: **start**（離線特徵預注入）
```python
# 在原始 DataFrame 上操作
df_original → gnn_after_start_fn(df) → df_enhanced
  ↓
[Pipeline] materialize → encoding → columnwise → decoding
```
- **執行時機**：在資料進入模型主幹之前
- **處理流程**：
  1. `input_proj`: `[N, F, 1] → [N, F, D]`（維度擴展）
  2. `attn_in`（自注意力）: `[N, F, D] → [N, F, D]`
  3. 注意力池化: `[N, F, D] → [N, D]`（生成 row-level 向量）
  4. `DGM_d`: 構建樣本間的動態圖 `[1, N, D] → ([N, D], edge_index)`
  5. `GCN`: `[N, D] → [N, G]`（圖卷積）
  6. `pred_head`: `[N, G] → [N, out_dim]`（監督訓練的輔助頭）
  7. 重建欄位尺度: 
     - `gcn_to_attn`: `[N, G] → [N, 1, D]`
     - `attn_out`: `[N, F, D] → [N, F, D]`
     - `out_proj`: `[N, F, D] → [N, F]`
  8. 產出新 DataFrame，保留 F 個欄位
- **特點**：
  - 離線式處理，不改變模型主幹
  - 可以視為一種圖增強的特徵工程
  - 需要監督信號來訓練 GNN

#### 策略 2: **materialize**（物化後離線注入）
```python
# 在 TensorFrame 上操作
df → materialize_fn → TensorFrame → gnn_after_materialize_fn → TensorFrame_enhanced
  ↓
[Pipeline] encoding → columnwise → decoding
```
- **執行時機**：在資料物化後、編碼前
- **處理流程**：與 start 類似，但在 TensorFrame 轉回 DataFrame 處理
- **差異**：
  - 資料已經過類別編碼、標準化等預處理
  - 可以利用物化階段的資訊排序結果
- **特點**：
  - 離線式處理
  - 在更結構化的資料上構圖

#### 策略 3: **encoding**（聯訓：編碼後、卷積前注入）
```python
# 在 tokens 上操作
x = encoder(data)               # [B, F, C]
x = x + column_embed            # 加入位置編碼
x = self_attn(x)                # [B, F, C]
x_pooled = attention_pool(x)   # [B, C] (row-level)
edge_index = DGM_d(x_pooled)    # 動態建圖
x_gnn = GCN(x_pooled, edge_index)  # [B, C]
x_decoded = self_attn_out(x_gnn)   # [B, F, C]
x = x + sigmoid(alpha) * x_decoded # 殘差融合
  ↓
[columnwise] → [decoding]
```
- **執行時機**：在 encoding 後、columnwise 前
- **處理流程**：
  1. 對 tokens 進行自注意力: `[B, F, C] → [B, F, C]`
  2. 注意力池化獲得 batch 內的 row-level 表徵: `[B, F, C] → [B, C]`
  3. 以 batch 內樣本為節點構建動態圖
  4. GCN 在圖上傳播: `[B, C] → [B, C]`
  5. 通過注意力解碼回 token 空間: `[B, C] → [B, F, C]`
  6. 殘差融合到原始 tokens
- **特點**：
  - 聯合訓練（end-to-end）
  - 在 token-level 引入圖結構
  - batch 內樣本間的交互

#### 策略 4: **columnwise**（聯訓：卷積後注入）
```python
# 在 columnwise 交互後操作
x = encoder(data)           # [B, F, C]
for conv in convs:
    x = conv(x)             # 多層列間交互 [B, F, C]
# 以下與 encoding 相同
x_pooled = attention_pool(x)
edge_index = DGM_d(x_pooled)
x_gnn = GCN(x_pooled, edge_index)
x_decoded = self_attn_out(x_gnn)
x = x + sigmoid(alpha) * x_decoded
  ↓
[decoding]
```
- **執行時機**：在 columnwise 完成後、decoding 前
- **處理流程**：與 encoding 相同，但作用於更高階的特徵
- **特點**：
  - 聯合訓練
  - 在列間交互後的高階特徵上構圖
  - **實驗發現最常帶來性能提升的階段**

#### 策略 5: **decoding**（聯訓：以 GNN 取代解碼器）
```python
# 完全取代 decoder
x = encoder(data)           # [B, F, C]
for conv in convs:
    x = conv(x)             # [B, F, C]
x = x + column_embed
x = self_attn(x)
x_pooled = attention_pool(x)     # [B, C]
edge_index = DGM_d(x_pooled)
out = GCN(x_pooled, edge_index)  # [B, out_channels]
# 不經過原始 decoder，直接輸出
```
- **執行時機**：取代整個 decoding 階段
- **處理流程**：
  1. 在 columnwise 後進行自注意力與池化
  2. 構建動態圖
  3. GCN 直接輸出預測 logits
  4. **完全繞過原始 decoder**
- **特點**：
  - 端到端圖預測
  - 用圖上的 row-level 表徵直接做預測
  - 架構改動最大

### 5.3 GNN 配置參數
```python
--gnn_hidden_dim 256      # GNN 隱藏層維度
--gnn_layers 2            # GNN 層數
--gnn_dropout 0.2         # GNN Dropout 比率
```

### 5.4 階段選擇指引
- **快速驗證**：只跑 `none` + `columnwise`
- **完整分析**：跑所有 6 個階段 `all`
- **特定假設**：根據研究問題選擇特定階段組合

## 6. 執行入口與主要參數（main.py）
- 命令範例：
  - 指定單資料集、單模型、全階段：
    - python main.py --dataset eye --models tabnet --gnn_stages all --epochs 300
  - 按類別批量：
    - python main.py --dataset_size small_datasets --task_type binclass --feature_type numerical \
      --models excelformer resnet tabnet --gnn_stages none columnwise encoding \
      --train_ratio 0.05 --val_ratio 0.15 --epochs 200 --few_shot
- 主要參數：
  - 資料：--dataset 或 --dataset_size/--task_type/--feature_type，--data_dir
  - 模型：--models（可多個）
  - 階段：--gnn_stages（可多個，none/start/materialize/encoding/columnwise/decoding/all）
  - 切分：--train_ratio --val_ratio --test_ratio（或用 few_shot/few_shot_ratio）
  - 訓練：--epochs --batch_size --lr --weight_decay --patience --seed
  - GNN：--gnn_hidden_dim --gnn_layers --gnn_dropout
  - 其他：--few_shot --debug_metrics
- 流程：解析參數 → DatasetLoader 準備資料 → ModelRunner 動態載入 → GNNInjector 依 stage 掛鉤 → 訓練/評估 → 輸出 log/CSV/可視化。

---

## 7. 第一階段實驗：全面比較分析

### 7.1 實驗設計

**實驗規模**：
- **模型**：目前為 21 個（10 可拆分 + 11 基線，含 TabPFN；self-contained GNN baselines 共 7 個）
- **資料集**：116 個（涵蓋所有大小/任務/特徵類型組合）
- **切分策略**：2 種（few-shot 0.05/0.15/0.80 和 full 0.80/0.15/0.05）
- **GNN 階段**：6 個（none, start, materialize, encoding, columnwise, decoding）
- **總實驗次數**：僅針對「10 個可拆分模型 × 116 datasets × 2 splits × 6 stages」的注入變體為 13,920 次（基線模型不做 stage 注入）。

**實驗流程**：
```bash
# 執行所有模型、所有資料集、所有階段
python main.py --dataset_size all --task_type all --feature_type all \
               --models all --gnn_stages all --epochs 300
```

**輸出位置**：
- 原始結果：`/home/skyler/ModelComparison/TaBLEau/summary_results/`
- 分析報告：`/home/skyler/ModelComparison/TaBLEau/gnn_injection_analysis/per_model_result/`

### 7.2 比較基線設定

對於每個可拆分模型的 GNN 插入變體，我們與以下基線進行比較：

**基線類別 1：自身基線**
- **Few-shot non-GNN**：同模型、few-shot 切分、無 GNN（gnn_stage=none）
  - 目的：評估 GNN 在少樣本情境下的純增益
  - 比較規則：**嚴格比較**（必須 strictly better 才算 beat）
- **Full-sample non-GNN**：同模型、full 切分、無 GNN（gnn_stage=none）
  - 目的：評估 GNN 能否彌補訓練樣本不足
  - 比較規則：容差 1e-3（平手視為 tie）

**基線類別 2：樹模型基線**
- **Few-shot tree models**：XGBoost, CatBoost, LightGBM（few-shot 切分）
- **Full-sample tree models**：XGBoost, CatBoost, LightGBM（full 切分）
  - 目的：對比傳統強基線在不同資料量下的表現
  - 特點：樹模型在中小型表格數據上往往很強

**基線類別 3：自含式（Self-contained）GNN 基線**
- **Few-shot GNN models**：TabGNN, T2G-Former, DGM, LAN-GNN, IDGL-GNN, GLCN, LDS-GNN（few-shot 切分）
- **Full-sample GNN models**：TabGNN, T2G-Former, DGM, LAN-GNN, IDGL-GNN, GLCN, LDS-GNN（full 切分）
  - 目的：對比「原生設計的 GNN」vs「插入 GNN 的傳統模型」
  - 區別：
    - 原生 GNN：整個架構都圍繞圖結構設計
    - GNN 插入：在現有架構中加入圖模組
  - 特別提醒（可比性聲明）：
    - TabGNN/DGM/LAN-GNN/IDGL-GNN/GLCN/LDS-GNN 屬於 **row-graph**，核心是 **row-level message passing**（transductive whole-graph 常見）。
    - T2G-Former 屬於 **feature/token-graph**，核心是 **feature interaction**（更接近 Transformer 的 columnwise interaction）。

**基線類別 4：預訓練基線**
- **Few-shot TabPFN**：TabPFN（few-shot 切分）
- **Full-sample TabPFN**：TabPFN（full 切分）
  - 目的：對比大規模預訓練帶來的少樣本優勢
  - 特點：TabPFN 在極少樣本下通常表現優異

### 7.3 評估指標

**主要指標**：
- **性能值**：根據任務類型選擇
  - 分類：AUC（二分類）、Accuracy（多分類）
  - 迴歸：MAE（平均絕對誤差）
- **平均排名（avg_rank）**：在同一資料集類別下的排名平均
  - 數值越小表示性能越好
  - 用於跨資料集的公平比較

**比較規則**：
- **容差**：1e-3（0.001）
  - 若兩個模型的 avg_rank 差距 < 1e-3，視為平手（tie）
- **嚴格比較**（僅用於 few-shot self-baseline）：
  - 必須 strictly lower（avg_rank 更小）才算 beat
  - 平手時標記為 "No (tie, few-shot strict)"
- **一般比較**（用於其他基線）：
  - 容差內視為平手，標記為 "Yes (tie)" 或 "No (tie)"

### 7.4 分析維度

**按資料集類別分組**：
```
1. large_datasets + binclass + numerical
2. large_datasets + binclass + categorical
3. large_datasets + binclass + balanced
4. large_datasets + multiclass + numerical
5. large_datasets + multiclass + categorical
6. large_datasets + multiclass + balanced
7. large_datasets + regression + numerical
8. large_datasets + regression + categorical
9. large_datasets + regression + balanced
10. small_datasets + binclass + numerical
11. small_datasets + binclass + categorical
12. small_datasets + binclass + balanced
... (以此類推，共 18 個組合)
```

**每個類別下的分析內容**：
- 各 GNN 階段的平均排名
- 對各基線類別的擊敗/平手/輸掉統計
- 例如："beats few-shot tree (2/3)" 表示在 3 個樹模型中擊敗 2 個

### 7.5 結果文件結構

**Per-Model 詳細報告**（以 ExcelFormer 為例）：
```
gnn_injection_analysis/per_model_result/
├── excelformer_gnn_enhancement.md           # 完整分析
├── excelformer_gnn_enhancement.txt          # 純文字版本
├── excelformer_gnn_enhancement_summary.md   # 總結報告
├── excelformer_per_dataset_metrics_test.csv # 測試集逐資料集指標
├── excelformer_per_dataset_metrics_val.csv  # 驗證集逐資料集指標
├── excelformer_sensitivity_test.csv        # 敏感度分析（測試集）
├── excelformer_sensitivity_val.csv         # 敏感度分析（驗證集）
├── excelformer_sensitivity_aggregate_category_test.md  # 按類別聚合
└── excelformer_sensitivity_split_category_test.md      # 按切分與類別聚合
```

**Summary 文件格式示例**：
```markdown
## Category: small_datasets+binclass+numerical (25 datasets)

| Injection | avg_rank | beats few-shot-non-gnn? | beats full-non-gnn? | 
|-----------|----------|-------------------------|---------------------|
| columnwise| 8.20     | Yes                     | No                  |
| encoding  | 8.45     | Yes                     | No                  |
| none      | 8.80     | -                       | No                  |
| decoding  | 9.10     | No (tie)                | No                  |
| start     | 10.50    | No                      | No                  |
| materialize| 10.80   | No                      | No                  |

(續表包含對樹模型、GNN、TabPFN 的比較統計)
```

### 7.6 主要發現（高層結論）

**有利於 GNN 的情境**：
- ✅ **小型資料集**（small_datasets）
- ✅ **少樣本設定**（few-shot 0.05）
- ✅ **數值主導型**（numerical feature 占比高）
- ✅ **二分類任務**（binclass）
- ✅ **Columnwise 階段**（最穩定的增益）
- ✅ **Encoding/Decoding 階段**（次優選擇）

**GNN 增益減弱的情境**：
- ❌ 大型資料集（large_datasets）
- ❌ 充足樣本（full 0.80）
- ❌ 類別主導型（categorical feature 占比高）
- ❌ 遇到強基線（樹模型 @ full、TabPFN）

**不同階段的效果排序**（根據 few-shot 平均表現）：
```
1. columnwise    ⭐⭐⭐⭐⭐ (最常帶來增益)
2. encoding      ⭐⭐⭐⭐
3. decoding      ⭐⭐⭐⭐
4. none (baseline) ⭐⭐⭐
5. materialize   ⭐⭐
6. start         ⭐
```

**跨模型差異**：
- FT-Transformer、ExcelFormer：對 GNN 較敏感，增益明顯
- TabNet、SCARF：中等敏感度
- ResNet、VIME：敏感度較低，增益不穩定

### 7.7 查閱方式

**快速查看某模型的 GNN 效果**：
```bash
# 查看 ExcelFormer 的總結報告
cat gnn_injection_analysis/per_model_result/excelformer_gnn_enhancement_summary.md

# 查看 ResNet 在特定資料集上的表現
grep "kaggle_Audit_Data" gnn_injection_analysis/per_model_result/resnet_per_dataset_metrics_test.csv
```

**對比所有模型的排名**：
```bash
cat gnn_injection_analysis/per_model_result/all_models_ranking_all_datasets_by_test.md
```

---

## 8. 第二階段實驗：三項消融研究計畫（Ablation Studies - 規劃階段）

> **重要說明**：本章節描述的是第二階段實驗的詳細計畫，目前**尚未開始實作**。相關腳本、資料處理與分析工作都還在規劃階段，需要後續 AI 代理協助完成。

基於第一階段的實驗觀察，我們發現 GNN injection 在特定情境下（small dataset、binclass、few-shot、numerical）相對於同模型的 few-shot non-GNN 基線有所超越，特別是在 columnwise、encoding、decoding 等階段。為了驗證這些觀察背後的因果關係，我們設計了三項系統性的消融實驗。

### 8.1 通用實驗設定

**共同配置**：
- **模型**：10 個可拆分模型
- **GNN 階段**：6 個（none, start, materialize, encoding, columnwise, decoding）
- **Epochs**：300
- **隨機種子**：原定 20 個（42-61），可降為 5 個以加速
- **並行度**：建議 80-120（依 GPU 記憶體調整）
- **評估指標**：依任務選 AUC/Accuracy/MAE

**輸出要求**：
- 每個模型生成 1 張大圖，包含 2 張子圖：
  1. **性能圖**：橫軸為自變量（train_ratio/numerical_share/dataset_size），縱軸為性能值
     - 6 條折線：none + 5 種 GNN 插入變體
  2. **增益圖**：橫軸同上，縱軸為相對於 none 的 performance gain
     - 5 條折線：5 種 GNN 插入變體相對於 none 的增益
     - Gain 定義：`(metric_stage - metric_none) / metric_none * 100%`

**統計要求**：
- 每個數據點是多個資料集 × 多個種子的聚合
- 必須計算並報告：
  - **均值（Mean）**
  - **方差（Variance）**
  - **標準差（Standard Deviation）**
- 可視化時在折線圖上繪製誤差棒（error bars）或陰影區域（confidence interval）

---

### 8.2 消融實驗一：訓練樣本量對 GNN 增益的影響

**研究假設**：
> GNN injection 在訓練樣本量較少時帶來顯著的性能優勢，但隨著訓練樣本量增加，這個優勢逐漸減弱甚至消失。

#### 8.2.1 實驗設計

**資料集選擇**：
- 數量：20 個
- 條件：small_datasets + binclass
- 優先：row count 越多越好（以確保在不同 train_ratio 下都有足夠樣本）

**訓練比例掃描**（Train Ratio Grid）：
- **訓練比例（train_ratio）**：16 個點
  ```python
  [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 
   0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
  ```
- **驗證比例（val_ratio）**：固定 0.15
- **測試比例（test_ratio）**：自動計算 `1 - train_ratio - 0.15`

**隨機種子**：
- 完整版：20 個（42-61）
- 精簡版：5 個（42-46）以加速實驗

#### 8.2.2 預期結果

**性能曲線特徵**：
- None 基線：隨 train_ratio 增加單調上升（更多數據 → 更好性能）
- GNN 變體：在低 train_ratio 時超越 none，高 train_ratio 時趨近或低於 none

**增益曲線特徵**：
- X 軸：train_ratio（0.05 → 0.8）
- Y 軸：gain 相對於 none（%）
- 預期趨勢：
  ```
  Gain
   ^
  5%|     columnwise ╲
    |                  ╲___
  3%|  encoding          ╲___
    |            ╲___________╲___
  1%|                             ╲___
    |                                  ╲___
  0%|________________________________________╲___
    |                                             ╲___
 -1%|________________________________________________╲
    +-------------------------------------------------->
     0.05  0.1   0.2   0.3   0.4   0.5   0.6   0.7  0.8
                         train_ratio
  ```
- 解讀：
  - 在 train_ratio=0.05 時，columnwise 可能有 ~3-5% 的增益
  - 隨著 train_ratio 增加，增益逐漸減少
  - 在 train_ratio=0.8 時，增益接近 0% 甚至略為負值

#### 8.2.3 統計聚合

對於每個 (model, gnn_stage, train_ratio) 組合：
```
- 跑 20 個資料集
- 每個資料集跑 N 個種子（20 或 5）
- 總計 20 × N 個數值

聚合統計：
  mean = Σ(all_values) / (20 × N)
  variance = Σ(value - mean)² / (20 × N)
  std = √variance
```

#### 8.2.4 執行命令示例

```bash
# 對單一資料集、單一模型、單一 train_ratio、單一種子
python main.py --dataset <dataset_name> --models excelformer \
               --gnn_stages all --train_ratio 0.05 --val_ratio 0.15 \
               --epochs 300 --seed 42

# 需要包裝腳本批量執行：
# - 遍歷 20 個資料集
# - 遍歷 16 個 train_ratio
# - 遍歷 10 個模型
# - 遍歷 6 個 gnn_stages
# - 遍歷 N 個 seeds
# 總計：20 × 16 × 10 × 6 × N 次實驗
```

---

### 8.3 消融實驗二：數值特徵占比對 GNN 增益的影響

**研究假設**：
> GNN injection 在數值特徵占比較高時帶來顯著增益，但隨著數值特徵占比下降（類別特徵增多），增益逐漸減弱。

#### 8.3.1 實驗設計

**資料集選擇**：
- 數量：20 個
- 條件：small_datasets + binclass
- 優先：
  - feature column 數量越多越好
  - 同時包含 numerical 與 categorical 欄位
  - 兩種類型盡量平均（balanced）

**數值占比調整策略**：
由於資料集本身的特徵類型是固定的，有兩種方式調整占比：

**方法 A：動態特徵選擇**（推薦）
1. 計算原始資料集的 numerical 與 categorical 欄位數量
2. 設定多個占比目標（例如：100%, 80%, 60%, 40%, 20%, 0%）
3. 對每個目標占比：
   - 保留對應比例的 numerical 欄位
   - 其餘用 categorical 欄位填充
   - 確保總欄位數不變或接近原始數量

**方法 B：挑選多樣資料集**（備選）
1. 選擇 20 個資料集，確保它們的 numerical 占比涵蓋完整範圍
2. 根據實際占比分組分析

**占比掃描點**：
```python
numerical_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
# 或精簡為：[1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
```

**固定設定**：
- train_ratio = 0.05（few-shot）
- val_ratio = 0.15
- test_ratio = 0.80

**隨機種子**：
- 完整版：20 個（42-61）
- 精簡版：5 個（42-46）

#### 8.3.2 預期結果

**增益曲線特徵**：
```
  Gain
   ^
  5%|  columnwise ╲
    |               ╲
  3%|  encoding      ╲___
    |                    ╲___
  1%|  decoding              ╲___
    |                            ╲___
  0%|________________________________╲___
    |                                    ╲___
 -1%|________________________________________╲
    +--------------------------------------------->
    100%  80%  60%  40%  20%   0%
              Numerical Feature Ratio
```
- 解讀：
  - 在 numerical_ratio=100% 時，GNN 增益最大
  - 隨著 numerical_ratio 下降，增益遞減
  - 在 numerical_ratio=0%（純類別型）時，GNN 增益接近 0 或負值

**原因推測**：
- GNN 建圖時依賴特徵相似度（通常用歐氏距離或餘弦相似度）
- 數值特徵提供連續的相似性信號，有利於構建有意義的圖結構
- 類別特徵（尤其經 one-hot 後）的相似性信號較弱

#### 8.3.3 統計聚合

對於每個 (model, gnn_stage, numerical_ratio) 組合：
```
- 跑 20 個資料集
- 每個資料集跑 N 個種子
- 總計 20 × N 個數值
- 計算 mean, variance, std
```

#### 8.3.4 實作細節

**特徵選擇邏輯**（偽代碼）：
```python
def adjust_numerical_ratio(df, target_ratio):
    """
    調整資料集的數值特徵占比
    
    Args:
        df: 原始 DataFrame
        target_ratio: 目標數值占比 (0-1)
    
    Returns:
        調整後的 DataFrame
    """
    numerical_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64']]
    categorical_cols = [c for c in df.columns if df[c].dtype == 'object']
    
    total_features = len(numerical_cols) + len(categorical_cols)
    target_num_count = int(total_features * target_ratio)
    target_cat_count = total_features - target_num_count
    
    # 根據特徵重要性或隨機選擇
    selected_num = numerical_cols[:target_num_count]
    selected_cat = categorical_cols[:target_cat_count]
    
    return df[selected_num + selected_cat + ['target']]
```

---

### 8.4 消融實驗三：資料集大小對 GNN 增益的影響

**研究假設**：
> GNN injection 在資料集較小時帶來顯著增益，但隨著資料集變大，增益逐漸減弱。

#### 8.4.1 實驗設計

**資料集選擇**：
- 數量：20 個
- 條件：**large_datasets** + binclass
- 優先：row count 越多越好（以確保下采樣後仍有足夠樣本）

**資料集下采樣策略**：
1. 從原始 large dataset 中按比例抽取樣本
2. 抽樣比例：10%, 20%, 30%, ..., 90%, 100%（共 10 個點）
3. 對每個抽樣子集，再套用 few-shot 切分（0.05/0.15/0.80）

**重要說明**：
> 無論抽取多少比例的資料，都要在子集上套用 few-shot 5% 的設定。

**範例**：
```
原始 large dataset: 10,000 rows

抽樣 10% → 1,000 rows
  ├─ Train (5%): 50 rows
  ├─ Val (15%): 150 rows
  └─ Test (80%): 800 rows

抽樣 50% → 5,000 rows
  ├─ Train (5%): 250 rows
  ├─ Val (15%): 750 rows
  └─ Test (80%): 4,000 rows

抽樣 100% → 10,000 rows
  ├─ Train (5%): 500 rows
  ├─ Val (15%): 1,500 rows
  └─ Test (80%): 8,000 rows
```

**抽樣比例點**：
```python
sampling_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```

**固定設定**：
- train_ratio = 0.05（few-shot，但是相對於子集）
- val_ratio = 0.15
- test_ratio = 0.80

**隨機種子**：
- 完整版：20 個（42-61）
  - 種子 1 用於下采樣
  - 種子 2 用於 train/val/test 切分
- 精簡版：5 個

#### 8.4.2 預期結果

**增益曲線特徵**：
```
  Gain
   ^
  8%|  columnwise ╲
    |               ╲
  5%|  encoding      ╲___
    |                    ╲___
  3%|  decoding              ╲___
    |                            ╲___
  1%|________________________________╲___
    |                                    ╲___
  0%|________________________________________╲___
    +----------------------------------------------->
    10%  20%  30%  40%  50%  60%  70%  80%  90% 100%
                   Dataset Sampling Ratio
```
- 解讀：
  - 在 sampling_ratio=10% 時（模擬最小資料集），GNN 增益最大
  - 隨著 sampling_ratio 增加，資料集變大，增益遞減
  - 在 sampling_ratio=100% 時，增益接近 0 或略為負值

**原因推測**：
- 小資料集：樣本間的圖結構能提供額外的歸納偏差（inductive bias）
- 大資料集：模型已有足夠樣本學習，GNN 的結構信息變得冗餘

#### 8.4.3 統計聚合

對於每個 (model, gnn_stage, sampling_ratio) 組合：
```
- 跑 20 個 large 資料集
- 每個資料集跑 N 個種子
- 總計 20 × N 個數值
- 計算 mean, variance, std
```

#### 8.4.4 實作細節

**下采樣邏輯**（偽代碼）：
```python
def subsample_dataset(df, sampling_ratio, seed):
    """
    對資料集進行下采樣
    
    Args:
        df: 原始 DataFrame
        sampling_ratio: 抽樣比例 (0-1)
        seed: 隨機種子
    
    Returns:
        抽樣後的 DataFrame
    """
    n_samples = int(len(df) * sampling_ratio)
    return df.sample(n=n_samples, random_state=seed)

# 使用範例
for sampling_ratio in [0.1, 0.2, ..., 1.0]:
    for seed in range(42, 47):  # 5 seeds
        # 下采樣
        df_sub = subsample_dataset(df_original, sampling_ratio, seed)
        
        # 再切分為 train/val/test (0.05/0.15/0.80)
        train, val, test = split_data(df_sub, 
                                       train_ratio=0.05, 
                                       val_ratio=0.15, 
                                       seed=seed)
        
        # 訓練與評估
        results = train_and_evaluate(model, train, val, test)
```

---

### 8.5 三項消融的對比總結

| 消融實驗 | 自變量 | 因變量 | 資料集類型 | 資料集數量 | 固定參數 | 掃描點數 |
|---------|--------|--------|-----------|-----------|---------|---------|
| **實驗一** | train_ratio | GNN gain | small + binclass | 20 | val=0.15 | 16 |
| **實驗二** | numerical_ratio | GNN gain | small + binclass (mixed) | 20 | train=0.05, val=0.15 | 6-11 |
| **實驗三** | sampling_ratio | GNN gain | large + binclass | 20 | train=0.05 (of subset), val=0.15 | 10 |

**共同點**：
- 都關注 GNN 在 few-shot 情境下的增益變化
- 都需要繪製性能圖與增益圖
- 都需要統計均值、方差、標準差

**差異點**：
- 實驗一：控制樣本量（訓練比例）
- 實驗二：控制特徵類型（數值 vs 類別）
- 實驗三：控制資料集規模（total rows）

---

### 8.6 時間與資源估算

**單次實驗耗時**（根據第一階段經驗）：
- 平均每個 (model, dataset, stage, seed) 組合：~3 秒

**實驗一總耗時**：
```
10 models × 20 datasets × 16 ratios × 6 stages × N seeds × 3 sec
= 57,600 × N seconds

N=20: 320 小時（並行 80 → 4 小時）
N=5:  80 小時（並行 80 → 1 小時）
```

**實驗二總耗時**：
```
10 models × 20 datasets × 11 ratios × 6 stages × N seeds × 3 sec
= 39,600 × N seconds

N=20: 220 小時（並行 80 → 2.75 小時）
N=5:  55 小時（並行 80 → 40 分鐘）
```

**實驗三總耗時**：
```
10 models × 20 datasets × 10 sampling_ratios × 6 stages × N seeds × 3 sec
= 36,000 × N seconds

N=20: 200 小時（並行 80 → 2.5 小時）
N=5:  50 小時（並行 80 → 37 分鐘）
```

**建議策略**：
- **快速驗證**：seeds=5，並行=100 → 每個實驗 <1 小時
- **完整發表**：seeds=20，並行=80 → 每個實驗 2-4 小時
- **分階段執行**：先跑 none + columnwise，確認流程後再跑全階段

---

### 8.7 消融實驗的實作規劃

**現有文件**：
- **位置**：`/home/skyler/ModelComparison/TaBLEau/ablation_study/`
- **主要文件**：`ABLATION_STUDY_PLAN.md` - 完整計畫文件（已存在）

**需要實作的腳本**（目前尚未建立）：
```bash
# 實驗一：train_ratio 掃描
python ablation_study/run_train_ratio_ablation.py \
    --models all --datasets 20 --seeds 5 --parallel 80

# 實驗二：numerical_ratio 掃描
python ablation_study/run_numerical_ratio_ablation.py \
    --models all --datasets 20 --seeds 5 --parallel 80

# 實驗三：sampling_ratio 掃描
python ablation_study/run_sampling_ratio_ablation.py \
    --models all --datasets 20 --seeds 5 --parallel 80
```

**規劃的輸出結構**：
```
ablation_study/
├── ABLATION_STUDY_PLAN.md          # 已存在
├── train_ratio_results/            # 待建立
│   ├── raw_data/                   # 原始 CSV
│   ├── aggregated_stats/           # 聚合統計
│   └── plots/                      # 性能與增益圖
│       ├── excelformer_ablation.png
│       ├── fttransformer_ablation.png
│       └── ...
├── numerical_ratio_results/        # 待建立
│   └── ...
└── sampling_ratio_results/         # 待建立
    └── ...
```

**待完成工作**：
1. 撰寫三個主執行腳本（run_*_ablation.py）
2. 實作資料聚合腳本（aggregate_results.py）
3. 實作繪圖腳本（plot_ablation_results.py）
4. 建立資料集選擇邏輯（dataset_selector.py）
5. 實作並行執行管理器（parallel_executor.py）
---

## 9. 執行入口與命令詳解

### 9.1 main.py 主程式

**基本語法**：
```bash
python main.py [資料集參數] [模型參數] [GNN參數] [訓練參數] [其他參數]
```

### 9.2 資料集參數

**方式一：指定單一資料集**
```bash
--dataset <dataset_name>

# 範例
python main.py --dataset eye --models tabnet --gnn_stages all --epochs 300
python main.py --dataset kaggle_Audit_Data --models excelformer --gnn_stages columnwise
```

**方式二：按類別批量選擇**
```bash
--dataset_size {small_datasets, large_datasets, all}
--task_type {binclass, multiclass, regression, all}
--feature_type {numerical, categorical, balanced, all}
--data_dir ./datasets

# 範例：所有小型二分類數值型資料集
python main.py --dataset_size small_datasets --task_type binclass --feature_type numerical \
               --models excelformer resnet --gnn_stages none columnwise --epochs 200

# 範例：所有資料集
python main.py --dataset_size all --task_type all --feature_type all \
               --models all --gnn_stages all --epochs 300
```

### 9.3 模型參數

```bash
--models <model1> <model2> ...

# 單一模型
--models excelformer

# 多個模型
--models excelformer resnet tabnet fttransformer

# 所有模型（目前包含 10 可拆分 + 11 基線，總計 21 模型）
--models all

# 可用模型列表
# 可拆分：excelformer, fttransformer, resnet, tabnet, tabtransformer, 
#         trompt, scarf, subtab, vime, tabm
# 基線：xgboost, catboost, lightgbm, tabgnn, t2gformer, dgm, lan_gnn, idgl_gnn, glcn, lds_gnn, tabpfn
```

### 9.4 GNN 階段參數

```bash
--gnn_stages <stage1> <stage2> ...

# 無 GNN（基線）
--gnn_stages none

# 單一階段
--gnn_stages columnwise

# 多個階段
--gnn_stages none columnwise encoding decoding

# 所有階段
--gnn_stages all
# 等同於：none start materialize encoding columnwise decoding

# 可用階段列表
# none, start, materialize, encoding, columnwise, decoding, all
```

### 9.5 資料切分參數

**方式一：手動指定比例**
```bash
--train_ratio 0.05
--val_ratio 0.15
--test_ratio 0.80  # 可省略，會自動計算

# 範例：few-shot 設定
python main.py --dataset eye --models tabnet --gnn_stages all \
               --train_ratio 0.05 --val_ratio 0.15 --epochs 300

# 範例：fully-supervised 設定
python main.py --dataset eye --models tabnet --gnn_stages all \
               --train_ratio 0.80 --val_ratio 0.15 --epochs 300
```

**方式二：使用 few-shot 快捷設定**
```bash
--few_shot                    # 啟用 few-shot 模式
--few_shot_ratio 0.05         # few-shot 訓練比例（預設 0.05）

# 範例
python main.py --dataset eye --models tabnet --gnn_stages all \
               --few_shot --few_shot_ratio 0.05 --epochs 300
```

### 9.6 GNN 配置參數

```bash
--gnn_hidden_dim 256          # GNN 隱藏層維度（預設 256）
--gnn_layers 2                # GNN 層數（預設 2）
--gnn_dropout 0.2             # GNN Dropout 比率（預設 0.2）

# 範例
python main.py --dataset eye --models tabnet --gnn_stages columnwise \
               --gnn_hidden_dim 512 --gnn_layers 3 --gnn_dropout 0.3
```

### 9.7 訓練配置參數

```bash
--epochs 300                  # 訓練輪數（預設 300）
--batch_size 256              # 批次大小（預設 256）
--lr 0.001                    # 學習率（預設 0.001）
--weight_decay 1e-5           # 權重衰減（預設 1e-5）
--patience 10                 # 早停 patience（預設 10）
--seed 42                     # 隨機種子（預設 42）

# 範例
python main.py --dataset eye --models tabnet --gnn_stages all \
               --epochs 200 --batch_size 128 --lr 0.0005 --seed 123
```

### 9.8 其他參數

```bash
--gpu 0                       # GPU ID（預設 0）
--debug_metrics               # 啟用除錯指標輸出
--output_dir ./results        # 結果輸出目錄

# 範例：使用多 GPU
python main.py --dataset eye --models tabnet --gnn_stages all --gpu 1
```

### 9.9 完整範例命令

**範例 1：單資料集、單模型、全階段 few-shot**
```bash
python main.py \
    --dataset eye \
    --models tabnet \
    --gnn_stages all \
    --train_ratio 0.05 \
    --val_ratio 0.15 \
    --epochs 300 \
    --seed 42
```

**範例 2：多資料集類別、多模型、特定階段**
```bash
python main.py \
    --dataset_size small_datasets \
    --task_type binclass \
    --feature_type numerical \
    --models excelformer resnet tabnet \
    --gnn_stages none columnwise encoding \
    --train_ratio 0.05 \
    --val_ratio 0.15 \
    --epochs 200 \
    --batch_size 128 \
    --seed 42
```

**範例 3：完整實驗（所有模型、所有資料集、所有階段）**
```bash
python main.py \
    --dataset_size all \
    --task_type all \
    --feature_type all \
    --models all \
    --gnn_stages all \
    --train_ratio 0.05 \
    --val_ratio 0.15 \
    --epochs 300
```

**範例 4：消融實驗（train_ratio 掃描）**
```bash
# 需要在外層包裝腳本中循環執行
for ratio in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8; do
    for seed in {42..46}; do
        python main.py \
            --dataset eye \
            --models excelformer \
            --gnn_stages all \
            --train_ratio $ratio \
            --val_ratio 0.15 \
            --epochs 300 \
            --seed $seed
    done
done
```

### 9.10 程式執行流程

```
1. 解析命令列參數
   ├─ 驗證參數有效性
   └─ 設置隨機種子

2. 資料載入（DatasetLoader）
   ├─ 根據參數篩選資料集
   ├─ 讀取 CSV 檔案
   ├─ 處理缺失值
   ├─ 編碼類別特徵
   └─ 切分 train/val/test

3. 模型建立（ModelRunner）
   ├─ 動態載入模型模組
   ├─ 初始化模型權重
   └─ 註冊 GNN 掛鉤（若 gnn_stage != none）

4. GNN 插入（GNNInjector）
   ├─ 根據 gnn_stage 選擇插入點
   ├─ 構建 DGM + GCN 模組
   └─ 包裝原始模型

5. 訓練迴圈
   ├─ For epoch in epochs:
   │   ├─ 訓練一輪
   │   ├─ 驗證集評估
   │   ├─ 早停檢查
   │   └─ 儲存最佳模型
   └─ 載入最佳模型

6. 測試評估
   ├─ 在測試集上評估
   ├─ 計算指標（AUC/Accuracy/MAE）
   └─ 輸出結果

7. 結果儲存
   ├─ 儲存 log 檔案
   ├─ 儲存 CSV 表格
   └─ 生成可視化圖表
```

---

## 10. 分析與繪圖工具（待實作）

> **重要說明**：本章節描述的是需要實作的分析與繪圖工具。目前專案根目錄雖然有一些視覺化相關的 Python 檔案（visualize_*.py），但這些腳本實際上**並未被使用**，因此不列入說明。以下內容是第二階段消融實驗所需的新工具規劃。

### 10.1 消融實驗繪圖腳本（需要實作）

**核心功能需求**：
```python
# 消融實驗繪圖的核心函數

def plot_ablation_results(model_name, x_variable, results_df):
    """
    繪製消融實驗的雙子圖
    
    Args:
        model_name: 模型名稱（如 'excelformer'）
        x_variable: 橫軸變量名稱（如 'train_ratio'）
        results_df: 包含以下欄位的 DataFrame
            - x_variable: 自變量值
            - gnn_stage: GNN 階段
            - mean_performance: 平均性能
            - std_performance: 性能標準差
            - mean_gain: 平均增益（相對於 none）
            - std_gain: 增益標準差
    
    Returns:
        matplotlib figure 物件
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子圖 1：性能曲線
    for stage in ['none', 'start', 'materialize', 'encoding', 'columnwise', 'decoding']:
        stage_data = results_df[results_df['gnn_stage'] == stage]
        ax1.plot(stage_data[x_variable], stage_data['mean_performance'], 
                 label=stage, marker='o')
        ax1.fill_between(stage_data[x_variable],
                         stage_data['mean_performance'] - stage_data['std_performance'],
                         stage_data['mean_performance'] + stage_data['std_performance'],
                         alpha=0.2)
    ax1.set_xlabel(x_variable)
    ax1.set_ylabel('Performance')
    ax1.set_title(f'{model_name}: Performance vs {x_variable}')
    ax1.legend()
    ax1.grid(True)
    
    # 子圖 2：增益曲線
    for stage in ['start', 'materialize', 'encoding', 'columnwise', 'decoding']:
        stage_data = results_df[results_df['gnn_stage'] == stage]
        ax2.plot(stage_data[x_variable], stage_data['mean_gain'], 
                 label=stage, marker='o')
        ax2.fill_between(stage_data[x_variable],
                         stage_data['mean_gain'] - stage_data['std_gain'],
                         stage_data['mean_gain'] + stage_data['std_gain'],
                         alpha=0.2)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel(x_variable)
    ax2.set_ylabel('Gain (%)')
    ax2.set_title(f'{model_name}: GNN Gain vs {x_variable}')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig
```

### 10.2 統計聚合腳本（需要實作）

**需要實作的聚合函數**：
```python
def aggregate_ablation_results(raw_results_dir, output_csv):
    """
    聚合消融實驗的原始結果
    
    Args:
        raw_results_dir: 原始結果目錄（包含所有 model×dataset×seed 的 CSV）
        output_csv: 輸出的聚合統計 CSV
    
    聚合邏輯：
        對每個 (model, gnn_stage, x_value) 組合：
            1. 收集所有 20 datasets × N seeds 的性能值
            2. 計算 mean, variance, std
            3. 計算相對於 none 的 gain
            4. 輸出到 CSV
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    results = []
    
    # 遍歷所有原始結果檔案
    for result_file in Path(raw_results_dir).glob("*.csv"):
        df = pd.read_csv(result_file)
        # 解析檔名以提取 model, dataset, seed, x_value, stage
        # ...
        results.append(df)
    
    # 合併與聚合
    all_results = pd.concat(results)
    aggregated = all_results.groupby(['model', 'gnn_stage', 'x_variable']).agg({
        'performance': ['mean', 'var', 'std'],
        'gain': ['mean', 'var', 'std']
    }).reset_index()
    
    aggregated.to_csv(output_csv, index=False)
    return aggregated
```

---

## 11. 關鍵結論與實務指引

### 11.1 最易獲益場景（GNN Friendly）

✅ **資料特徵**
- 資料集規模：小型（row count < 5000）
- 任務類型：二分類（binclass）
- 特徵類型：數值主導（numerical ratio > 70%）
- 訓練樣本：稀缺（train_ratio < 0.2）

✅ **最佳 GNN 階段**
1. **columnwise**（最穩定，首選）
2. **encoding**（次優，token-level 增強）
3. **decoding**（端到端圖預測，效果視模型而定）

✅ **適用模型**（根據第一階段發現）
- FT-Transformer：對 GNN 敏感度高
- ExcelFormer：在數值型資料上增益明顯
- TabNet：中等敏感度，穩定增益

### 11.2 增益減弱場景（GNN Unfriendly）

❌ **不利條件**
- 資料集規模：大型（row count > 10000）
- 訓練樣本：充足（train_ratio > 0.6）
- 特徵類型：類別主導（categorical ratio > 70%）
- 強基線存在：
  - 充足樣本下的樹模型（XGBoost, CatBoost, LightGBM）
  - 預訓練模型（TabPFN）

❌ **不推薦階段**
- **start/materialize**：離線處理，增益不穩定且耗時
- 在大型資料集或充足樣本下，任何階段的 GNN 都可能無增益

### 11.3 實驗設計建議

**快速驗證（Sanity Check）**
```bash
# 只跑 none + columnwise，5 個種子，精簡資料集
python ablation_study/quick_test.py --stages none columnwise --seeds 5 --datasets 5
```
- 時間：~10 分鐘
- 目的：驗證流程、初步觀察趨勢

**中期檢查（Pilot Study）**
```bash
# 6 階段，10 個種子，20 個資料集
python ablation_study/run_ablation.py --stages all --seeds 10 --datasets 20
```
- 時間：~1 小時（並行 100）
- 目的：得到初步結論、驗證假設

**完整發表（Full Study）**
```bash
# 6 階段，20 個種子，20 個資料集
python ablation_study/run_ablation.py --stages all --seeds 20 --datasets 20
```
- 時間：~3-4 小時（並行 80）
- 目的：獲取完整統計、用於論文

### 11.4 資源管理建議

**GPU 記憶體**
- 單個任務：~600-800MB
- 並行 80：需 ~50-64GB（8× A100 40GB 或類似）
- 並行 120：需 ~72-96GB（監控使用率，避免 OOM）

**時間規劃**
- 第一階段（已完成）：13,920 次實驗，約 1-2 天（並行 80）
- 第二階段三項消融：
  - 實驗一（train_ratio）：~4 小時（seeds=20, 並行=80）
  - 實驗二（numerical_ratio）：~3 小時
  - 實驗三（sampling_ratio）：~2.5 小時
  - 總計：~10 小時（可分批執行）

**儲存空間**
- 每個實驗結果：~1-5MB（log + CSV）
- 第一階段總計：~50-100GB
- 第二階段總計：~20-30GB
- 建議預留：150-200GB

### 11.5 故障排除

**常見問題 1：GPU 記憶體不足**
```
解決方案：
1. 降低並行度（120 → 80 → 40）
2. 減少 batch_size（256 → 128 → 64）
3. 減少 epochs（300 → 100，僅用於測試）
```

**常見問題 2：進程掛起**
```
檢查項目：
1. 磁碟空間是否充足（df -h）
2. GPU 溫度是否過高（nvidia-smi）
3. 資料集是否損壞（檔案完整性）
```

**常見問題 3：結果檔案缺失**
```
檢查項目：
1. 輸出目錄權限
2. 檔案命名是否衝突
3. 程式是否正常結束（查看 log）
```

---

## 12. 檔案導覽與路徑速查

### 12.1 核心程式

| 檔案 | 路徑 | 功能 |
|------|------|------|
| 主程式 | `/home/skyler/ModelComparison/TaBLEau/main.py` | 實驗入口、參數解析 |
| GNN 插入 | `/home/skyler/ModelComparison/TaBLEau/gnn_injection.py` | GNN 模組與掛鉤邏輯 |
| 模型運行器 | `/home/skyler/ModelComparison/TaBLEau/model_runner.py` | 動態載入與管理模型 |
| 資料載入器 | `/home/skyler/ModelComparison/TaBLEau/utils/data_utils.py` | 資料讀取與預處理 |

### 12.2 模型實作

| 類別 | 路徑 | 內容 |
|------|------|------|
| PyTorch Frame | `/home/skyler/ModelComparison/TaBLEau/models/pytorch_frame/` | 6 個可拆分模型 |
| Custom | `/home/skyler/ModelComparison/TaBLEau/models/custom/` | 4 個可拆分模型 |
| Comparison | `/home/skyler/ModelComparison/TaBLEau/models/comparison/` | 6 個基線模型 |
| Base | `/home/skyler/ModelComparison/TaBLEau/models/base/` | 基礎類別 |
| Data | `/home/skyler/ModelComparison/TaBLEau/models/data/` | 資料處理工具 |

### 12.3 資料集

| 類別 | 路徑 | 數量 |
|------|------|------|
| 所有資料集 | `/home/skyler/ModelComparison/TaBLEau/datasets/` | 116 個 CSV |
| 小型資料集 | `/home/skyler/ModelComparison/TaBLEau/datasets/small_datasets/` | ~60 個 |
| 大型資料集 | `/home/skyler/ModelComparison/TaBLEau/datasets/large_datasets/` | ~56 個 |

### 12.4 實驗結果

| 類別 | 路徑 | 內容 |
|------|------|------|
| 全域結果 | `/home/skyler/ModelComparison/TaBLEau/summary_results/` | 所有模型的彙總表格 |
| 模型別分析 | `/home/skyler/ModelComparison/TaBLEau/gnn_injection_analysis/per_model_result/` | 每個模型的詳細分析 |
| 消融結果 | `/home/skyler/ModelComparison/TaBLEau/ablation_study/` | 三項消融實驗 |
| 可視化 | `/home/skyler/ModelComparison/TaBLEau/visualization_results/` | 生成的圖表 |

### 12.5 文檔與說明

| 類別 | 路徑 | 內容 |
|------|------|------|
| 主文檔 | `/home/skyler/ModelComparison/TaBLEau/README.md` | 專案總覽 |
| 完整說明 | `/home/skyler/ModelComparison/TaBLEau/docs/SAGE_full_overview.md` | 本檔案 |
| 消融計畫 | `/home/skyler/ModelComparison/TaBLEau/ablation_study/ABLATION_STUDY_PLAN.md` | 第二階段計畫 |
| 參考論文 | `/home/skyler/ModelComparison/TaBLEau/Paper/` | 相關研究論文 |

---

## 13. AI 代理可執行任務清單

### 13.1 實驗執行任務

- [ ] 運行單一模型在特定資料集上的全階段測試
- [ ] 批量執行多模型在多資料集上的實驗
- [ ] 執行 train_ratio 掃描消融實驗
- [ ] 執行 numerical_ratio 掃描消融實驗
- [ ] 執行 sampling_ratio 掃描消融實驗
- [ ] 根據 GPU 資源自動調整並行度

### 13.2 資料處理任務

- [ ] 聚合原始實驗結果（20×N 個數值 → 均值/方差/標準差）
- [ ] 生成 per-category 的排名表
- [ ] 計算各 GNN 階段相對於 none 的 gain
- [ ] 產生擊敗/平手統計表（vs 各類基線）
- [ ] 匯出標準格式的 CSV 供後續分析

### 13.3 可視化任務（需先實作繪圖工具）

- [ ] **實作繪圖腳本**：建立消融實驗的雙子圖繪製函數
- [ ] 繪製消融實驗的雙子圖（性能 + 增益）
- [ ] 生成 10 個模型的完整圖集
- [ ] 繪製跨模型的熱力圖（哪些模型對 GNN 最敏感）
- [ ] 生成 per-category 的箱型圖（性能分佈）
- [ ] 製作動態圖表（互動式 HTML，可選）

### 13.4 分析與報告任務

- [ ] 自動判斷哪些 (模型, 階段, 資料類型) 組合效果最佳
- [ ] 生成 Markdown 格式的總結報告
- [ ] 對比第一階段與第二階段的發現
- [ ] 提取統計顯著性（t-test, ANOVA）
- [ ] 撰寫論文用的 LaTeX 表格

### 13.5 維護與優化任務

- [ ] 檢查實驗結果的完整性（缺失值、異常值）
- [ ] 估算剩餘實驗的時間與資源需求
- [ ] 根據初步結果調整實驗參數（adaptive sampling）
- [ ] 自動重跑失敗的實驗
- [ ] 定期備份實驗結果

---

## 14. 快速檢查清單（實驗前必查）

### 14.1 環境檢查

- [ ] Python 版本：3.8+
- [ ] PyTorch 版本：1.12+（含 CUDA 支援）
- [ ] PyTorch Geometric 已安裝
- [ ] PyTorch Frame 已安裝
- [ ] GPU 可用：`nvidia-smi` 有輸出
- [ ] 磁碟空間：至少 200GB 可用

### 14.2 資料檢查

- [ ] 資料集目錄存在：`/home/skyler/ModelComparison/TaBLEau/datasets/`
- [ ] 116 個 CSV 檔案齊全：`find datasets -name "*.csv" | wc -l` 應顯示 116
- [ ] 資料集可讀取：隨機抽查幾個 CSV 檔案

### 14.3 程式檢查

- [ ] `main.py` 可執行：`python main.py --help` 有輸出
- [ ] 模型可載入：測試載入 excelformer
- [ ] GNN 插入無錯誤：測試 `--gnn_stages columnwise`

### 14.4 輸出檢查

- [ ] 輸出目錄存在且可寫：`summary_results/`, `ablation_study/`
- [ ] Log 檔案正常產生：`experiment.log`
- [ ] CSV 結果可讀取

### 14.5 參數檢查

- [ ] seeds 數量合理：5-20 之間
- [ ] train_ratio 範圍正確：0-1 之間，且 train + val + test = 1
- [ ] 並行度不超過 GPU 記憶體限制
- [ ] epochs 數量適中：測試用 10-50，正式用 200-300

---

## 15. 結語與後續規劃

### 15.1 當前進度

✅ **已完成**：
- TaBLEau 框架建立（116 資料集、21 個模型：10 可拆分 + 11 基線）
- 統一的五階段流水線設計
- GNN 插入機制實作（6 種策略）
- 第一階段全面實驗（13,920 次）
- Per-model 詳細分析與報告
- 第二階段三項消融實驗的詳細計畫（ABLATION_STUDY_PLAN.md）
- 本全面說明文檔（SAGE_full_overview.md）

📅 **待完成**（第二階段）：
- 實作消融實驗的執行腳本（run_*_ablation.py）
- 實作資料聚合與統計分析腳本
- 實作繪圖與可視化工具
- 執行三項消融實驗
- 生成完整的可視化圖集
- 撰寫論文與投稿

### 15.2 論文撰寫要點

**主要貢獻**：
1. 提出 SAGE：首個系統性分析 GNN 與表格模型整合的框架
2. 統一的五階段流水線：實現跨模型的公平比較
3. 全面的實證分析：116 資料集 × 10 個可拆分模型 × 6 階段（並提供 11 個基線模型作為對照）
4. 三項消融實驗：揭示 GNN 增益的關鍵因素（樣本量、特徵類型、資料規模）

**實驗章節結構建議**：
- 4.1 第一階段：全面比較（已有數據）
- 4.2 第二階段：消融實驗一（train_ratio）
- 4.3 第二階段：消融實驗二（numerical_ratio）
- 4.4 第二階段：消融實驗三（sampling_ratio）
- 4.5 討論與分析

**圖表規劃**：
- 總圖數：~40-50 張
  - 主圖：10-15 張（關鍵發現）
  - 附錄：30-35 張（完整結果）

### 15.3 未來擴展方向

**模型擴展**：
- 本研究的 self-contained GNN baselines 已定稿為 7 個（TabGNN/T2G-Former/DGM/LAN-GNN/IDGL-GNN/GLCN/LDS-GNN），目前不再規劃新增。
- 如需擴展，建議另開「延伸研究」章節，以避免與第一階段的可重現結果混淆。

**實驗擴展**：
- 多模態表格數據（文字 + 數值）
- 時間序列表格數據
- 不平衡資料集的專項分析

**方法創新**：
- 自適應 GNN 插入（根據資料特性自動選擇階段）
- 多階段聯合插入（同時在多個階段插入）
- 可學習的圖構建策略（替代固定的 k-NN）

---

---

## 16. 本文檔的使用說明

### 16.1 文檔目的

本文檔旨在為 AI 代理提供 TaBLEau + SAGE 研究的**完整上下文與全面理解**，包括：
- 研究背景與動機
- 已完成的第一階段實驗與發現
- 規劃中的第二階段消融實驗（詳細設計但尚未實作）
- 所有相關的資料、模型、流程說明
- 執行命令、參數配置、檔案路徑

### 16.2 當前狀態總結

**第一階段（已完成）**：
- ✅ 116 個資料集已就緒
- ✅ 21 個模型（10 可拆分 + 11 基線）已實作（self-contained GNN baselines 共 7 個）
- ✅ 6 種 GNN 插入策略已實作
- ✅ 13,920 次實驗已執行完成
- ✅ Per-model 分析報告已生成（位於 gnn_injection_analysis/per_model_result/）

**第二階段（規劃階段，待實作）**：
- 📋 三項消融實驗已詳細規劃（見第 8 章）
- ⏳ 執行腳本尚未實作
- ⏳ 資料聚合與統計工具尚未實作
- ⏳ 繪圖與可視化工具尚未實作
- ⏳ 實驗尚未執行

### 16.3 AI 代理的首要任務

如果您是接手此研究的 AI 代理，建議按以下優先順序進行：

**階段 A：理解與驗證**
1. 閱讀本文檔，理解研究框架與目標
2. 檢查第一階段結果的完整性
3. 驗證資料集與模型可用性
4. 測試 main.py 的基本功能

**階段 B：第二階段實作準備**
1. 根據第 8 章規劃，設計資料集選擇邏輯
2. 實作三個主執行腳本（run_*_ablation.py）
3. 實作資料聚合與統計分析工具
4. 實作繪圖與可視化工具

**階段 C：執行與分析**
1. 先執行小規模測試（5 seeds, 少量資料集）
2. 驗證結果合理性與繪圖正確性
3. 執行完整的三項消融實驗
4. 生成完整的分析報告與圖表

**階段 D：論文撰寫**
1. 整理所有實驗結果
2. 生成論文用的表格與圖表
3. 撰寫實驗章節
4. 準備投稿

### 16.4 重要提醒

⚠️ **關於視覺化腳本**：
- 專案根目錄雖然有一些 `visualize_*.py` 檔案，但這些**並未實際使用**
- 第二階段需要的繪圖工具需要**重新實作**
- 請參考第 10 章的規劃進行實作

⚠️ **關於 ablation_study 目錄**：
- 目前只有 `ABLATION_STUDY_PLAN.md` 這個計畫文件
- 所有執行腳本、資料處理腳本、繪圖腳本都**尚未建立**
- 需要根據計畫文件與本文檔第 8 章的設計進行實作

⚠️ **關於時間與資源**：
- 三項消融實驗預計需要 8-10 小時（seeds=20, 並行=80）
- 可先用精簡版（seeds=5）快速驗證，約 2-3 小時
- 確保有足夠的 GPU 資源與磁碟空間

---

**本文檔為自洽的完整說明，為 AI 代理提供了全面的研究上下文。雖然第二階段實驗尚未開始實作，但所有必要的設計、規劃與指引均已詳細說明，可作為實作與執行的完整參考。**

## 9. 典型操作範例
- 單模型全階段（few-shot 預設）：
  - python main.py --dataset eye --models tabnet --gnn_stages all --epochs 300
- 多模型、多階段、限制於小型數值二分類：
  - python main.py --dataset_size small_datasets --task_type binclass --feature_type numerical \
    --models excelformer resnet tabnet --gnn_stages none columnwise encoding \
    --train_ratio 0.05 --val_ratio 0.15 --epochs 200 --few_shot
- 消融（示意，需在腳本內改 seeds/ratios 後執行）：
  - python ablation_study/quick_columnwise_run_v3.py --parallel-jobs 100

## 10. 分析與繪圖要點
- 性能指標：依任務選 AUC/accuracy/MAE 等（程式自動判斷）。
- gain 定義： (metric_stage - metric_none) / metric_none。
- 排名比較：per_model_result/*_gnn_enhancement_summary.md 提供 vs few/full non-GNN、樹、GNN 原生、tabpfn 的擊敗/平手統計，容差 1e-3。
- 可視化腳本：visualize_gnn_enhancement.py、visualize_model_variants.py、visualize_results.py；消融繪圖在 ablation_study/（如 plot_columnwise_comparison.py）。

## 11. 關鍵結論與直覺（供 AI 判斷任務優先）
- 最易獲益場景：小型、few-shot、數值主導、binclass；columnwise 最常帶來正增益，其次 encoding/decoding。
- 增益遞減：訓練比例升高、數據集變大、數值占比降低，或遇到強基線（樹、tabpfn）。
- 若需快速驗證：先跑 none vs columnwise、精簡 seeds 與 ratio，再擴大。

## 12. 檔案導覽
- README.md：專案總覽與資料下載連結。
- main.py：主入口與參數解析、訓練迴圈。
- gnn_injection.py：GNN 模組與各 stage 掛鉤實作。
- model_runner.py：動態載入模型與掛鉤管理。
- models/：各模型實作（pytorch_frame/custom/comparison）與映射說明。
- summary_results/：全域結果彙整（兩種切分）。
- gnn_injection_analysis/per_model_result/：模型別 GNN 增強與敏感度分析文件。
- ablation_study/：第二階段消融腳本與計畫文件。
- docs/：其他說明（含本檔）。

## 13. AI 代理可執行的典型任務清單
- 依指令批量跑實驗（指定 models、gnn_stages、ratio、seeds）。
- 聚合 20×seeds 的均值/方差/標準差，產生性能與 gain 曲線圖。
- 生成 per-category 的排名/擊敗統計表，對照 baselines。
- 根據特徵占比或子集大小自動生成實驗配置，並繪製兩欄圖（性能、gain）。
- 檢查 GPU/時間限制，自動調整 seeds、ratio 點數與並行度。

## 14. 快速檢查清單（跑實驗前）
- 確認資料路徑：datasets/ 是否齊全、可讀。
- 確認輸出目錄：summary_results/ 或自定 output_dir 是否存在/可寫。
- GPU/並行設定：並行 80–120 是否足夠記憶體；必要時降低 batch/epochs。
- seeds 與 ratio 清單：是否已按需求精簡。
- gnn_stages：若只做 sanity，可用 none+columnwise；完整則 all。

---
本檔為自洽的說明，AI 代理可直接依據上述結構與範例命令，進行批量實驗、消融、統計與繪圖。