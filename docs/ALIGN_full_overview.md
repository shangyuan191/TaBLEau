# TaBLEau + ALIGN 全面說明檔（供 AI 使用）

> 目的：為 AI 代理提供對 TaBLEau 框架與 ALIGN 研究的完整上下文，涵蓋資料、模型、流程、GNN 插入策略、實驗設計與後續消融計畫，方便自動化運行、分析與繪圖。

## 1. 專案定位與目標
- TaBLEau：統一的表格深度學習基準框架，整合多個 SOTA 模型與多類資料集，提供一致的輸入輸出與訓練流程。
- ALIGN：研究在可拆分的表格模型中，於五個階段插入 GNN 的效果，特別關注少樣本 (few-shot) 情境下的泛化增益，並分析不同資料型態下的差異。
- 研究核心：
  - 問題：GNN 能否在表格學習中帶來結構感知與小數據增益？
  - 方法：對 10 個可拆分模型，在 five-stage pipeline 的五個插入點分別插入 GNN，與多種基線比較。
  - 指標：性能值（依任務選 AUC/accuracy/MAE 等）、與 none 基線的 performance gain、平均排名、方差與標準差。

## 2. 資料集總覽
- 位置：datasets/（含 small_datasets, large_datasets；底下再分 binclass, multiclass, regression；並依特徵比例分 numerical, categorical, balanced）。
- 總數：116 個 CSV。
- 典型路徑示例：
  - 小型二分類數值型：datasets/small_datasets/binclass/numerical/
  - 大型多分類混合型：datasets/large_datasets/multiclass/balanced/
- 標準切分：
  - few-shot：train/val/test = 0.05 / 0.15 / 0.80
  - full：train/val/test = 0.80 / 0.15 / 0.05
- 欄位類型假設：數值、類別（可能經 one-hot/embedding），欄位數量差異大；需要 DatasetLoader 自動處理。

## 3. 模型族群與分類
- 可拆分（10 個，可插 GNN）：excelformer, fttransformer, resnet, tabnet, tabtransformer, trompt, scarf, subtab, vime, tabm。
  - 主要程式：models/pytorch_frame/*.py、models/custom/*.py。
  - 映射文件（示例）：models/pytorch_frame/excelformer_pytorch_frame_mapping.md（描述各自的 materialize/encoding/columnwise/decoding 對應）。
- 比較基線（不可拆分，不插 GNN）：
  - 樹模型：xgboost, catboost, lightgbm。
  - 原生 GNN：tabgnn, t2gformer（未來可再增 2~3 個）。
  - 預訓練：tabpfn。
  - 位置：models/comparison/。

## 4. 五階段統一流水線（stage）
- start：dummy，實際不改資料，存在目的是允許在最前插入 GNN。
- materialize：資料物化成 TensorFrame/DataLoader，含類別處理、資訊排序等。
- encoding：列/特徵編碼，輸出 tokens x:[B, F, C]。
- columnwise：列間交互（多頭注意等），保持 x:[B, F, C]。
- decoding：解碼到 out:[B, out_channels]。

## 5. GNN 插入策略（gnn_injection.py）
- 共同組件：DGM_d 動態建圖 + GCNConv 堆疊 + 殘差融合門（sigmoid alpha）。
- 插入點：none, start, materialize, encoding, columnwise, decoding。
- 行為概述：
  - start：對原始 DataFrame 做注意力編碼→建圖→GCN→重建欄位，再進主幹。
  - materialize：先物化，再類似 start 的離線式處理，然後回到主幹。
  - encoding：在 tokens 後插入 GNN（DGM+GCN+注意力解碼+殘差），再進 convs/decoder。
  - columnwise：在 convs 後插入 GNN，再進 decoder；實務上最常帶來增益。
  - decoding：用 GNN 直接取代 decoder（池化建圖→GCN 輸出 logits）。

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

## 7. 已產出結果與參考文件
- 總結結果：summary_results/（各模型、兩種切分的表格與排名）。
- 模型別 GNN 增強分析：gnn_injection_analysis/per_model_result/ 下的 *_gnn_enhancement.md/.txt/.csv，含：
  - 按資料類別 (大小/任務/特徵) 的平均排名、擊敗/平手統計。
  - 對照基線：few-shot/full non-GNN、樹模型、GNN 原生、tabpfn。
  - 評分規則：容差 1e-3；對 few-shot 自身基線需嚴格更好才算 beat。
- 重點觀察：
  - 小型、few-shot、數值占比高、binclass：columnwise/encoding/decoding 常優於 none。
  - 大型或 full-sample：GNN 增益縮小，強基線（樹、tabpfn）常佔優。

## 8. 第二階段消融計畫（ablation_study/ABLATION_STUDY_PLAN.md）
### 8.1 核心設定
- 模型：10 個可拆分模型。
- 階段：6 個（none/start/materialize/encoding/columnwise/decoding）。
- seeds：預設 20 (42–61)，可降為 5 以加速。
- 並行：建議 80–120；可依 GPU 記憶體調整。

### 8.2 任務一：train_ratio 掃描（樣本量）
- 數據：20 個 small+binclass（row 越多越好）。
- ratio：16 點 train_ratio=[0.05, 0.1, ..., 0.8]，val=0.15，test=自動。
- 指標：每模型畫兩張小圖（共 10 大圖）：
  - 性能 vs train_ratio（6 曲線：none+5 注入）。
  - gain vs train_ratio（5 曲線：相對 none）。
- 聚合：每點 20 資料集 × seeds 平均，需均值/方差/標準差。

### 8.3 任務二：數值特徵占比掃描
- 數據：20 個 small+binclass（欄位多、數值與類別混合，盡量均衡）。
- 方法：調降 numerical column 占比（高→低）。
- 指標：性能與 gain 對 numerical 占比的曲線，6/5 曲線同上；報均值/方差/標準差。

### 8.4 任務三：大表下采樣（size→small）
- 數據：20 個 large+binclass。
- 抽樣：子集比例 {10%,20%,...,100%}，再套 few-shot 5%（train=子集的 5%，val=15%，其餘 test）。
- 指標：性能與 gain 對抽樣比例的曲線，6/5 曲線；報均值/方差/標準差。

### 8.5 時間與精簡建議
- seeds 由 20 → 5 可省 75% 時間。
- train_ratio 由 16 → 8（如 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8）。
- 方案示例（計畫文件內）：
  - 方案 D：seeds=10、ratio=8、stages=2 (none+columnwise)，並行 100，約 6 分鐘（快速 sanity）。
  - 方案 C：seeds=5、ratio=8、stages=6，並行 100，約 27 分鐘（精簡）。
  - 方案 B：seeds=10、ratio=8、stages=6，並行 100，約 56 分鐘（中期）。
  - 方案 A：seeds=20、ratio=16、stages=6，並行 80，約 3.7 小時（完整版）。

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