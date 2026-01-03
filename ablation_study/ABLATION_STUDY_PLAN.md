# GNN Injection Ablation Study - 完整實驗計劃

## 📋 實驗目標

進行全面的 GNN injection 階段對比實驗，評估不同 GNN 注入位置（包括 none baseline）對模型性能的影響。

---

## 📊 實驗規模與參數

### 模型列表（10個可拆分 GNN stage 的模型）
```
1. excelformer
2. fttransformer
3. resnet
4. tabnet
5. tabtransformer
6. trompt
7. scarf
8. subtab
9. vime
10. tabm
```

### 數據集
- **數據集類型**：small_datasets + binclass
- **數據集數量**：20 個
- **數據集位置**：`/home/skyler/ModelComparison/TaBLEau/datasets/small_datasets/binclass/`
- **說明**：同時包含 numerical, categorical, balanced 三種特徵類型的數據集

### Split Ratio 配置
- **訓練數據比例 (train_ratio)**：16 種
  ```
  [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
  ```
- **驗證數據比例**：固定 0.15
- **測試數據比例**：自動計算 (1 - train_ratio - 0.15)
- **說明**：通過變動 train_ratio，研究 GNN 在不同樣本量下的效果

### 隨機種子
- **種子數量**：20 個
- **種子範圍**：42-61
- **說明**：用於數據分割的隨機性，確保結果穩定性

### GNN 注入階段（6 個）
```
1. none          (Baseline - 不注入 GNN)
2. start         (在模型開始時注入)
3. materialize   (在物化階段注入)
4. encoding      (在編碼階段注入)
5. columnwise    (在列級別注入 - 最關鍵)
6. decoding      (在解碼階段注入)
```

### 訓練配置
- **Epochs**：300
- **GPU**：cuda:0
- **批大小**：256 (默認)
- **學習率**：0.001 (默認)

---

## ⏱️ 時間分析

### 單次實驗耗時
```
根據實際測試結果（UCI_Quality_Assessment_of_Digital_Colposcopies 數據集）：

excelformer @ none:        4.16秒
excelformer @ start:       3.66秒
excelformer @ materialize: 2.25秒
excelformer @ encoding:    2.98秒
excelformer @ columnwise:  1.73秒
excelformer @ decoding:    2.02秒
─────────────────────────────────
平均每個 stage：2.8 秒
平均每個 model×dataset×ratio×seed：16.8秒
```

### 完整實驗規模計算
```
總實驗次數 = 10 models × 16 ratios × 20 datasets × 20 seeds × 6 GNN stages
           = 384,000 次實驗
```

### 時間估算（不同並行度）

| 並行度 | 時間估算 | 實際運行時間 | 可行性 |
|--------|---------|-----------|--------|
| 串行 | 1,075,200秒 = **298小時 = 12.4天** | ❌ 過長 | 不可行 |
| 40並行 | 26,880秒 = **7.5小時** | ⚠️ 可行 | 慢 |
| 80並行 | 13,440秒 = **3.7小時** | ✅ 可行 | 好 |
| 100並行 | 10,752秒 = **3小時** | ✅ 可行 | 較好 |
| 120並行 | 8,960秒 = **2.5小時** | ⚠️ 需驗證GPU | 最快 |

---

## 🚀 加速策略

### 策略 1：減少隨機種子（最有效）
- **20 → 10 seeds**：節省 50% 時間
- **20 → 5 seeds**：節省 75% 時間
- **建議**：統計上 5-10 個 seeds 已足夠確保結果穩定

### 策略 2：減少 Split Ratios（保留關鍵點）
- **16 → 8 個**：節省 50% 時間
- **推薦選擇**：`[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]`
- **優勢**：覆蓋樣本量從極少 (5%) 到充足 (80%) 的完整範圍

### 策略 3：分階段執行
第一階段 - 快速驗證（6分鐘）
- Seeds: 5
- Ratios: 6 個關鍵點
- Stages: none + columnwise (最關鍵的 2 個)
- 並行: 100
- 目的：快速驗證管道、獲取初步結果

第二階段 - 完整比較（1小時）
- Seeds: 10
- Ratios: 8 個
- Stages: 全部 6 個
- 並行: 80
- 目的：得到完整的多階段對比

### 策略 4：增加並行數
- 測試 GPU 是否能支援 120-160 並行
- 監控 GPU 記憶體使用率
- 每個 job 約 600-800MB GPU 記憶體

---

## 📋 推薦實驗方案

### 方案 A：完整版（推薦用於最終發表）
```
配置：
  Models: 10
  Datasets: 20
  Seeds: 20
  Ratios: 16
  Stages: 6
  並行: 80

總實驗數：384,000
預估時間：3.7 小時
適用場景：最終結果、完整論文
```

### 方案 B：快速版（推薦用於中期檢查）
```
配置：
  Models: 10
  Datasets: 20
  Seeds: 10
  Ratios: 8
  Stages: 6
  並行: 100

總實驗數：96,000
預估時間：56 分鐘
適用場景：驗證假設、快速迭代
```

### 方案 C：精簡版（推薦用於快速驗證）
```
配置：
  Models: 10
  Datasets: 20
  Seeds: 5
  Ratios: 8
  Stages: 6
  並行: 100

總實驗數：48,000
預估時間：27 分鐘
適用場景：快速驗證邏輯、初期測試
```

### 方案 D：初探版（推薦用於快速出初步結果）
```
配置：
  Models: 10
  Datasets: 20
  Seeds: 10
  Ratios: 8
  Stages: 2 (none + columnwise)
  並行: 100

總實驗數：32,000
預估時間：6 分鐘
適用場景：快速獲取 columnwise 對比結果
```

---

## 🎯 核心分析指標

### 1. 主要對比：各 GNN 階段 vs Baseline (none)
```
對於每個 (ratio, dataset)：
  Improvement(stage) = (AUC(stage) - AUC(none)) / AUC(none) × 100%
```

### 2. 樣本量依賴性分析
```
關鍵發現（基於背景知識）：
  - 低樣本 (0.05-0.2)：GNN 帶來明顯增益 (1-2.5%)
  - 中樣本 (0.3-0.5)：GNN 增益逐漸減少 (0.3-1%)
  - 高樣本 (0.6-0.8)：GNN 增益很小或略為負面 (-0.3% 到 +0.3%)
```

### 3. 模型差異分析
```
比較 10 個模型對不同 GNN 階段的響應差異：
  - 哪個模型對 GNN 最敏感？
  - 不同模型的最優 GNN 階段是否不同？
```

### 4. 統計穩定性
```
關於 seeds 數量的建議：
  - 5 seeds：基本統計穩定，標準差可能在 ±2-3%
  - 10 seeds：較好的統計穩定性
  - 20 seeds：最佳統計穩定性
  
  建議取決於所需的統計嚴格程度
```

---

## 📁 相關文件與代碼

### 主要腳本位置
```
/home/skyler/ModelComparison/TaBLEau/
├── main.py                                    # 主實驗運行器
├── ablation_study/
│   ├── quick_columnwise_run_v3.py            # 當前執行腳本（columnwise vs none）
│   ├── plot_columnwise_comparison.py         # 繪圖腳本
│   ├── generate_dummy_comparison_plots.py    # Dummy 數據生成（參考）
│   └── columnwise_results/                   # 結果輸出目錄
└── run_columnwise_v3.sh                      # 執行包裝腳本
```

### main.py 支援的關鍵參數
```
--dataset              數據集名稱
--models               模型名稱（支援多個）
--gnn_stages           GNN 注入階段（支援多個）
--train_ratio          訓練數據比例 (0-1)
--val_ratio            驗證數據比例 (0-1)
--seed                 隨機種子
--epochs               訓練輪數
--gpu                  GPU ID (0-based)
--output_dir           結果輸出目錄
```

---

## 💡 實施建議

### 第一步：快速驗證（推薦方案 D）
```bash
# 執行時間：6 分鐘
# 目的：驗證全個管道是否正常運作
cd /home/skyler/ModelComparison/TaBLEau
python ablation_study/quick_columnwise_run_v3.py --parallel-jobs 100
```

### 第二步：中期檢查（推薦方案 B）
```bash
# 執行時間：56 分鐘
# 目的：獲取初步結論、驗證主要假設
# 需要修改腳本配置：
#   SEEDS = list(range(42, 52))  # 10 seeds
#   SPLIT_RATIOS 改為 8 個關鍵點
```

### 第三步：最終實驗（推薦方案 A）
```bash
# 執行時間：3.7 小時
# 目的：獲取完整的統計結果，用於發表
# 使用完整配置，所有模型/數據集/seeds/ratios
```

---

## 🔧 故障排除與最佳實踐

### 常見問題

1. **GPU 記憶體不足**
   - 減少並行度 (80 → 40)
   - 減少 epochs (300 → 100，用於測試)

2. **進程掛起**
   - 檢查磁盤空間
   - 監控 GPU 溫度 (nvidia-smi)

3. **結果文件缺失**
   - 檢查 output_dir 權限
   - 驗證 main.py 是否正確保存結果

### 監控命令
```bash
# 監控 GPU 使用情況
watch -n 1 nvidia-smi

# 監控進程數
ps aux | grep main.py | grep -v grep | wc -l

# 監控結果文件生成
ls ablation_study/columnwise_comparison/*.json | wc -l
```

---

## 📊 預期結果格式

### JSON 結果文件示例
```json
{
  "dataset": "kaggle_Customer_Classification",
  "train_ratio": 0.05,
  "gnn_stage": "columnwise",
  "aucs": [0.932, 0.945, 0.928, ...],  // 20 個 seeds 的結果
  "mean": 0.9357,
  "std": 0.0082,
  "count": 20
}
```

### 最終對比圖表
```
X 軸：Train Ratio (0.05 到 0.80)
Y 軸 (左)：AUC (0.90 到 1.00)
Y 軸 (右)：GNN Gain (%)

曲線 1：None (Baseline)
曲線 2：Columnwise GNN
曲線 3+：其他 GNN 階段（方案 A 時）
```

---

## ✅ 檢查清單

在開始實驗前：
- [ ] 驗證數據集是否完整（20 個 binclass 數據集）
- [ ] 確認 GPU 驅動程序和 CUDA 版本正常
- [ ] 檢查磁盤空間是否充足（預計需要 10-20GB）
- [ ] 測試單個 main.py 運行是否正常
- [ ] 備份重要代碼和腳本

在運行過程中：
- [ ] 監控 GPU 溫度和記憶體使用
- [ ] 定期檢查結果文件生成進度
- [ ] 保存運行日誌（用於調試）

在完成後：
- [ ] 驗證所有結果文件完整性
- [ ] 生成匯總統計
- [ ] 繪製對比圖表
- [ ] 分析主要發現

---

**文件創建日期**：2026-01-03  
**狀態**：待執行  
**優先級**：高