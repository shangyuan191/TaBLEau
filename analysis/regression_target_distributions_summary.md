## 回顧：Regression 目標分布（target distribution）對有 GNN 插入模型表現的影響

日期：2025-11-06

來源檔案（主要參考）
- `analysis/regression_target_distributions_analysis/target_distribution_summary.csv`
- `analysis/regression_target_distributions_analysis/target_distribution_classified.csv`
- `analysis/regression_target_distributions_analysis/primary_label_table.md`
- `analysis/regression_target_distributions_analysis/model_vs_primarylabel_scores.csv`
- `analysis/regression_target_distributions_analysis/model_ranks_by_primary_label_rmse_lower_is_better.csv`

目的

評估：當任務為回歸（regression）時，是否因為 target 欄位的分布特性不同，而導致有 GNN 插入的模型（不同基礎模型與不同插入階段）的相對表現出現系統性差異。

主要觀察（重點式 bullet points）
- 資料集已被自動分類為主要 target-distribution 類別：`approx_normal`, `constant`, `heavy_tailed`, `highly_skewed`, `moderately_skewed`, `multimodal`（見 `primary_label_table.md`）。每類別包含數個代表性 dataset，並附有分布統計（skewness、kurtosis、peaks、n 等）。
- 從群組化的平均排名（以 RMSE 排名，越小越好）觀察到：不同 GNN 變體在不同 target-distribution 類別中的平均相對排名有系統性差異，表示 target 分布屬性會影響 GNN 插入後的效果。
- 具體趨勢摘要（描述性，基於 `model_ranks_by_primary_label_rmse_lower_is_better.csv` 與 `model_vs_primarylabel_scores.csv`）：
  - multimodal（多峰）：`resnet_0.8_decoding` 與 `trompt_0.8_decoding` 在 multimodal 類別上平均排名低（表現較好）；部分 `subtab` 變體在 multimodal 上表現較差（或更變動）。這表示在多峰 target（可能代表多個子群／regime）中，能捕捉複雜模式的變體更有優勢。
  - heavy_tailed（重尾）：若 target 有長尾或常見極端值，部分 `subtab`（尤其 `subtab_0.8_start`）與 `trompt` 的變體平均排名較好；一些 `resnet` 變體在 heavy_tailed 類別上表現不穩定或較差。
  - constant（近常數目標）：`trompt_0.8_decoding` 在 constant 類別表現特別好（mean rank 約 1.67），暗示某些變體在低變異/近常數情境下更穩健。
  - approx_normal / moderately_skewed：不同變體在這些較接近常態的分布下各有優勢，沒有一個變體在所有此類資料集上都穩佔上風（需以 per-dataset pairwise 分析確認）。
- 不同「GNN 插入階段（stage）」會改變結果：相同基礎模型但在 encoding/decoding/columnwise/start/materialize 插入 GNN 的效果並非等同，stage-specific 的差異在多個分布類別中可見。

結論（簡短、可行的判斷）
- 結論 1：target 分布的性質確實會影響有 GNN 插入模型在回歸任務中的相對表現——也就是說，GNN 的有無與插入方式，與目標欄位的分布互動，會改變模型排序。
- 結論 2：一般模式：
  - 多峰（multimodal）資料集較可能從像 `decoding` 類的 GNN 插入受益（resnet/trompt 顯示穩健表現）。
  - 重尾（heavy_tailed）資料集則可能由某些 subtab 或特定插入階段受益（對 outlier / 長尾處理較穩定）。
  - 常數或非常低變異目標（constant）情況需特別處理，有些變體在此類表現良好，但這類資料集本身對模型比較的資訊量有限。
- 結論 3：目前結果屬探索性（descriptive），尚需配對的統計檢定與效果量報告，以證實跨分布類別的系統性差異是否具顯著性與實務意義。

主要限制（必讀）
- 目前分析以 dataset-level 的平均排名（rank）為主，未對每個 dataset 做成對（paired）統計檢定；平均排名可能被 group 內大 dataset 或少數極端 dataset 所主導。
- target-distribution 的分類是啟發式（heuristic）產生，部分 dataset 可能位於分類邊界或被錯誤分派（如 multimodal vs heavy-tailed 的判別在某些案例會模糊）。
- 當前使用的評估度量為 RMSE（rank）；改用 MAE 或 R2 可能會改變結論。

建議的後續步驟（以取得更具說服力的證據）
1. 對每一 dataset 進行 paired performance comparison（每個 dataset 上計算 baseline vs 每個 GNN 變體的差值，或 %change）。
2. 在每個 primary_label group 上做匯總統計（例如：Wilcoxon signed-rank test 或 Friedman test + Dunn post-hoc），並計算效果量（Cliff's delta）來衡量實務大小。
3. 做敏感度分析：匯總時改用不同的加權策略（例如按 dataset n 加權 vs 皆等權）以檢視結果穩健性。
4. 對 constant 類 dataset 單獨處理或排除（因近常數 target 使比較失去意義）。
5. 視覺化建議：在每個 primary_label 下繪製「per-dataset 的變體 vs baseline 差分箱型圖」，與變體在該 group 的散點（每點為一 dataset），以便辨識一致性與異常值。

我可以代為執行（如果你同意）
- 若你要我進一步執行配對分析並產出 report，請回覆「Run paired analysis」並提供：
  - 每個 primary_label 的最小 dataset 數（例如要求 group 至少包含 3 或 5 個 dataset 才執行統計），
  - 是否採用效果量（建議 Cliff's delta）以及事後檢定方法（建議 Dunn's test）。

參考／附註
- 原始分類與每 dataset 的統計詳見 `analysis/regression_target_distributions_analysis/` 目錄下的 CSV 與 `primary_label_table.md`。

---

（本檔為描述性總結；如需我執行下一步的配對統計分析與效果量報告請回覆指示）
