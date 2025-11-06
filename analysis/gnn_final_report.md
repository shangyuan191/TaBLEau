# GNN 增強效果總結報告（簡體/中文）

說明：本文基於已解析的排名表格（來源：`gnn_enhancement_analysis/*_gnn_enhancement.md`），整理至 `analysis/gnn_enhancement_parsed.csv`，再由 `scripts/summarize_gnn_parsed.py` 匯總，輸出以下結果與證據。主要比較規則如下：
- few-shot 類別：`ratio` 字串以 `0.05/...` 開頭（parser 標記為 `few`）；full-sample 類別：以 `0.8/...` 開頭（標記為 `full`）。
- 以平均排名 `avg_rank` 作比較（數值越小越好）。若某模型在同一群組（group）中存在非 `none` 的 GNN 變體在 few-shot 下的 best avg_rank 小於該模型同群組的 few-shot baseline (`gnn_stage=='none' & ratio_cat=='few'`)，則視為在該群組下 few-shot GNN 優於 few-shot baseline。

來源檔案：
- 解析後的表格：`analysis/gnn_enhancement_parsed.csv`
- per-model 機器摘要：`analysis/gnn_summary.json`、`analysis/gnn_summary.md`
- stage × dataset-type 匯總：`analysis/gnn_stage_dataset_summary.csv`

----

## 一、總覽結論（逐點）

1) few-shot GNN 相較於相同模型的 few-shot baseline：
   - 明顯正面（在多個群組能勝出）的模型：
     - `resnet`：在 10 個群組中有 9 個群組顯示 few-shot GNN 優於 few-shot baseline（參見 `analysis/gnn_summary.json` 中 `resnet.examples_vs_fewshot_baseline` 與 `analysis/gnn_summary.md`）。範例：
       - group=`large_datasets+binclass+numerical`，best GNN（`decoding`，avg_rank=1.0）優於 baseline_few=2.0（來源：`/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis/resnet_gnn_enhancement.md`）
       - group=`small_datasets+regression+numerical`，best GNN（`decoding`，avg_rank=3.39）優於 baseline_few=4.67（來源同上）。
     - `subtab`：6 / 10 群組顯示勝出（參見 `analysis/gnn_summary.json`）。範例：group=`small_datasets+binclass+balanced`，`columnwise` avg_rank=3.4 vs baseline_few=4.0（來源：`/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis/subtab_gnn_enhancement.md`）。
     - `fttransformer`、`tabnet`、`tabm`、`tabtransformer`、`trompt`：分別在多個群組出現 few-shot GNN 優於 few-shot baseline（具體數量請見下方逐模型細表）。
   - 幾乎沒有或很少正面效果的模型：`excelformer`、`scarf`、`vime` （在本次匯總中 few-shot GNN 贏的群組數為 0）。

2) few-shot GNN 相較於相同模型的 full-sample baseline：
   - 部分模型的 few-shot GNN 甚至能優於其 full-sample baseline（例如 `resnet` 在 9 個群組、`tabnet`/`tabm` 在數個群組）。這表示在某些群組中，加入 GNN（few-shot 設定）能帶來比直接在 full-sample (ratio_cat='full') 訓練下的 none baseline 更好的平均排名。
   - 證據與範例：請見 `analysis/gnn_summary.json` 中各 model 的 `examples_vs_full_baseline` 條目（每一條目包含 `group`, `baseline_full`, `best_gnn_few`, `row.source_file`）。

3) few-shot GNN 是否勝過參考模型（tabpfn, t2g-former, tabgnn, xgboost, catboost, lightgbm）：
   - 統計結果（summary）：若干模型在多個群組中能在 few-shot 設定下擊敗參考模型（例如 `resnet` 在 9 個群組 beat references（`vs_references_fewshot_better_count`=9），`fttransformer`=5 等）。
   - 這代表 few-shot 的 GNN 變體在部分情況下能超越這些普遍表現良好的 reference 模型（但並非一致在所有群組中都優勢）。請見 `analysis/gnn_summary.json` 的 `vs_references_fewshot_better_count` 欄位與 `analysis/gnn_enhancement_parsed.csv` 中相對的具體排名列以驗證。

4) few-shot GNN 是否勝過參考模型在 full-sample：
   - summary 中 `vs_references_full_better_count` 多數為 0（在本次自動化匯總中，few-shot GNN 在 full-sample 參考比較上較少出現勝出的情況），代表在 full-sample 設定下 reference 模型較為穩健；若要更嚴謹比較，建議直接比對相同 ratio_cat（full）的 best GNN 與各參考模型的 avg_rank。

----

## 二、逐模型重點（摘錄自 `analysis/gnn_summary.json`，各行均含證據路徑）

（我列出每個模型的三個要點：群組數、few-shot beat few-shot baseline 的群組數與 1–3 個具體範例）

- resnet
  - groups considered: 10
  - few-shot GNN > few-shot baseline: 9
  - few-shot GNN > full-sample baseline: 9
  - 範例證據（從 `analysis/gnn_summary.json`）：
    - group=`large_datasets+binclass+numerical`：best_gnn_few=1.0 (gnn_stage=`decoding`) vs baseline_few=2.0；來源：`/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis/resnet_gnn_enhancement.md`（請於檔案中搜尋該 group 表格列）
    - group=`small_datasets+regression+numerical`：best_gnn_few=3.39 (decoding) vs baseline_few=4.67；來源同上

- subtab
  - groups considered: 10
  - few-shot GNN > few-shot baseline: 6
  - few-shot GNN > full-sample baseline: 3
  - 範例證據：group=`small_datasets+binclass+balanced`，best_gnn_few=3.4 (columnwise) vs baseline_few=4.0；來源：`/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis/subtab_gnn_enhancement.md`

- fttransformer
  - groups considered: 10
  - few-shot GNN > few-shot baseline: 5
  - few-shot GNN > full-sample baseline: 3
  - 範例證據：group=`small_datasets+binclass+balanced`，best_gnn_few=3.67 (columnwise) vs baseline_few=4.0；來源：`/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis/fttransformer_gnn_enhancement.md`

- tabnet
  - groups considered: 10
  - few-shot GNN > few-shot baseline: 4
  - few-shot GNN > full-sample baseline: 7
  - 範例證據：group=`large_datasets+binclass+numerical`，best_gnn_few=1.0 vs baseline_few=2.0；來源：`/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis/tabnet_gnn_enhancement.md`

- tabm
  - groups considered: 10
  - few-shot GNN > few-shot baseline: 3
  - few-shot GNN > full-sample baseline: 6
  - 範例證據：group=`large_datasets+binclass+numerical`，best_gnn_few=1.0 vs baseline_few=1.7；來源：`/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis/tabm_gnn_enhancement.md`

- tabtransformer
  - groups considered: 10
  - few-shot GNN > few-shot baseline: 3
  - few-shot GNN > full-sample baseline: 4
  - 範例證據：group=`large_datasets+binclass+numerical`，best_gnn_few=1.0 vs baseline_few=1.6；來源：`/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis/tabtransformer_gnn_enhancement.md`

- trompt
  - groups considered: 10
  - few-shot GNN > few-shot baseline: 3
  - few-shot GNN > full-sample baseline: 6
  - 範例證據：group=`large_datasets+binclass+numerical`，best_gnn_few=1.0 vs baseline_few=1.8；來源：`/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis/trompt_gnn_enhancement.md`

- excelformer
  - groups considered: 10
  - few-shot GNN > few-shot baseline: 0
  - few-shot GNN > full-sample baseline: 3
  - 範例證據（full-baseline）：group=`small_datasets+regression+balanced`，best_gnn_few=4.8 vs baseline_full=5.8；來源：`/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis/excelformer_gnn_enhancement.md`

- scarf
  - groups considered: 10
  - few-shot GNN > few-shot baseline: 0
  - few-shot GNN > full-sample baseline: 2

- vime
  - groups considered: 10
  - few-shot GNN > few-shot baseline: 0
  - few-shot GNN > full-sample baseline: 3

----

## 三、stage × dataset-type 趨勢（摘要，詳見 `analysis/gnn_stage_dataset_summary.csv`）

觀察重點：
- 在 small dataset 類群（特別是 small/classification/numerical 與 small/regression/*）中，`decoding`、`encoding`、`columnwise` 這些 stage 出現較高的「勝過 few-shot baseline」比例（例如 small/classification/numerical：decoding prop_better=0.075, encoding prop_better=0.075，見 CSV）。
- 在 large dataset 與多類（multiclass）或分類任務中，雖然出現較多的嘗試，但比例一般較低（prop_better 多為 0.01–0.02 範圍），表示 GNN 的邊際改善在大型資料群組裡較少。

建議：若以有限資源（few-shot）嘗試加入 GNN，優先在 small datasets、分類（classification）或 regression 的 small 資料集上嘗試 `decoding` / `encoding` / `columnwise` 插入策略，因為這些組合在匯總中顯示較高的成功率。

----

## 四、如何追溯與驗證（操作說明）
1. 若要驗證某條證據，打開 `analysis/gnn_enhancement_parsed.csv`，搜尋對應的 `model` 與 `group`，會看到該群組內所有 competitor / ratio / gnn_stage / avg_rank 的列。該 CSV 的 `source_file` 欄位會指向原始 Markdown 報告（例如 `/home/shangyuan/ModelComparison/TaBLEau/gnn_enhancement_analysis/resnet_gnn_enhancement.md`），可在該 MD 中看到原始表格上下文。
2. 本報告中每個所舉的範例（group）其完整欄位資訊已包含在 `analysis/gnn_summary.json` 的 `examples_vs_fewshot_baseline` 或 `examples_vs_full_baseline` 中；若需批次輸出所有證據列，我可以再產生 `analysis/gnn_evidence_rows.csv`，把每一筆勝出對應的完整表格列導出。

----

## 五、建議的下一步（可選）
- 若要更嚴謹的統計證明，對每個 model+group 做 paired test（例如比較每資料集的性能值而非平均排名），或定義一個最小改進閾值（例如 avg_rank 改善至少 0.5）以避免極小差異被計為勝利。
- 如需圖表化（bar/heatmap）我可以把 `analysis/gnn_stage_dataset_summary.csv` 與 per-model勝率畫出 PNG 檔並放到 `analysis/figs/`。
- 如果你要我產生完整的 `analysis/gnn_evidence_rows.csv`（所有被視為 few-shot GNN 勝出的列），我可以立刻產出。

----

## 附錄：主要檔案位置
- 解析後表格：`analysis/gnn_enhancement_parsed.csv`
- 機器摘要（JSON）：`analysis/gnn_summary.json`
- 機器摘要（MD）：`analysis/gnn_summary.md`
- 本最終報告：`analysis/gnn_final_report.md`（此檔）
- stage×dataset 匯總：`analysis/gnn_stage_dataset_summary.csv`

----

## 六、跨模型觀察（分類） — 基於 `gnn_enhancement_analysis/all_models_ranking_by_classification.md`

以下結論直接來源於 `gnn_enhancement_analysis/all_models_ranking_by_classification.md`（檔案內對每個分類子群組列出 競爭者 / ratio / gnn_stage / 平均排名 / 資料集數 的總排名），我把可重複觀察列在下面並附上最直接的證據位置（群組名稱 + 檔案）。若需原始表格上下文，請打開該 MD 檔並搜尋對應群組名稱。

- 大型分類（large_datasets+binclass+numerical）由傳統/強 baseline 與少數幾個模型主導：
  - top 5 分別為：`tabpfn (none, full)`、`t2g-former (none, full)`、`catboost (none, full)`、`resnet (none, full)`、`lightgbm (none, full)`（見檔案中 section `large_datasets+binclass+numerical` 的排名表，該組第1–5名）。這代表在大數據、二元分類且數值特徵的場景，reference models 與部分 no-GNN baseline 仍居上位（證據：`all_models_ranking_by_classification.md`，group=`large_datasets+binclass+numerical`，前五名）。

- GNN 插入階段出現的模式（跨模型）：
  - `encoding` / `decoding` / `columnwise` 常見於排名靠前的 GNN 變體；例如 `resnet` 在多個 large/small 群組以 `encoding` 或 `decoding` 登上前段（見多個 group 行，例如 `large_datasets+binclass+numerical` 中 `resnet (encoding/decoding/columnwise)` 的排名）。
  - 在 small datasets 的情境（classification 或 regression），一些專門的表現良好模型（如 `subtab` 的 few/rate=0.05 variants、`vime` 的 encoding）會以 GNN 變體佔優（見 `small_datasets+regression+numerical`、`small_datasets+binclass+numerical` 的前幾名）。

- small datasets（classification/numerical）傾向：
  - `small_datasets+binclass+numerical` 的總排名顯示 `vime (encoding, ratio=0.8)`、`tabpfn (none)`、以及 `trompt (none)` 在前列；同時多個 GNN 化的 `resnet (encoding)` 也靠前（見 group=`small_datasets+binclass+numerical`）。這說明在 small & numerical 的二元分類中，既有的 few-shot baseline 與某些 GNN 插入（encoding/decoding）都能帶來優勢，視模型與 ratio 而定。

- small datasets（balanced / categorical）傾向：
  - `small_datasets+binclass+balanced` 的排名裡，`catboost` 和 `scarf`（在某些 stage，如 `materialize` / `start`）名次靠前，代表在一些小型且類別較平衡的分類資料集，傳統樹模型與特定 transformer-like 模型仍相當競爭（見 group=`small_datasets+binclass+balanced` 的前 6 名）。

- few-shot (ratio=0.05 開頭) vs full (ratio=0.8 開頭) 的跨模型分布：
  - 在多個 large-dataset 組（例如 `large_datasets+binclass+numerical`、`large_datasets+multiclass+numerical`），占優者多為 `ratio=0.8/0.15/0.05`（full）下的 none baseline 或 reference models（例如 `tabpfn (0.8)`、`t2g-former (0.8)`），而 few (0.05) 的變體通常名次靠後或分布更分散（證據：檔案內每一 group 的排名表，full 類別在前段出現頻率高）。

- 哪些模型在分類任務跨群組表現穩定（簡短列舉）：
  - 非 GNN 參考/強基線穩定：`tabpfn`, `t2g-former`, `catboost`, `lightgbm`, `xgboost`（尤其在 large datasets）。證據：在 `large_datasets+*` 的多個 group 裡，這些競爭者常見於前 1–10 名。
  - GNN-friendly / 對 few-shot 敏感的模型：`resnet`（多組以 `encoding`/`decoding` 表現良好）、`subtab`（在 small/regression/numerical 與 small/regression/balanced 裡的 few variants 順位高）、`fttransformer` 與 `tabnet` 在若干群組也有可重複的正面表現（見對應 group 排名）。

- 實務建議（基於跨模型觀察）：
  1. 若使用者目標是 large dataset 的分類（numerical features），優先先 baseline test `tabpfn`, `t2g-former`, 並以它們的 none/full 表現作為參考；few-shot 加 GNN 的邊際提升不保證能超越這些 strong baselines（證據：`large_datasets+binclass+numerical` 前列）。
  2. 若資源/資料為 small dataset 且 focus 在 classification (numerical 或 balanced)，優先嘗試在 `resnet`/`subtab`/`vime` 上以 `encoding` 或 `decoding` 插入 GNN（few-shot 嘗試），因為該類組在排名表中顯示這些組合屢屢出現於前段（證據：`small_datasets+*` 分節）。
  3. 若想快速驗證 GNN 是否值得投資（快速試驗建議）：在少數代表性 small datasets 上，跑 resnet/subtab 的 `encoding` 与 `decoding` variants（few ratio），觀察 avg_rank 是否明顯優於本模型的 few-shot none baseline；若多數 dataset 有穩定改善，再擴大到更多 dataset 或更完整比對（詳見本報告第 4 節的「如何追溯」）。

註：本節所有具體排名/名次/群組皆可在 `gnn_enhancement_analysis/all_models_ranking_by_classification.md` 中逐行核對（該 MD 已列出每個 group 的完整 競爭者排名表）。若需要，我可以把每個 group 的 top-5（或 top-10）匯出為單獨 CSV 以便圖表化或交叉比對。
