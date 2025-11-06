## GNN × 標籤類別數（label-cardinality）分析 — 中文說明

日期：2025-11-06

本文件為先前分析的中文說明，針對 10 個基礎模型 × 5 個 GNN 插入階段（共 50 個變體）的標籤類別數影響做簡要總結。分析所用的主要產物位於 `analysis/`：

- `analysis/variant_cardinality_summary.csv` — 每個變體（variant）在不同 ratio_cat 與標籤類別分箱（2、3–10、>10）下的彙總統計（n、平均排名、中位數、標準差、四分位距）。
- `analysis/variant_cardinality_stats.csv` — Kruskal–Wallis 檢定的 H 值與 p 值，以及經 Bonferroni 校正的兩兩比較 p 值。
- `analysis/figs/variant_cardinality_boxplots/` — 每個變體對應的箱型圖（PNG）。

快速結論
- 在所有變體中，只有少數變體在標籤類別數（2、3–10、>10）間的平均排名（avg_rank）分布顯著不同（Kruskal–Wallis）。從 `analysis/variant_cardinality_stats.csv` 可見的顯著例子包括：
  - `resnet::decoding`（ratio_cat = full）：KW H = 7.877，p ≈ 0.019；兩兩比較顯示 2 vs >10 p ≈ 0.020，以及 3–10 vs >10 p ≈ 3.4e-05。
  - `subtab::columnwise`（few）：KW H = 6.864，p ≈ 0.032；主要由 3–10 vs >10 驅動（p ≈ 3.4e-05）。
  - `subtab::encoding`（few）：KW H = 6.991，p ≈ 0.030；2 vs 3–10 p ≈ 0.032，且 3–10 vs >10 p ≈ 3.4e-05。
  - `subtab::start`（few）：KW H = 19.445，p ≈ 6.0e-05（強訊號）；兩兩比較顯示 2 vs 3–10 p ≈ 3.3e-04，3–10 vs >10 p ≈ 3.4e-05。
  - `subtab::start`（full）：KW H = 13.852，p ≈ 9.8e-04；2 vs 3–10 p ≈ 0.017，2 vs >10 p ≈ 0.035，3–10 vs >10 p ≈ 3.4e-05。
  - `trompt::decoding`（full）：KW H = 34.731，p ≈ 2.9e-08（非常顯著）；兩兩比較多對顯著（例如 2 vs 3–10 p ≈ 2.6e-06、2 vs >10 p ≈ 0.0023）。

p 值解讀（簡短）：Kruskal–Wallis 用以檢驗三個標籤類別箱的分布是否不同；若 KW 顯著，則可檢查已做 Bonferroni 校正的兩兩比較 p 值，找出哪些箱間差異最顯著。

重要注意事項（關鍵 caveats）
- 備援（fallback）之每資料集排名：若某變體在 `gnn_enhancement_analysis/per_dataset_variant_ranks_mapped.csv` 中缺乏明確的 per-dataset 排名，分析會以已解析出的變體層級平均排名（parsed avg_rank）對應到該變體所屬的資料集清單（依 `per_dataset_target_value_counts_full.json` 的鍵匹配），並重複使用該平均值作為該資料集的「替代」排名。這會減少 per-dataset 的變異（因為同一 avg_rank 會重複出現），可能使非參數檢定與效果量估計偏向偵測到由樣本數差異所造成的中心趨勢差，而不是真實的 per-dataset 差異。
- 高類別數分箱的樣本量偏小：目前輸出中，3–10 與 >10 這兩箱的樣本數通常偏小（常見為 n=18 與 n=9）。樣本數過小會降低檢定力，使 p 值與效果量估計不穩定。
- 多重比較與解讀：總共約 50 個變體 × 2 個 ratio_cat，檢定數量龐大；雖然目前在 `variant_cardinality_stats.csv` 中已對個別變體的兩兩比較做 Bonferroni 校正，但對邊緣性 p 值仍需謹慎解讀，優先信任同時具備顯著性與合理樣本量的結果。

保守解讀建議（精簡）
1. 將以上列出的顯著變體視為「提示性」而非確證性發現。像 `trompt::decoding::full`、`subtab::start::few` 這類具有極小 p 值的情況值得優先追蹤，但仍需後續驗證效果大小與穩健性。
2. 建議的後續保守步驟：
   - 先套用每箱最小樣本數門檻（建議 n ≥ 10），再重新做 KW 以及事後比較，移除在 >10 分箱僅有 n=9 等脆弱情形。
   - 對於篩選後仍顯著的比較，計算兩兩的效果量（建議使用 Cliff's delta）並報告效果量等級（微弱/小/中/大）。
   - 事後檢定建議採用 Dunn’s test（搭配適當的多重比較校正），比單純用 Bonferroni 校正的 Mann–Whitney U 更為典型且合適。
   - 優先報告那些至少有部分實際 per-dataset 排名支援（非完全靠 fallback）的變體，並在彙總表中明確標示哪些變體含有大量 fallback 行為。

細節檔案（可檢視）
- `analysis/variant_cardinality_summary.csv` — 檢查每個 (variant × ratio_cat × card_bin) 的樣本數與群體統計。
- `analysis/variant_cardinality_stats.csv` — 包含 KW 與兩兩比較 p 值，`sig_kruskal=True` 的列為候選項目。
- `analysis/figs/variant_cardinality_boxplots/` — 對上述顯著變體開啟對應的 PNG，視覺上檢驗中位數與分布是否與統計結果一致。

可選的後續動作（擇一）
- 選項 A（建議）：執行保守重分析（套用 min-n ≥ 10、對顯著比較計算 Cliff's delta、用 Dunn’s test 做事後比較），並產出 `analysis/variant_cardinality_stats_conservative.csv`，同時在報告附錄中嵌入前 6 張最具代表性的箱型圖。
- 選項 B（你先前選過）：接受目前這份中文說明作為交付成果。我也可以在此 Markdown 中直接內嵌前 6 張圖檔（將 PNG 插入文件）如果你想要。
- 選項 C：我可以先對所有比較先計算效果量（Cliff's delta），依效果量大小排序並呈現前幾名變體（同時報出 p 值與群組樣本數）。

若要我執行選項 A 或 C，請確認：
- 每箱的最小樣本數（建議 10），以及
- 是否要求該變體至少含有一筆真實的 per-dataset 排名（即排除完全靠 fallback 生成的變體）。我建議至少保留有部分真實 per-dataset 排名的變體，以免結論完全建立在重複的 avg_rank 上。

附錄 — 原始顯著列（摘自 `analysis/variant_cardinality_stats.csv`）

```
resnet::decoding, full: n=(495,18,9), H=7.877, p=0.01947 — pairwise: 2 vs >10 p≈0.0200; 3-10 vs >10 p≈3.4e-05
subtab::columnwise, few: n=(495,18,9), H=6.864, p=0.0323 — pairwise: 3-10 vs >10 p≈3.4e-05
subtab::encoding, few: n=(495,18,9), H=6.991, p=0.0303 — pairwise: 2 vs 3-10 p≈0.0317; 3-10 vs >10 p≈3.4e-05
subtab::start, few: n=(495,18,9), H=19.445, p=5.99e-05 — pairwise: 2 vs 3-10 p≈3.26e-04; 3-10 vs >10 p≈3.4e-05
subtab::start, full: n=(495,18,9), H=13.852, p=9.82e-04 — pairwise: 2 vs 3-10 p≈0.0177; 2 vs >10 p≈0.0347; 3-10 vs >10 p≈3.4e-05
trompt::decoding, full: n=(495,18,9), H=34.731, p=2.87e-08 — pairwise: 2 vs 3-10 p≈2.6e-06; 2 vs >10 p≈0.0023
```

結語
- 我已依據現有 CSV 與圖形產出本中文說明。上述統計訊號值得進一步追蹤，但在做出確證性結論前，應先進行保守篩選與效果量報告（選項 A）。如果你同意，我可以立刻開始執行選項 A（預設 min-n=10），並產出更新後的 CSV 與含圖檔的簡短報告。

— 報告結束

