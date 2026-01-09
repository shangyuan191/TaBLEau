# ALIGN：10 種表格模型的模型類型與 GNN Injection Stage 增益機制（論文用整理）

> 本文件目的：提供「10 種模型各自屬於什麼類型」以及「在這類模型中，為何在不同 GNN injection stage 可能有/無增益」的詳細論述。
>
> 範圍說明：本文件聚焦於模型類型與機制層級的解釋，不納入任何結果數字或 dataset-by-dataset 的實驗整理。

---

## 1. TaBLEau 的「五階段 + 兩個前置階段」框架（GNN 可插入的位置）

TaBLEau 將表格模型的資料流抽象成以下 stage（本研究把 GNN 插入視為一種可控的 inductive bias/representation transform）：

- **none**：不插入 GNN，走原始模型路徑。
- **start**：在資料仍是 DataFrame（或等價的原始表格）時，先做一個「圖式特徵轉換/預處理」，再交給後續 pipeline。
- **materialize**：將 DataFrame 物化成 tensor / TensorFrame / DataLoader 後，在「已數值化的特徵張量」上插入 GNN（通常仍是離線轉換）。
- **encoding**：在模型把每個欄位/特徵編碼成 embedding/token 之後，插入 GNN（多數是端到端 joint training）。
- **columnwise**：模型完成欄位間交互（注意力/MLP blocks/TabNet steps）後，再插入 GNN（端到端 joint training 或 head-level mixing）。
- **decoding**：將 GNN 當作最後的 decoder/head（把 row-level 表徵映射到輸出），或用 GNN head 取代/增強原 decoder。

在 TaBLEau 的設計語境中，一個核心觀點是：**不同 stage 對應「圖的節點特徵」含義不同**，也就改變了 GNN message passing 的行為。

---

## 2. 先講清楚：表格上的 GNN injection 在做什麼？（機制層級）

在這個專案裡，絕大多數 injection 都可以被理解成：

1. **把每一筆資料（row/sample）當作節點**
2. 用 kNN 或可學習的動態建圖（例如 DGM 類）建立節點鄰接關係
3. 用 GCN/GCN-like message passing 對 node embedding 做平滑/聚合
4. 把更新後的 node embedding 以某種方式「回灌」回原模型

因此，GNN injection 的增益或失效，通常由以下因素共同決定：

### 2.1 圖結構是否「語義對齊」任務
- 若 kNN 是在「能表徵 label 相關相似性」的空間建立，message passing 等價於一種溫和的 label smoothing / local averaging，常能提升泛化。
- 若 kNN 建在與任務不對齊、或被噪聲主導的空間，message passing 會把不相似樣本混在一起，造成 representation collapse 或 decision boundary 被抹平。

### 2.2 GNN 被插入的位置：是在「輸入端」還是「決策端」
- **輸入端（start/materialize）**：更像是對原始特徵做一個「圖式去噪/重構/平滑」的前處理。
- **決策端（decoding）**：更像把 GNN 當成一個 supervised head，讓它直接在 row-level embedding 上擬合目標；此時即使圖品質不完美，也可能透過監督信號把圖與任務對齊。
- **中間（encoding/columnwise）**：屬於高風險高報酬。若圖與任務對齊，能把全模型的表徵拉到更「局部結構友善」的方向；若不對齊，整個 backbone 可能被拖偏。

### 2.3 目標函數（supervised vs self-supervised）是否和 message passing 相容
- **純 supervised**（分類/迴歸）：GNN 的平滑 bias 往往與「同類樣本相近」假設相容，尤其在少樣本/噪聲標註/特徵稀疏時。
- **reconstruction 類自監督**（mask/recon/denoise）：GNN 可能有幫助（把可重構的局部群集拉近），也可能害（把細節洗掉，讓 reconstruction 變得不精確）。
- **對比式自監督（contrastive）**：GNN 的鄰域聚合會降低 instance discrimination 所需的「樣本可分性」，因此常與對比目標衝突（除非 GNN 被限制在 downstream head 或採用更合適的圖/正負樣本設計）。

### 2.4 transductive vs inductive：資料切分與建圖的時機
- 若在 start/materialize 把 train/val/test 合併建一張圖，屬於 **transductive flavor**（測試節點參與了表徵更新），可能提高 performance 但也提高評估偏差風險。
- 若只用 train 建圖再推到 val/test（或在 mini-batch 內建圖），屬於 **inductive flavor**，更貼近部署情境，但圖訊息較弱。

本研究的 stage 設計，實際上就是在「更 transductive 的離線轉換」與「更 inductive 的端到端 joint training」之間切換。

---

## 3. 兩大 injection 風格：ExcelFormer-aligned vs Simple kNN+GCN

在 TaBLEau 的 10 模型中，GNN injection 主要分成兩種實作風格：

### 3.1 ExcelFormer-aligned（偏「token→row pooling→建圖→GCN→回寫 token」）
典型特徵：
- 把表格編成 token（欄位 embedding）後，先透過 attention（或類似機制）產生 row-level 表徵
- 在 row-level 表徵上建圖（kNN 或 DGM 動態圖）
- 用 GCN 更新 row-level 表徵
- 再把 row-level 的資訊「解碼/回寫」回 token 表徵（常見：用 attention decode + residual gate）

這種設計多出現在 pytorch-frame 系列模型：
- ExcelFormer / FTTransformer / TabTransformer / TabNet / TromPT / ResNet（wrapper 內做了對齊改造）

### 3.2 Simple kNN+GCN（偏「把 row 當節點，直接做特徵轉換或 head」）
典型特徵：
- start/materialize：把所有 row 合併後建 kNN 圖，訓練一個 GCN 做 reconstruction 或 supervised 預測，最後把 GCN 的輸出拼回 DataFrame 作為新特徵
- encoding/columnwise：常見做法是在 batch 層級把 embedding 做一個 kNN+GCN 的 mixing（有時甚至不是端到端學到的圖）
- decoding：用 GNN 當 head

這類設計多出現在 custom 系列：
- SubTab / VIME / TabM / SCARF

---

## 4. 十個模型逐一整理（模型類型 × stage 機制）

以下每個模型都用同一個模板描述：
1) 模型類型（監督/自監督、row-wise/token-wise、pretext 類型）
2) 在 TaBLEau 的實作中，GNN 大致怎麼接
3) 為何在各 stage 可能有/無增益（以機制解釋）

---

### 4.1 ExcelFormer

- **模型類型**：監督式（supervised）token-wise 表格 Transformer；核心是欄位編碼 + 多層 column-wise attention，最後由 decoder 輸出。
- **實作位置**：[models/pytorch_frame/excelformer.py](../models/pytorch_frame/excelformer.py)
- **GNN 注入風格**：ExcelFormer-aligned（attention pooling →（kNN/DGM）→ GCN → attention decode → residual/gate 回灌）。

**為何各 stage 可能有/無增益**：
- **start / materialize**（離線轉換）：
  - *可能有增益*：等價於把原始特徵做「圖式去噪/平滑」，在少樣本或特徵噪聲較高時可能提高 downstream 的可分性。
  - *可能無增益/甚至有害*：因為此階段建圖是基於原始或淺層特徵，圖語義未必對齊 label；平滑可能把關鍵區分訊號洗掉。
- **encoding**（joint training，較常見的好位置）：
  - *可能有增益*：token 已經是模型學到的 embedding，圖更容易對齊任務；把 row-level 的鄰域信息回寫回 token，可視為一種結構化的 inductive bias。
  - *無增益情境*：若資料本身不具備「局部相似→同標籤」假設，或 embedding 空間的最近鄰不穩定，則會引入噪聲。
- **columnwise**（在欄位交互後再建圖）：
  - *可能有增益*：此時的 row 表徵通常更接近決策邊界，kNN 更容易對齊 label；GNN 相當於在「更語義化的空間」做局部聚合。
  - *可能無增益*：如果 columnwise attention 已充分捕捉交互，GNN 可能只是多一層平滑而不帶來新訊息。
- **decoding**（GNN 當 head/decoder）：
  - *可能有增益*：GNN 直接承擔 supervised mapping，可把建圖誤差透過 loss 校正。
  - *可能無增益*：若原 decoder 已很強，GNN head 會變成不必要的 inductive bias，甚至限制模型容量。

---

### 4.2 FTTransformer

- **模型類型**：監督式 token-wise Transformer（FT-Transformer family）；以 feature token + transformer blocks 做表徵學習。
- **實作位置**：[models/pytorch_frame/fttransformer.py](../models/pytorch_frame/fttransformer.py)
- **GNN 注入風格**：ExcelFormer-aligned + 允許 decoding 用 GNN 作 head。

**stage 解釋**：
- **start/materialize**：
  - 有利於「輸入端去噪」的資料；但對本來就可線性分離/特徵乾淨的資料，可能會造成過度平滑。
- **encoding/columnwise**：
  - 常是較合理的位置：FTTransformer 的中間表徵包含欄位交互資訊，kNN 建在這裡比較像在做 label-aligned smoothing。
- **decoding**：
  - 把 row embedding 丟到 GNN head，可把鄰域聚合當成最後一步的 inductive bias；但也可能把本來需要的非局部/非鄰域決策抹掉。

---

### 4.3 ResNet（tabular ResNet）

- **模型類型**：監督式 row-wise MLP/ResNet（不以 attention 做欄位交互，而是以殘差 MLP blocks 擬合）。
- **實作位置**：[models/pytorch_frame/resnet.py](../models/pytorch_frame/resnet.py)
- **GNN 注入風格**：在 wrapper 中加入 DGM/kNN + GCN（讓 row-wise backbone 也能套用類 ExcelFormer 的 row-graph mixing）。

**stage 解釋**：
- **start/materialize**：
  - 對 MLP 類模型來說，這兩個 stage 的 GNN 更像「graph-based feature engineering」。若圖對齊任務，可能補上 MLP 不具備的局部結構 bias。
  - 若圖不對齊，MLP 的優勢（自由擬合）反而會被前處理的錯誤平滑破壞。
- **encoding/columnwise**（此處對 ResNet 的意義更接近“中間層後”）：
  - 若在較深的 representation 建圖，GNN 等價於在語義空間做局部聚合，常比在原始特徵建圖更安全。
- **decoding**：
  - 用 GNN head 直接預測，能在最後一層引入「相似樣本共享信息」的偏好；但若任務需要強烈的局部例外（outlier 決策），GNN head 可能不利。

---

### 4.4 TabNet

- **模型類型**：監督式、以 sequential decision steps + sparse feature selection（attentive masks）為特色的 row-wise 模型。
- **實作位置**：[models/pytorch_frame/tabnet.py](../models/pytorch_frame/tabnet.py)
- **GNN 注入風格**：在 wrapper 中對齊「row-level 建圖 + GCN + 回寫」，使其能在不同 stage 插入。

**stage 解釋**：
- **start/materialize**：
  - TabNet 的強項是以 step-wise mask 挑出關鍵特徵；若先做圖式平滑，有可能讓 TabNet 更容易捕捉穩定訊號。
  - 但也可能害：TabNet 的 sparse selection 依賴特徵細節，過度平滑會讓 mask 難以學到真正的稀疏性。
- **encoding/columnwise**：
  - 若把 GNN 放在 TabNet 的中間表徵上，可能幫助「同類樣本」共享被選出的子特徵訊息；
  - 但也可能與 TabNet 的逐步選特徵機制競爭（模型同時想做 sparse selection，又被迫做鄰域平均）。
- **decoding**：
  - 作為 head 時，TabNet 仍能保留其內部 selection 的好處；GNN head 只負責最後 mapping，較不會破壞 TabNet 的 inductive bias。

---

### 4.5 TabTransformer

- **模型類型**：監督式 token-wise Transformer，特別強於 categorical embedding + contextualization。
- **實作位置**：[models/pytorch_frame/tabtransformer.py](../models/pytorch_frame/tabtransformer.py)
- **GNN 注入風格**：ExcelFormer-aligned（token 表徵→row pooling→建圖→GCN→回灌）。

**stage 解釋**：
- **start/materialize**：
  - 若 categorical 的關係需要“語義”才顯現（例如同一類別值在不同欄位組合下代表不同意義），過早建圖會失真。
- **encoding**：
  - 這通常是更自然的切入點：categorical token 已被 contextualize，row embedding 更能反映任務語義，kNN 較可信。
- **columnwise**：
  - 在更深的交互後做 GNN，常等價於在語義空間做局部 smooth，有機會提升泛化。
- **decoding**：
  - 當 head 時能把圖偏好限制在決策端；但若 tabtransformer 的 decoder 已能擬合複雜邊界，GNN head 可能不必要。

---

### 4.6 TromPT

- **模型類型**：監督式 token-wise（prompt-based tabular model）；通常包含 prompt token / prompt-like adaptation 機制，使模型能以提示引導表徵與預測。
- **實作位置**：[models/pytorch_frame/trompt.py](../models/pytorch_frame/trompt.py)
- **GNN 注入風格**：ExcelFormer-aligned。

**stage 解釋**：
- **start/materialize**：
  - 可能有增益的情境是：資料在原始空間就具有明顯群聚結構；GNN 先把局部群集訊號凸顯出來，再交由 prompt-based backbone。
  - 無增益/有害情境：prompt 機制本身已在表徵空間做偏置，若前處理平滑與 prompt 偏置方向衝突，會造成不穩定。
- **encoding/columnwise**：
  - 若 prompt token 使得表徵更 task-aligned，kNN 建在這裡更可靠；GNN 像是額外的 structure prior。
  - 反之若 prompt 造成表徵空間被“拉扯”（不同 prompt 造成不同局部幾何），kNN 會變得不穩定。
- **decoding**：
  - GNN head 容易把 prompt 的語義壓縮成「鄰域共享」行為；可能有益也可能讓 prompt 的細緻差異被抹平。

---

### 4.7 SCARF

- **模型類型**：以對比式自監督（contrastive learning）為核心的 tabular 表徵學習，常見流程是：
  - corruption/augmentation 產生正樣本對
  - 以 NT-Xent 類 loss 讓同一筆資料的兩個 view 接近、不同資料遠離
  - 下游用簡單線性分類器/回歸器評估（或在本框架中加上 supervised head 做 joint training）
- **實作位置**：[models/custom/scarf.py](../models/custom/scarf.py)
- **GNN 注入風格**：Simple kNN+GCN（不同 stage 可能是離線轉換、或在某處做 mixing、或用 GNN 作 decoder/head）。

**為何在這類模型中，GNN 常出現「不穩定」的增益**（機制角度）：
- **核心張力**：
  - contrastive 目標需要 instance discrimination（同筆資料的兩個 view 接近、不同筆資料遠離）。
  - 但 GNN message passing 本質上是把鄰居聚合，會讓不同節點的表徵更相似。
  - 若圖是基於表徵空間建立，GNN 會加速“群聚”，降低對比學習需要的可分性。

**stage 解釋**：
- **start/materialize**：
  - 若是以 reconstruction/平滑方式離線轉換，可能破壞 contrastive 所需要的細微差異（尤其是 corruption 設計要讓模型學會抓到特徵層級的辨識線索）。
- **encoding/columnwise**：
  - 更高風險：把 message passing 放進對比學習 backbone，會直接改變 representation geometry，容易使 NT-Xent 的梯度方向與 GNN 平滑方向衝突。
- **decoding**：
  - 相對安全的用法：把 GNN 限制在 supervised head/downstream predictor（即便 backbone 仍做 contrastive），讓 GNN 的影響只出現在決策端。

（總結一句）對比式模型要用 GNN，通常需要更細緻的設計：例如把鄰域定義成 label-aligned（但這又引入監督）、或把 GNN 只用在 downstream。

---

### 4.8 SubTab

- **模型類型**：reconstruction / subset-based 的自監督表示學習（把特徵分子集、加噪/遮蔽，學習重構或一致性），再做下游監督。
- **實作位置**：[models/custom/subtab.py](../models/custom/subtab.py)
- **GNN 注入風格**：Simple kNN+GCN；同時支援離線特徵轉換與 decoding head。

**stage 解釋**：
- **start/materialize**：
  - SubTab 的 pretext 常希望模型能從部分特徵重建整體；GNN 的鄰域聚合可以提供“相似樣本”的額外線索，等價於把 reconstruction 目標從單點變成局部群集。
  - 但若鄰域本身不可靠，會把錯誤樣本的訊息引入，造成 reconstruction 走偏。
- **encoding/columnwise**：
  - 如果此時的 embedding 已經是較穩定的 latent，GNN mixing 有助於形成群集式表徵；
  - 若 latent 必須保留細節以支援重構，過強的平滑會讓重構品質下降。
- **decoding**：
  - 作為 head 時通常更容易看到效果：GNN 直接用監督信號把“鄰域共享”導向正確的預測方向；同時不會破壞 SubTab pretext 的內部幾何太多。

---

### 4.9 VIME

- **模型類型**：mask + reconstruction 的自監督（VIME family）：
  - 自監督階段做 mask prediction 與 feature reconstruction
  - 下游再做監督學習
- **實作位置**：[models/custom/vime.py](../models/custom/vime.py)
- **GNN 注入風格**：Simple kNN+GCN（包含離線轉換與在 batch 內建圖的 mixing，以及 decoding head）。

**stage 解釋**：
- **start/materialize**：
  - 有機會幫助的原因：mask/recon 目標等同於 denoising autoencoder；GNN 可提供鄰域先驗，讓模型在缺失/遮蔽時借用相似樣本的訊息。
  - 可能無增益：若遮蔽機制本來就能迫使 encoder 學到 robust feature；多一層圖平滑只是額外 bias。
- **encoding/columnwise**：
  - 若是在 encoder 輸出的語義空間建圖，kNN 更穩定，GNN mixing 可能提升 representation 的可遷移性。
  - 但若 batch-level 建圖造成鄰域估計不穩，則會引入訓練噪聲。
- **decoding**：
  - GNN head 的優點在於：下游監督可校正圖偏差；尤其在小數據下，head-level 的鄰域共享通常比 backbone-level 更安全。

---

### 4.10 TabM

- **模型類型**：監督式集成（ensemble / multiple predictions）導向的 tabular 模型（以多個 view/backbone 或 batch-ensemble 結構提高泛化）。
- **實作位置**：[models/custom/tabm.py](../models/custom/tabm.py)
- **GNN 注入風格**：Simple kNN+GCN；start/materialize 做離線轉換，encoding/columnwise 做 batch-level mixing，decoding 可用 GNN 當 decoder。

**stage 解釋**：
- **start/materialize**：
  - 這兩階段對 TabM 最像「graph-enhanced feature engineering」：先把 row 轉成 GNN embedding，再交由 TabM ensemble backbone 擬合。
  - 增益條件：資料確實存在局部結構（相似樣本近似同分佈），GNN embedding 可以作為更平滑、更可泛化的輸入。
  - 無增益條件：TabM 本身已靠 ensemble 降低方差並擬合複雜模式，額外平滑可能只會限制表達能力。
- **encoding/columnwise**：
  - 若在 backbone 的中間層做 GNN mixing，可能與 ensemble 的多樣性假設衝突：
    - ensemble 希望不同子模型/子路徑保留差異
    - GNN mixing 會把樣本表徵往群聚方向拉
  - 因此中間插入常屬「看資料」的高風險位置。
- **decoding**：
  - 把 GNN 放在最後，通常較不會破壞 TabM 的 ensemble 表徵；它更像是最後一步的局部平滑決策。

---

## 5. 寫進論文時可用的「總結句型」

你可以把上面機制濃縮成論文裡的通用敘述（可直接引用/改寫）：

- **監督式（supervised）模型**中，GNN injection 的主要作用是把「相似樣本共享資訊」當成 inductive bias。當鄰域圖與任務語義對齊時（尤其在少樣本或高噪聲資料），GNN 在 encoding/columnwise/decoding 往往能提升泛化；反之在輸入端（start/materialize）過早平滑可能洗掉關鍵判別訊號。
- **重構式（reconstruction-based）自監督模型**中，GNN 可視為一種結構化 denoising prior：它能在遮蔽/缺失/噪聲情境下提供鄰域補全訊息，但也可能因過度平滑而損害重構細節；因此其增益高度依賴圖品質與重構目標對“細節”的需求。
- **對比式（contrastive）自監督模型**中，GNN 的鄰域聚合傾向降低樣本可分性，可能與 instance discrimination 的訓練目標產生張力；相對更穩健的做法是將 GNN 限制在 decoding/head，讓監督信號將鄰域共享導向任務相關的方向。

---

## 6. 文件索引（TaBLEau 內對應實作）

- ExcelFormer: [models/pytorch_frame/excelformer.py](../models/pytorch_frame/excelformer.py)
- FTTransformer: [models/pytorch_frame/fttransformer.py](../models/pytorch_frame/fttransformer.py)
- ResNet: [models/pytorch_frame/resnet.py](../models/pytorch_frame/resnet.py)
- TabNet: [models/pytorch_frame/tabnet.py](../models/pytorch_frame/tabnet.py)
- TabTransformer: [models/pytorch_frame/tabtransformer.py](../models/pytorch_frame/tabtransformer.py)
- TromPT: [models/pytorch_frame/trompt.py](../models/pytorch_frame/trompt.py)
- SCARF: [models/custom/scarf.py](../models/custom/scarf.py)
- SubTab: [models/custom/subtab.py](../models/custom/subtab.py)
- VIME: [models/custom/vime.py](../models/custom/vime.py)
- TabM: [models/custom/tabm.py](../models/custom/tabm.py)
