# 多層感知器（MLP）分類器

[English](../en/mlp-classifier.md) · [專題首頁](../../README.md) · [講義對照](lecture-reference.md)

## 網路結構與分數

多層感知器（MLP）從投影後的特徵與二元標籤學習非線性的決策邊界。輸入寬度由訓練矩陣推得。預設架構為 K → 16 → 1。設定 `HiddenLayerSizes = [32 16]` 可改為 K → 32 → 16 → 1。各層的權重與偏差皆可訓練。

```mermaid
flowchart LR
    Features["一列 PCA 或 LDA 特徵（K 維）"] --> Normalize["啟用時套用已擬合的標準化"]
    Normalize --> Hidden["全連接隱藏層與 tanh"]
    Hidden --> Output["全連接輸出層（一個 logit）"]
    Output --> Score["原始分數，標籤 = double(score > 0)"]
```

對含 B 筆樣本的小批次，令 $A_0$ 為標準化後（若啟用）的 B × K 矩陣。第 $l$ 層的 $W_l$ 為 $n_{l-1}\times n_l$ 矩陣，$b_l$ 為套用至每筆樣本的 $1\times n_l$ 列向量。前饋計算 $H_l=A_{l-1}W_l+b_l$，每個隱藏層再計算 $A_l=\tanh(H_l)$。輸出 $t=H_L$ 是每筆樣本的一個 logit。模型隱含的女性（1）機率為 $p=\sigma(t)=1/(1+e^{-t})$。本專題尚未驗證經驗機率校準。

`predictMLP` 回傳 logit 作為連續分數。分數愈高，愈支持女性（1）。分數小於或等於零時預測為男性（0）。即使 sigmoid 的結果因浮點捨入而變成 0 或 1，logit 仍可保留原有排序。每一列都使用擬合後的固定係數獨立計分。

## 目標函數與反向傳播

對二元標籤 $y$，機率域的二元交叉熵（BCE）為 $-y\log(p)-(1-y)\log(1-p)$。實作改在 logit 域計算等價且穩定的式子。令 $m=(1-2y)t$，每筆損失為 $\max(m,0)+\log(1+e^{-|m|})$。用 `log1p` 保留微小損失。即使 logit 約為 ±1000，指數函數的輸入也不大於零。

正則化目標為 $J=\operatorname{mean}(\mathrm{BCE})+\frac{\lambda}{2}\sum_l\|W_l\|_F^2$。所有層的權重（含輸出層）均套用同一正則化係數，最後一批較小時也一樣。偏差項不受正則化。

將 softplus 對 logit 微分，得到 $\sigma(t)-y$。程式使用等價的帶符號錯類機率 $(1-2y)\sigma(m)$，保留高信心且分類正確時的微小導數。令 $r=e^{-|m|}$。當 $m\le0$ 時以 $r/(1+r)$ 計算 $\sigma(m)$，否則用 $1/(1+r)$。在輸出層除以 B 一次，即對應平均 BCE。接著由鏈式法則得到

$$
D_L=(\sigma(t)-y)/B,\qquad
D_{l-1}=(D_l W_l^{\mathsf T})\odot(1-A_{l-1}^2) \quad (l>1).
$$

權重梯度為 $A_{l-1}^{\mathsf T}D_l+\lambda W_l$，偏差梯度為 $D_l$ 的逐欄總和。程式檢查梯度尺寸與有限性，再使用由同一組前饋權重算出的梯度更新各層。程式也會檢查前饋值、目標值、標準化特徵及更新後的參數。超出支援的雙精度數值範圍時回報 `genderRecognition:MLPNumericalRange`。

## 訓練、初始化與標準化

```mermaid
flowchart LR
    Rows["訓練特徵與標籤"] --> Prep["擬合標準化並初始化網路"]
    Prep --> Batches["洗牌並分成小批次"]
    Batches --> Step["前饋、反向傳播及更新"]
    Step --> EndEpoch["記錄完整訓練資料的 BCE 與目標"]
    EndEpoch -->|"還有下一輪"| Batches
    EndEpoch -->|"達到設定輪數"| Model["回傳擬合模型"]
```

1. 只使用傳入的訓練列擬合逐欄平均值與樣本標準差。MATLAB 的 `std(ZTrain,0,1)` 採用分母 N−1。常數欄以欄內的觀測值作為精確平均，尺度設為 1。若有變化的欄位算出的 `std` 為零或非有限值，則先縮放殘差再重算，避免直接平方極端數值。有效的正尺度會原樣保留。設定 `Standardize = false` 時，兩個標準化陣列均為空。
2. 用 Glorot／Xavier 初始化各層（含輸出層）的權重，在 ±√(6/(輸入寬度 + 輸出寬度)) 之間均勻取樣。偏差及動量緩衝值起始為零。
3. 初始化與每個 epoch 的一次 `randperm` 共用同一個有種子的區域 `mt19937ar` 亂數流。按洗牌後的列順序分小批，包含尾端不足一批的樣本。批次設定大於樣本數時使用單一完整批次。MATLAB 全域亂數狀態保持不變。
4. 依 $v\gets\beta v-\eta\nabla J$ 及 $\theta\gets\theta+v$ 更新各權重與偏差。設定 `Momentum = 0` 即為一般梯度下降，設定 `L2Regularization = 0` 則停用懲罰項。
5. 初始化（第 0 個 epoch）及每個 epoch 結束後，計算完整訓練資料的 BCE 與正則化目標。訓練在 `MaxEpochs` 結束。兩條紀錄各有 `MaxEpochs + 1` 個值，允許波動。訓練、標準化與停止規則皆只使用傳入的訓練列。

相同資料、設定及種子會重現初始化及批次順序。在本機驗證中，擬合狀態與分數完全重現。結果可能因 MATLAB 版本、硬體或數值函式庫而異。唯一支援的隱藏層活化函數為 `tanh`。預設值是工程基準，訓練輪數固定。早停、超參數選擇及正式比較留待後續工作。

## 介面與模型欄位

```matlab
mlpModel = fitMLP(ZTrain, yTrain, config);
[yPred, scores] = predictMLP(mlpModel, ZValidation);
```

特徵須為完整、實數且有限的 N × K single 或 double 矩陣，每列一筆樣本且 K ≥ 1。內部採雙精度計算。訓練標籤須為對齊的 N × 1 double 欄向量，元素為 0 或 1 且兩類皆須出現。兩項推論輸出都是保留輸入順序的 N × 1 double 欄向量。`zeros(0,K)` 會得到兩個 0 × 1 欄向量。推論沿用已擬合的標準化係數，模型與亂數狀態保持不變。無效的輸入、選項或推論欄位會回報 `genderRecognition:InvalidMLPFeatures`、`InvalidMLPLabels`、`MissingMLPClass`、`InvalidMLPConfig` 或 `InvalidMLPModel`，後四者使用相同前綴。

```matlab
config = struct('HiddenLayerSizes', 16, 'Activation', "tanh", ...
    'LearningRate', 0.01, 'Momentum', 0.9, 'MaxEpochs', 300, 'BatchSize', 16, ...
    'L2Regularization', 1e-4, 'Standardize', true, 'RandomSeed', 42);
```

可省略設定或用純量結構覆寫部分欄位。隱藏層寬度、epoch 數及批次大小須為正整數。學習率須為有限正數，動量係數須在 [0, 1)，L2 係數須為有限非負數。種子須為 0 至 2³²−1 的整數，`Standardize` 須為 logical 純量。

| 欄位 | 內容 |
|---|---|
| `ClassLabels`、`ClassCounts`、`NumTrainingSamples` | 類別順序 [0 1] 與訓練計數 |
| `NumFeatures`、`LayerSizes`、`Weights`、`Biases`、`Activation` | 網路尺寸、各層的權重矩陣與偏差列、隱藏層活化函數 |
| `StandardizationMean`、`StandardizationScale`、`NumNormalizationParameters` | 由訓練資料擬合的標準化係數及儲存數量 |
| `NumClassifierParameters` | 所有權重與偏差陣列的元素總數 |
| `EpochsCompleted`、`StoppingReason` | 完成輪數與停止原因 `"MaxEpochs"` |
| `InitialTrainingLoss`、`FinalTrainingLoss`、`InitialTrainingObjective`、`FinalTrainingObjective` | 完整訓練資料的初始與最終 BCE 和正則化目標 |
| `TrainingLossHistory`、`TrainingObjectiveHistory`、`Options` | 完整訓練紀錄與實際設定，包含 `Initialization = "glorotUniform"` 及 `RandomStream = "mt19937ar"` |

陣列元素總數會以 $\sum_{l=1}^L (n_{l-1}+1)n_l$ 獨立核對。模型不儲存訓練特徵、逐筆預測、活化值或最佳化器的緩衝值。啟用標準化時儲存 2K 個係數，停用時為零。

| 輸入寬度 | K → 16 → 1 | K → 32 → 16 → 1 | 啟用標準化的係數數量 |
|---|---:|---:|---:|
| 1 | 49 | 609 | 2 |
| 40 | 673 | 1,857 | 80 |

議題 #10 可使用連續正類 logit、`NumFeatures`、`NumClassifierParameters`、`NumNormalizationParameters` 及投影模型的 `NumFittedParameters`。三種參數數量分別回報。分類器擬合及整批推論分別以秒計時，磁碟 I/O、前處理及投影在計時之外。議題 #11 應讓各流程使用相同的外層分折索引，在每折內獨立擬合各階段，並將設定選擇限定在訓練資料內。

## PCA 與 LDA 使用範例

```matlab
setup;
[images, y, subjects] = loadDataset("train");
folds = createFolds(y, subjects);
tr = folds(1).TrainIndices;
va = folds(1).ValidationIndices;
prep = fitPreprocessor(images(tr));
XTrain = transformImages(prep, images(tr));
XValidation = transformImages(prep, images(va));

pcaModel = fitPCA(XTrain, struct('NumComponents', 40));
pcaMLP = fitMLP(transformPCA(pcaModel, XTrain), y(tr));
[yPCA, scorePCA] = predictMLP(pcaMLP, transformPCA(pcaModel, XValidation));

ldaModel = fitLDA(XTrain, y(tr));
ldaMLP = fitMLP(transformLDA(ldaModel, XTrain), y(tr));
[yLDA, scoreLDA] = predictMLP(ldaMLP, transformLDA(ldaModel, XValidation));
```

執行 `run_mlp_analysis` 查看 PCA 彙總結果，或以 `run_mlp_analysis("train","lda")` 查看 LDA 彙總結果。位置引數依序為訓練路徑、特徵方法、MLP 設定、PCA 或 LDA 設定、前處理設定、折號以及是否繪製兩條訓練曲線（logical，預設 false）。可用 `run_mlp_analysis("train","pca",struct('HiddenLayerSizes',[32 16]))` 設定網路寬度。分析函式回傳彙總數值，無檔案輸出。

## 實測數值

依受試者分組的第 1 折有 160 張訓練影像（每類 80 張）與 40 張從訓練資料集中保留的影像（每類 20 張），受試者識別碼重疊數為零。正式訓練資料共 200 張 640 × 640 × 3 RGB 影像。預設 64 × 64 灰階前處理產生 200 × 4,096 的 double 矩陣，強度範圍為 [0.00157612745, 0.99418915]。

| 第 1 折診斷 | PCA + MLP | LDA + MLP |
|---|---:|---:|
| 投影後訓練／驗證特徵尺寸 | 160 × 40 / 40 × 40 | 160 × 1 / 40 × 1 |
| 網路結構 | 40 → 16 → 1 | 1 → 16 → 1 |
| 可訓練參數／標準化／投影係數 | 673 / 80 / 167,936 | 49 / 2 / 8,192 |
| 完成輪數 | 300 | 300 |
| 初始 → 最終訓練 BCE | 0.868467145 → 0.000761819203 | 0.388212996 → 0.000505941768 |
| 初始 → 最終訓練目標 | 0.869757184 → 0.00288934358 | 0.388392720 → 0.00147054127 |
| 標籤／分數尺寸 | 40 × 1 / 40 × 1 | 40 × 1 / 40 × 1 |
| 有限分數／相異分數 | 40/40 / 40 | 40/40 / 40 |
| 分數範圍 | [−9.95149305, 10.1933641] | [−9.11846970, 10.3216234] |
| 預測類別計數 [男性, 女性] | [20, 20] | [20, 20] |
| 分類器擬合／40 筆批次推論（秒） | 0.453999 / 0.006368 | 0.229166 / 0.002474 |

計時為本機單次測量，包含函式呼叫成本。兩種流程各自重跑後，擬合模型與分數完全一致。在記憶體內改動保留影像後，分數改變，前處理、投影、標準化及網路的擬合狀態維持完全一致。此折僅供數值驗證。獨立測試資料未讀取。

## 測試與講義對照

```powershell
matlab -wait -batch "run_tests;"
matlab -wait -batch "setup; run_mlp_analysis; run_mlp_analysis('train','lda');"
```

驗證環境為 Windows 上的 MATLAB R2026a Update 5（26.1.0.3346908），僅需 MATLAB 本體。第一條指令使用合成數值資料及暫存 PNG，115 項測試全部通過（原有 96 項，新增 MLP 19 項），無失敗或未完成項目。第二條指令需要本機的正式訓練資料。

梯度檢查在兩個隱藏層且 L2 非零的網路上，對每個權重及偏差分別擾動 ±$10^{-5}$。以同一個生產目標函數的中央差分對照反向傳播解析梯度。誤差分母採用 $\max(10^{-6},|g_{\mathrm{num}}|+|g_{\mathrm{analytic}}|)$，最大相對誤差為 $1.38\times10^{-10}$。一維、二維線性案例均正確分類 6 筆訓練樣本及 2 筆額外樣本。XOR 案例正確分類 4 筆訓練樣本及 4 筆額外樣本，訓練 BCE 從 0.728666155 降至 0.00761152654。

已核對本機講義第 6 章 §§6.2–6.3。合併 PDF 第 386–394 頁說明仿射輸入、非線性前饋及 XOR，第 396–420 頁說明鏈式法則、更新方式及學習曲線。也可參考 §§6.8.3、6.8.8、6.8.10、6.8.16，分別為尺度調整（435–436 頁）、初始化（445–447 頁）、動量（448–452 頁）及交叉熵（471–480 頁）。單 logit BCE、數值穩定公式、Glorot 範圍、有種子的小批次順序、動量更新慣例、只用訓練資料標準化及固定輪數，都是本專題的實作選擇。[講義對照](lecture-reference.md)
