# PCA 特徵擷取

[English](../en/pca-feature-extraction.md) · [專案首頁](../../README.md)

主成分分析（PCA）將影像特徵向量轉換為較少的座標，並盡量保留訓練特徵的變異。擬合只使用特徵值，無須類別標籤。PCA + Bayes 與 PCA + MLP 可共用投影介面，標籤仍與特徵矩陣各列一一對應。

## 從講義的散布矩陣到精簡 SVD

已核對合併講義 PDF 第 277–290 頁，即第 3 章 §3.8.1「Principal Component Analysis」。講義推導樣本平均值與最小平方投影，選取散布矩陣的主要特徵向量，並將推導推廣至多個方向。詳見 [講義對照](lecture-reference.md)。

訓練矩陣 $X$ 每列放一筆樣本，共 $N$ 列、每列 $D$ 個特徵。先計算訓練平均 $\mu = \frac{1}{N}\sum_{i=1}^{N}x_i$，再從每列減去 $\mu$ 得到中心化矩陣 $X_c$。按此矩陣方向，講義的散布矩陣為 $S = X_c^\top X_c$，樣本共變異數矩陣為 $C = S/(N-1)$。兩者有相同的特徵向量，特徵值相差 $N-1$ 倍。

實作使用精簡奇異值分解（economy SVD）$X_c = U\Sigma V^\top$。$V$ 的各欄是主成分方向，依奇異值 $\sigma_i$ 由大到小排列。由 $X_c^\top X_c = V\Sigma^\top\Sigma V^\top$ 可得樣本共變異數的特徵值 $\lambda_i = \sigma_i^2/(N-1)$。第 $i$ 個成分的解釋變異比率為 $\mathrm{EVR}_i = \lambda_i/\sum_j\lambda_j$，前 $k$ 個的累積比率為 $\mathrm{CEVR}_k = \sum_{i=1}^k\mathrm{EVR}_i$。分母納入精簡 SVD 的完整頻譜，包含低於秩門檻及未保留的成分。精簡 SVD 省略的方向在精確算術下對應零特徵值。程式以 $(\sigma_i/\sigma_1)^2$ 計算比率，避免計算分母時超出數值範圍。

保留 $W = V(:,1{:}K)$，以 $Z = (X-\mu)W$ 取得 $N \times K$ 投影矩陣。各列皆減去同一個擬合平均值。給定 $K$ 時，主要方向可最小化訓練資料的平方重建誤差。奇異向量的正負號可反轉，重複特徵值對應的子空間內也可旋轉基底。同一子空間的解釋變異與重建結果不受這些變化影響。

## 數值秩與成分選擇

中心化後的秩最多為 $\min(N-1,D)$。SVD 後，計算嚴格大於 `max(N,D) * eps(sigmaMax)` 的奇異值個數，再以此理論值為上限。`eps(sigmaMax)` 為最大奇異值處的雙精度數值間距。冗餘特徵、重複樣本及其他線性關係都可能使秩降低。在 4,096 維下，160 筆訓練樣本的理論上限為 159，200 筆的理論上限為 199。精簡 SVD 只需至多 $\min(N,D)$ 個右奇異向量，可直接處理特徵多於樣本的情形，無須建立 $D \times D$ 的共變異數矩陣或散布矩陣。

設定使用純量結構，最多指定下列一種選擇方式。

| 設定 | 行為 |
|---|---|
| 省略或 `struct()` | 保留 `min(40, NumericalRank)` 個成分。40 維是根據先前實驗採用的工程基準。 |
| `NumComponents` | 指定正整數，最多可保留至數值秩。超出時回報 `genderRecognition:PCAComponentsExceedRank`。 |
| `ExplainedVarianceThreshold` | 在數值秩內選擇達到指定累積比率的最少成分數。比率須在 $(0,1]$，絕對容許差為 `8 * eps(1)`。無法達標時回報 `genderRecognition:PCAThresholdExceedsRank`。 |

未知、無效或衝突的選項會回報 `genderRecognition:InvalidConfig`。設定與模型以比例表示變異，分析輸出則使用百分比。維度基準、容許差與選擇策略皆是專案的工程決定。解釋變異描述訓練特徵，無法據此判定分類的最佳維度。

## 擬合與轉換

```matlab
setup;
pcaModel = fitPCA(XTrain, struct('NumComponents', 40));
ZTrain = transformPCA(pcaModel, XTrain);
ZValidation = transformPCA(pcaModel, XValidation);
```

`fitPCA` 至少需要兩筆樣本與一個特徵欄。兩個函式接受非稀疏、有限且為實數的 `single` 或 `double` 矩陣。計算使用雙精度，`transformPCA` 回傳 `double`。輸入 `0 × D` 矩陣會得到 `0 × K` 矩陣。轉換會檢查係數的形狀、有限性、各欄的範數及維度資訊。每列皆使用同一組平均值與投影矩陣，所以單筆和批次轉換結果在浮點容許差內一致。

完全固定的訓練矩陣會回報 `genderRecognition:ZeroPCAVariance`。若中心化或投影超出雙精度範圍、SVD 產生非有限值，或可用成分的共變異數特徵值因捨入成為零，會回報 `genderRecognition:PCANumericalRange`。本實作使用平均值中心化與正交正規投影。

| 擬合欄位 | 意義 |
|---|---|
| `Mean` | $1 \times D$ 訓練平均列向量。 |
| `Components` | $D \times K$ 的保留右奇異向量矩陣。 |
| `Eigenvalues` | 由大到小排列的樣本共變異數特徵值，為長度 $\min(N,D)$ 的欄向量，包含零值與數值上可忽略的值。 |
| `ExplainedVarianceRatio`、`CumulativeExplainedVariance` | 完整頻譜的解釋變異比率欄向量及累積比率欄向量。累積比率的最後一項捨入為 1。 |
| `NumericalRank`、`RankTolerance` | 可用方向數與奇異值門檻。 |
| `NumTrainingSamples`、`NumInputFeatures`、`NumFeatures` | 樣本數 $N$、原始特徵數 $D$ 與保留維度 $K$。 |
| `NumFittedParameters` | 投影需儲存的係數數量 $D + DK$。 |
| `RetainedExplainedVariance` | 保留 $K$ 個成分時的累積解釋變異比率。 |
| `Options` | `SelectionMode` 為 `"default"`、`"fixed"` 或 `"explainedVariance"`。`NumComponents`、`ExplainedVarianceThreshold` 記錄採用的設定，未採用欄位留空。 |

參數量包含所儲存的全部平均值與投影係數。正交正規條件限制統計上的獨立自由度。請將 PCA 轉換係數量與分類器擬合參數量分開記錄。真實影像擬合的 PCA 模型含有人臉衍生係數，應留在本機。

### 使用共用分折的其中一折

```matlab
[images, y, subjectIDs] = loadDataset(trainingPath);
[folds, ~] = createFolds(y, subjectIDs);

trainIdx = folds(1).TrainIndices;
valIdx = folds(1).ValidationIndices;

preprocessor = fitPreprocessor(images(trainIdx));
XTrain = transformImages(preprocessor, images(trainIdx));
XValidation = transformImages(preprocessor, images(valIdx));

pcaModel = fitPCA(XTrain, struct('NumComponents', 40));
ZTrain = transformPCA(pcaModel, XTrain);
ZValidation = transformPCA(pcaModel, XValidation);
yTrain = y(trainIdx);
yValidation = y(valIdx);
```

每折用當折訓練索引重新擬合前處理與 PCA，以固定的擬合狀態轉換當折驗證資料，避免驗證樣本影響投影。沿用索引可維持影像、特徵、標籤與受試者識別碼的對應。若訓練子集較小，請指定符合數值秩的維度，或使用解釋變異門檻模式。詳見 [資料載入與前處理](data-preprocessing.md) 與 [資料集分折](dataset-partitioning.md)。

## 本機分析與實測彙總

```matlab
setup;
run_pca_analysis;  % 以預設設定分析本機 train/ 的全部訓練資料
stats = run_pca_analysis("path/to/training-data", struct('NumComponents', 40));
foldStats = run_pca_analysis("path/to/training-data", ...
    struct('ExplainedVarianceThreshold', 0.95), struct('ImageSize', [64 64]), 1);
```

引數依序為訓練路徑、PCA 設定、前處理設定、預設受試者分組切分的折編號，以及選用的繪圖旗標。折編號傳入 `[]` 時分析全部訓練資料。第五個引數為 `true` 時，顯示個別與累積解釋變異比率，並標記保留維度與 90%、95%、99% 門檻。預設關閉繪圖。函式輸出並回傳彙總值，不寫入檔案。全訓練資料的分析用途是描述資料，交叉驗證時應逐折重新擬合。

在 Windows 的 MATLAB R2026a Update 5 上驗證。正式訓練資料共 200 張 640 × 640 × 3 RGB 影像，每類各 100 張。預設前處理產生 64 × 64 灰階的 `double` 特徵，強度在 $[0,1]$ 內（實測 0.00157613–0.994189）。第 1 折每類有 80 張訓練影像及 20 張驗證影像，兩部分的受試者識別碼無重疊。兩次擬合均產生 1 × 4,096 的平均向量與 4,096 × 40 的投影矩陣，共儲存 167,936 個擬合係數。

| 項目 | 全部訓練資料 | 第 1 折訓練 | 第 1 折驗證 |
|---|---:|---:|---:|
| 降維前 → 後 | 200 × 4,096 → 200 × 40 | 160 × 4,096 → 160 × 40 | 40 × 4,096 → 40 × 40 |
| 數值秩／理論上限 | 199 / 199 | 159 / 159 | 沿用當折訓練模型 |
| 40 維的解釋變異 | 93.331369% | 93.764618% | 沿用當折訓練模型 |
| 達 90%／95%／99% 所需維度 | 26 / 53 / 128 | 25 / 49 / 109 | 沿用當折訓練模型 |
| 降維前 → 後矩陣儲存量（位元組） | 6,553,600 → 64,000 | 5,242,880 → 51,200 | 1,310,720 → 12,800 |
| PCA 擬合時間（秒） | 0.065598 | 0.018214 | 沿用當折訓練模型 |
| PCA 轉換時間（秒） | 0.008630 | 0.004668 | 0.003003 |

時間是單次本機執行的批次量測，載入與前處理在計時範圍之外。儲存量只計特徵矩陣的內容，模型、輸入影像及暫存陣列另計。執行時間會隨硬體及暖機狀態變動。獨立測試集保留供最終評估。

## 自動化驗證

從儲存庫根目錄執行 `matlab -batch "run_tests;"`，或在 MATLAB 執行 `run_tests;`。只需 MATLAB 基礎功能。測試以獨立計算的小型樣本共變異數矩陣為參考，驗證重建與個別／批次轉換的一致性，並涵蓋數值秩、輸入邊界及 160 × 4,096 的合成矩陣。暫存的合成 PNG 影像用於驗證受試者分組後的資料對齊，以及變更驗證影像時，擬合狀態與訓練投影維持一致。
