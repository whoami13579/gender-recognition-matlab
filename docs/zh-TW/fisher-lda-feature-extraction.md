# Fisher LDA 特徵擷取

[English](../en/fisher-lda-feature-extraction.md) · [專案首頁](../../README.md)

Fisher 線性判別分析利用訓練標籤尋找一個方向，使各類平均值在投影後相距較遠、各類內部的變異較小。PCA 無需標籤，尋找整體變異較大的方向。兩種方法各自使用原始前處理特徵。LDA 為求解判別方向而分解**類內殘差**，PCA 則分解以全體平均值中心化的樣本以保留變異。

## Fisher 準則與講義對照

已查核本機合併講義 PDF 第 291–312 頁（原始投影片第 103–124 頁），即第 3 章 §3.8.2 *Fisher Linear Discriminant*。第 297–305 頁定義投影平均值、類內散布、類間散布與 Fisher 商數。第 306–307 頁推導類內散布非奇異時的傳統解。下文的嶺正則化與低秩求解是本專題處理奇異散布的實作選擇。 [講義對照與系統架構](lecture-reference.md)

令 $X_k$ 包含第 $k$ 類的 $N_k$ 筆訓練列，其中男性為 $k=0$，女性為 $k=1$（正類）。類別平均列向量為 $\mu_k=\operatorname{mean}(X_k,1)$，全體訓練平均列向量為 $\mu=\operatorname{mean}(X,1)$，類別平均差欄向量為 $\delta=(\mu_1-\mu_0)^\top$。將 $X_0-\mu_0$ 與 $X_1-\mu_1$ 依列堆疊成殘差矩陣 $R$。類內散布為各殘差外積的總和 $S_W=R^\top R$。秩為一的類間散布為 $S_B=\delta\delta^\top$。依各類樣本數加權的定義相當於將 $S_B$ 乘以正純量 $N_0N_1/(N_0+N_1)$，因此得到相同的最佳方向。

$$
J(w)=\frac{w^\top S_Bw}{w^\top S_Ww}
    =\frac{(w^\top\delta)^2}{\lVert Rw\rVert_2^2}.
$$

$S_W$ 非奇異時，最佳方向與 $S_W^{-1}\delta$ 成正比。每類的殘差列總和為零，因此 $\operatorname{rank}(S_W)=\operatorname{rank}(R)\le\min(N-2,D)$。在 $D=4096$ 時，完整訓練集 200 筆的秩至多 198，單折訓練部分 160 筆的秩至多 158，兩者的類內散布皆為奇異矩陣。兩類平均值相異時，$S_B$ 只有一個非零方向，因此二類 LDA 產生一個特徵。

## 嶺正則化與低秩求解

加入正的嶺係數 $\alpha$ 後，系統為正定。使正則化 Fisher 準則最大化的方向，與 $(S_W+\alpha I)w_{\rm raw}=\delta$ 的解成正比。

$$
J_\alpha(w)=\frac{(w^\top\delta)^2}{\lVert Rw\rVert_2^2+\alpha\lVert w\rVert_2^2},
\qquad
\alpha=\rho\,\frac{\operatorname{tr}(S_W)}{D}
      =\rho\,\frac{\lVert R\rVert_F^2}{D}.
$$

設定為純量結構，選填 `Regularization`。此值為有限且大於零的實數純量 $\rho$，預設 `1e-2` 是工程基準。$\rho$ 無單位，$\alpha$ 的單位與散布相同，為特徵單位的平方。將所有輸入一致乘以 $a>0$ 時，嶺係數與散布都乘以 $a^2$，除了捨入誤差外，方向保持相同。較小的嶺係數提高低類內散布方向的權重，包含零空間。較大的嶺係數讓方向更接近 $\delta$。

類內散布恰為零時，改用 $\alpha=\rho\lVert\delta\rVert_2^2/D$，所得方向平行於 $\delta$。若兩類平均值相同，回報 `genderRecognition:IdenticalLDAClassMeans`。散布跡須可表示（殘差非零時須為正），有效與縮放後的嶺係數須為正的雙精度正規數。若結果非有限或無法使用、Fisher 準則值非正、相對求解殘差超過 `1e-6`，或嶺係數無法表示，回報 `genderRecognition:LDANumericalRange`。可用的嶺係數範圍取決於資料。本次使用預設值，未以驗證資料或獨立測試資料調參。

若對 $R=U\Sigma V^\top$ 進行精簡奇異值分解（economy SVD），與原解方向相同的縮放解可寫成

$$
\alpha w_{\rm raw}
=V\,\operatorname{diag}\!\left(\frac{\alpha}{s_i^2+\alpha}\right)V^\top\delta
 +\bigl(\delta-VV^\top\delta\bigr).
$$

第二項保留與精簡右奇異向量正交的平均差成分。實作藉由改變求解座標，避免以兩個近乎相等的特徵空間向量相減來計算此項。先以 $c=\max(\lVert R\rVert_F,\sqrt{\alpha})$ 將 $R$ 縮放為 $A=R/c$，並設定 $b=\delta/\lVert\delta\rVert_2$ 與 $\beta=\alpha/c^2$。特徵數多於樣本數時，對 $[b\;\; A^\top]=QT$ 做精簡 QR 分解，將殘差列向量與平均差納入同一個最多 $N+1$ 維的子空間。嶺項在其正交補空間只做純量乘法，而右邊向量在該處為零，因此解落在此子空間。縮減後的右邊向量為 $T(:,1)$，殘差矩陣為 $T(:,2{:}\mathrm{end})^\top$。

在縮減後的殘差矩陣補上一列零，再做精簡 SVD，以取得縮減問題的完整右奇異方向，包含零空間方向。每個縮放奇異值 $s_i$ 對應的右邊向量係數乘以 $\beta/(s_i^2+\beta)$，再透過 $Q$ 轉回原座標。若 $N\ge D$，直接對 $A$ 做精簡 SVD 即可取得全部 $D$ 個右奇異向量。兩個做法皆透過正交因子與純量權重計算 $u=\beta(A^\top A+\beta I)^{-1}b$。實作無需建立特徵空間的散布矩陣、共變異數矩陣、單位矩陣或反矩陣。特徵數多於樣本數時，基底最多含 $D(N+1)$ 個元素，SVD 矩陣大小為 $(N+1)\times(N+1)$。

相對求解殘差為 $\lVert A^\top(Au)+\beta(u-b)\rVert_2/(\beta\lVert b\rVert_2)$，只需矩陣向量乘法。數值秩統計嚴格大於 `max(N,D) * eps(sigmaMax)` 的**未縮放殘差奇異值**個數，上限為 $\min(N-2,D)$。散布特徵值是殘差奇異值的平方。秩門檻僅用於診斷，求解時使用縮減問題的完整頻譜。判別方向正規化為單位歐幾里得範數，並取 $w^\top\delta>0$ 的符號。

## 擬合、轉換與模型欄位

```matlab
setup;
ldaModel = fitLDA(XTrain, yTrain, struct('Regularization', 1e-2));
ZTrain = transformLDA(ldaModel, XTrain);
ZValidation = transformLDA(ldaModel, XValidation);
```

輸入為完整、實數且有限的 `single` 或 `double` 矩陣，每列一筆樣本。擬合需要對齊的 $N\times1$ 雙精度標籤欄向量，且每類至少各有一筆。計算與輸出使用雙精度。轉換時對每列使用已固定的全體訓練平均值與判別方向，計算 $Z=(X-\mu)w$。轉換無需標籤、保留列順序並回傳 $N\times1$ 欄向量。欄數正確的空批次會回傳 $0\times1$。減去訓練平均值與 PCA 介面一致，且不改變 Fisher 準則。投影結果是一維特徵，供分類器使用。

| 擬合欄位 | 意義 |
|---|---|
| `Mean`、`Components` | 轉換使用的 $1\times D$ 全體平均值與 $D\times1$ 單位方向。 |
| `ClassMeans`、`ClassCounts` | $2\times D$ 類別平均值及 $1\times2$ 類別樣本數，依 `[0 1]` 排列。 |
| `WithinClassRank`、`RankTolerance` | 殘差數值秩與奇異值門檻。 |
| `RegularizationCoefficient`、`EffectiveRegularization`、`Options` | 設定的 $\rho$、計算得到的 $\alpha$ 與生效的設定結構。 |
| `NumTrainingSamples`、`NumInputFeatures`、`NumFeatures` | $N$、$D$ 與一個輸出欄。 |
| `NumFittedParameters` | 投影使用的 $2D$ 個平均值與判別方向係數。 |
| `MeanDifferenceNorm`、`WithinClassScatterTrace`、`UsedZeroScatterFallback` | $\delta$ 的範數、$\lVert R\rVert_F^2$ 與零散布替代方式旗標。 |
| `ProjectedClassMeans`、`ProjectedMeanSeparation`、`ProjectedWithinClassScatter` | 依標籤順序排列的中心化投影類別平均值、女性減男性的平均差及 $\lVert Rw\rVert_2^2$。 |
| `RegularizedFisherCriterion`、`RelativeResidual` | $J_\alpha(w)$ 與縮放後的線性系統相對殘差（解析替代解為零）。 |

當 $D=4096$ 時，投影需儲存 8,192 個係數，計數方式與 PCA 一致。統計自由度、序列化模型大小以及 Bayes 或 MLP 的擬合參數另計。訓練列、殘差與 SVD 因子只在擬合時計算。請將模型及其中的類別平均值與判別方向留在本機。

### 使用共用分折的其中一折

```matlab
[images, y, subjectIDs] = loadDataset(trainingPath);
[folds, ~] = createFolds(y, subjectIDs);
trainIdx = folds(1).TrainIndices;
valIdx = folds(1).ValidationIndices;

preprocessor = fitPreprocessor(images(trainIdx));
XTrain = transformImages(preprocessor, images(trainIdx));
XValidation = transformImages(preprocessor, images(valIdx));
yTrain = y(trainIdx);

ldaModel = fitLDA(XTrain, yTrain);
ZTrain = transformLDA(ldaModel, XTrain);
ZValidation = transformLDA(ldaModel, XValidation);
```

影像、標籤、受試者與特徵列使用相同索引。每折僅用該折訓練部分各自擬合前處理與 LDA，防止驗證資料洩漏。LDA + Bayes 與 LDA + MLP 可使用 `ZTrain` 與 `yTrain` 訓練，輸入特徵數為一。各分類器自行產生連續正類分數。 [資料載入與前處理](data-preprocessing.md) · [資料集分折](dataset-partitioning.md)

## 本機分析與實測彙總

```matlab
setup;
run_lda_analysis;  % 使用本機 train/ 做描述性擬合
stats = run_lda_analysis("path/to/training-data", struct('Regularization', 1e-2));
foldStats = run_lda_analysis("path/to/training-data", ...
    struct('Regularization', 1e-2), struct('ImageSize', [64 64]), 1);
```

第四個引數選擇預設依受試者分組的折號，使用 `[]` 代表完整訓練集。分析讀取指定的訓練資料夾，列印彙總統計並以記憶體中的結構回傳。完整訓練集的擬合用於描述資料。指定折時只以該折的訓練列擬合前處理與 LDA。

以 Windows 上的 MATLAB R2026a Update 5 驗證。正式訓練集包含 200 張 640 × 640 × 3 的 RGB 影像。預設前處理產生 64 × 64 灰階雙精度特徵，強度位於 $[0,1]$（實測訓練範圍為 0.00157613–0.994189）。第 1 折提供的受試者識別碼無重疊。兩次擬合皆使用 $\rho=0.01$，全體平均值為 1 × 4,096，類別平均值為 2 × 4,096，判別方向為 4,096 × 1，共 8,192 個投影係數。殘差形狀與擬合特徵相同，最多輸出一維。

| 指標 | 完整訓練集 | 第 1 折訓練 | 第 1 折驗證 |
|---|---:|---:|---:|
| 男性／女性樣本數 | 100 / 100 | 80 / 80 | 20 / 20 |
| 降維前後特徵尺寸 | 200 × 4,096 → 200 × 1 | 160 × 4,096 → 160 × 1 | 40 × 4,096 → 40 × 1 |
| 類內數值秩／理論上限 | 198 / 198 | 158 / 158 | — |
| 有效嶺係數 $\alpha$ | 0.0598684235 | 0.0472629789 | — |
| 投影平均值［男，女］ | −1.48221597, 1.48221597 | −1.76143457, 1.76143457 | −1.76640547, 1.82323874 |
| 投影樣本變異數［男，女］ | 1.68731962e−6, 2.30960494e−6 | 1.10045735e−6, 1.27955996e−6 | 0.155990164, 0.322504941 |
| 投影平均差（女性減男性） | 2.96443193 | 3.52286913 | 3.58964421 |
| 正則化 Fisher 準則值 | 145.822370 | 261.545739 | — |
| 相對求解殘差 | 3.73312133e−12 | 4.32189081e−12 | — |
| 降維前後特徵矩陣位元組 | 6,553,600 → 1,600 | 5,242,880 → 1,280 | 1,310,720 → 320 |
| LDA 擬合（秒） | 0.097200 | 0.030176 | — |
| LDA 轉換（秒） | 0.006294 | 0.003629 | 0.002458 |

耗時為單次批次量測，載入與前處理不計入。儲存量只計特徵矩陣。類別變異數採 MATLAB 樣本變異數，單一樣本時為零。平均差與變異數描述各批次的投影分布。分類器訓練與評估屬於後續階段。獨立測試資料集保留供最終評估。

## 自動化驗證

從儲存庫根目錄執行 `matlab -batch "run_tests;"`，或在 MATLAB 內執行 `run_tests;`。只需基礎 MATLAB。`TestLDA` 對小型滿秩與奇異案例使用獨立的稠密求解驗證，並檢查零空間辨識訊號與正則化準則。另以 160 × 4,096 的合成矩陣驗證秩與線性系統關係，並以暫存 PNG 驗證受試者分組、列對齊、逐折擬合及變更保留影像的影響。合成測試不需任何私人資料。
