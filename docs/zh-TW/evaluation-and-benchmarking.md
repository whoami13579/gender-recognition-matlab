# 評估指標與效能量測

[English](../en/evaluation-and-benchmarking.md) · [專題說明](project-requirements.md)

PCA + Bayes、LDA + Bayes、PCA + MLP 與 LDA + MLP 共用分類準確率、ROC、AUC、特徵維度、參數量、訓練時間與推論時間的定義。分析入口僅使用合成資料與基礎 MATLAB，已在 Windows 的 MATLAB R2026a Update 5 驗證。

```matlab
setup;
results = run_evaluation_analysis;  % 僅在命令視窗顯示摘要
results = run_evaluation_analysis("results/evaluation-demo");  % 匯出 JSON 與 CSV
run_evaluation_analysis("", true);  % 選擇性繪製 ROC
```

若在無圖形介面環境執行，可先將 `groot` 的 `DefaultFigureVisible` 設為 `'off'`，或將不可見圖窗中的座標軸傳給 `plotROCCurves(results, ax)`。

## 準確率、ROC 與 AUC

```matlab
[yPred, scores] = predictBayes(bayesModel, ZValidation);
% MLP 使用 [yPred, scores] = predictMLP(mlpModel, ZValidation)。
metrics = evaluateClassification(yValidation, yPred, scores);
[roc, auc] = computeROC(yValidation, scores);
```

男性標籤為 0，女性標籤為 1，並以女性（1）為正類。準確率為 `nnz(yTrue == yPred) / numel(yTrue)`，以比例儲存，顯示為百分比時乘以 100。混淆矩陣為 `[TN FP; FN TP]`，列依實際類別 `[0, 1]` 排列，欄依預測類別 `[0, 1]` 排列。兩項結果均直接採用傳入的預測標籤。Bayes 回傳後驗對數勝算，MLP 回傳原始 logit。兩者只有在 `scores > 0` 時預測類別 1，因此分數為零時預測 0。ROC 使用原始連續分數。Sigmoid 等機率轉換在浮點運算中可能飽和為 0 或 1，使原本可區分的分數變成同分。

ROC 依不同分數由高至低掃描，各閾值以 `score >= threshold` 判定為正類。起始閾值為 `+Inf`，所有有限分數均未納入，曲線從 `(0,0)` 起始。分數完全相同的樣本同時納入。在最低的有限分數閾值，兩個類別的全部樣本均已納入，曲線到達 `(1,1)`。TPR（真正率）是已納入的正類數除以實際正類總數，FPR（偽正率）是已納入的負類數除以實際負類總數。閾值為零時，ROC 會納入分數為零的樣本，分類器在這些樣本上則預測 0。排序與分組累計的時間複雜度為 O(N log N)。

AUC 以 `trapz(roc.FPR, roc.TPR)` 對依序排列的 ROC 曲線做梯形積分。亦可解讀為隨機取一對正、負樣本時，正類分數較高的機率加上同分機率的一半。完美排序的 AUC 為 1，完全反向為 0，全部同分為 0.5，標籤 `[0;0;1;1]` 配合分數 `[0;1;1;2]` 時為 0.875。完全同分的正負配對各得半分，結果與輸入順序無關。準確率可隨分類決策改變，AUC 則獨立於分類器的決策閾值。

輸入須為非空、等長、實數、非稀疏的 N × 1 欄向量。標籤接受值為 0/1 的 double、single 或 logical。分數接受有限的 double 或 single，並將 single 精確轉為 double。不支援的型別、形狀、標籤值或非有限分數會觸發 `genderRecognition:InvalidEvaluationInput`。若只有一個類別，仍可計算準確率與混淆矩陣。此時 AUC 為 NaN，三個 ROC 向量為空，`ROC.Valid` 為 false，`ROC.Status` 為 `MissingPositiveClass` 或 `MissingNegativeClass`。有效 ROC 的狀態為 `OK`。

## 擬合後的特徵維度與參數量

```matlab
counts = summarizeModelParameters(preprocessor, projection, classifier);
```

摘要從擬合後的投影模型取得原始維度 D 與保留維度 K，並檢查前處理輸出與分類器輸入是否相符。正式的維度指標為 `NumFeatures`。依解釋變異量選擇維度時，各折保留的維度可能不同。`DimensionalityReductionFraction` 為 `1 - K/D`。

- 前處理與投影分別使用各自的 `NumFittedParameters`。固定灰階轉換、影像縮放與強度正規化所學得的係數數量為零。未使用前處理器的數值資料流程可傳入 `[]`。PCA 儲存 D 個平均值與 D × K 個載荷係數。Fisher LDA 儲存 D 個平均值與 D 個方向係數。
- 分類器的正式比較採用 `NumClassifierParameters`。高斯貝氏分類器有 2K 個平均值。完整共變異數有兩組 K(K+1)/2 個對稱矩陣元素，共用共變異數有一組，對角共變異數有 2K 個變異數。以資料估計的二類先驗機率加一個獨立參數，固定等先驗則加零。衍生的分解因子與對數行列式不另計。
- MLP 按各層的 `(輸入寬度 + 1) * 輸出寬度` 加總權重與偏差項。預設 K → 16 → 1 架構共有 16K + 33 個參數。啟用標準化時，`NumNormalizationParameters` 另計 K 個平均值和 K 個尺度係數，停用時為零。已擬合的 MLP 必須提供此欄位。Bayes 的額外標準化係數為零。
- `CombinedParameterCount` 加總四項參數量，合併所儲存的轉換係數與獨立統計參數或可訓練分類參數。這是專案定義的計數方式。序列化檔案大小及數學自由度是不同的量。

前處理學得的參數量為零時，各方法的預設計數如下。

| 方法 | D → K | 投影 | 標準化 | 分類器 | 合計 |
|---|---:|---:|---:|---:|---:|
| PCA + Bayes | 4096 → 40 | 167936 | 0 | 1721 | 169657 |
| LDA + Bayes | 4096 → 1 | 8192 | 0 | 5 | 8197 |
| PCA + MLP | 4096 → 40 | 167936 | 80 | 673 | 168689 |
| LDA + MLP | 4096 → 1 | 8192 | 2 | 49 | 8243 |

## 計時規範

```matlab
trainConfig = struct('WarmupRuns', 2, 'NumRepeats', 5);
[fitTiming, model] = benchmarkOperation(@() fitBayes(ZTrain, yTrain, bayesConfig), trainConfig);
inferenceConfig = trainConfig;
inferenceConfig.BatchSize = size(ZValidation, 1);
[predictionTiming, labels, scores] = ...
    benchmarkOperation(@() predictBayes(model, ZValidation), inferenceConfig);
```

每次計時執行一次操作。先準備好無輸入參數的函式控制代碼、輸入資料與設定。預設執行兩次不計時的暖身和五次計時。每次呼叫要求相同數量的回傳值，並保留第一次計時呼叫的輸出。每次以 `tic/toc` 量測，含呼叫與回傳輸出的時間。結果包含各次耗時、實際計時和暖身次數、中位數、平均值、樣本標準差（分母為 N−1，只量一次時為零）、最小值與最大值。延遲以中位數為主。操作發生錯誤時直接回報。傳入的操作負責管理自身的亂數行為。每次訓練須以相同的記憶體內資料、超參數與種子重新擬合。MLP 擬合本身使用獨立的固定種子亂數流。若有完整流程的量測，評估使用第一次計時所得的完整模型。單獨計時所得的分類器僅供量測。推論一律使用保留的固定模型與同一批輸入。

建立結果時指定各計時範圍。完整流程須將整段操作一次直接計時。各階段中位數的加總是另一個統計量。

| 建立結果時的計時欄位 | 量測範圍 |
|---|---|
| `DatasetLoading` | 檔案搜尋、磁碟讀取、解碼及載入驗證 |
| `PreprocessingFit` | 對記憶體內的訓練影像擬合前處理 |
| `TrainingPreprocessing` | 將已擬合的前處理器套用至訓練影像 |
| `ProjectionFit` | 對前處理後的訓練特徵擬合 PCA 或 LDA |
| `TrainingProjection` | 將已擬合的 PCA 或 LDA 套用至訓練特徵 |
| `ClassifierFit` | 對記憶體內的降維特徵擬合 Bayes 或 MLP，包含 MLP 標準化 |
| `PipelineFit` | 直接量測自前處理擬合至分類器擬合的完整耗時，包含訓練資料的兩項轉換 |
| `ValidationPreprocessing` | 對記憶體內的評估批次套用固定前處理 |
| `ValidationProjection` | 將固定的投影套用至前處理後的評估特徵 |
| `ClassifierInference` | 由記憶體內的降維特徵預測標籤並計算分數，包含模型驗證及固定標準化 |
| `PipelineInference` | 直接量測固定的前處理、投影與預測的完整耗時 |
| `MetricCalculation` | 由記憶體內的輸出計算準確率、混淆矩陣、ROC 與 AUC |
| `ResultExport` | 視需要另行量測既有結果序列化與寫檔 |

分類器及完整流程的訓練與推論排除磁碟和模型讀寫、指標計算、序列化、繪圖與無關的命令視窗輸出。推論期間不擬合模型。合成示範不量測磁碟載入與匯出耗時。若須記錄既有結果的匯出耗時，可對指定的本機目錄另外量測一次寫檔。用於量測的檔案無法記錄該次寫檔本身的結果。

所有耗時以秒為單位。推論須提供 `BatchSize`，同一筆結果若包含兩種推論範圍，須採用相同的批次大小。`SecondsPerSample` 為批次耗時中位數除以批次樣本數，`MeanSecondsPerSample` 則使用批次平均耗時。兩者皆為批次推算的每筆平均耗時。單張影像的延遲須以一張的批次實測。量測批次大小與評估樣本數分別記錄。各方法使用相同批次及計時設定。依目前規劃，Issue #11 驗證批次有 40 筆，Issue #12 獨立測試批次有 100 筆。硬體、MATLAB 版本、執行緒、快取狀態與同時執行的工作都會影響量測，比較時應記錄環境與計時設定。

## 結果格式與匯出

```matlab
context = struct('Method', "pca-bayes", 'EvaluationStage', "cross-validation", 'FoldNumber', 1);
timings = struct('ClassifierFit', fitTiming, 'ClassifierInference', predictionTiming);
result = buildEvaluationResult(context, metrics, preprocessor, projection, model, timings);
files = exportEvaluationResults(result, "results/evaluation-demo");
rows = jsondecode(fileread(files.JSON));
summary = readtable(files.CSV, 'TextType', 'string');
```

`Method` 可為 `pca-bayes`、`lda-bayes`、`pca-mlp` 或 `lda-mlp`。`EvaluationStage` 可為 `synthetic`、`cross-validation` 或 `independent-test`。交叉驗證的 `FoldNumber` 為正整數，其餘階段用 `[]`。

| 第 1 版欄位 | 意義 |
|---|---|
| `SchemaVersion`、`Method`、`EvaluationStage`、`FoldNumber` | 紀錄識別、格式版本 1 及選填折次 |
| `NumTrainingSamples`、`NumEvaluationSamples`、`NumCorrect`、`Accuracy`、`ConfusionMatrix` | 擬合使用的樣本數、評估計數與比例形式的準確率 |
| `AUC`、`ROC` | AUC 及 ROC 內的 `FPR`、`TPR`、`Thresholds`、`NumPositiveSamples`、`NumNegativeSamples`、`NumDistinctScores`、`Valid`、`Status` |
| `NumOriginalFeatures`、`NumFeatures`、`DimensionalityReductionFraction` | 擬合後的維度及降維比例 |
| `NumPreprocessingParameters`、`NumProjectionParameters`、`NumNormalizationParameters`、`NumClassifierParameters`、`CombinedParameterCount` | 各類擬合參數量及其總和 |
| `Timing.Units`、`Timing.Training`、`Timing.Inference`、`Timing.Stages` | 單位 `seconds`、訓練與推論各自包含 `Classifier` 與 `Pipeline` 量測，以及上表所列的分階段量測 |
| `Configuration` | 各模型中實際生效的 `Preprocessing`、`Projection`、`Classifier` 擬合設定 |

每筆計時紀錄包含 `SamplesSeconds`、`WarmupRuns`、`NumRepeats`、`MedianSeconds`、`MeanSeconds`、`StdSeconds`、`MinSeconds`、`MaxSeconds`、`BatchSize`、`SecondsPerSample` 與 `MeanSecondsPerSample`。未量測的範圍和不適用的欄位使用 `[]`。匯出會驗證結果，只接受已知的模型設定，排除路徑、擬合係數與逐筆樣本欄位。

匯出會在指定目錄寫入 `evaluation-results.json` 與 `evaluation-summary.csv`，若同名檔案已存在則覆寫。JSON 最外層為紀錄陣列。有效曲線的兩組座標以原點起始，閾值陣列只儲存其後的有限值。若 `ROC.InitialThreshold` 為 `"positive-infinity"`，在 `ROC.Thresholds` 前補上 `Inf` 即可重建。未定義時，ROC 向量與 `InitialThreshold` 均為空，`AUC` 在 JSON 中為 `null`（MATLAB 將此純量欄位解碼為 `[]`）。以下節錄有效的完全同分曲線與未定義曲線。

```json
[
  {"AUC":0.5,"ROC":{"FPR":[0,1],"TPR":[0,1],"Thresholds":[0],"InitialThreshold":"positive-infinity","Valid":true,"Status":"OK"}},
  {"AUC":null,"ROC":{"FPR":[],"TPR":[],"Thresholds":[],"InitialThreshold":[],"Valid":false,"Status":"MissingNegativeClass"}}
]
```

CSV 每筆結果占一列，只含純量欄位。訓練和推論耗時欄位為秒數中位數。缺少的值留白。以下為欄位節錄。

```csv
Method,EvaluationStage,FoldNumber,Accuracy,AUC,ROCValid,NumFeatures,NumClassifierParameters
pca-bayes,synthetic,,0.75,0.875,1,3,19
pca-bayes,synthetic,,0.5,,0,3,19
```

`plotROCCurves` 讀取記憶體內的曲線座標，繪製隨機排序對角參考線，並在圖例標示各有效方法及其 AUC。無效曲線會在發出診斷後略過。正式資料的匯出結果須置於已忽略的 `results/` 或其他本機私有位置，因有限 ROC 閾值會保留各個不同的原始分數值。

## 講義依據與後續整合

第 2 章 §2.8.3（合併講義 PDF 第 181–186 頁）定義命中率與誤警率，並示範閾值掃描。第 3 章 §3.7.2（第 266–274 頁）討論維度與樣本數對高斯參數估計及判別成本的影響。兩處均已核對本機合併講義。梯形 AUC、實測 ROC 的同分分組、計時設定及序列化皆為本專案的實作約定。另見[講義對照](lecture-reference.md)。

Issue #11 可依方法與折次保存結果，依各折準確率與 AUC 計算平均值和標準差，再利用保存的座標與純量建立比較圖表。各折獨立模型的原始分數尺度可能不同。混合各折分數所得的 ROC 可與逐折 AUC 統計不同，平均各折閾值前須確認分數尺度可比。跨折 ROC 彙整方式及理由由 Issue #11 記錄。Issue #12 可沿用相同格式記錄凍結的最終模型與獨立評估批次。
