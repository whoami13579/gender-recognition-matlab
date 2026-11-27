# 五折資料集切分

[English](../en/dataset-partitioning.md) · [專案首頁](../../README.md)

## 為什麼要依受試者分組

五折交叉驗證將訓練資料分成五折，輪流保留一折做驗證。每筆樣本恰好驗證一次，並在其餘四折用於訓練。PCA + Bayes、LDA + Bayes、PCA + MLP、LDA + MLP 共用同一組索引，讓四種方法使用相同的驗證樣本。

- 一般交叉驗證逐筆切分樣本，未考慮類別或受試者。
- 分層交叉驗證依類別分配樣本，使各折的類別比例接近整體資料。
- 依受試者分組的分層交叉驗證以受試者為分配單位，平衡各折中每類的影像張數。同一受試者識別碼的影像若在某折供驗證，就不會出現在該折的訓練部分。

載入器從檔名取得包含類別前綴的識別碼。以合成識別碼為例，`man_900001` 與 `woman_900001` 的數字部分相同，仍是不同識別碼。分組只保證傳入的識別碼不跨折。僅憑檔名無法確認真實身分是否唯一，也無法辨識使用不同識別碼的同一人。

[正式專題說明](project-requirements.md)要求在訓練資料上進行五折交叉驗證。第 1 章 §1.4.5 與第 6 章 §6.3.3（合併講義 PDF 第 38、418–420 頁）討論驗證、測試、泛化與過度擬合。完整的五折流程超出講義的涵蓋範圍。以下分組、平衡與同分處理規則是專案實作決定。詳見[講義對照](lecture-reference.md)。

## 介面與設定

```matlab
setup;
[folds, partitionInfo] = createFolds(y, subjectIDs, config);
```

`y` 是非稀疏的 N × 1 實數 double 欄向量，男性標籤為 `0`，女性為 `1`（正類）。`subjectIDs` 是與樣本對齊的 N × 1 字串欄向量，每個值皆不得缺漏、為空或僅含空白。識別碼依完整字串比對，區分大小寫。載入器已將解析出的識別碼正規化。函式僅使用標籤、識別碼與設定。

`config` 是可省略的純量結構。省略的欄位使用以下預設值。

| 選項 | 預設值 | 支援值 |
|---|---:|---|
| `NumFolds` | `5` | 大於等於 2 且不超過 `flintmax` 的整數。每類至少須有相同數量的可分配單位。 |
| `RandomSeed` | `42` | [0, 2³² − 1] 內的整數。 |
| `GroupBySubject` | `true` | 邏輯純量。 |

`folds` 是 K × 1 的結構陣列。每折包含 `TrainIndices` 與 `ValidationIndices`，均為遞增、從 1 起算的 double 欄向量，指向原始輸入的位置。相同索引可直接套用至對齊的影像、標籤、識別碼、中繼資料列及特徵。

| `partitionInfo` 欄位 | 意義 |
|---|---|
| `NumFolds`、`RandomSeed`、`GroupBySubject`、`Stratified` | 實際設定。分層固定啟用。 |
| `SampleToFold` | N × 1 欄向量，記錄各輸入列所屬的驗證折（1 至 K）。真實分折結果應留在本機。 |
| `TrainSampleCounts`、`ValidationSampleCounts` | K × 1 樣本數。 |
| `TrainClassCounts`、`ValidationClassCounts` | K × 2 各類樣本數，欄順序為 `[男性 女性]`。 |
| `TrainSubjectCounts`、`ValidationSubjectCounts`、`SubjectOverlapCounts` | K × 1 識別碼數，分別統計訓練部分、驗證部分及兩者交集。省略識別碼時回傳空陣列。 |

若缺乏可靠的識別碼，請明確指定逐筆樣本分層。

```matlab
[folds, partitionInfo] = createFolds(y, [], struct('GroupBySubject', false));
```

逐筆樣本分層可能將同一人的影像分到訓練與驗證兩邊。若有有效的識別碼欄向量，可在停用分組時仍傳入識別碼，以計算重疊數。識別碼不完整時，改用逐筆切分應傳入 `[]`。啟用分組時，識別碼缺漏、格式有誤或同一識別碼對應多個標籤均會報錯。

## 平衡策略與重現性

1. 依識別碼首次出現的順序建立受試者群組。停用分組時，依輸入順序將每筆樣本視為一個單位。每個單位只有一種類別，權重為其影像張數。
2. 以 `RandomSeed` 建立區域 `mt19937ar` 亂數流。先隨機排列折的優先順序，再依男性、女性順序分別打亂各類的分配單位。各類再依影像張數由大到小排序，同張數時保留打亂後的順序。
3. 每次將當前單位分配至該類影像張數最少的折。若有多折同為最少，選總影像張數最少者。若仍同分，依種子所決定的折優先順序選擇。各類先填滿尚無該類影像的折，才會將該類的第二個單位分到已有該類的折。

貪婪分配以影像張數為依據，也能處理各受試者擁有不同張數的情況。影像張數特別多的群組可能使折的大小或類別比例難以一致。此策略不會窮舉所有可能的分配。逐筆切分沿用相同步驟，每個單位的權重為 1。

函式會拒絕空標籤、非欄向量或不支援的標籤、缺少類別、無效或未知的選項，以及長度不符或格式有誤的識別碼。為供後續計算 ROC 與 AUC，每折驗證資料都須包含兩類。分組模式要求每類至少有 K 個不同群組，逐筆模式則要求每類至少有 K 筆樣本。函式在回傳前確認每折訓練與驗證索引有效、不重複、互斥，且合起來涵蓋所有樣本。也會確認各樣本恰好驗證一次、驗證折包含兩類，以及分組模式下沒有識別碼重疊。因此，每筆樣本會在其餘 K − 1 折用於訓練。

輸入值、列順序及選項相同時，重跑會得到相同的切分，並保留 MATLAB 的全域亂數狀態。請保留列順序及選項以重現結果，並將一次呼叫所回傳的 `folds` 結構共用於四種分類流程。

## 執行訓練資料分折分析

從儲存庫根目錄在 MATLAB 執行下列指令。

```matlab
setup;
run_partition_analysis;  % 讀取本機 train/
stats = run_partition_analysis("path/to/training-data", ...
    struct('NumFolds', 5, 'RandomSeed', 42, 'GroupBySubject', true));
```

`run_partition_analysis(trainingPath, config)` 使用既有載入器。它會輸出並回傳彙總數值、每折識別碼重疊數、驗證涵蓋率、分折完整性與固定種子的重現性，並檢查全域亂數狀態。結果保留於命令視窗及可選的回傳結構。分析僅使用指定的訓練資料夾，無須執行前處理或模型擬合。真實分折結果、檔名及中繼資料均應留在本機。

在 Windows 的 MATLAB R2026a Update 5 上驗證本機正式訓練資料，共 200 張影像（每類 100 張）與 100 組檔名推得的受試者識別碼（每類 50 組，每組兩張）。預設設定的實測彙總如下。

| 折 | 訓練影像（男／女） | 驗證影像（男／女） | 訓練識別碼 | 驗證識別碼（男／女） | 重疊識別碼 |
|---:|---:|---:|---:|---:|---:|
| 1 | 160（80／80） | 40（20／20） | 80 | 20（10／10） | 0 |
| 2 | 160（80／80） | 40（20／20） | 80 | 20（10／10） | 0 |
| 3 | 160（80／80） | 40（20／20） | 80 | 20（10／10） | 0 |
| 4 | 160（80／80） | 40（20／20） | 80 | 20（10／10） | 0 |
| 5 | 160（80／80） | 40（20／20） | 80 | 20（10／10） | 0 |

200 張影像各驗證恰好一次。五個驗證折都有兩類，重跑得到相同的索引與統計值。

## 後續分類流程如何使用共同分折

```matlab
[images, y, subjectIDs, metadata] = loadDataset(trainingPath);
config = struct('NumFolds', 5, 'RandomSeed', 42, 'GroupBySubject', true);
preprocessingConfig = struct();
[folds, partitionInfo] = createFolds(y, subjectIDs, config);

for k = 1:numel(folds)
    trainIdx = folds(k).TrainIndices;
    valIdx = folds(k).ValidationIndices;

    trainingImages = images(trainIdx);
    validationImages = images(valIdx);
    yTrain = y(trainIdx);
    yValidation = y(valIdx);

    preprocessor = fitPreprocessor(trainingImages, preprocessingConfig);
    XTrain = transformImages(preprocessor, trainingImages);
    XValidation = transformImages(preprocessor, validationImages);

    % PCA、LDA 及分類器擬合由後續議題實作。
end
```

`metadata(trainIdx,:)` 與 `metadata(valIdx,:)` 依照相同順序排列。`XTrain` 與 `XValidation` 每列對應一張輸入影像，均有 D 個特徵欄。每折只使用該折的訓練影像擬合前處理器。詳見[資料載入與前處理](data-preprocessing.md)。

## 測試

從儲存庫根目錄執行 `matlab -batch "run_tests;"`，或在 MATLAB 執行 `run_tests;`。只需 MATLAB 基礎功能。測試涵蓋平衡與不平衡的分折、不等大的群組、替代切分、無效輸入及重現性。整合測試會建立暫存的合成 PNG，確認影像、標籤、中繼資料與特徵對齊。獨立測試集保留給最終模型評估。
