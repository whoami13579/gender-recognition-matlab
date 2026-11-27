# 資料載入與前處理

[English](../en/data-preprocessing.md) · [專案首頁](../../README.md)

## 環境設定與本機資料

已在 Windows 的 MATLAB R2026a Update 5 驗證。資料載入、前處理及合成測試使用 MATLAB 基礎功能。將訓練 PNG 直接放入本機 `train/`，或傳入其他資料夾。請從儲存庫根目錄執行以下指令。分析僅開啟傳入的訓練資料夾。

```matlab
setup;
run_dataset_analysis;
stats = run_dataset_analysis("path/to/training-data", struct('ImageSize', [96 96]));
run_tests;
```

`run_dataset_analysis` 會顯示彙總張數、維度、觀察到的強度範圍、特徵矩陣位元組數與耗時，也可回傳這些統計值。分析結果保留於命令視窗及回傳的結構。若需匯出特徵、模型或私有中繼資料，請留在已由 Git 忽略的本機 `results/`。

## 載入與列順序

```matlab
[images, y, subjectIDs, metadata] = loadDataset("train");
[label, subjectID, imageID, sampleID] = parseImageFilename("man_900001_1.png");
```

- 載入器接受平面資料夾內的具標籤 PNG，影像須為不透明的灰階或 RGB。遇到非預期檔案或子目錄、無效檔名、重複識別碼、不可讀影像、索引色彩或透明度時會回報錯誤。正式訓練集的張數由分析功能另行檢查。
- 檔名採 `man_<person>_<image>.png` 或 `woman_<person>_<image>.png`，兩段編號皆由十進位數字組成。解析時接受英文字母大小寫差異，並移除編號多餘的前置零。上方範例為合成檔名。`man_900001` 與 `woman_900001` 屬於不同受試者。樣本識別碼由受試者識別碼加上影像編號組成。
- 檔案依不區分大小寫的檔名字典順序排列。`images` 為 N × 1 cell 陣列。`y` 為 N × 1 的 double 欄向量，男性為 `0`，女性為 `1`。後續分類分數、ROC 與 AUC 均以女性為正類。`subjectIDs` 為 N × 1 字串欄向量。
- `metadata` 的第 `i` 列對應 `images{i}`、`y(i)`、`subjectIDs(i)` 與後續的 `X(i,:)`。欄位包含 `SampleID`、`SubjectID`、`ImageID`、`Label`、`Filename`、`Height`、`Width`、`Channels`、`PixelType`。表格及真實識別碼應留在本機。受試者分組依賴資料集的檔名慣例，遇到不支援的檔名時會明確回報，避免捏造身分。

## 擬合與轉換

```matlab
config = struct('Grayscale', true, 'ImageSize', [64 64], ...
    'OutputType', "double", 'IntensityScaling', "unit");
preprocessor = fitPreprocessor(images, config);
X = transformImages(preprocessor, images);
processedImages = transformImages(preprocessor, images, "images");
```

| 選項 | 預設值 | 支援值 |
|---|---|---|
| `Grayscale` | `true` | 邏輯純量。`false` 保留各張影像原有的通道數。 |
| `ImageSize` | `[64 64]` | 由兩個正整數組成的列向量 `[高度 寬度]`，或以 `[]` 保留原始解析度。 |
| `OutputType` | `"double"` | `"double"` 或 `"single"`。 |
| `IntensityScaling` | `"unit"` | `"unit"` 依無號輸入型別的最大值縮放。`"native"` 以來源型別的強度尺度回傳浮點數。 |

影像必須為非空、完整的實數灰階或 RGB 陣列，支援 `logical`、`uint8`、`uint16`、`single`、`double`。浮點輸入須為 [0, 1] 內的有限值。前處理將像素轉成浮點數，依設定進行灰階轉換，再以含抗混疊處理的雙線性插值調整尺寸。`"native"` 在上述操作後將無號輸入還原至 [0, 255] 或 [0, 65535] 的尺度。邏輯與浮點輸入維持原有的單位尺度。保留原始解析度時，各張影像的高度與寬度必須相同。停用灰階轉換時，通道數也須相同。保留供驗證的影像須能轉成與擬合時相同的輸出形狀。

`transformImages` 預設依輸入順序回傳 N × D 特徵矩陣，每列一張影像。傳入 `"images"` 可取得處理後的 N × 1 cell 陣列。以 H × W × C 的輸出影像為例，依 MATLAB 欄優先順序展平，像素 `(r,c,k)` 對應特徵索引 `r + (c-1)*H + (k-1)*H*W`，索引自 1 起算。各通道依影像的通道順序串接。

交叉驗證時，請以當折訓練部分執行 `fitPreprocessor(trainingImages, config)`，再對保留資料執行 `transformImages(preprocessor, validationImages)`。目前擬合步驟驗證選項並固定輸出表示，保存 `OutputImageSize`、`NumFeatures` 與 `NumFittedParameters`（0）。轉換採用固定的逐張影像運算。

## 建議的後續模組介面

```matlab
pcaModel = fitPCA(X, config);
Z = transformPCA(pcaModel, X);

ldaModel = fitLDA(X, y, config);
Z = transformLDA(ldaModel, X);

bayesModel = fitBayes(Z, y, config);
[yPred, scores] = predictBayes(bayesModel, Z);

mlpModel = fitMLP(Z, y, config);
[yPred, scores] = predictMLP(mlpModel, Z);
```

這些函式簽章供後續議題採用。`Z` 為 N × K，`yPred` 及連續的正類 `scores` 均為 N × 1。分數越高，代表越支持女性（`1`）。擬合模型應保存對保留資料推論時所需的設定與狀態。投影模型建議提供 `NumInputFeatures`、`NumFeatures`（輸出維度）及 `NumFittedParameters`。分類器建議提供 `NumFeatures`（輸入維度）及 `NumClassifierParameters`。請分開記錄分類器和各特徵轉換的擬合參數量，並說明計數慣例。

## 訓練資料驗證

| 項目 | 本機驗證結果 |
|---|---|
| 影像 | 200 張 PNG，男性、女性各 100 張 |
| 原始尺寸與型別 | 640 × 640 × 3，RGB `uint8` |
| 由檔名取得的受試者識別碼 | 100 組，每類 50 組，每組 2 張，跨類別識別碼衝突為 0 |
| 無效或不可讀影像 | 0 張 |
| 預設影像表示 | 64 × 64 灰階，每張 4,096 維特徵 |
| 預設輸出 | `X` 為 200 × 4,096 的 `double`，`y` 為 200 × 1 |
| 特徵矩陣記憶體 | 6,553,600 位元組（6.25 MiB），未包含輸入影像及暫存陣列 |

受試者分組依檔名慣例推定。64 × 64 基準及標籤編碼是專案的工程決定。若保留原始解析度並轉灰階，每張有 409,600 維，同一批 200 張影像的 double 特徵矩陣需 625 MiB。特徵量隨解析度及通道數增加，也會提高儲存、後續擬合及推論的成本。

## 測試與上課講義

從儲存庫根目錄在 MATLAB 執行 `run_tests;`，或在終端機執行 `matlab -batch "run_tests;"`。測試使用程式產生的合成影像，並由 MATLAB 測試 fixture 清理暫存資料夾。測試涵蓋驗證、中繼資料對應、向量順序及重現性。

第 1 章 §1.2、§1.3 對應合併講義 PDF 第 12、25–29 頁，涵蓋辨識流程與特徵表示。請見[講義對照與系統架構](lecture-reference.md)與[正式專題說明](project-requirements.md)。
