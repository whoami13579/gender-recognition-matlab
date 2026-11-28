# Gender Recognition Grand Challenge

[繁體中文導覽](#繁體中文導覽) · [English overview](#english-overview)

## 繁體中文導覽

本專題為圖形識別課程的期末專題，使用 MATLAB 建立人臉影像性別辨識系統，並比較四種特徵降維與分類器組合的辨識效能及計算成本。

在 Windows 上以 MATLAB R2026a 驗證資料載入、影像前處理及測試，無需額外工具箱。將訓練 PNG 放在本機 `train/`，或提供其他本機路徑。從儲存庫根目錄執行 `setup`、`run_dataset_analysis` 及 `run_tests`。完整設定與介面請見[資料載入與前處理](docs/zh-TW/data-preprocessing.md)。

執行 `setup; run_partition_analysis;` 可檢查本機訓練資料的五折切分、類別分布及受試者識別碼是否重疊。設定與共用索引請見[資料集分折](docs/zh-TW/dataset-partitioning.md)。

執行 `setup; run_pca_analysis;` 可檢查本機訓練資料的 PCA 秩、保留維度及解釋變異。設定及逐折擬合範例請見 [PCA 特徵擷取](docs/zh-TW/pca-feature-extraction.md)。

執行 `setup; run_lda_analysis;` 可檢查本機訓練資料的 Fisher LDA 秩、正則化與一維投影後的類別分布。使用 `fitLDA(XTrain, yTrain)` 擬合，再以 `transformLDA(ldaModel, X)` 轉換特徵。設定與逐折範例請見 [Fisher LDA 特徵擷取](docs/zh-TW/fisher-lda-feature-extraction.md)。

執行 `setup; run_bayes_analysis;` 可檢查第 1 折的 PCA + Bayes 彙總數值，或以 `run_bayes_analysis("train", "lda");` 檢查 LDA + Bayes。用 `fitBayes(ZTrain, yTrain)` 擬合分類器，再呼叫 `[yPred, scores] = predictBayes(bayesModel, ZValidation)` 取得標籤及女性正類的連續分數。設定與範例見[高斯貝氏分類器](docs/zh-TW/gaussian-bayes-classifier.md)。

## English overview

This Pattern Recognition final project uses MATLAB to build a gender recognition system from face images and compare the recognition performance and computational cost of four feature reduction and classifier combinations.

- PCA + Bayes Classifier
- LDA + Bayes Classifier
- PCA + MLP Classifier
- LDA + MLP Classifier

| Documentation | 繁體中文 | English |
|---|---|---|
| Project Requirements | [專題說明](docs/zh-TW/project-requirements.md) | [Project requirements](docs/en/project-requirements.md) |
| Lecture Reference and System Architecture | [講義對照與系統架構](docs/zh-TW/lecture-reference.md) | [Lecture reference and system architecture](docs/en/lecture-reference.md) |
| Dataset Loading and Preprocessing | [資料載入與前處理](docs/zh-TW/data-preprocessing.md) | [Dataset loading and preprocessing](docs/en/data-preprocessing.md) |
| Five-Fold Dataset Partitioning | [資料集分折](docs/zh-TW/dataset-partitioning.md) | [Dataset partitioning](docs/en/dataset-partitioning.md) |
| PCA Feature Extraction | [PCA 特徵擷取](docs/zh-TW/pca-feature-extraction.md) | [PCA feature extraction](docs/en/pca-feature-extraction.md) |
| Fisher LDA Feature Extraction | [Fisher LDA 特徵擷取](docs/zh-TW/fisher-lda-feature-extraction.md) | [Fisher LDA feature extraction](docs/en/fisher-lda-feature-extraction.md) |
| Gaussian Bayes Classifier | [高斯貝氏分類器](docs/zh-TW/gaussian-bayes-classifier.md) | [Gaussian Bayes classifier](docs/en/gaussian-bayes-classifier.md) |

### MATLAB setup

Dataset loading, preprocessing, partitioning, PCA, Fisher LDA, Gaussian Bayes, and their tests were verified on Windows with MATLAB R2026a Update 5. These modules use base MATLAB.

Place the local training PNGs directly in `train/`, or pass a different local directory to the analysis functions. Keep datasets, generated artifacts, and private metadata local.

Run these commands from the repository root in MATLAB.

```matlab
setup;
run_dataset_analysis;  % Uses the local train/ directory
run_partition_analysis;  % Reports aggregate five-fold training statistics
run_pca_analysis;  % Reports whole-training PCA dimensions and explained variance
run_lda_analysis;  % Reports whole-training Fisher LDA rank and projected class statistics
run_bayes_analysis;  % Reports aggregate PCA + Bayes diagnostics for training fold 1
run_tests;
```

Pass a local path and adjust the analysis options as needed.

```matlab
run_dataset_analysis("path/to/training-data", struct('ImageSize', [96 96]));
run_partition_analysis("path/to/training-data", struct('RandomSeed', 17));
run_pca_analysis("path/to/training-data", struct('NumComponents', 40), struct(), 1);
run_lda_analysis("path/to/training-data", struct('Regularization', 1e-2), struct(), 1);
run_bayes_analysis("path/to/training-data", "lda", struct('CovarianceType', "full"));
```

Use `ldaModel = fitLDA(XTrain, yTrain)` to fit a binary projection and `Z = transformLDA(ldaModel, X)` to project an aligned feature matrix to one column. Fit a separate model on each fold’s training rows and reuse it for that fold’s validation rows. [Fisher LDA feature extraction](docs/en/fisher-lda-feature-extraction.md)

Use `bayesModel = fitBayes(ZTrain, yTrain)` with either PCA or LDA features. Call `[yPred, scores] = predictBayes(bayesModel, ZValidation)` for labels and female-class log-posterior odds. [Gaussian Bayes classifier](docs/en/gaussian-bayes-classifier.md)

From a terminal with MATLAB on PATH, run the synthetic suite with `matlab -batch "run_tests;"`. The tests create and clean up their own temporary fixtures.

| Path | Purpose |
|---|---|
| `setup.m`, `run_tests.m` | MATLAB path setup and test entry point |
| `src/data/` | Dataset loading, filename parsing, and preprocessing |
| `src/validation/` | Shared cross-validation fold generation |
| `src/features/pca/` | PCA fitting and projection |
| `src/features/lda/` | Supervised Fisher LDA fitting and projection |
| `src/classifiers/bayes/` | Gaussian Bayes fitting and prediction |
| `scripts/` | Executable analysis functions |
| `tests/` | Synthetic MATLAB tests |
| `docs/en/`, `docs/zh-TW/` | Bilingual requirements and implementation guides |
| `train/`, `test/` | Local datasets. Analysis reads only the supplied training directory. |
| `results/` | Ignored directory for local generated outputs |
