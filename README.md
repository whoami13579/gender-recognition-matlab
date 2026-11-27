# Gender Recognition Grand Challenge

[繁體中文導覽](#繁體中文導覽) · [English overview](#english-overview)

## 繁體中文導覽

本專題為圖形識別課程的期末專題，使用 MATLAB 建立人臉影像性別辨識系統，並比較四種特徵降維與分類器組合的辨識效能及計算成本。

在 Windows 上以 MATLAB R2026a 驗證資料載入、影像前處理及測試，無需額外工具箱。將訓練 PNG 放在本機 `train/`，或提供其他本機路徑。從儲存庫根目錄執行 `setup`、`run_dataset_analysis` 及 `run_tests`。完整設定與介面請見[資料載入與前處理](docs/zh-TW/data-preprocessing.md)。

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

### MATLAB setup

Dataset loading, preprocessing, and tests were verified on Windows with MATLAB R2026a Update 5. These modules use base MATLAB.

Place the local training PNGs directly in `train/`, or pass a different local directory to `run_dataset_analysis`. Keep datasets, generated artifacts, and private metadata local.

Run these commands from the repository root in MATLAB.

```matlab
setup;
run_dataset_analysis;  % Uses the local train/ directory
run_tests;
```

Pass a local path and a different resolution as needed.

```matlab
run_dataset_analysis("path/to/training-data", struct('ImageSize', [96 96]));
```

From a terminal with MATLAB on PATH, run the synthetic suite with `matlab -batch "run_tests;"`. The tests create and clean up their own temporary fixtures.

| Path | Purpose |
|---|---|
| `setup.m`, `run_tests.m` | MATLAB path setup and test entry point |
| `src/data/` | Dataset loading, filename parsing, and preprocessing |
| `scripts/` | Executable analysis functions |
| `tests/` | Synthetic MATLAB tests |
| `docs/en/`, `docs/zh-TW/` | Bilingual requirements and implementation guides |
| `train/`, `test/` | Local datasets. Analysis reads only the supplied training directory. |
| `results/` | Ignored directory for local generated outputs |
