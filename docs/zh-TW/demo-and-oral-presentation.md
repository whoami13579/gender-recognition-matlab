# MATLAB 展示與口頭報告操作指南

[English](../en/demo-and-oral-presentation.md) · [專題說明第九節](project-requirements.md) · [專案首頁](../../README.md)

## 準備離線副本

使用已啟用 JVM 的 MATLAB R2026a。本機以 Windows 11、R2026a Update 5 與基礎 MATLAB 驗證，沒有額外工具箱。從 repo 根目錄執行 `setup`。已存成果可離線執行。新訓練用 Git 記錄程式版本。Windows PowerPoint 只用於重建簡報。

以私下管道複製所選實驗的完整目錄，保留內部檔名與內容。已存結果展示需要兩份 CV 與一份最終模型。單圖辨識需一張本機 RGB 或灰階影像。新擬合才需課程訓練集。獨立評估需另行提供的保留資料。

```text
train/                                        # 課程訓練 PNG，已有模型時可不帶
test/                                         # 課程正式保留資料，取得後再放入
results/cross-validation/baseline/run-*/       # Issue #11 基準
results/cross-validation/nested/run-*/         # Issue #11 巢狀實驗
results/final-evaluation/run-*/                # Issue #12 最終模型與訓練索引
  independent-test/attempt-*/                 # 如有已存狀態或評估
results/demo/                                 # 新圖表、實驗與測試記錄
results/presentation/                         # 已建簡報與備援
```

相對路徑以 repo 根目錄為基準。也可使用本機絕對路徑。各參數可獨立設定。

| 參數 | 預設與用途 |
| --- | --- |
| `TrainingPath` | `train`。明確執行訓練及驗證訓練示例位置時使用。 |
| `TestingPath` | `test`。只有獨立測試相關動作會存取。 |
| `BaselineRun`、`NestedRun` | 指定有 `experiment-metadata.json` 的完整實驗。省略時搜尋 `results/cross-validation` 與展示輸出的對應子目錄。 |
| `FinalRun` | 指定含 `model-manifest.json` 的驗證模型目錄。省略時搜尋 `results/final-evaluation` 與展示輸出的對應子目錄。 |
| `IndependentAttempt` | 指定含 `test-metadata.json` 的獨立測試。獨立測試動作會自動尋找與模型相符的唯一已完成記錄，搜尋模型或展示輸出的子目錄。 `figures` 使用明確指定的獨立測試結果。 |
| `OutputDirectory` | `results/demo`。新產出應放在 Git 忽略的本機目錄。 |

有多個相符實驗時，依提示填入預先選好的路徑。報告採用 Issue #11 的 `91c6d5e` 基準與巢狀實驗。另一份本機基準版本為 `66f82ff`。正式最終模型在 `984e14d` 擬合。程式顯示各筆來源及資料範圍。合成測試使用 `Synthetic=true` 與合成產物。

## 執行展示

從 repo 根目錄啟動乾淨的 MATLAB：

```matlab
setup;
run_demo("quick", BaselineRun="results/cross-validation/baseline/<chosen-run>");
```

以本機目錄取代預留字串。本機報告所用的基準目錄是 `results/cross-validation/baseline/run-tp2b5627d2_ac6f_463a_845f_aad7bd4e2dad`。若巢狀與最終實驗各只有一份相符記錄，可自動載入。上台前，建議用下例鎖定所有路徑並選一張訓練影像，避免將影像檔名寫入追蹤檔。

```matlab
setup;
demoOptions = struct( ...
    "TrainingPath", "train", "TestingPath", "test", ...
    "BaselineRun", "results/cross-validation/baseline/<chosen-run>", ...
    "NestedRun", "results/cross-validation/nested/<chosen-run>", ...
    "FinalRun", "results/final-evaluation/<chosen-run>", ...
    "OutputDirectory", "results/demo");
[file, folder] = uigetfile("*.png", "Select a local training example", fullfile(pwd, "train"));
if isequal(file, 0), error("Select a local image before the live demo."); end
demoOptions.ImagePath = fullfile(folder, file);
demoOptions.ImageKind = "training";
a = namedargs2cell(demoOptions);
run_demo("quick", a{:}, ShowFigures=true);
run_demo("figures", a{:}, ShowFigures=true);
run_demo("independent-status", a{:});
```

檔案選擇視窗只在事前準備使用。批次請直接提供已選定的 `ImagePath`，保留預設 `ShowFigures=false`。`TrainingPath` 內的影像會自動標明為訓練示例。用 `ImageKind="training"` 可要求影像位於該目錄。其他本機圖片使用預設 `ImageKind="local"`。互動選單用 `run_demo("menu", a{:})`。各預設實驗唯一時也可直接 `setup; run_demo;`。選單 2、3、7 明確觸發新運算。

```powershell
matlab -wait -batch "setup; run_demo('quick',BaselineRun='results/cross-validation/baseline/run-tp2b5627d2_ac6f_463a_845f_aad7bd4e2dad');"
```

| 動作 | 展示內容 |
| --- | --- |
| `quick` | 讀取基準、巢狀 CV、模型設定、參數及擬合時間。可選單張影像，不擬合、不搜尋參數、不碰保留資料。 |
| `results` | 僅查看 CV 比較。加 `FoldDetails=true` 查看逐折指標。 |
| `models`、`image` | 驗證模型，或用同一張影像執行四條流程。用 `ShowFigures=true` 可顯示原圖和前處理結果。 |
| `figures` | 重建平均 ROC、準確率／AUC 與計算成本。加 `FoldROC=true` 匯出逐折曲線。`ShowFigures=true` 在 MATLAB 顯示 ROC PNG，展示後可關閉。 |
| `independent-status` | 查已存結果或狀態，例如可執行、資料缺少、所選失敗記錄。只檢視，不執行評估。 |
| `independent` | 重用已完成評估，或用凍結模型與既有完整性檢查評估指定測試路徑。檢查回傳狀態。 |
| `baseline`、`train` | 明確執行基準五折或擬合與驗證最終模型，輸出至新目錄。 |

CV 表有準確率、AUC 與折間樣本標準差、降維後維度、分類器／投影／標準化／合計參數、分類器與完整流程的訓練和推論時間。ROC 保留實測座標及隨機排序對角線。正式已存四條曲線重合。折間標準差描述各折變動。基準、巢狀外折與最終設定可能不同。PCA + MLP 最終使用 20 維，巢狀外折使用 10 維。

時間以毫秒表示。正式 CV 的數字為各折中位數的平均，每批推論 40 張。最終模型擬合中位數以 200 張量測。完整推論處理已載入記憶體的影像，分類器計時從投影後特徵開始。MAT 位元組另受保存方式、中繼資料與壓縮影響。現場單次執行時間適用於不同工作負荷。

影像展示顯示原始及處理後尺寸、特徵數、各預測標籤與分數。男 0、女 1，1 為正類。分數大於 0 預測為 1，恰為 0 時預測為 0。Bayes 是後驗對數勝算，MLP 是 logit，分數尺度不同。系統預測課程標籤，外貌無法判定性別認同。請把個別影像、預測和分數留在本機。

## 獨立測試與備援

有正式已完成記錄時，用 `run_demo("independent-status", a{:}, IndependentAttempt=attempt, ShowFigures=true)` 顯示正確凍結模型的準確率、AUC、ROC、混淆矩陣及兩種記憶體內批次推論時間。要匯出獨立圖表，用 `run_demo("figures", a{:}, IndependentAttempt=attempt)`。

若已取得正式保留資料但尚未評估，確認所選資料夾與凍結模型，再執行：

```matlab
r = run_demo("independent", a{:});
disp(r.Independent.Status);
```

新檔案會寫入 `OutputDirectory/independent-test`。狀態必須是 `complete` 才能展示結果。正式測試目錄缺少時，程式說明模型已備妥、正式結果待補。合成圖表與結果有獨立標記，只供驗證功能。保留受阻或不完整記錄以調查原因，維持保留資料的最終評估用途。

缺少模型時，私下複製驗證過的完整目錄，或在展示前用課程訓練資料準備：

```matlab
setup;
r = run_final_training("train", BaselineRun=baselineDir, NestedRun=nestedDir);
run_demo("models", FinalRun=r.OutputDirectory);
```

令 `baselineDir`、`nestedDir` 為已存資料夾。巢狀建議缺少時，既有流程可依已公告設定重建並記錄來源。詳見完整性規則。[最終訓練與測試](final-training-and-testing.md)

若繪圖慢或無法開圖窗，用 `ShowFigures=false` 並打開預先匯出的 PNG／PDF。可在上台前用 `run_demo("figures", a{:}, FoldROC=true)` 準備。投影片失效時，切到 PDF 與已存圖表。模型驗證失敗時，還原完整的原實驗，勿混搭檔案。找不到 `setup` 時，先切換到 repo 根目錄。出現多份相符記錄時，依診斷設定指定路徑。切回簡報前關閉展示視窗。

## 報告資料

- [可編輯投影片來源與建置](../presentation/README.md)
- [英文投影片大綱](../en/oral-presentation-outline.md)
- [繁體中文逐頁講稿](oral-presentation-notes.md)
- [問答準備](oral-presentation-qa.md)
- [彩排檢核與技術紀錄](demo-rehearsal-checklist.md)
