# 12 月 15 日口頭報告彩排檢核

[操作指南](demo-and-oral-presentation.md) · [逐頁講稿](oral-presentation-notes.md) · [問答](oral-presentation-qa.md)

## 建議安排

- **12 月 11 日** 將選定的模型、CV 目錄、PDF 與圖表複製到簡報機，從乾淨工作階段測試。
- **12 月 12–13 日** 由另一位組員獨立照指南重現。輪流講原理與結果，計時並演練交接。
- **12 月 14 日** 彩排整場。確認實際授課時間限制後調整講稿，重練 MATLAB 視窗切換與離線備援。
- **12 月 15 日** 開始前檢查顯示解析度、MATLAB 路徑、PDF 備份及測試資料狀態。

## 本機技術驗證

- [x] Windows 11、MATLAB R2026a Update 5、基礎 MATLAB、JVM 可用。
- [x] 由 repo 根目錄在全新 MATLAB 程序執行 `setup`。
- [x] 讀取 Issue #11 指定的基準和巢狀實驗，驗證 Issue #12 四個模型，列出參數與兩種計時。
- [x] 顯示一張本機訓練影像與前處理圖，執行四條實際辨識流程。個別輸出留在本機。
- [x] 由已存結果重新產生兩組平均及逐折 ROC、成效與成本圖，輸出離線 PNG／PDF。
- [x] 確認正式 `test/` 缺少，獨立指標待補。合成測試驗證找不到測試目錄及已完成評估的重載。
- [x] 建置並檢視 PowerPoint 匯出的 20 張投影片。核對對應 PDF、講稿與問答。
- [x] 原有 163 項測試與新增 6 項測試均通過。

## 首次彩排實測｜2026 年 11 月 29 日（臺灣時間）

乾淨程序執行指南中的 quick 指令約 16.24 秒，包含 MATLAB 啟動。另一次技術演練的 quick 6.171 秒、獨立狀態 0.207 秒、四法單圖與圖窗 7.865 秒、完整 PNG／PDF 匯出 48.166 秒。演練的 MATLAB 工作段共 62.417 秒，從 PowerShell 啟動至返回共 74.27 秒。以上是本機一次執行的耗時，與投影片中的受控基準範圍不同。所有步驟成功。MATLAB 工作流程只使用本機檔案；測量時未停用網路介面。

```powershell
matlab -wait -batch "run_tests;"
matlab -wait -batch "setup; r=runtests('tests/TestDemoWorkflow.m'); assertSuccess(r);"
matlab -wait -batch "setup; run_demo('quick',BaselineRun='results/cross-validation/baseline/run-tp2b5627d2_ac6f_463a_845f_aad7bd4e2dad');"
matlab -wait -batch "setup; demoOptions=struct('BaselineRun',string(fullfile('results','cross-validation','baseline','run-tp2b5627d2_ac6f_463a_845f_aad7bd4e2dad'))); a=namedargs2cell(demoOptions); total=tic; d=run_demo('quick',a{:}); fprintf('QUICK_SECONDS %.3f\n',toc(total)); t=tic; s=run_demo('independent-status',a{:}); assert(s.Independent.Status=='unavailable'); fprintf('STATUS_SECONDS %.3f\n',toc(t)); sampleFiles=dir(fullfile('train','*.png')); assert(~isempty(sampleFiles)); t=tic; p=run_demo('image',a{:},ImagePath=fullfile(sampleFiles(1).folder,sampleFiles(1).name),ImageKind='training',ShowFigures=true); assert(height(p.Prediction.Summary)==4); drawnow; close all; fprintf('IMAGE_SECONDS %.3f\n',toc(t)); t=tic; f=run_demo('figures',a{:},FoldROC=true); fprintf('FIGURES_SECONDS %.3f\n',toc(t)); fprintf('TOTAL_SECONDS %.3f\n',toc(total));"
powershell -NoProfile -ExecutionPolicy Bypass -File docs/presentation/build.ps1 -Render
```

技術演練的單圖自第一個本機 PNG 選入。正式展示前，由報告者確認使用哪張圖並填好本機路徑。先匯出並開好平均 ROC，保留 PowerPoint 與 PDF。若課程在彩排後提供獨立資料，沿用已驗證凍結模型、查完整性與完成狀態，再更新投影片及話術。

## 組員與現場彩排

- [ ] 另一位組員在指定版次照指南獨立重現，記錄 MATLAB 版本、路徑與結果。
- [ ] 簡報機在無網路連線時啟動 MATLAB。試開備用投影片、PNG／PDF 與模型。
- [ ] 從乾淨工作階段，現場顯示 baseline、nested、四法 ROC、維度、參數量、訓練與推論時間。
- [ ] 對同一張已核可的本機影像執行四法，確認原圖分享範圍、訓練示例與分數說法。
- [ ] 顯示正式獨立結果或待補訊息，測試切換到預先產生的圖及 PDF。
- [ ] 全員查核第九節六項內容、公式與實作、文字可讀性、數字出處、結論及參考文獻。
- [ ] 練習交接與問答，量出整場時長，與教師確認後的允許時間比較。
- [ ] 2026 年 12 月 15 日前完成真人報告彩排並填記錄。

| 彩排日期 | 參與者 | 完整報告時間 | 實際允許時間 | 交接與備援問題 |
| --- | --- | --- | --- | --- |
| 待填 | 待填 | 待填 | 待確認 | 待填 |
