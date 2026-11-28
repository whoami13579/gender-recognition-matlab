classdef TestCrossValidationResults < matlab.unittest.TestCase
    methods (Test)
        function aggregatesWeightsCountsAndUndefinedFolds(testCase)
            [images, y, ~, config] = cvFixture();
            [~, ~, pipeline] = evaluatePipelineFold(images([1:10 13:22]), y([1:10 13:22]), ...
                images([11 12 23 24]), y([11 12 23 24]), "pca-bayes", 1, config.Pipeline, config.Benchmark);
            config.Pipeline.PCA.NumComponents = 1;
            smaller = fitPipeline(images([1:10 13:22]), y([1:10 13:22]), "pca-bayes", config.Pipeline);
            one = makeRecord(pipeline, 1, [0; 1], [0; 1], [-1; 1]);
            two = makeRecord(smaller, 2, [0; 0; 1; 1], [0; 1; 0; 1], [-2; 1; -1; 2]);
            [summary, curves] = summarizeCrossValidation([one; two], "pca-bayes", 2);
            testCase.verifyEqual(summary.FoldAccuracy, [1 0.5]);
            testCase.verifyEqual(summary.FoldAUC, [1 0.75]);
            testCase.verifyEqual(summary.MeanAccuracy, 0.75);
            testCase.verifyEqual(summary.StdAccuracy, sqrt(1/8), 'AbsTol', 1e-14);
            testCase.verifyEqual(summary.MeanAUC, 0.875);
            testCase.verifyEqual(summary.StdAUC, sqrt(1/32), 'AbsTol', 1e-14);
            testCase.verifyEqual([summary.MinAccuracy summary.MaxAccuracy summary.MinAUC summary.MaxAUC], ...
                [0.5 1 0.75 1]);
            testCase.verifyEqual([summary.TN summary.FP; summary.FN summary.TP], [2 1; 1 2]);
            testCase.verifyEqual([summary.TotalCorrect summary.TotalEvaluated], [4 6]);
            testCase.verifyEqual(summary.WeightedAccuracy, 4/6);
            testCase.verifyEqual(summary.FoldFeatures, [2 1]);
            testCase.verifyEqual(summary.FeaturesRange, "1–2");
            testCase.verifyEqual(curves.Area, 0.875);

            undefined = makeRecord(smaller, 2, [0; 0], [0; 1], [-1; 1]);
            [partial, curve] = summarizeCrossValidation([one; undefined], "pca-bayes", 3);
            testCase.verifyEqual(partial.ValidAUCFolds, 1);
            testCase.verifyEqual(partial.UndefinedAUCFolds, "2");
            testCase.verifyEqual(partial.MissingFolds, "3");
            testCase.verifyTrue(isnan(partial.MeanAUC));
            testCase.verifyEqual(curve.InvalidCurveIndices, [2; 3]);
            testCase.verifyError(@() summarizeCrossValidation([one; one], "pca-bayes", 3), ...
                'genderRecognition:InvalidCVRecords');
        end

        function reconstructsSingletonTieAndUndefinedJSON(testCase)
            fixture = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            [images, y, ~, config] = cvFixture();
            model = fitPipeline(images, y, 'pca-bayes', config.Pipeline);
            tie = makeRecord(model, 1, [0; 1], [0; 0], [0; 0]);
            undefined = makeRecord(model, 2, [0; 0], [0; 0], [0; 0]);
            files = exportEvaluationResults([tie; undefined], fixture.Folder);
            restored = loadEvaluationResults(files.JSON);
            testCase.verifyEqual(restored, [tie; undefined]);
            testCase.verifyEqual(restored(1).ROC.Thresholds, [Inf; 0]);
            testCase.verifySize(restored(2).ROC.Thresholds, [0 1]);
            testCase.verifyEmpty(restored(1).Timing.Inference.Pipeline);
        end

        function exportsReloadsAndReplotsWithoutTrainingImages(testCase)
            fixture = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            train = fullfile(fixture.Folder, 'train');
            mkdir(train);
            [~, ~, ~, config] = cvFixture(train);
            config.OutputDirectory = fullfile(fixture.Folder, 'local-results');
            config.ExportResults = true;
            baseline = run_cross_validation(train, config);
            config.Mode = "nested";
            config.Search.Candidates = struct('pca_bayes', {{}}, 'lda_bayes', {{}}, ...
                'pca_mlp', {{}}, 'lda_mlp', {{}});
            nested = run_cross_validation(train, config);
            testCase.verifyTrue(isfolder(baseline.OutputDirectory));
            testCase.verifyTrue(isfolder(nested.OutputDirectory));
            testCase.verifyNotEqual(baseline.OutputDirectory, nested.OutputDirectory);
            testCase.verifyTrue(isfile(fullfile(nested.OutputDirectory, 'hyperparameter-search.json')));
            testCase.verifyFalse(isfile(fullfile(baseline.OutputDirectory, 'hyperparameter-search.json')));
            rmdir(train, 's');
            restored = loadCrossValidationRun(baseline.OutputDirectory);
            testCase.verifyEqual(restored.Folds, baseline.Folds);
            testCase.verifyEqual(restored.Summary.MeanAccuracy, baseline.Summary.MeanAccuracy);
            testCase.verifyEqual(restored.Summary.MeanAUC, baseline.Summary.MeanAUC, 'AbsTol', 1e-14);
            testCase.verifyEqual([restored.MeanROC.Area], [baseline.MeanROC.Area], 'AbsTol', 1e-14);
            testCase.verifyTrue(all(arrayfun(@(r) r.Timing.Inference.Pipeline.NumRepeats == 1, restored.Records)));
            csv = readtable(fullfile(baseline.OutputDirectory, 'evaluation-summary.csv'));
            testCase.verifyEqual(height(csv), 20);
            testCase.verifyEqual(csv.Accuracy, [restored.Records.Accuracy].', 'AbsTol', 1e-14);
            details = readtable(fullfile(baseline.OutputDirectory, 'fold-details.csv'));
            testCase.verifyEqual(details.NumCorrect, [restored.Records.NumCorrect].');
            testCase.verifyTrue(all(details.PositiveSamples + details.NegativeSamples == details.NumEvaluationSamples));

            destination = fullfile(fixture.Folder, 'regenerated');
            plotCrossValidationRun(restored, destination, true);
            plotCrossValidationComparison(restored, loadCrossValidationRun(nested.OutputDirectory), destination);
            for name = ["mean-roc", "fold-roc", "accuracy-auc", "computational-cost", "baseline-vs-nested"]
                testCase.verifyTrue(isfile(fullfile(destination, name + ".png")));
                testCase.verifyTrue(isfile(fullfile(destination, name + ".pdf")));
            end
            testCase.verifyError(@() loadCrossValidationRun(baseline.OutputDirectory + ".incomplete"), ...
                'genderRecognition:CVReloadFailed');
        end
    end
end

function row = makeRecord(pipeline, fold, truth, predictions, scores)
context = struct('Method', pipeline.Method, 'EvaluationStage', "cross-validation", 'FoldNumber', fold);
metrics = evaluateClassification(truth, predictions, scores);
row = buildEvaluationResult(context, metrics, pipeline.Preprocessor, pipeline.Projection, pipeline.Classifier);
end
