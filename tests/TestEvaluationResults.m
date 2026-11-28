classdef TestEvaluationResults < matlab.unittest.TestCase
    methods (Test)
        function fittedDefaultParameterCounts(testCase)
            stream = RandStream('mt19937ar', 'Seed', 71);
            y = [zeros(26, 1); ones(26, 1)];
            X = 0.25 * rand(stream, numel(y), 4096) + 0.2 * y;
            images = arrayfun(@(idx) reshape(X(idx, :), 64, 64), (1:numel(y)).', 'UniformOutput', false);
            prep = fitPreprocessor(images);
            X = transformImages(prep, images);
            projections = {fitPCA(X), fitLDA(X, y)};
            transforms = {@transformPCA, @transformLDA};
            expected = [40 167936 0 1721 169657; 1 8192 0 5 8197; ...
                40 167936 80 673 168689; 1 8192 2 49 8243];
            for idx = 1:2
                projection = projections{idx};
                Z = transforms{idx}(projection, X);
                models = {fitBayes(Z, y), fitMLP(Z, y)};
                for kind = 1:2
                    counts = summarizeModelParameters(prep, projection, models{kind});
                    testCase.verifyEqual(counts.NumOriginalFeatures, 4096);
                    testCase.verifyEqual(counts.NumPreprocessingParameters, 0);
                    testCase.verifyEqual(counts.DimensionalityReductionFraction, 1 - size(Z, 2) / 4096);
                    testCase.verifyEqual([counts.NumFeatures counts.NumProjectionParameters ...
                        counts.NumNormalizationParameters counts.NumClassifierParameters counts.CombinedParameterCount], ...
                        expected(idx + 2 * (kind - 1), :));
                end
            end
        end

        function alternativeBayesDimensionsCovarianceAndPriors(testCase)
            stream = RandStream('mt19937ar', 'Seed', 33);
            y = [zeros(5, 1); ones(8, 1)];
            X = randn(stream, numel(y), 6) + y;
            for pcaConfig = {struct('NumComponents', 2), struct('NumComponents', 4), ...
                    struct('ExplainedVarianceThreshold', 0.6)}
                projection = fitPCA(X, pcaConfig{1});
                Z = transformPCA(projection, X);
                k = size(Z, 2);
                expectedCovarianceCounts = [k * (k + 1), k * (k + 1) / 2, 2 * k];
                covariances = ["full", "shared", "diagonal"];
                for idx = 1:numel(covariances)
                    for prior = ["empirical", "equal"]
                        bayes = fitBayes(Z, y, struct('CovarianceType', covariances(idx), 'PriorType', prior));
                        summary = summarizeModelParameters([], projection, bayes);
                        classifierCount = 2 * k + expectedCovarianceCounts(idx) + double(prior == "empirical");
                        testCase.verifyEqual(summary.NumFeatures, k);
                        testCase.verifyEqual(summary.NumProjectionParameters, 6 + 6 * k);
                        testCase.verifyEqual(summary.NumClassifierParameters, classifierCount);
                        testCase.verifyEqual(summary.NumNormalizationParameters, 0);
                        testCase.verifyEqual(summary.CombinedParameterCount, 6 + 6 * k + classifierCount);
                    end
                end
            end
            bayes = fitBayes(Z, y);
            [labels, scores] = predictBayes(bayes, Z);
            result = buildEvaluationResult(struct('Method', "pca-bayes", 'EvaluationStage', "synthetic"), ...
                evaluateClassification(y, labels, scores), [], projection, bayes);
            testCase.verifyEqual(result.Configuration.Projection.ExplainedVarianceThreshold, 0.6);
            testCase.verifyEqual(result.NumFeatures, k);
        end

        function alternativeMLPParametersAndNormalization(testCase)
            [~, projection, ~, Z, y] = smallFixture();
            context = struct('Method', "pca-mlp", 'EvaluationStage', "synthetic");
            for standardize = [true false]
                model = fitMLP(Z, y, struct('HiddenLayerSizes', [5 3], 'MaxEpochs', 5, 'Standardize', standardize));
                [labels, scores] = predictMLP(model, Z);
                summary = buildEvaluationResult(context, evaluateClassification(y, labels, scores), [], projection, model);
                classifier = 3 * 5 + 6 * 3 + 4;
                normalization = 2 * 2 * double(standardize);
                testCase.verifyEqual(summary.NumClassifierParameters, classifier);
                testCase.verifyEqual(summary.NumNormalizationParameters, normalization);
                testCase.verifyEqual(summary.CombinedParameterCount, 3 * (2 + 1) + classifier + normalization);
                testCase.verifyEqual(summary.Configuration.Classifier.Standardize, standardize);
                testCase.verifyEqual(summary.Configuration.Classifier.HiddenLayerSizes, [5 3]);
            end
        end

        function invalidModelMetadataIsRejected(testCase)
            [prep, projection, bayes, Z, y] = smallFixture();
            id = 'genderRecognition:InvalidEvaluationModel';
            testCase.verifyError(@() summarizeModelParameters(prep, rmfield(projection, 'NumFittedParameters'), bayes), id);
            testCase.verifyError(@() summarizeModelParameters(prep, projection, rmfield(bayes, 'NumClassifierParameters')), id);
            testCase.verifyError(@() summarizeModelParameters(rmfield(prep, 'NumFittedParameters'), projection, bayes), id);
            wrongInput = bayes;
            wrongInput.NumFeatures = 1;
            testCase.verifyError(@() summarizeModelParameters(prep, projection, wrongInput), id);
            mlp = fitMLP(Z, y, struct('MaxEpochs', 2));
            testCase.verifyError(@() summarizeModelParameters(prep, projection, rmfield(mlp, 'NumNormalizationParameters')), id);
        end

        function fourMethodsProduceAndExportAllRequiredOutputs(testCase)
            initialRNG = rng;
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            results = run_evaluation_analysis(folder.Folder);
            json = jsondecode(fileread(fullfile(folder.Folder, 'evaluation-results.json')));
            csv = readtable(fullfile(folder.Folder, 'evaluation-summary.csv'), 'TextType', 'string');
            methods = ["pca-bayes"; "lda-bayes"; "pca-mlp"; "lda-mlp"];
            testCase.verifyEqual(string({json.Method}).', methods);
            testCase.verifyEqual(csv.Method, methods);
            testCase.verifyEqual(csv.NumFeatures, [3; 1; 3; 1]);
            testCase.verifyEqual(csv.NumClassifierParameters, [19; 5; 81; 49]);
            testCase.verifyEqual(csv.NumProjectionParameters, [48; 24; 48; 24]);
            testCase.verifyEqual(csv.NumNormalizationParameters, [0; 0; 6; 2]);
            testCase.verifyEqual(csv.CombinedParameterCount, [67; 29; 135; 75]);
            testCase.verifyEqual(csv.InferenceBatchSize, repmat(8, 4, 1));
            for idx = 1:numel(results)
                result = results(idx);
                timing = result.Timing;
                testCase.verifyGreaterThanOrEqual(result.Accuracy, 0);
                testCase.verifyLessThanOrEqual(result.Accuracy, 1);
                testCase.verifyTrue(result.ROC.Valid);
                testCase.verifyEqual(result.AUC, trapz(result.ROC.FPR, result.ROC.TPR));
                testCase.verifyEqual(csv.Accuracy(idx), result.Accuracy);
                testCase.verifyEqual(csv.AUC(idx), result.AUC, 'AbsTol', 8 * eps);
                testCase.verifyEqual(json(idx).ROC.FPR, result.ROC.FPR);
                testCase.verifyEqual(json(idx).ROC.TPR, result.ROC.TPR);
                testCase.verifyEqual([Inf; json(idx).ROC.Thresholds], result.ROC.Thresholds);
                testCase.verifyEqual(json(idx).Timing.Training.Pipeline.SamplesSeconds, timing.Training.Pipeline.SamplesSeconds);
                testCase.verifyEqual(timing.Units, "seconds");
                testCase.verifyEqual(timing.Training.Classifier.NumRepeats, 5);
                testCase.verifyEqual(timing.Training.Pipeline.NumRepeats, 5);
                testCase.verifyEqual(timing.Inference.Classifier.BatchSize, 8);
                testCase.verifyEqual(timing.Inference.Pipeline.BatchSize, 8);
                testCase.verifyEqual(timing.Inference.Pipeline.SecondsPerSample, timing.Inference.Pipeline.MedianSeconds / 8);
                testCase.verifyEqual(timing.Stages.MetricCalculation.NumRepeats, 5);
                testCase.verifyEmpty(timing.Stages.DatasetLoading);
                testCase.verifyEmpty(timing.Stages.ResultExport);
                testCase.verifyGreaterThanOrEqual(csv.ClassifierFitSeconds(idx), 0);
                testCase.verifyGreaterThanOrEqual(csv.PipelineFitSeconds(idx), 0);
                testCase.verifyGreaterThanOrEqual(csv.ClassifierInferenceSeconds(idx), 0);
                testCase.verifyGreaterThanOrEqual(csv.PipelineInferenceSeconds(idx), 0);
            end
            testCase.verifyEqual(json(3).Configuration.Classifier.RandomSeed, 11);
            testCase.verifyEqual(rng, initialRNG);
        end

        function jsonNullSentinelAndExactThresholdRoundTrip(testCase)
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            extreme = makeResult([0; 1; 0; 1], [-realmax; 1; 1 + eps(1); realmax]);
            tied = makeResult([0; 1], [0; 0]);
            invalid = makeResult([1; 1], [0; 1]);
            files = exportEvaluationResults([extreme tied invalid], folder.Folder);
            raw = fileread(files.JSON);
            testCase.verifyTrue(contains(raw, '"AUC":null'));
            testCase.verifyTrue(contains(raw, '"InitialThreshold":"positive-infinity"'));
            testCase.verifyTrue(contains(raw, '"Thresholds":[0]'));
            testCase.verifyTrue(contains(raw, '"Thresholds":[]'));
            decoded = jsondecode(raw);
            testCase.verifyEqual(decoded(1).SchemaVersion, 1);
            testCase.verifyEqual([Inf; decoded(1).ROC.Thresholds], extreme.ROC.Thresholds);
            testCase.verifyEqual(decoded(2).AUC, 0.5);
            testCase.verifyFalse(decoded(3).ROC.Valid);
            testCase.verifyEqual(decoded(3).ROC.Status, 'MissingNegativeClass');
            testCase.verifyEmpty(decoded(3).AUC);
        end

        function exportRejectsInvalidRecordsAndOverwritesVerifiedFiles(testCase)
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            valid = makeResult([0; 1], [-1; 1]);
            files = exportEvaluationResults(valid, folder.Folder);
            original = fileread(files.JSON);
            testCase.verifyTrue(startsWith(original, '[{'));
            id = 'genderRecognition:InvalidEvaluationResult';
            modified = valid;
            modified.Configuration.Classifier.DatasetPath = "private/path";
            testCase.verifyError(@() exportEvaluationResults(modified, folder.Folder), id);
            modified = valid;
            modified.Configuration.Classifier.PriorType = "private/path";
            testCase.verifyError(@() exportEvaluationResults(modified, folder.Folder), id);
            modified = valid;
            modified.Predictions = [0; 1];
            testCase.verifyError(@() exportEvaluationResults(modified, folder.Folder), id);
            modified = valid;
            modified.ROC.Thresholds = [Inf; 1; 1];
            testCase.verifyError(@() exportEvaluationResults(modified, folder.Folder), id);
            modified = valid;
            modified.Accuracy = 0;
            testCase.verifyError(@() exportEvaluationResults(modified, folder.Folder), id);
            testCase.verifyEqual(fileread(files.JSON), original);
            invalid = makeResult([0; 0], [0; 1]);
            exportEvaluationResults(invalid, folder.Folder);
            testCase.verifyNotEqual(fileread(files.JSON), original);
            rawCSV = splitlines(string(fileread(files.CSV)));
            columns = split(rawCSV(1), ',');
            values = split(rawCSV(2), ',');
            testCase.verifyEqual(values(ismember(columns, ["AUC", "FoldNumber", "PipelineFitSeconds"])), strings(3, 1));
            csv = readtable(files.CSV, 'TextType', 'string');
            % Force scalar blank columns to numeric for readers that infer text from blank data.
            opts = detectImportOptions(files.CSV);
            opts = setvartype(opts, {'AUC', 'PipelineFitSeconds', 'FoldNumber'}, 'double');
            blanks = readtable(files.CSV, opts);
            testCase.verifyTrue(isnan(blanks.AUC) && isnan(blanks.PipelineFitSeconds) && isnan(blanks.FoldNumber));
            testCase.verifyEqual(csv.ROCStatus, "MissingPositiveClass");
            testCase.verifyError(@() exportEvaluationResults(valid, ""), 'genderRecognition:InvalidEvaluationOutput');
        end

        function foldAndInferenceTimingCompatibility(testCase)
            [prep, projection, model] = smallFixture();
            metrics = evaluateClassification([0; 1], [0; 1], [-1; 1]);
            context = struct('Method', "pca-bayes", 'EvaluationStage', "cross-validation", 'FoldNumber', 3);
            result = buildEvaluationResult(context, metrics, prep, projection, model);
            testCase.verifyEqual(result.FoldNumber, 3);
            testCase.verifyEqual(result.NumTrainingSamples, 6);
            context = rmfield(context, 'FoldNumber');
            id = 'genderRecognition:InvalidEvaluationResult';
            testCase.verifyError(@() buildEvaluationResult(context, metrics, prep, projection, model), id);
            context.EvaluationStage = "independent-test";
            result = buildEvaluationResult(context, metrics, prep, projection, model);
            testCase.verifyEmpty(result.FoldNumber);
            testCase.verifyEmpty(result.Timing.Training.Pipeline);
            settings = struct('WarmupRuns', 0, 'NumRepeats', 1, 'BatchSize', 1);
            timings.ClassifierInference = benchmarkOperation(@() 4, settings);
            settings.BatchSize = 2;
            timings.PipelineInference = benchmarkOperation(@() 4, settings);
            testCase.verifyError(@() buildEvaluationResult(context, metrics, prep, projection, model, timings), id);
            timings.PipelineInference = [];
            result = buildEvaluationResult(context, metrics, prep, projection, model, timings);
            testCase.verifyEqual(result.Timing.Inference.Classifier.BatchSize, 1);
            timings.ClassifierInference.MeanSeconds = [1 2];
            testCase.verifyError(@() buildEvaluationResult(context, metrics, prep, projection, model, timings), id);
        end

        function headlessPlotShowsValidCurvesAndDiagnosesUndefinedROC(testCase)
            f = figure('Visible', 'off');
            testCase.addTeardown(@() close(f));
            ax = axes('Parent', f);
            valid = makeResult([0; 1; 0; 1], [-2; 1; 0; 2]);
            tied = makeResult([0; 1], [0; 0]);
            invalid = makeResult([1; 1], [0; 1]);
            testCase.verifyWarning(@() plotROCCurves([valid tied invalid], ax), 'genderRecognition:UndefinedROC');
            testCase.verifyEqual(ax.XLim, [0 1]);
            testCase.verifyEqual(ax.YLim, [0 1]);
            testCase.verifyEqual(string(ax.XLabel.String), "False-positive rate");
            testCase.verifyEqual(string(ax.YLabel.String), "True-positive rate");
            lines = findobj(ax, 'Type', 'line');
            testCase.verifyNumElements(lines, 3);
            reference = findobj(lines, 'DisplayName', 'Random ranking');
            testCase.verifyEqual(reference.XData, reference.YData);
            curve = findobj(lines, 'DisplayName', 'PCA + BAYES (AUC 1.000)');
            testCase.verifyEqual(curve.XData(:), valid.ROC.FPR);
            testCase.verifyEqual(curve.YData(:), valid.ROC.TPR);
            testCase.verifyFalse(ishold(ax));
        end
    end
end

function result = makeResult(y, scores)
[prep, projection, classifier] = smallFixture();
context = struct('Method', "pca-bayes", 'EvaluationStage', "synthetic");
metrics = evaluateClassification(y, double(scores > 0), scores);
result = buildEvaluationResult(context, metrics, prep, projection, classifier);
end

function [prep, projection, classifier, Z, y] = smallFixture()
X = [0 2 1; 1 2 0; 2 0 1; 0 1 6; 4 2 1; 6 0 1] / 6;
y = [0; 0; 0; 1; 1; 1];
images = mat2cell(X, ones(size(X, 1), 1), size(X, 2));
prep = fitPreprocessor(images, struct('ImageSize', []));
X = transformImages(prep, images);
projection = fitPCA(X, struct('NumComponents', 2));
Z = transformPCA(projection, X);
classifier = fitBayes(Z, y);
end
