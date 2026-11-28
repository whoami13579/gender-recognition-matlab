classdef TestCrossValidation < matlab.unittest.TestCase
    methods (Test)
        function baselineSharesFiveFoldsAndIsDeterministic(testCase)
            [images, labels, subjects, config] = cvFixture();
            randomState = rng;
            results = crossValidatePipelines(images, labels, subjects, config);
            repeated = crossValidatePipelines(images, labels, subjects, config);
            testCase.verifyEqual(rng, randomState);
            testCase.verifyEqual(results.Status, "complete");
            testCase.verifyNumElements(results.Records, 20);
            testCase.verifyEmpty(results.Failures);
            testCase.verifyEqual(results.Folds, repeated.Folds);
            testCase.verifyEqual(results.Predictions, repeated.Predictions);
            testCase.verifyEqual(rmfield(results.Records, 'Timing'), rmfield(repeated.Records, 'Timing'));
            for method = results.Methods
                visits = zeros(size(labels));
                for fold = 1:5
                    fit = results.Folds(fold).TrainIndices;
                    validation = results.Folds(fold).ValidationIndices;
                    visits(validation) = visits(validation) + 1;
                    testCase.verifyEmpty(intersect(fit, validation));
                    testCase.verifyEmpty(intersect(subjects(fit), subjects(validation)));
                    testCase.verifyEqual(unique(labels(validation)), [0; 1]);
                    idx = [results.Records.Method] == method & [results.Records.FoldNumber] == fold;
                    testCase.assertEqual(nnz(idx), 1);
                    row = results.Records(idx);
                    testCase.verifyEqual(results.Predictions(idx).Labels, labels(validation));
                    testCase.verifyEqual(row.NumTrainingSamples, numel(fit));
                    testCase.verifyEqual(row.NumEvaluationSamples, numel(validation));
                    testCase.verifyTrue(row.ROC.Valid);
                    testCase.verifyTrue(isfinite(row.AUC));
                    testCase.verifyTrue(isfinite(row.Accuracy));
                    testCase.verifyGreaterThan(row.NumFeatures, 0);
                    testCase.verifyGreaterThan(row.NumClassifierParameters, 0);
                    testCase.verifyEqual(row.Timing.Inference.Pipeline.BatchSize, numel(validation));
                    testCase.verifyNotEmpty(row.Timing.Training.Pipeline.SamplesSeconds);
                    testCase.verifyNotEmpty(row.Timing.Inference.Pipeline.SamplesSeconds);
                end
                testCase.verifyEqual(visits, ones(size(labels)));
            end
        end

        function outerHoldoutCannotChangeAnyFittedState(testCase)
            [images, labels, subjects, config] = cvFixture();
            folds = createFolds(labels, subjects);
            fit = folds(1).TrainIndices;
            val = folds(1).ValidationIndices;
            changed = images;
            changed(val) = repmat({uint8(zeros(6))}, numel(val), 1);
            modifiedLabels = labels;
            modifiedLabels(val) = 1 - labels(val);
            for method = ["pca-bayes", "lda-bayes", "pca-mlp", "lda-mlp"]
                [~, originalPrediction, original] = evaluatePipelineFold(images(fit), labels(fit), ...
                    images(val), labels(val), method, 1, config.Pipeline, config.Benchmark);
                [~, alteredPrediction, altered] = evaluatePipelineFold(changed(fit), modifiedLabels(fit), ...
                    changed(val), modifiedLabels(val), method, 1, config.Pipeline, config.Benchmark);
                testCase.verifyEqual(original, altered);
                testCase.verifyNotEqual(originalPrediction.Scores, alteredPrediction.Scores);
            end
        end

        function entryPointRejectsTestingPathBeforeLoading(testCase)
            fixture = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            directory = fullfile(fixture.Folder, 'TeSt');
            mkdir(directory);
            % This directory would fail the loader with EmptyDataset if it were opened.
            testCase.verifyError(@() run_cross_validation(directory), 'genderRecognition:IndependentTestPath');
            testCase.verifyError(@() run_cross_validation("test"), 'genderRecognition:IndependentTestPath');
            testCase.verifyError(@() run_cross_validation([], struct('TestingPath', directory)), ...
                'genderRecognition:InvalidCVConfig');
        end
    end
end
