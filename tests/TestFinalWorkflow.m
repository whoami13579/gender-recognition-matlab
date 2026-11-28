classdef TestFinalWorkflow < matlab.unittest.TestCase
    properties
        Work
        Training
        Testing
        Run
    end

    methods (TestClassSetup)
        function trainOnce(testCase)
            fixture = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            testCase.Work = string(fixture.Folder);
            testCase.Training = fullfile(testCase.Work, "train-fixture");
            testCase.Testing = fullfile(testCase.Work, "holdout-fixture");
            writeFixture(testCase.Training, 12, 101);
            writeFixture(testCase.Testing, 4, 202);
            testCase.Run = run_final_training(testCase.Training, ...
                OutputDirectory=testCase.Work, Synthetic=true, Benchmark=quickBenchmark());
        end
    end

    methods (Test)
        function allMethodsUseCompleteDataset(testCase)
            run = testCase.Run;
            testCase.verifyEqual(run.Status, "complete");
            testCase.verifyEqual(run.Summary.NumTrainingSamples, 24 * ones(4, 1));
            testCase.verifyEqual(run.Summary.NumFeatures.', [10 1 20 1]);
            testCase.verifyEqual(run.Summary.NumProjectionParameters.', [45056 8192 86016 8192]);
            testCase.verifyEqual(run.Summary.NumNormalizationParameters.', [0 0 40 2]);
            testCase.verifyEqual(run.Summary.NumClassifierParameters.', [131 4 353 25]);
            testCase.verifyEqual(run.Summary.CombinedParameterCount.', [45187 8196 86409 8219]);
            metadata = readJSON(run.OutputDirectory, "training-metadata.json");
            testCase.verifyEqual(string(metadata.InitialIndependentTestStatus), "pending");
            testCase.verifyFalse(isfolder(fullfile(run.OutputDirectory, "independent-test")) && ...
                isfile(fullfile(run.OutputDirectory, "independent-test", "evaluation-results.json")));
        end

        function persistenceAndReproducibility(testCase)
            state = rng;
            first = loadFinalModels(testCase.Run.OutputDirectory);
            repeat = run_final_training(testCase.Training, OutputDirectory=testCase.Work, ...
                Synthetic=true, Benchmark=quickBenchmark());
            second = loadFinalModels(repeat.OutputDirectory);
            testCase.verifyNotEqual(repeat.OutputDirectory, testCase.Run.OutputDirectory);
            testCase.verifyEqual(first, second);
            testCase.verifyEqual(rng, state);
            checks = testCase.Run.Verification;
            testCase.verifyTrue(all([checks.FittedStateEqual] & [checks.LabelsEqual] & ...
                [checks.ScoresExactlyEqual] & [checks.ScoreOrderEqual] & [checks.RNGUnchanged]));
            testCase.verifyEqual([checks.MaxAbsoluteDifference], zeros(1, 4));
            testCase.verifyEqual([checks.MaxRelativeDifference], zeros(1, 4));
            [holdout, ~] = loadDataset(testCase.Testing);
            for k = 1:4
                path = fullfile(testCase.Work, first(k).Method + "-standalone.mat");
                savePipelineModel(first(k), path);
                info = whos('-file', path);
                testCase.verifyEqual({info.name}, {'artifact'});
                model = loadPipelineModel(path);
                testCase.verifyEqual(model, first(k));
                [labels, scores] = predictPipeline(model, holdout);
                testCase.verifyEqual(labels, double(scores > 0));
                testCase.verifyError(@() savePipelineModel(model, path), 'genderRecognition:FinalExportExists');
            end
        end

        function metricsExportAndReconstruction(testCase)
            state = rng;
            result = run_independent_testing(testCase.Run.OutputDirectory, testCase.Testing);
            metadata = readJSON(result.OutputDirectory, "test-metadata.json");
            testCase.assertEqual(result.Status, "complete", result.Reason);
            testCase.verifyEqual(rng, state);
            testCase.verifyFalse(metadata.OfficialEvaluation);
            testCase.verifyLessThanOrEqual(string(metadata.FrozenUTC), string(metadata.TrainingCompletedUTC));
            testCase.verifyLessThanOrEqual(string(metadata.TrainingCompletedUTC), string(metadata.BundleVerifiedUTC));
            testCase.verifyLessThanOrEqual(string(metadata.BundleVerifiedUTC), string(metadata.DatasetLoadStartedUTC));
            testCase.verifyEqual(numel(result.Records), 4);
            restored = loadEvaluationResults(fullfile(result.OutputDirectory, "evaluation-results.json"));
            predictions = load(fullfile(result.OutputDirectory, "predictions.mat"));
            for k = 1:4
                record = result.Records(k);
                local = predictions.predictions(k);
                y = local.Labels;
                scores = local.Scores;
                neg = scores(y == 0).';
                pos = scores(y == 1);
                referenceAUC = mean((pos > neg) + 0.5 * (pos == neg), 'all');
                testCase.verifyGreaterThan(numel(unique(scores)), 2);
                testCase.verifyEqual(record.EvaluationStage, "synthetic");
                testCase.verifyEmpty(record.FoldNumber);
                testCase.verifyEqual(record.ConfusionMatrix, accumarray([y local.Predictions] + 1, 1, [2 2]));
                testCase.verifyEqual(record.Accuracy, mean(y == local.Predictions));
                testCase.verifyEqual(record.ROC.FPR, ...
                    mean(neg >= record.ROC.Thresholds(:), 2));
                testCase.verifyEqual(record.ROC.TPR, ...
                    mean(pos.' >= record.ROC.Thresholds(:), 2));
                testCase.verifyEqual(record.AUC, referenceAUC, 'AbsTol', eps);
                testCase.verifyEqual(record.Timing.Inference.Pipeline.BatchSize, 8);
                testCase.verifyEqual(record.Timing.Inference.Classifier.SecondsPerSample, ...
                    record.Timing.Inference.Classifier.MedianSeconds / 8);
                testCase.verifyEqual(restored(k).ROC, record.ROC, 'RelTol', 4 * eps);
            end
            figures = fullfile(testCase.Work, "reconstructed");
            plotIndependentTestRun(result.OutputDirectory, figures);
            for name = ["test-roc", "accuracy-auc", "computational-cost"]
                testCase.verifyTrue(isfile(fullfile(figures, name + ".png")));
                testCase.verifyTrue(isfile(fullfile(figures, name + ".pdf")));
            end
            values = struct('MeanAccuracy', [0.75 0.8 0.9 0.8], 'MeanAUC', [0.8 0.9 0.95 0.9]);
            comparison = struct('Baseline', values, 'Nested', values);
            plotIndependentTestFigures(restored, comparison, fullfile(testCase.Work, "comparison"), true);
            testCase.verifyTrue(isfile(fullfile(testCase.Work, "comparison", "cross-validation-vs-independent.pdf")));
        end

        function testChangesNeverChangeTheFrozenModels(testCase)
            first = loadFinalModels(testCase.Run.OutputDirectory);
            [images, ~] = loadDataset(testCase.Testing);
            switched = fullfile(testCase.Work, "switched");
            mkdir(switched);
            % Swap both labels and the two class blocks in filename order.
            for k = 1:4
                imwrite(images{k + 4}, fullfile(switched, sprintf('man_%02d_01.png', k)));
                imwrite(images{k}, fullfile(switched, sprintf('woman_%02d_01.png', k)));
            end
            changed = fullfile(testCase.Work, "changed");
            writeFixture(changed, 4, 303);
            original = run_independent_testing(testCase.Run.OutputDirectory, testCase.Testing, GeneratePlots=false);
            again = run_independent_testing(testCase.Run.OutputDirectory, testCase.Testing, GeneratePlots=false);
            swapped = run_independent_testing(testCase.Run.OutputDirectory, switched, GeneratePlots=false);
            replaced = run_independent_testing(testCase.Run.OutputDirectory, changed, GeneratePlots=false);
            for result = [original again swapped replaced]
                testCase.assertEqual(result.Status, "complete", result.Reason);
            end
            a = load(fullfile(original.OutputDirectory, "predictions.mat"));
            b = load(fullfile(swapped.OutputDirectory, "predictions.mat"));
            c = load(fullfile(again.OutputDirectory, "predictions.mat"));
            d = load(fullfile(replaced.OutputDirectory, "predictions.mat"));
            testCase.verifyEqual(a.predictions, c.predictions);
            order = [5:8 1:4];
            for k = 1:4
                testCase.verifyEqual(b.predictions(k).Predictions, a.predictions(k).Predictions(order));
                testCase.verifyEqual(b.predictions(k).Scores, a.predictions(k).Scores(order), 'RelTol', 8 * eps);
                testCase.verifyNotEqual(d.predictions(k).Scores, a.predictions(k).Scores);
            end
            testCase.verifyEqual(loadFinalModels(testCase.Run.OutputDirectory), first);
        end

        function missingBlockedAndIncompleteAreExplicit(testCase)
            path = testCase.Run.OutputDirectory;
            missing = run_independent_testing(path, fullfile(testCase.Work, "missing"));
            testCase.verifyEqual(missing.Status, "unavailable");
            same = run_independent_testing(path, fullfile(testCase.Training, ".", "..", "train-fixture"));
            testCase.verifyEqual(same.Status, "blocked");
            copy = fullfile(testCase.Work, "copy-of-training");
            copyfile(testCase.Training, copy);
            duplicate = run_independent_testing(path, copy);
            testCase.verifyEqual(duplicate.Status, "blocked");
            integrity = readJSON(duplicate.OutputDirectory, "dataset-integrity.json");
            testCase.verifyEqual(integrity.TestImagesWithTrainingMatch, 24);
            invalid = fullfile(testCase.Work, "invalid");
            mkdir(invalid);
            writelines("invalid", fullfile(invalid, "invalid.png"));
            partial = run_independent_testing(path, invalid);
            testCase.verifyEqual(partial.Status, "incomplete");
            testCase.verifyEqual(partial.ErrorID, "genderRecognition:InvalidFilename");
            testCase.verifyEqual(string(readJSON(partial.OutputDirectory, "test-metadata.json").Status), "incomplete");
            testCase.verifyError(@() plotIndependentTestRun(partial.OutputDirectory, testCase.Work), ...
                'genderRecognition:IncompleteIndependentTest');
        end

        function corruptBundleFailsBeforeOpeningTestData(testCase)
            corrupt = fullfile(testCase.Work, "corrupt-bundle");
            copyfile(testCase.Run.OutputDirectory, corrupt);
            writelines("bad model", fullfile(corrupt, "models", "lda-mlp.mat"));
            testCase.verifyError(@() run_independent_testing(corrupt, testCase.Testing), ...
                'genderRecognition:FinalArtifactInvalid');
            incomplete = fullfile(testCase.Work, "training.incomplete");
            mkdir(incomplete);
            testCase.verifyError(@() run_independent_testing(incomplete, testCase.Testing), ...
                'genderRecognition:FinalArtifactInvalid');
        end

        function unsupportedMissingAndMalformedArtifacts(testCase)
            missing = fullfile(testCase.Work, "absent.mat");
            testCase.verifyError(@() loadPipelineModel(missing), 'genderRecognition:MissingPipelineArtifact');
            corrupt = fullfile(testCase.Work, "corrupt.mat");
            writelines("invalid MAT", corrupt);
            testCase.verifyError(@() loadPipelineModel(corrupt), 'genderRecognition:InvalidPipelineArtifact');
            pipeline = loadFinalModels(testCase.Run.OutputDirectory);
            artifact = struct('ModelFormatVersion', 2, 'Pipeline', pipeline(1));
            incompatible = fullfile(testCase.Work, "incompatible.mat");
            save(incompatible, 'artifact');
            testCase.verifyError(@() loadPipelineModel(incompatible), 'genderRecognition:UnsupportedPipelineVersion');
            artifact.ModelFormatVersion = 1;
            artifact.Pipeline.Classifier.NumFeatures = 2;
            malformed = fullfile(testCase.Work, "invalid-shape.mat");
            save(malformed, 'artifact');
            testCase.verifyError(@() loadPipelineModel(malformed), 'genderRecognition:InvalidPipelineArtifact');
        end
    end
end

function options = quickBenchmark
options = struct('WarmupRuns', 0, 'NumRepeats', 1);
end

function value = readJSON(directory, filename)
value = jsondecode(fileread(fullfile(directory, filename)));
end

function writeFixture(path, perClass, seed)
mkdir(path);
stream = RandStream('mt19937ar', 'Seed', seed);
for label = 0:1
    classes = ["man", "woman"];
    for subject = 1:perClass
        pixels = uint8(randi(stream, [0 200], 16, 16));
        pixels(1:4, 1:4) = pixels(1:4, 1:4) + uint8(50 * label);
        imwrite(pixels, fullfile(path, sprintf('%s_%02d_01.png', classes(label + 1), subject)));
    end
end
end
