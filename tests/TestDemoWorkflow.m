classdef TestDemoWorkflow < matlab.unittest.TestCase
    properties
        Work
        Training
        Testing
        Baseline
        Nested
        Final
        Image
        Output
    end

    methods (TestClassSetup)
        function prepareSyntheticRuns(testCase)
            fixture = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            testCase.Work = string(fixture.Folder);
            testCase.Training = fullfile(testCase.Work, "training");
            testCase.Testing = fullfile(testCase.Work, "holdout");
            testCase.Output = fullfile(testCase.Work, "demo");
            mkdir(testCase.Training);
            mkdir(testCase.Testing);
            [~, ~, ~, config] = cvFixture(testCase.Training);
            config.OutputDirectory = fullfile(testCase.Output, "cross-validation");
            config.ExportResults = true;
            run = run_cross_validation(testCase.Training, config);
            testCase.Baseline = run.OutputDirectory;
            config.Mode = "nested";
            methodKeys = ["pca_bayes", "lda_bayes", "pca_mlp", "lda_mlp"];
            for key = methodKeys
                config.Search.Candidates.(key) = {};
            end
            run = run_cross_validation(testCase.Training, config);
            testCase.Nested = run.OutputDirectory;

            files = dir(fullfile(testCase.Training, '*.png'));
            testCase.Image = string(fullfile(files(1).folder, files(1).name));
            stream = RandStream('mt19937ar', 'Seed', 140);
            for label = ["man", "woman"]
                for idx = 1:3
                    pixels = uint8(randi(stream, [1, 254], 8, 8));
                    imwrite(pixels, fullfile(testCase.Testing, label + "_" + idx + "_1.png"));
                end
            end
            final = run_final_training(testCase.Training, Synthetic=true, ...
                OutputDirectory=fullfile(testCase.Output, "final-evaluation"), ...
                Benchmark=struct('WarmupRuns', 0, 'NumRepeats', 1));
            testCase.Final = final.OutputDirectory;
        end
    end

    methods (Test)
        function quickUsesSavedArtifactsAndIsDeterministic(testCase)
            config = testCase.options();
            config.TestingPath = missing; % Path resolution would reject it if accessed.
            config.ImagePath = testCase.Image;
            config.FoldDetails = true;
            args = namedargs2cell(config);
            before = testCase.fileList();
            state = rng;
            output = evalc('first = run_demo("quick", args{:});');
            second = run_demo("quick", args{:});
            testCase.verifyEqual(first.Prediction, second.Prediction);
            testCase.verifyEqual(first.Baseline.Summary, second.Baseline.Summary);
            testCase.verifyEqual(first.Nested.Summary, second.Nested.Summary);
            testCase.verifyEqual(first.Final.Summary, second.Final.Summary);
            testCase.verifyEqual(testCase.fileList(), before);
            testCase.verifyEqual(rng, state);
            for label = ["Synthetic fixture | baseline CV", "Synthetic fixture | nested CV", ...
                    "Synthetic fixture final models", "Training example | software demonstration", ...
                    "AccuracyPct", "AccuracySDPct", "AUCSD", "ReducedFeatures", ...
                    "Projection", "Normalization", "ClassifierFitMs", "PipelineFitMs", ...
                    "ClassifierBatchMs", "PipelineBatchMs", "BatchSize", "revision"]
                testCase.verifyTrue(contains(output, label), label);
            end
            testCase.verifyFalse(isfield(first, 'Independent'));
            testCase.verifyEqual(first.Final.OutputDirectory, testCase.Final);
            testCase.verifyEqual(first.Baseline.OutputDirectory, testCase.Baseline);
            summary = first.Prediction.Summary;
            testCase.verifyEqual(summary.Method.', ["pca-bayes","lda-bayes","pca-mlp","lda-mlp"]);
            testCase.verifyEqual(summary.Label, double(summary.Score > 0));
            testCase.verifyEqual(summary.ReducedFeatures.', [10 1 20 1]);
            testCase.verifyEqual(summary.OriginalFeatures, 4096*ones(4,1));
            testCase.verifyEqual(summary.ScoreType.', ["Bayes log-posterior odds", ...
                "Bayes log-posterior odds", "MLP logit", "MLP logit"]);
            pipelines = loadFinalModels(testCase.Final);
            for k = 1:numel(pipelines)
                [y, score] = predictPipeline(pipelines(k), {imread(testCase.Image)});
                testCase.verifyEqual(summary.Label(k), y);
                testCase.verifyEqual(summary.Score(k), score);
            end
        end

        function unavailableIsNonfatalAndMakesNoFigure(testCase)
            config = testCase.options();
            config.TestingPath = fullfile(testCase.Work, "absent");
            before = testCase.fileList();
            [output, result] = capturedDemo("independent-status", config);
            testCase.verifyEqual(result.Independent.Status, "unavailable");
            testCase.verifyEmpty(result.Independent.Records);
            testCase.verifyTrue(contains(output, "Synthetic independent testing dataset unavailable."));
            testCase.verifyTrue(contains(output, "Independent accuracy, ROC, AUC, and inference time remain pending."));
            testCase.verifyEqual(testCase.fileList(), before);
        end

        function evaluatesAndReloadsSyntheticAttemptUnderConfiguredOutput(testCase)
            config = testCase.options();
            config.OutputDirectory = fullfile(testCase.Output, "test-evaluation");
            args = namedargs2cell(config);
            ready = run_demo("independent-status", args{:}, TestingPath=testCase.Testing);
            testCase.verifyEqual(ready.Independent.Status, "ready");
            result = run_demo("independent", args{:}, TestingPath=testCase.Testing);
            testCase.assertEqual(result.Independent.Status, "complete");
            path = result.Independent.OutputDirectory;
            testCase.verifyTrue(startsWith(path, fullfile(config.OutputDirectory, "independent-test")));
            testCase.verifyEqual([result.Independent.Records.EvaluationStage], repmat("synthetic", 1, 4));
            testCase.verifyEqual([result.Independent.Records.NumEvaluationSamples], 6*ones(1,4));
            metadata = jsondecode(fileread(fullfile(path, "test-metadata.json")));
            testCase.verifyFalse(metadata.OfficialEvaluation);
            before = testCase.fileList();
            saved = run_demo("independent", args{:}, IndependentAttempt=path, TestingPath=missing);
            testCase.verifyEqual(saved.Independent.Summary, result.Independent.Summary);
            testCase.verifyEqual(testCase.fileList(), before);
            testCase.verifyError(@() run_demo("independent-status", FinalRun=testCase.Final, ...
                IndependentAttempt=path), 'genderRecognition:DemoDataScope');
            other = fullfile(testCase.Work, "different-run");
            copyfile(path, other);
            metadata.TrainingCompletedUTC = "Different final training timestamp";
            writelines(jsonencode(metadata), fullfile(other, "test-metadata.json"));
            testCase.verifyError(@() run_demo("independent-status", args{:}, IndependentAttempt=other), ...
                'genderRecognition:DemoIndependentMismatch');

            copy = fullfile(fileparts(path), "attempt-second");
            copyfile(path, copy);
            testCase.verifyError(@() run_demo("independent-status", args{:}), ...
                'genderRecognition:DemoAmbiguousRun');
        end

        function ambiguousCVRequiresExplicitSelection(testCase)
            copy = fullfile(fileparts(testCase.Baseline), "run-second");
            copyfile(testCase.Baseline, copy);
            testCase.addTeardown(@() rmdir(copy, 's'));
            config = testCase.options();
            config.BaselineRun = "";
            args = namedargs2cell(config);
            testCase.verifyError(@() run_demo("results", args{:}), 'genderRecognition:DemoAmbiguousRun');
            selected = run_demo("results", BaselineRun=copy, NestedRun=testCase.Nested, Synthetic=true);
            testCase.verifyEqual(selected.Baseline.OutputDirectory, copy);
        end

        function missingAndCorruptModelBundleAreExplicit(testCase)
            testCase.verifyError(@() run_demo("models", FinalRun=fullfile(testCase.Work, "missing"), ...
                Synthetic=true), 'genderRecognition:DemoMissingArtifact');
            corrupt = fullfile(testCase.Work, "bad-bundle");
            copyfile(testCase.Final, corrupt);
            writelines("tampered", fullfile(corrupt, "models", "lda-mlp.mat"));
            testCase.verifyError(@() run_demo("independent-status", FinalRun=corrupt, ...
                Synthetic=true, TestingPath=missing), 'genderRecognition:FinalArtifactInvalid');
        end

        function regeneratesLabeledMeanAndFoldFigures(testCase)
            args = namedargs2cell(testCase.options());
            figures = run_demo("figures", args{:}, FoldROC=true);
            folder = figures.FigureDirectory;
            testCase.verifyTrue(startsWith(folder, testCase.Output));
            testCase.verifyTrue(isfile(fullfile(folder, "baseline", "mean-roc.png")));
            testCase.verifyTrue(isfile(fullfile(folder, "nested", "fold-roc.pdf")));
            testCase.verifyTrue(isfile(fullfile(folder, "baseline-vs-nested.pdf")));
            testCase.verifyFalse(isfolder(fullfile(folder, "independent")));
        end
    end

    methods (Access = private)
        function config = options(testCase)
            config = struct('TrainingPath', testCase.Training, 'BaselineRun', testCase.Baseline, ...
                'NestedRun', testCase.Nested, 'FinalRun', testCase.Final, ...
                'OutputDirectory', testCase.Output, 'Synthetic', true);
        end

        function names = fileList(testCase)
            entries = dir(fullfile(testCase.Work, '**', '*'));
            names = sort(string(fullfile({entries.folder}, {entries.name})));
        end
    end
end

function [output, result] = capturedDemo(action, config)
args = namedargs2cell(config);
operation = @() run_demo(action, args{:}); %#ok<NASGU> Used by evalc.
[output, result] = evalc('operation()');
end
