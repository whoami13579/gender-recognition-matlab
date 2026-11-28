classdef TestBenchmarking < matlab.unittest.TestCase
    methods (Test)
        function warmupsAndRepeatedStatistics(testCase)
            calls = 0;
            [timing, firstCall] = benchmarkOperation(@visit, struct('WarmupRuns', 3, 'NumRepeats', 4));
            samples = timing.SamplesSeconds;
            testCase.verifyEqual(calls, 7);
            testCase.verifyEqual(firstCall, 4);
            testCase.verifyEqual(timing.WarmupRuns, 3);
            testCase.verifyEqual(timing.NumRepeats, 4);
            testCase.verifySize(samples, [4 1]);
            testCase.verifyTrue(all(isfinite(samples) & samples >= 0));
            sorted = sort(samples);
            independentMean = sum(samples) / 4;
            independentStd = sqrt(sum((samples - independentMean).^2) / 3);
            testCase.verifyEqual(timing.MedianSeconds, (sorted(2) + sorted(3)) / 2);
            testCase.verifyEqual(timing.MeanSeconds, independentMean);
            testCase.verifyEqual(timing.StdSeconds, independentStd, 'AbsTol', eps(max(samples)));
            testCase.verifyEqual(timing.MinSeconds, sorted(1));
            testCase.verifyEqual(timing.MaxSeconds, sorted(end));
            testCase.verifyEmpty(timing.BatchSize);
            testCase.verifyEmpty(timing.SecondsPerSample);

            function number = visit()
                calls = calls + 1;
                number = calls;
            end
        end

        function defaultSettingsSupportOperationsWithoutOutputs(testCase)
            calls = 0;
            timing = benchmarkOperation(@visit);
            testCase.verifyEqual(calls, 7);
            testCase.verifyEqual(timing.WarmupRuns, 2);
            testCase.verifyEqual(timing.NumRepeats, 5);

            function visit()
                calls = calls + 1;
            end
        end

        function singleRepetitionAndBatchDerivedLatency(testCase)
            config = struct('WarmupRuns', 0, 'NumRepeats', 1, 'BatchSize', 13);
            [timing, data] = benchmarkOperation(@() ones(13, 1), config);
            testCase.verifyEqual(timing.BatchSize, size(data, 1));
            testCase.verifyEqual(timing.SamplesSeconds, timing.MedianSeconds);
            testCase.verifyEqual(timing.MeanSeconds, timing.MedianSeconds);
            testCase.verifyEqual(timing.StdSeconds, 0);
            testCase.verifyEqual(timing.SecondsPerSample, timing.MedianSeconds / size(data, 1));
            testCase.verifyEqual(timing.MeanSecondsPerSample, timing.MeanSeconds / size(data, 1));

            config.BatchSize = 1;
            timing = benchmarkOperation(@() 1, config);
            testCase.verifyEqual(timing.SecondsPerSample, timing.MedianSeconds);
        end

        function invalidConfigurationAndOperation(testCase)
            id = 'genderRecognition:InvalidBenchmarkConfig';
            configs = {[], struct('Unknown', 1), repmat(struct(), 1, 2), ...
                struct('WarmupRuns', -1), struct('WarmupRuns', 0.5), struct('WarmupRuns', [0 1]), ...
                struct('NumRepeats', 0), struct('NumRepeats', Inf), struct('NumRepeats', NaN), ...
                struct('NumRepeats', true), struct('BatchSize', 0), struct('BatchSize', 2i), ...
                struct('BatchSize', "4"), struct('BatchSize', sparse(4))};
            for config = configs
                testCase.verifyError(@() benchmarkOperation(@() 1, config{1}), id);
            end
            testCase.verifyError(@() benchmarkOperation('sum'), 'genderRecognition:InvalidBenchmarkOperation');
        end

        function operationErrorPropagatesFromWarmupAndTimedRun(testCase)
            id = 'genderRecognition:TestOperationFailure';
            testCase.verifyError(@() benchmarkOperation(@fail), id);
            testCase.verifyError(@() benchmarkOperation(@fail, struct('WarmupRuns', 0)), id);

            function fail()
                error(id, 'Synthetic operation failure.');
            end
        end

        function freshSeededFitsAndFixedModelPrediction(testCase)
            initialRNG = rng;
            features = [-3; -2; -1; 1; 2; 3];
            y = double(features > 0);
            settings = struct('MaxEpochs', 5, 'RandomSeed', 17, 'HiddenLayerSizes', [3 2]);
            fits = cell(1, 4);
            fitCalls = 0;
            [fitTiming, model, firstCall] = benchmarkOperation(@fitOnce, struct('WarmupRuns', 1, 'NumRepeats', 3));
            testCase.verifyEqual(firstCall, 2);
            testCase.verifyEqual(fitTiming.NumRepeats, 3);
            testCase.verifyEqual(fitCalls, 4);
            testCase.verifyEqual(fits, repmat({model}, size(fits)));

            saved = model;
            batch = [-1; 0; 2];
            [expectedLabels, expectedScores] = predictMLP(saved, batch);
            predictionCalls = 0;
            [inference, predicted, scores] = benchmarkOperation(@predictOnce, ...
                struct('WarmupRuns', 1, 'NumRepeats', 3, 'BatchSize', size(batch, 1)));
            testCase.verifyEqual(predictionCalls, 4);
            testCase.verifyEqual(model, saved);
            testCase.verifyEqual(predicted, expectedLabels);
            testCase.verifyEqual(scores, expectedScores);
            testCase.verifyEqual(inference.BatchSize, size(batch, 1));
            testCase.verifyEqual(fitCalls, 4);
            testCase.verifyEqual(rng, initialRNG);

            function [fitted, callNumber] = fitOnce()
                fitCalls = fitCalls + 1;
                fitted = fitMLP(features, y, settings);
                fits{fitCalls} = fitted;
                callNumber = fitCalls;
            end

            function [labels, values] = predictOnce()
                predictionCalls = predictionCalls + 1;
                testCase.assertEqual(model, saved);
                [labels, values] = predictMLP(model, batch);
            end
        end
    end
end
