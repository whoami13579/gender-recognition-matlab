classdef TestHyperparameterSelection < matlab.unittest.TestCase
    methods (Test)
        function nestedSearchUsesSharedInnerSplitsAndRefitsOuterTraining(testCase)
            [images, labels, subjects, config] = cvFixture();
            config.Mode = "nested";
            config.Search.RecommendFinal = true;
            config.Search.Candidates = struct('pca_bayes', {{struct('PCA', struct('NumComponents', 1)), ...
                struct('PCA', struct('NumComponents', 99))}}, 'lda_bayes', {{}}, ...
                'pca_mlp', {{}}, 'lda_mlp', {{}});
            rngBefore = rng;
            results = crossValidatePipelines(images, labels, subjects, config);
            testCase.verifyEqual(rng, rngBefore);
            testCase.assertEqual(results.Status, "complete");
            testCase.verifyNumElements(results.Records, 20);
            testCase.verifyNumElements(results.FinalSearch.Selections, 4);
            testCase.verifySize(results.Recommendations, [1 4]);
            for outer = 1:5
                train = results.Folds(outer).TrainIndices;
                holdout = results.Folds(outer).ValidationIndices;
                inner = results.Inner{outer};
                testCase.verifyEqual(inner.Partition.RandomSeed, config.Search.RandomSeed);
                for fold = 1:2
                    ii = train(inner.Folds(fold).TrainIndices);
                    iv = train(inner.Folds(fold).ValidationIndices);
                    testCase.verifyEmpty(intersect([ii; iv], holdout));
                    testCase.verifyEmpty(intersect(subjects(ii), subjects(iv)));
                end
                for methodIdx = 1:4
                    method = results.Methods(methodIdx);
                    selected = results.Searches{outer, methodIdx};
                    testCase.verifyEqual(selected.Candidates(1).Configuration, config.Pipeline);
                    for k = find([selected.Candidates.Valid])
                        candidate = selected.Candidates(k);
                        for fold = 1:2
                            ii = train(inner.Folds(fold).TrainIndices);
                            iv = train(inner.Folds(fold).ValidationIndices);
                            ref = fitPipeline(images(ii), labels(ii), method, candidate.Configuration);
                            [pred, score] = predictPipeline(ref, images(iv));
                            metrics = evaluateClassification(labels(iv), pred, score);
                            testCase.verifyEqual(candidate.Folds(fold).Accuracy, metrics.Accuracy);
                            testCase.verifyEqual(candidate.Folds(fold).AUC, metrics.AUC);
                            testCase.verifyEqual(candidate.Folds(fold).NumTrainingSamples, numel(ii));
                        end
                    end
                    idx = [results.Records.Method] == method & [results.Records.FoldNumber] == outer;
                    ref = fitPipeline(images(train), labels(train), method, selected.SelectedConfiguration);
                    [pred, score] = predictPipeline(ref, images(holdout));
                    testCase.verifyEqual(results.Predictions(idx).Predictions, pred);
                    testCase.verifyEqual(results.Predictions(idx).Scores, score);
                    testCase.verifyEqual(results.Records(idx).NumTrainingSamples, numel(train));
                end
                invalid = results.Searches{outer, 1}.Candidates(3);
                testCase.verifyFalse(invalid.Valid);
                testCase.verifyEqual([invalid.Folds.FailureID], ...
                    repmat("genderRecognition:PCAComponentsExceedRank", 1, 2));
            end
        end

        function candidateSelectionIsUnaffectedByAlteredOuterHoldout(testCase)
            [images, labels, subjects, config] = cvFixture();
            folds = createFolds(labels, subjects);
            fit = folds(2).TrainIndices;
            val = folds(2).ValidationIndices;
            changed = images;
            changed(val) = repmat({uint8(255 * ones(6))}, numel(val), 1);
            changedLabels = labels;
            changedLabels(val) = 1 - labels(val);
            options = struct('NumFolds', 2, 'RandomSeed', 42, 'GroupBySubject', true);
            inner = createFolds(labels(fit), subjects(fit), options);
            innerAltered = createFolds(changedLabels(fit), subjects(fit), options);
            testCase.verifyEqual(inner, innerAltered);
            choices = {config.Pipeline, config.Pipeline};
            first = selectPipelineConfig(images(fit), labels(fit), inner, 'lda-mlp', choices);
            second = selectPipelineConfig(changed(fit), changedLabels(fit), innerAltered, 'lda-mlp', choices);
            testCase.verifyEqual(rmfield(first.Candidates, 'Seconds'), rmfield(second.Candidates, 'Seconds'));
            testCase.verifyEqual(first.SelectedConfiguration, second.SelectedConfiguration);
            testCase.verifyEqual([first.SelectedIndex second.SelectedIndex], [1 1]);
        end

        function allCandidateFailuresYieldAnExplicitIncompleteExperiment(testCase)
            [images, labels, subjects, config] = cvFixture();
            config.Mode = "nested";
            config.Pipeline.PCA.NumComponents = 100;
            config.Search.Candidates = struct('pca_bayes', {{}}, 'lda_bayes', {{}}, ...
                'pca_mlp', {{}}, 'lda_mlp', {{}});
            results = crossValidatePipelines(images, labels, subjects, config);
            testCase.verifyEqual(results.Status, "incomplete");
            testCase.verifyNumElements(results.Failures, 10);
            testCase.verifyTrue(all([results.Failures.Identifier] == "genderRecognition:NoValidCandidate"));
            testCase.verifyNumElements(results.Records, 10);
            testCase.verifyEmpty(results.Searches{1, 1}.SelectedConfiguration);
            testCase.verifyEqual(results.Searches{1, 1}.Candidates(1).Folds(1).FailureID, ...
                "genderRecognition:PCAComponentsExceedRank");
            [summary, ~] = summarizeCrossValidation(results.Records);
            testCase.verifyTrue(isnan(summary.MeanAUC(1)));
            testCase.verifyEqual(summary.CompletedFolds, [0; 5; 0; 5]);
        end
    end
end
