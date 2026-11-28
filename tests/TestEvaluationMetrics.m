classdef TestEvaluationMetrics < matlab.unittest.TestCase
    methods (Test)
        function handCalculatedConfusionAndAccuracy(testCase)
            truth = [0; 0; 1; 1; 1; 0; 1; 0];
            predicted = [0; 1; 1; 0; 1; 0; 1; 0];
            metrics = evaluateClassification(truth, predicted, (1:8).');
            testCase.verifyEqual(metrics.ConfusionMatrix, [3 1; 1 3]);
            testCase.verifyEqual(metrics.Accuracy, 0.75);
            testCase.verifyEqual(metrics.NumCorrect, 6);
            testCase.verifyEqual(metrics.NumEvaluationSamples, 8);
        end

        function canonicalRankingCases(testCase)
            truth = [0; 0; 1; 1];
            scoreColumns = [-2 2 3 0; -1 1 3 1; 1 -1 3 1; 2 -2 3 2];
            expected = [1 0 0.5 0.875];
            for idx = 1:numel(expected)
                scores = scoreColumns(:, idx);
                [roc, auc] = computeROC(truth, scores);
                testCase.verifyEqual(auc, expected(idx));
                testCase.verifyEqual(auc, pairwiseAUC(truth, scores));
                testCase.verifyTrue(roc.Valid);
                testCase.verifyEqual(roc.Status, "OK");
            end
            tied = computeROC(truth, scoreColumns(:, 3));
            testCase.verifyEqual(tied.FPR, [0; 1]);
            testCase.verifyEqual(tied.TPR, [0; 1]);
            testCase.verifyEqual(tied.Thresholds, [Inf; 3]);
            testCase.verifyEqual(tied.NumDistinctScores, 1);
        end

        function groupedRatesThresholdsAndUnequalCounts(testCase)
            truth = [0; 1; 0; 1; 0; 0; 1];
            scores = [1; 2; 1; 1; -1; 0; 0];
            [roc, auc] = computeROC(truth, scores);
            testCase.verifyEqual(roc.Thresholds, [Inf; 2; 1; 0; -1]);
            testCase.verifyEqual(roc.FPR, [0; 0; 2; 3; 4] / 4);
            testCase.verifyEqual(roc.TPR, [0; 1; 2; 3; 3] / 3);
            testCase.verifyEqual([roc.NumNegativeSamples roc.NumPositiveSamples roc.NumDistinctScores], [4 3 4]);
            testCase.verifyEqual(auc, 17/24, 'AbsTol', 2 * eps);
            testCase.verifyEqual(auc, pairwiseAUC(truth, scores), 'AbsTol', 2 * eps);
            for idx = 1:numel(roc.Thresholds)
                predictedPositive = scores >= roc.Thresholds(idx);
                testCase.verifyEqual(roc.TPR(idx), nnz(predictedPositive & truth == 1) / 3);
                testCase.verifyEqual(roc.FPR(idx), nnz(predictedPositive & truth == 0) / 4);
            end
        end

        function tiedGroupAndFullPermutationInvariance(testCase)
            y = [0; 1; 0; 1; 0; 1];
            scores = [1; 1; 1; 2; 0; 0];
            [expected, expectedAUC] = computeROC(y, scores);
            for permutation = [2 3 1 4 6 5; 6 4 5 2 1 3; 6 5 4 3 2 1].'
                [actual, auc] = computeROC(y(permutation), scores(permutation));
                testCase.verifyEqual(actual, expected);
                testCase.verifyEqual(auc, expectedAUC);
            end
        end

        function randomizedAUCMatchesIndependentRankCounting(testCase)
            initialRNG = rng;
            stream = RandStream('mt19937ar', 'Seed', 90);
            for trial = 1:100
                n = randi(stream, [3 35]);
                y = [0; 1; randi(stream, [0 1], n - 2, 1)];
                scores = randi(stream, [-5 5], n, 1);
                order = randperm(stream, n);
                [roc, auc] = computeROC(y(order), scores(order));
                testCase.verifyEqual(auc, pairwiseAUC(y, scores), 'AbsTol', 8 * eps);
                testCase.verifyGreaterThanOrEqual(auc, -eps);
                testCase.verifyLessThanOrEqual(auc, 1 + eps);
                testCase.verifyEqual([roc.FPR([1 end]), roc.TPR([1 end])], [0 0; 1 1]);
                testCase.verifyTrue(issorted(roc.Thresholds, 'strictdescend'));
                testCase.verifyTrue(issorted(roc.FPR) && issorted(roc.TPR));
                [transformed, transformedAUC] = computeROC(y, 3 * scores + 17);
                testCase.verifyEqual(transformed.FPR, roc.FPR);
                testCase.verifyEqual(transformed.TPR, roc.TPR);
                testCase.verifyEqual(transformedAUC, auc);
            end
            testCase.verifyEqual(rng, initialRNG);
        end

        function veryLargeAndAdjacentFiniteScores(testCase)
            y = [0; 1; 0; 1; 0; 1];
            scores = [-realmax; -realmax + eps(realmax); 1; 1 + eps(1); realmax - eps(realmax); realmax];
            [roc, auc] = computeROC(y, scores);
            testCase.verifyEqual(roc.Thresholds, [Inf; flipud(scores)]);
            testCase.verifyEqual(roc.NumDistinctScores, 6);
            testCase.verifyEqual(auc, 2/3, 'AbsTol', 2 * eps);
            testCase.verifyEqual(auc, pairwiseAUC(y, scores), 'AbsTol', 2 * eps);
        end

        function zeroDecisionAndIndependentAccuracy(testCase)
            truth = [0; 0; 1; 1];
            scores = [-1000; -2; 0; 1000];
            supplied = evaluateClassification(truth, double(scores > 0), scores);
            testCase.verifyEqual(supplied.ConfusionMatrix, [2 0; 1 1]);
            indexAtZero = supplied.ROC.Thresholds == 0;
            testCase.verifyEqual(supplied.ROC.TPR(indexAtZero), 1);
            testCase.verifyEqual(supplied.AUC, 1);
            differentPredictions = evaluateClassification(truth, truth, scores);
            testCase.verifyEqual(differentPredictions.Accuracy, 1);
            testCase.verifyEqual(differentPredictions.ROC, supplied.ROC);
            samePredictions = evaluateClassification(truth, double(scores > 0), -scores);
            testCase.verifyEqual(samePredictions.Accuracy, supplied.Accuracy);
            testCase.verifyEqual(samePredictions.AUC, 0);
        end

        function bothSingleClassDiagnostics(testCase)
            for label = [0 1]
                result = evaluateClassification(repmat(label, 3, 1), [0; 1; 1], [-1; 1; 1]);
                testCase.verifyFalse(result.ROC.Valid);
                testCase.verifyTrue(isnan(result.AUC));
                testCase.verifyEmpty(result.ROC.FPR);
                testCase.verifyEmpty(result.ROC.TPR);
                testCase.verifyEmpty(result.ROC.Thresholds);
                testCase.verifyEqual(result.ROC.NumDistinctScores, 2);
                if label == 0
                    testCase.verifyEqual(result.ROC.Status, "MissingPositiveClass");
                    testCase.verifyEqual(result.ConfusionMatrix, [1 2; 0 0]);
                    testCase.verifyEqual(result.Accuracy, 1/3);
                else
                    testCase.verifyEqual(result.ROC.Status, "MissingNegativeClass");
                    testCase.verifyEqual(result.ConfusionMatrix, [0 0; 1 2]);
                    testCase.verifyEqual(result.Accuracy, 2/3);
                end
            end
        end

        function supportedLabelAndScoreTypes(testCase)
            scores = single([1; 1 + eps('single')]);
            metrics = evaluateClassification(logical([0; 1]), single([0; 1]), scores);
            testCase.verifyEqual(metrics.ROC.Thresholds, [Inf; double(flipud(scores))]);
            testCase.verifyEqual(metrics.Accuracy, 1);
            testCase.verifyEqual(metrics.AUC, 1);
        end

        function invalidInputsFailExplicitly(testCase)
            id = 'genderRecognition:InvalidEvaluationInput';
            truth = [0; 1];
            scores = [-1; 1];
            invalidLabels = {[], [0 1], [0 1; 1 0], reshape([0 1], 1, 1, 2), ...
                sparse(truth), [0; 1i], [0; NaN], [0; Inf], [-1; 1], [0; 0.5], ...
                uint8(truth), ["0"; "1"]};
            invalidScores = {[], 1, [0 1], [0 1; 1 0], reshape([0 1], 1, 1, 2), ...
                sparse(scores), [0; 1i], [0; NaN], [0; Inf], [0; -Inf], uint8(truth), logical(truth), {0; 1}};
            for vector = invalidLabels
                testCase.verifyError(@() computeROC(vector{1}, scores), id);
                testCase.verifyError(@() evaluateClassification(truth, vector{1}, scores), id);
            end
            for vector = invalidScores
                testCase.verifyError(@() computeROC(truth, vector{1}), id);
            end
            testCase.verifyError(@() evaluateClassification(truth, 0, scores), id);
            testCase.verifyError(@() evaluateClassification([], [], []), id);
        end
    end
end

function auc = pairwiseAUC(labels, scores)
% Count each positive-negative pair independently of sorting, ROC, and trapz.
positive = scores(labels == 1);
negative = scores(labels == 0);
wins = nnz(positive > negative.');
ties = nnz(positive == negative.');
auc = (wins + 0.5 * ties) / (numel(positive) * numel(negative));
end
