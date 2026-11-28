classdef TestGaussianBayes < matlab.unittest.TestCase
    methods (Test)
        function estimatesMaximumLikelihoodParameters(testCase)
            [Z, y] = correlatedFeatures();
            rho = 0.12;
            [means, raw, pooled] = referenceEstimates(Z, y);
            expectedScale = trace(pooled) / 2;
            for mode = ["full" "shared" "diagonal"]
                model = fitBayes(Z, y, struct('CovarianceType', mode, 'Regularization', rho));
                testCase.verifyEqual(model.ClassLabels, [0 1]);
                testCase.verifyEqual(model.ClassCounts, [5 4]);
                testCase.verifyEqual(model.NumTrainingSamples, 9);
                testCase.verifyEqual(model.NumFeatures, 2);
                testCase.verifyEqual(model.Priors, [5 4] / 9);
                testCase.verifyEqual(model.ClassMeans, means, 'AbsTol', 1e-14);
                testCase.verifyEqual(model.CovarianceScale, expectedScale, 'RelTol', 1e-14);
                testCase.verifyEqual(model.CovarianceScaleSource, "withinClass");
                testCase.verifyEqual(model.EffectiveRegularization, rho * expectedScale, 'RelTol', 1e-14);
                expected = regularizedReference(raw, pooled, mode, rho * expectedScale);
                testCase.verifyEqual(size(model.Covariances, 3), size(expected, 3));
                for c = 1:size(expected, 3)
                    actual = model.Covariances(:, :, c);
                    L = model.CovarianceFactors(:, :, c);
                    if mode == "diagonal"
                        actual = diag(actual);
                        L = diag(L);
                    end
                    testCase.verifyEqual(actual, expected(:, :, c), 'AbsTol', 1e-13);
                    testCase.verifyEqual(L * L.', actual, 'AbsTol', 1e-13);
                    testCase.verifyEqual(model.LogDeterminants(c), sum(log(eig(actual))), 'AbsTol', 1e-13);
                end
            end
            testCase.verifyNotEqual(raw(:, :, 1), cov(Z(y == 0, :), 0));
        end

        function matchesUnivariateReferenceWithTwoTails(testCase)
            Z = [-1; 0; 1; -3; -1; 1; 3];
            y = [zeros(3, 1); ones(4, 1)];
            model = fitBayes(Z, y);
            testCase.verifyEqual(model.Options, ...
                struct('CovarianceType', "full", 'PriorType', "empirical", 'Regularization', 0.01));
            alpha = 0.01 * 22 / 7;
            v = [2/3 5] + alpha;
            queries = [-4; -1; 0; 1; 4];
            g = log([3 4] / 7) - 0.5 * log(2*pi*v) - queries.^2 ./ (2*v);
            expected = g(:, 2) - g(:, 1);
            [labels, scores] = predictBayes(model, queries);
            testCase.verifyEqual(model.EffectiveRegularization, alpha, 'RelTol', 1e-14);
            testCase.verifyEqual(scores, expected, 'AbsTol', 1e-13);
            testCase.verifyEqual(labels, [1; 0; 0; 0; 1]);
        end

        function matchesIndependentMultivariateReference(testCase)
            [Z, y] = correlatedFeatures();
            queries = [-3 -4; 0 0; 1 2; 2 1; 3 4; 5 4; 7 1];
            [means, raw, pooled] = referenceEstimates(Z, y);
            rho = 0.035;
            for mode = ["full" "shared" "diagonal"]
                sigma = regularizedReference(raw, pooled, mode, rho * trace(pooled) / 2);
                for prior = ["empirical" "equal"]
                    priors = [5 4] / 9;
                    if prior == "equal"
                        priors = [0.5 0.5];
                    end
                    model = fitBayes(Z, y, struct('CovarianceType', mode, ...
                        'PriorType', prior, 'Regularization', rho));
                    expected = denseReference(queries, means, sigma, priors);
                    [labels, scores] = predictBayes(model, queries);
                    testCase.verifyEqual(model.Priors, priors);
                    testCase.verifyEqual(scores, expected, 'AbsTol', 1e-12);
                    testCase.verifyEqual(labels, double(expected > 0));
                end
            end
        end

        function countsIndependentParameters(testCase)
            modes = ["full" "shared" "diagonal"];
            expected = [5 1721; 4 901; 5 161];
            dimensions = [1 40];
            for d = 1:numel(dimensions)
                K = dimensions(d);
                Z = (1:6).' * (1:K);
                y = [0; 0; 0; 1; 1; 1];
                for m = 1:numel(modes)
                    config = struct('CovarianceType', modes(m), 'PriorType', "empirical");
                    model = fitBayes(Z, y, config);
                    testCase.verifyEqual(model.NumClassifierParameters, expected(m, d));
                    config.PriorType = "equal";
                    fixedModel = fitBayes(Z, y, config);
                    testCase.verifyEqual(fixedModel.NumClassifierParameters, expected(m, d) - 1);
                end
            end
        end

        function regularizesWideRedundantAndNearlyCollinearFeatures(testCase)
            Z = [0 0 0 7 1 0; 2 2 2+1e-9 7 1 0; 3 3 3 7 2 0; 5 5 5-1e-9 7 2 0];
            y = [0; 0; 1; 1];
            for mode = ["full" "shared" "diagonal"]
                model = fitBayes(Z, y, struct('CovarianceType', mode));
                [~, scores] = predictBayes(model, [Z; 12 12 12 7 5 1]);
                testCase.verifyTrue(all(isfinite(scores)));
                testCase.verifyGreaterThan(model.EffectiveRegularization, 0);
                if mode == "diagonal"
                    testCase.verifyEqual(model.CovarianceRanks, [3 3]);
                elseif mode == "shared"
                    testCase.verifyLessThanOrEqual(model.CovarianceRanks, 2);
                else
                    testCase.verifyEqual(model.CovarianceRanks, [1 1]);
                end
            end
        end

        function handlesZeroWithinClassVarianceAndSingleton(testCase)
            Z = [1 2; 4 6; 4 6; 4 6];
            y = [0; 1; 1; 1];
            expectedScale = (1/4) * (3/4) * (3^2 + 4^2) / 2;
            for mode = ["full" "shared" "diagonal"]
                model = fitBayes(Z, y, struct('CovarianceType', mode, 'PriorType', "equal"));
                testCase.verifyEqual(model.ClassCounts, [1 3]);
                testCase.verifyEqual(model.CovarianceScaleSource, "betweenClass");
                testCase.verifyEqual(model.CovarianceScale, expectedScale, 'RelTol', 1e-14);
                testCase.verifyTrue(all(model.CovarianceRanks == 0));
                [labels, scores] = predictBayes(model, [Z(1, :); Z(2, :)]);
                testCase.verifyEqual(labels, [0; 1]);
                testCase.verifyEqual(scores, [-1; 1] * 25 / (2 * 0.01 * expectedScale), 'RelTol', 1e-14);

                other = fitBayes([Z(1:3, :); 6 8], y, struct('CovarianceType', mode));
                testCase.verifyEqual(other.CovarianceScaleSource, "withinClass");
                [~, otherScores] = predictBayes(other, Z);
                testCase.verifyTrue(all(isfinite(otherScores)));
            end
        end

        function usesOnlyPriorsForIdenticalConstantFeatures(testCase)
            Z = repmat([0.1 -realmax], 5, 1);
            y = [0; 0; 1; 1; 1];
            for mode = ["full" "shared" "diagonal"]
                model = fitBayes(Z, y, struct('CovarianceType', mode));
                testCase.verifyEqual(model.CovarianceScaleSource, "constant");
                testCase.verifyEqual(model.CovarianceScale, 1);
                testCase.verifyEqual(model.EffectiveRegularization, 0.01);
                [labels, scores] = predictBayes(model, [0 0; realmax realmax]);
                testCase.verifyEqual(scores, repmat(log(3/2), 2, 1), 'AbsTol', eps);
                testCase.verifyEqual(labels, [1; 1]);
            end
        end

        function preservesLdaScaleScoresWhenDensitiesUnderflow(testCase)
            offset = [-sqrt(2); -1; 0; 1; sqrt(2)] * 1e-3;
            Z = [-1.75 + offset; 1.75 + 1.08 * offset];
            y = [zeros(5, 1); ones(5, 1)];
            model = fitBayes(Z, y);
            frozen = model;
            queries = [-2.4; -1.75; -1.1; -0.7; 0; 0.7; 1.1; 1.75; 2.4];
            [labels, scores] = predictBayes(model, queries);
            [means, raw, pooled] = referenceEstimates(Z, y);
            reference = denseReference(queries, means, raw + 0.01 * pooled, [0.5 0.5]);
            variances = reshape(model.Covariances, 1, 2);
            densities = exp(-0.5 * (queries - model.ClassMeans.').^2 ./ variances) ./ sqrt(2*pi*variances);
            testCase.verifyTrue(all(densities(4:6, :) == 0, 'all'));
            testCase.verifyTrue(all(isfinite(scores)));
            testCase.verifyEqual(scores, reference, 'AbsTol', 1e-7);
            testCase.verifyEqual(labels, double(reference > 0));
            testCase.verifyEqual(numel(unique(scores)), numel(queries));
            testCase.verifyGreaterThan(min(scores(6:9)), 1e3);
            testCase.verifyEqual(1 ./ (1 + exp(-scores(6:9))), ones(4, 1));
            testCase.verifyEqual(model, frozen);
        end

        function preservesScoreOrderAcrossPredictionBatches(testCase)
            Z = [-3; -2; -1; 1; 2; 3];
            y = [0; 0; 0; 1; 1; 1];
            queries = [2; -2; 0; 1; -1];
            for mode = ["full" "shared" "diagonal"]
                model = fitBayes(Z, y, struct('CovarianceType', mode));
                [labels, scores] = predictBayes(model, queries);
                testCase.verifyEqual(labels, [1; 0; 0; 1; 0]);
                testCase.verifyEqual(scores(3), 0);
                testCase.verifyGreaterThan(diff(scores([2 5 3 4 1])), zeros(4, 1));
                for k = 1:numel(queries)
                    [singleLabel, singleScore] = predictBayes(model, queries(k));
                    testCase.verifyEqual(singleLabel, labels(k));
                    testCase.verifyEqual(singleScore, scores(k), 'AbsTol', 1e-13);
                end
                [~, reordered] = predictBayes(model, [1e5; queries(4); queries(2); -1e5]);
                testCase.verifyEqual(reordered(2:3), scores([4 2]), 'AbsTol', 1e-13);
            end
        end

        function supportsSingleEmptyAndExtremeValidScales(testCase)
            [Z, y] = correlatedFeatures();
            queries = [1 3; 3 0; -2 1];
            for mode = ["full" "shared" "diagonal"]
                config = struct('CovarianceType', char(mode), 'Regularization', single(0.05));
                model = fitBayes(single(Z), y, config);
                testCase.verifyClass(model.ClassMeans, 'double');
                [labels, scores] = predictBayes(model, single(queries));
                testCase.verifyClass(labels, 'double');
                testCase.verifyClass(scores, 'double');
                [emptyLabels, emptyScores] = predictBayes(model, zeros(0, 2, 'single'));
                testCase.verifyEqual(emptyLabels, zeros(0, 1));
                testCase.verifyEqual(emptyScores, zeros(0, 1));
                for scale = [1e-145 1e150]
                    scaled = fitBayes(Z * scale, y, config);
                    [scaledLabels, scaledScores] = predictBayes(scaled, queries * scale);
                    testCase.verifyEqual(scaledLabels, labels);
                    testCase.verifyEqual(scaledScores, scores, 'AbsTol', 4e-13);
                    testCase.verifyEqual(scaled.EffectiveRegularization / scale^2, ...
                        model.EffectiveRegularization, 'RelTol', 1e-13);
                end
            end
            large = fitBayes([-1e154; 1e154; -5e153; 5e153], [0; 0; 1; 1]);
            [~, largeScores] = predictBayes(large, [-1e154; 0; 1e154]);
            testCase.verifyTrue(all(isfinite(largeScores)));
        end

        function honorsMultipleRegularizationStrengths(testCase)
            [Z, y] = correlatedFeatures();
            for rho = [1e-8 0.01 0.75]
                model = fitBayes(Z, y, struct('Regularization', rho));
                testCase.verifyEqual(model.RegularizationCoefficient, rho);
                testCase.verifyEqual(model.EffectiveRegularization, rho * model.CovarianceScale);
                [~, raw] = referenceEstimates(Z, y);
                testCase.verifyEqual(model.Covariances(:, :, 1) - raw(:, :, 1), ...
                    model.EffectiveRegularization * eye(2), 'AbsTol', 1e-13);
            end
        end

        function rejectsInvalidFeaturesAndLabels(testCase)
            Z = [1; 2; 3; 4];
            y = [0; 0; 1; 1];
            invalidFeatures = {[], zeros(0, 1), zeros(4, 0), sparse(Z), uint8(Z), ...
                logical(Z), {Z}, "features", Z + 1i, [NaN; Z(2:4)], [Inf; Z(2:4)], zeros(4, 1, 2)};
            for bad = invalidFeatures
                testCase.verifyError(@() fitBayes(bad{1}, y), 'genderRecognition:InvalidBayesFeatures');
            end
            invalidLabels = {y.', y(1:3), single(y), sparse(y), y + 1i, [y y], ...
                [0; 1; 2; 0], [0; 1; NaN; 0], [0; 1; Inf; 0]};
            for bad = invalidLabels
                testCase.verifyError(@() fitBayes(Z, bad{1}), 'genderRecognition:InvalidBayesLabels');
            end
            testCase.verifyError(@() fitBayes(Z, zeros(4, 1)), 'genderRecognition:MissingBayesClass');
            testCase.verifyError(@() fitBayes(Z, ones(4, 1)), 'genderRecognition:MissingBayesClass');
        end

        function rejectsInvalidOptions(testCase)
            invalid = {[], "full", repmat(struct(), 2, 1), struct('Unused', 1), ...
                struct('CovarianceType', "auto"), struct('CovarianceType', ["full" "shared"]), ...
                struct('CovarianceType', missing), struct('PriorType', "custom"), struct('PriorType', 1), ...
                struct('PriorType', missing), struct('PriorType', ["equal" "empirical"])};
            for value = {0, -1, Inf, NaN, 1i, sparse(1), "0.01", [1 2], true}
                invalid{end+1} = struct('Regularization', value{1}); %#ok<AGROW>
            end
            for bad = invalid
                testCase.verifyError(@() fitBayes([1; 2], [0; 1], bad{1}), ...
                    'genderRecognition:InvalidBayesConfig');
            end
        end

        function rejectsMalformedModelsAndPredictionInputs(testCase)
            [Z, y] = correlatedFeatures();
            model = fitBayes(Z, y);
            invalid = {[], struct(), [model model], rmfield(model, 'Priors'), ...
                replaceField(model, 'ClassMeans', [NaN 0; 1 1]), replaceField(model, 'NumFeatures', 7), ...
                replaceField(model, 'ClassLabels', [1 0]), replaceField(model, 'Priors', [0 1]), ...
                replaceField(model, 'Priors', [0.8 0.7]), replaceField(model, 'CovarianceType', "shared"), ...
                replaceField(model, 'LogDeterminants', [0 Inf]), ...
                replaceField(model, 'LogDeterminants', model.LogDeterminants + 1), ...
                replaceField(model, 'CovarianceFactors', cat(3, [1 1; 0 1], eye(2))), ...
                replaceField(model, 'CovarianceFactors', cat(3, zeros(2), eye(2)))};
            for bad = invalid
                testCase.verifyError(@() predictBayes(bad{1}, Z), 'genderRecognition:InvalidBayesModel');
            end
            testCase.verifyError(@() predictBayes(model, zeros(0, 3)), ...
                'genderRecognition:BayesFeatureDimensionMismatch');
            testCase.verifyError(@() predictBayes(model, [0 NaN]), 'genderRecognition:InvalidBayesFeatures');
            testCase.verifyError(@() predictBayes(model, sparse(Z)), 'genderRecognition:InvalidBayesFeatures');
        end

        function rejectsUnusableCovarianceAndUnrepresentableDistances(testCase)
            id = 'genderRecognition:BayesNumericalRange';
            y = [0; 0; 1; 1];
            testCase.verifyError(@() fitBayes([-1e200; 1e200; -1; 1], y), id);
            testCase.verifyError(@() fitBayes([-1e-180; 1e-180; -1e-180; 1e-180], y), id);
            testCase.verifyError(@() fitBayes([1; 1+eps; 2; 2+eps(2)], y, ...
                struct('Regularization', realmin)), id);
            testCase.verifyError(@() fitBayes([0; 8; 1; 9], y, struct('Regularization', realmax)), id);
            redundant = repmat([-1; -1; 1; 1; 2; 2; 4; 4], 1, 2);
            testCase.verifyError(@() fitBayes(redundant, repelem([0; 1], 4), ...
                struct('Regularization', realmin)), 'genderRecognition:BayesCovarianceFactorization');
            model = fitBayes([-1; 1; 2; 4], y);
            testCase.verifyError(@() predictBayes(model, realmax), id);
            testCase.verifyError(@() predictBayes(model, 1e-200), id);
        end

        function integratesPcaAndLdaOnSubjectGroupedPngs(testCase)
            fixture = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            writeSyntheticImages(fixture.Folder);
            [images, y, subjects, metadata] = loadDataset(fixture.Folder);
            [folds, partition] = createFolds(y, subjects);
            train = folds(1).TrainIndices;
            validation = folds(1).ValidationIndices;
            testCase.verifyEqual(y, [zeros(10, 1); ones(10, 1)]);
            testCase.verifyEqual(metadata.Label, y);
            testCase.verifyEqual(metadata.SubjectID, subjects);
            testCase.verifyEqual(partition.SubjectOverlapCounts(1), 0);
            testCase.verifyEmpty(intersect(subjects(train), subjects(validation)));
            for i = validation.'
                imwrite(255 - images{i}, fullfile(fixture.Folder, metadata.Filename(i)));
            end
            [changed, changedY, changedSubjects] = loadDataset(fixture.Folder);
            testCase.verifyEqual(changedY, y);
            testCase.verifyEqual(changedSubjects, subjects);
            testCase.verifyEqual(createFolds(changedY, changedSubjects), folds);
            testCase.verifyEqual(changed(train), images(train));
            testCase.verifyNotEqual(changed(validation), images(validation));
            config = struct('ImageSize', []);
            prep = fitPreprocessor(images(train), config);
            changedPrep = fitPreprocessor(changed(train), config);
            testCase.verifyEqual(changedPrep, prep);
            X = transformImages(prep, images(train));
            changedX = transformImages(changedPrep, changed(train));
            testCase.verifyEqual(changedX, X);
            XVal = transformImages(prep, images(validation));
            changedVal = transformImages(changedPrep, changed(validation));
            for i = 1:numel(train)
                source = train(i);
                [label, subject, imageNumber] = parseImageFilename(metadata.Filename(source));
                expected = syntheticPixels(label, str2double(extractAfter(subject, "_")), str2double(imageNumber));
                testCase.verifyEqual(label, y(source));
                testCase.verifyEqual(X(i, :), double(expected) / 255, 'AbsTol', eps);
            end
            for K = [2 1]
                if K == 2
                    project = @transformPCA;
                    projection = fitPCA(X, struct('NumComponents', K));
                    refit = fitPCA(changedX, struct('NumComponents', K));
                else
                    project = @transformLDA;
                    projection = fitLDA(X, y(train));
                    refit = fitLDA(changedX, changedY(train));
                end
                testCase.verifyEqual(refit, projection);
                Z = project(projection, X);
                changedZ = project(refit, changedX);
                testCase.verifySize(Z, [16 K]);
                testCase.verifyEqual(changedZ, Z);
                model = fitBayes(Z, y(train));
                refitModel = fitBayes(changedZ, changedY(train));
                testCase.verifyEqual(refitModel, model);
                testCase.verifyEqual(model.ClassCounts, [8 8]);
                [means, ~, ~] = referenceEstimates(Z, y(train));
                testCase.verifyEqual(model.ClassMeans, means, 'AbsTol', 1e-14);
                [labels, scores] = predictBayes(model, project(projection, XVal));
                [~, changedScores] = predictBayes(refitModel, project(refit, changedVal));
                testCase.verifySize(labels, [4 1]);
                testCase.verifySize(scores, [4 1]);
                testCase.verifyTrue(all(isfinite([scores; changedScores])));
                testCase.verifyNotEqual(changedScores, scores);
                testCase.verifyEqual(refitModel, model);
            end
        end

        function analysisReportsOnlyAggregateInformation(testCase)
            fixture = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            writeSyntheticImages(fixture.Folder);
            prep = struct('ImageSize', []);
            feature = struct('NumComponents', 2);
            bayes = struct('CovarianceType', "shared");
            [report, stats] = evalc('run_bayes_analysis(fixture.Folder, "pca", bayes, feature, prep, 2)');
            testCase.verifyEqual(stats.FoldNumber, 2);
            testCase.verifyEqual(stats.TrainClassCounts, [8 8]);
            testCase.verifyEqual(stats.ValidationClassCounts, [2 2]);
            testCase.verifyEqual(stats.SubjectOverlapCount, 0);
            testCase.verifyEqual(stats.ReducedTrainSize, [16 2]);
            testCase.verifyEqual(stats.CovarianceSize, [2 2 1]);
            testCase.verifyEqual(stats.NumClassifierParameters, 8);
            testCase.verifyEqual(stats.NumProjectionParameters, 12);
            testCase.verifyEqual(stats.NumFiniteScores, 4);
            testCase.verifyEqual(stats.ScoreSize, [4 1]);
            testCase.verifyTrue(stats.AllScoresFinite);
            testCase.verifyFalse(any(contains(report, [".png", "man_", "woman_", string(fixture.Folder)])));
            testCase.verifyFalse(any(isfield(stats, {'ClassMeans', 'Covariances', 'Scores', ...
                'Predictions', 'TrainIndices', 'ValidationIndices'})));
            testCase.verifyError(@() run_bayes_analysis(fixture.Folder, "other"), ...
                'genderRecognition:InvalidBayesAnalysisMethod');
            testCase.verifyError(@() run_bayes_analysis(fixture.Folder, "pca", bayes, feature, prep, 6), ...
                'genderRecognition:InvalidFoldNumber');
        end
    end
end

function [Z, y] = correlatedFeatures()
Z = [-2 -3; -1 -1; 0 1; 1 2; 2 1; 1 3; 3 -1; 5 2; 6 5];
y = [zeros(5, 1); ones(4, 1)];
end

function [means, raw, pooled] = referenceEstimates(Z, y)
male = Z(y == 0, :);
female = Z(y == 1, :);
means = [mean(male, 1); mean(female, 1)];
raw = cat(3, cov(male, 1), cov(female, 1));
pooled = (size(male, 1) * raw(:, :, 1) + size(female, 1) * raw(:, :, 2)) / size(Z, 1);
end

function sigma = regularizedReference(raw, pooled, mode, alpha)
ridge = alpha * eye(size(raw, 1));
switch mode
    case "full"
        sigma = raw + ridge;
    case "shared"
        sigma = pooled + ridge;
    case "diagonal"
        sigma = cat(3, diag(diag(raw(:, :, 1))) + ridge, diag(diag(raw(:, :, 2))) + ridge);
end
end

function scores = denseReference(queries, means, sigma, priors)
K = size(queries, 2);
discriminants = zeros(size(queries, 1), 2);
for c = 1:2
    covariance = sigma(:, :, min(c, size(sigma, 3)));
    residual = (queries - means(c, :)).';
    quadratic = dot(residual, covariance \ residual, 1).';
    logDet = sum(log(eig(covariance)));
    discriminants(:, c) = log(priors(c)) - 0.5 * (K * log(2*pi) + logDet + quadratic);
end
scores = discriminants(:, 2) - discriminants(:, 1);
end

function model = replaceField(model, name, value)
model.(name) = value;
end

function pixels = syntheticPixels(label, subject, imageNumber)
pixels = uint8([30 + 80 * label, 40 + 3 * subject, ...
    60 + 5 * imageNumber, 90 + subject * imageNumber]);
end

function writeSyntheticImages(folder)
classNames = ["man", "woman"];
for label = 0:1
    for subject = 1:5
        for imageNumber = 1:2
            filename = sprintf('%s_%d_%d.png', classNames(label + 1), subject, imageNumber);
            imwrite(reshape(syntheticPixels(label, subject, imageNumber), 2, 2), fullfile(folder, filename));
        end
    end
end
end
