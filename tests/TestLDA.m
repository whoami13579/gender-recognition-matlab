classdef TestLDA < matlab.unittest.TestCase
    methods (Test)
        function agreesWithFullRankDenseReference(testCase)
            [X, y] = fullRankFeatures();
            model = fitLDA(X, y);
            [reference, alpha, R, delta] = denseReference(X, y, 1e-2);
            testCase.verifyEqual(model.ClassCounts, [4 3]);
            testCase.verifyEqual(model.ClassMeans, [0 0 2; 3 1 5]);
            testCase.verifyEqual(model.Mean, [9 3 23] / 7, 'AbsTol', 1e-15);
            testCase.verifyEqual(R, [-1 0 0; 1 0 0; 0 -1 0; 0 1 0; 0 0 -2; 0 0 0; 0 0 2]);
            testCase.verifyEqual(model.WithinClassScatterTrace, 12, 'AbsTol', 1e-14);
            testCase.verifyEqual(model.ProjectedWithinClassScatter, norm(R * reference)^2, 'AbsTol', 1e-13);
            testCase.verifyEqual(model.WithinClassRank, 3);
            testCase.verifyEqual(model.RankTolerance, max(size(R)) * eps(norm(R, 2)));
            testCase.verifyEqual(model.Options, struct('Regularization', 0.01));
            testCase.verifyEqual(model.EffectiveRegularization, alpha, 'RelTol', 1e-14);
            testCase.verifyEqual(model.Components, reference, 'AbsTol', 1e-13);
            testCase.verifyEqual(norm(model.Components), 1, 'AbsTol', 1e-14);
            testCase.verifyGreaterThan(model.Components.' * delta, 0);
            testCase.verifyEqual(model.NumTrainingSamples, 7);
            testCase.verifyEqual(model.NumInputFeatures, 3);
            testCase.verifyEqual(model.NumFeatures, 1);
            testCase.verifyEqual(model.NumFittedParameters, 6);
            Z = transformLDA(model, X);
            testCase.verifySize(Z, [7 1]);
            testCase.verifyEqual(Z, (X - mean(X, 1)) * reference, 'AbsTol', 1e-13);
            testCase.verifyEqual(model.ProjectedClassMeans, ...
                [mean(Z(y == 0)) mean(Z(y == 1))], 'AbsTol', 1e-13);
        end

        function preservesDiscriminationOutsideWithinClassSubspace(testCase)
            [X, y] = wideFeatures();
            model = fitLDA(X, y);
            [reference, alpha, R, delta] = denseReference(X, y, 1e-2);
            testCase.verifyLessThan(size(X, 1), size(X, 2));
            testCase.verifyEqual(model.WithinClassRank, 1);
            testCase.verifyEqual(model.EffectiveRegularization, alpha, 'RelTol', 1e-14);
            testCase.verifyEqual(model.Components, reference, 'AbsTol', 5e-13);
            % Feature 3 has no within-class variation and carries a class mean shift.
            testCase.verifyEqual(R(:, 3), zeros(size(X, 1), 1));
            testCase.verifyGreaterThan(delta(3), 0);
            testCase.verifyGreaterThan(model.Components(3), 0.5);
            testCase.verifyLessThan(systemResidual(model.Components, R, delta, alpha), 1e-10);
        end

        function maximizesRegularizedFisherCriterion(testCase)
            [X, y] = wideFeatures();
            [reference, alpha, R, delta] = denseReference(X, y, 0.3);
            model = fitLDA(X, y, struct('Regularization', 0.3));
            stream = RandStream('mt19937ar', 'Seed', 48);
            candidates = [randn(stream, size(X, 2), 40), eye(size(X, 2)), delta, reference];
            objectives = (delta.' * candidates).^2 ./ ...
                (sum((R * candidates).^2, 1) + alpha * sum(candidates.^2, 1));
            testCase.verifyEqual(model.RegularizedFisherCriterion, objectives(end), 'RelTol', 1e-13);
            testCase.verifyGreaterThanOrEqual(model.RegularizedFisherCriterion + 1e-12, max(objectives));
            testCase.verifyGreaterThan(model.RegularizedFisherCriterion, max(objectives(1:40)));
        end

        function usesSpectrumBelowNumericalRankCutoff(testCase)
            X = [-1 0 0; 1 0 0; 0 9e-16 1e-15; 0 1.1e-15 1e-15];
            y = [0; 0; 1; 1];
            rho = 3e-33;
            model = fitLDA(X, y, struct('Regularization', rho));
            % The scatter is diagonal, so its tiny but nonzero eigenvalue
            % contributes to the direction even below the rank cutoff.
            [R, delta] = referenceResiduals(X, y);
            scatterDiagonal = sum(R.^2, 1).';
            alpha = rho * sum(scatterDiagonal) / size(X, 2);
            reference = delta ./ (scatterDiagonal + alpha);
            reference = reference / norm(reference);
            testCase.verifyEqual(model.WithinClassRank, 1);
            testCase.verifyGreaterThan(model.RankTolerance, sqrt(2) * 1e-16);
            testCase.verifyEqual(model.Components, reference, 'AbsTol', 1e-13);
            testCase.verifyLessThan(model.Components(2), 0.1);
        end

        function supportsSingleFeatureAndUnequalClassCounts(testCase)
            X = [8; 1; 3];
            y = [1; 0; 0];
            model = fitLDA(X, y);
            testCase.verifyEqual(model.ClassCounts, [2 1]);
            testCase.verifyEqual(model.WithinClassRank, 1);
            testCase.verifyEqual(model.Components, 1);
            testCase.verifyEqual(transformLDA(model, X), [4; -3; -1]);
            swapped = fitLDA(X, 1 - y);
            testCase.verifyEqual(swapped.ClassMeans, flipud(model.ClassMeans));
            testCase.verifyEqual(swapped.Components, -1);
        end

        function handlesDuplicateSamplesAndRedundantFeatures(testCase)
            [X, y] = fullRankFeatures();
            X = [X(:, [1 1 2]), 4 * ones(size(y))];
            X = [X; X(1, :)];
            y = [y; y(1)];
            model = fitLDA(X, y);
            reference = denseReference(X, y, 1e-2);
            testCase.verifyEqual(model.WithinClassRank, 2);
            testCase.verifyEqual(model.Components, reference, 'AbsTol', 1e-13);
            testCase.verifyEqual(model.Components(1), model.Components(2), 'AbsTol', 1e-13);
            testCase.verifyEqual(model.Components(4), 0, 'AbsTol', 1e-14);
        end

        function handlesExactlyZeroWithinClassScatter(testCase)
            X = [repmat([0.1 0.4], 3, 1); repmat([0.6 0.2], 4, 1)];
            y = [zeros(3, 1); ones(4, 1)];
            model = fitLDA(X, y);
            delta = [0.5; -0.2];
            testCase.verifyTrue(model.UsedZeroScatterFallback);
            testCase.verifyEqual(model.WithinClassRank, 0);
            testCase.verifyEqual(model.WithinClassScatterTrace, 0);
            testCase.verifyEqual(model.ProjectedWithinClassScatter, 0);
            testCase.verifyEqual(model.EffectiveRegularization, 1e-2 * norm(delta)^2 / 2, 'RelTol', 1e-14);
            testCase.verifyEqual(model.Components, delta / norm(delta), 'AbsTol', 1e-14);
            testCase.verifyEqual(model.RegularizedFisherCriterion, 200, 'RelTol', 1e-14);
            testCase.verifyEqual(model.RelativeResidual, 0);

            % The ridge remains representable even though the raw squared mean shift does not.
            largeModel = fitLDA([0; 1e155], [0; 1], struct('Regularization', 1e-4));
            testCase.verifyEqual(largeModel.Components, 1);
            testCase.verifyEqual(largeModel.EffectiveRegularization, 1e306, 'RelTol', 1e-14);
        end

        function rejectsIdenticalClassMeans(testCase)
            testCase.verifyError(@() fitLDA([-1 0; 1 0; 0 -1; 0 1], [0; 0; 1; 1]), ...
                'genderRecognition:IdenticalLDAClassMeans');
            testCase.verifyError(@() fitLDA([0.1; 0.1; 0.1; 0.1], [0; 0; 0; 1]), ...
                'genderRecognition:IdenticalLDAClassMeans');
        end

        function scalesRidgeWithFeatureUnits(testCase)
            [X, y] = wideFeatures();
            baseline = fitLDA(X, y);
            Z = transformLDA(baseline, X);
            for scale = [1e-140 1e140]
                scaled = fitLDA(X * scale, y);
                testCase.verifyEqual(scaled.EffectiveRegularization / scale^2, ...
                    baseline.EffectiveRegularization, 'RelTol', 1e-13);
                testCase.verifyEqual(scaled.WithinClassRank, baseline.WithinClassRank);
                testCase.verifyEqual(scaled.Components, baseline.Components, 'AbsTol', 2e-13);
                testCase.verifyEqual(scaled.RegularizedFisherCriterion, ...
                    baseline.RegularizedFisherCriterion, 'RelTol', 1e-13);
                testCase.verifyEqual(transformLDA(scaled, X * scale) / scale, Z, 'AbsTol', 1e-12);
            end
            singleModel = fitLDA(single(X), y, struct('Regularization', single(0.01)));
            testCase.verifyEqual(singleModel, fitLDA(X, y, struct('Regularization', double(single(0.01)))));
            testCase.verifyClass(transformLDA(singleModel, single(X)), 'double');
        end

        function honorsMultipleRegularizationStrengths(testCase)
            [X, y] = fullRankFeatures();
            for rho = [1e-6 0.01 10 1e8]
                model = fitLDA(X, y, struct('Regularization', rho));
                [reference, alpha, ~, delta] = denseReference(X, y, rho);
                testCase.verifyEqual(model.RegularizationCoefficient, rho);
                testCase.verifyEqual(model.EffectiveRegularization, alpha, 'RelTol', 1e-14);
                testCase.verifyEqual(model.Components, reference, 'AbsTol', 1e-13);
            end
            testCase.verifyEqual(model.Components, delta / norm(delta), 'AbsTol', 2e-8);
        end

        function usesFixedProjectionAndPreservesRowOrder(testCase)
            [X, y] = wideFeatures();
            model = fitLDA(X, y);
            snapshot = model;
            validation = [101 9 -13 2 6 10; -4 3 8 1 2 -3];
            Z = transformLDA(model, validation);
            testCase.verifyEqual(Z, (validation - mean(X, 1)) * model.Components, 'AbsTol', 1e-13);
            testCase.verifyEqual(transformLDA(model, validation(2, :)), Z(2), 'AbsTol', 1e-13);
            changedBatch = transformLDA(model, [1e6 * validation(1, :); validation(2, :)]);
            testCase.verifyEqual(changedBatch(2), Z(2), 'AbsTol', 1e-13);
            testCase.verifyEqual(model, snapshot);
            % Inference only depends on the fitted mean, direction, and dimensions.
            projection = struct('Mean', model.Mean, 'Components', model.Components, ...
                'NumInputFeatures', model.NumInputFeatures, 'NumFeatures', 1);
            testCase.verifyEqual(transformLDA(projection, validation), Z);
            order = [4 1 3 2].';
            ZTrain = transformLDA(model, X);
            testCase.verifyEqual(transformLDA(model, X(order, :)), ZTrain(order), 'AbsTol', 1e-13);
            testCase.verifyGreaterThan(mean(ZTrain(y == 1)), mean(ZTrain(y == 0)));
        end

        function solvesRepresentativeHighDimensionalProblem(testCase)
            stream = RandStream('mt19937ar', 'Seed', 701);
            X = randn(stream, 160, 4096);
            y = [zeros(80, 1); ones(80, 1)];
            X(y == 1, 1:20) = X(y == 1, 1:20) + 1;
            model = fitLDA(X, y);
            Z = transformLDA(model, X);
            testCase.verifyLessThanOrEqual(model.WithinClassRank, 158);
            testCase.verifyEqual(model.WithinClassRank, 158);
            testCase.verifySize(model.Components, [4096 1]);
            testCase.verifySize(Z, [160 1]);
            testCase.verifyEqual(model.NumFittedParameters, 8192);
            fields = struct2cell(model);
            for k = 1:numel(fields)
                if isnumeric(fields{k})
                    testCase.verifyTrue(all(isfinite(fields{k}), 'all'));
                end
            end
            testCase.verifyTrue(all(isfinite(Z)));
            [R, delta] = referenceResiduals(X, y);
            alpha = 1e-2 * sum(R.^2, 'all') / size(X, 2);
            testCase.verifyEqual(model.EffectiveRegularization, alpha, 'RelTol', 1e-12);
            testCase.verifyLessThan(systemResidual(model.Components, R, delta, alpha), 1e-9);
            testCase.verifyLessThan(model.RelativeResidual, 1e-9);
        end

        function rejectsInvalidFeatureRepresentations(testCase)
            validLabels = [0; 1];
            wrong = {[], zeros(2, 0), zeros(0, 2), zeros(2, 2, 2), uint8([1; 3]), ...
                true(2, 1), {1; 2}, [1; 2i], sparse([1; 2]), [NaN; 1], [1; Inf]};
            for k = 1:numel(wrong)
                testCase.verifyError(@() fitLDA(wrong{k}, validLabels), 'genderRecognition:InvalidLDAFeatures');
            end
        end

        function rejectsInvalidLabelsAndMissingClasses(testCase)
            X = [0; 1; 3; 4];
            wrong = {[], [0 0 1 1], [0; 1], [0; 0; 1; 1; 1], [0; 0; 1; 2], [0; 0; 1; NaN], ...
                [0; 0; 1; Inf], [0; 0; 1; 1i], single([0; 0; 1; 1]), ...
                logical([0; 0; 1; 1]), sparse([0; 0; 1; 1])};
            for k = 1:numel(wrong)
                testCase.verifyError(@() fitLDA(X, wrong{k}), 'genderRecognition:InvalidLabels');
            end
            testCase.verifyError(@() fitLDA(X, zeros(4, 1)), 'genderRecognition:MissingClass');
            testCase.verifyError(@() fitLDA(1, 1), 'genderRecognition:MissingClass');
        end

        function rejectsInvalidConfiguration(testCase)
            [X, y] = fullRankFeatures();
            badConfig = {[], "default", struct('Unknown', 1), repmat(struct(), 2, 1), ...
                struct('Regularization', 0, 'Unknown', 1)};
            badValues = {0, -1, Inf, NaN, 1i, [1 2], [], "0.01", sparse(1)};
            for k = 1:numel(badConfig)
                testCase.verifyError(@() fitLDA(X, y, badConfig{k}), 'genderRecognition:InvalidConfig');
            end
            for k = 1:numel(badValues)
                config = struct('Regularization', badValues{k});
                testCase.verifyError(@() fitLDA(X, y, config), 'genderRecognition:InvalidConfig');
            end
        end

        function rejectsUnrepresentableNumericalResults(testCase)
            [X, y] = fullRankFeatures();
            id = 'genderRecognition:LDANumericalRange';
            testCase.verifyError(@() fitLDA(X * 1e-170, y), id);
            testCase.verifyError(@() fitLDA(X * 1e155, y), id);
            testCase.verifyError(@() fitLDA([-realmax; realmax], [0; 1]), id);
            testCase.verifyError(@() fitLDA(X, y, struct('Regularization', realmax)), id);
            testCase.verifyError(@() fitLDA(X, y, struct('Regularization', realmin / 10)), id);
            model = fitLDA(realmax * [0.25; 0.25 + eps(1)], [0; 1], ...
                struct('Regularization', realmin));
            testCase.verifyError(@() transformLDA(model, -realmax), id);
        end

        function validatesModelAndTransformDimensions(testCase)
            [X, y] = fullRankFeatures();
            model = fitLDA(X, y);
            malformed = {[], struct(), rmfield(model, 'Mean'), [model model], ...
                replaceField(model, 'Mean', model.Mean.'), replaceField(model, 'Mean', [0 0 Inf]), ...
                replaceField(model, 'Components', zeros(3, 1)), ...
                replaceField(model, 'Components', [model.Components model.Components]), ...
                replaceField(model, 'Components', sparse(model.Components)), ...
                replaceField(model, 'Components', [1; 0; NaN]), ...
                replaceField(model, 'NumFeatures', 2), replaceField(model, 'NumInputFeatures', 4)};
            for k = 1:numel(malformed)
                testCase.verifyError(@() transformLDA(malformed{k}, X), 'genderRecognition:InvalidLDAModel');
            end
            testCase.verifyError(@() transformLDA(model, X(:, 1)), 'genderRecognition:LDAFeatureDimensionMismatch');
            testCase.verifyError(@() transformLDA(model, zeros(0, 2)), 'genderRecognition:LDAFeatureDimensionMismatch');
            testCase.verifyError(@() transformLDA(model, uint8(X)), 'genderRecognition:InvalidLDAFeatures');
            testCase.verifyError(@() transformLDA(model, [1 2 Inf]), 'genderRecognition:InvalidLDAFeatures');
            testCase.verifySize(transformLDA(model, zeros(0, 3, 'single')), [0 1]);
        end

        function integratesSubjectGroupedPngFoldsWithoutLeakage(testCase)
            fixture = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            writeSyntheticImages(fixture.Folder);
            [images, y, subjects, metadata] = loadDataset(fixture.Folder);
            [folds, partitionInfo] = createFolds(y, subjects);
            trainIdx = folds(1).TrainIndices;
            validationIdx = folds(1).ValidationIndices;
            testCase.verifyEqual(metadata.Label, y);
            testCase.verifyEqual(metadata.SubjectID, subjects);
            testCase.verifyEmpty(intersect(subjects(trainIdx), subjects(validationIdx)));
            testCase.verifyEqual(partitionInfo.SubjectOverlapCounts(1), 0);

            config = struct('ImageSize', []);
            preprocessor = fitPreprocessor(images(trainIdx), config);
            XTrain = transformImages(preprocessor, images(trainIdx));
            XValidation = transformImages(preprocessor, images(validationIdx));
            yTrain = y(trainIdx);
            model = fitLDA(XTrain, yTrain);
            ZTrain = transformLDA(model, XTrain);
            ZValidation = transformLDA(model, XValidation);
            testCase.verifyEqual(model.NumTrainingSamples, 16);
            testCase.verifyEqual(model.ClassCounts, [8 8]);
            testCase.verifySize(ZTrain, [16 1]);
            testCase.verifySize(ZValidation, [4 1]);
            testCase.verifyGreaterThan(mean(ZTrain(yTrain == 1)), mean(ZTrain(yTrain == 0)));
            for k = 1:numel(trainIdx)
                i = trainIdx(k);
                [label, subject, imageNumber] = parseImageFilename(metadata.Filename(i));
                pixels = syntheticPixels(label, str2double(extractAfter(subject, "_")), str2double(imageNumber));
                testCase.verifyEqual(label, yTrain(k));
                testCase.verifyEqual(subject, subjects(i));
                testCase.verifyEqual(images{i}, reshape(pixels, 2, 2));
                testCase.verifyEqual(XTrain(k, :), double(pixels) / 255, 'AbsTol', eps);
                testCase.verifyEqual(ZTrain(k), (XTrain(k, :) - model.Mean) * model.Components, 'AbsTol', 1e-13);
            end

            for i = validationIdx.'
                imwrite(255 - images{i}, fullfile(fixture.Folder, metadata.Filename(i)));
            end
            [changedImages, sameLabels, sameSubjects, sameMetadata] = loadDataset(fixture.Folder);
            testCase.verifyEqual(sameLabels, y);
            testCase.verifyEqual(sameSubjects, subjects);
            testCase.verifyEqual(sameMetadata, metadata);
            refitPrep = fitPreprocessor(changedImages(trainIdx), config);
            refitX = transformImages(refitPrep, changedImages(trainIdx));
            refitModel = fitLDA(refitX, sameLabels(trainIdx));
            changedValidation = transformImages(refitPrep, changedImages(validationIdx));
            testCase.verifyEqual(refitPrep, preprocessor);
            testCase.verifyEqual(refitX, XTrain);
            testCase.verifyEqual(refitModel, model);
            testCase.verifyEqual(transformLDA(refitModel, refitX), ZTrain);
            testCase.verifyNotEqual(transformLDA(refitModel, changedValidation), ZValidation);
        end

        function reportsAggregateAnalysisForCompleteDataAndFold(testCase)
            fixture = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            writeSyntheticImages(fixture.Folder);
            config = struct('ImageSize', []);
            [report, stats] = evalc('run_lda_analysis(fixture.Folder, struct(''Regularization'', 0.2), config, 1)');
            testCase.verifyEqual(stats.TrainFeatureSize, [16 4]);
            testCase.verifyEqual(stats.ValidationFeatureSize, [4 4]);
            testCase.verifyEqual(stats.ReducedTrainSize, [16 1]);
            testCase.verifyEqual(stats.ReducedValidationSize, [4 1]);
            testCase.verifyEqual(stats.RegularizationCoefficient, 0.2);
            testCase.verifyEqual(stats.SubjectOverlapCount, 0);
            testCase.verifyFalse(any(contains(report, [".png", "man_", "woman_", string(fixture.Folder)])));
            testCase.verifyFalse(any(isfield(stats, {'Mean', 'Components', 'ClassMeans', 'TrainIndices'})));
            [~, complete] = evalc('run_lda_analysis(fixture.Folder, struct(), config)');
            testCase.verifyEqual(complete.TrainFeatureSize, [20 4]);
            testCase.verifyEqual(complete.ReducedTrainSize, [20 1]);
            testCase.verifyTrue(contains(complete.FitScope, "descriptive"));
            testCase.verifyFalse(isfield(complete, 'ValidationFeatureSize'));
            testCase.verifyError(@() run_lda_analysis(fixture.Folder, struct(), config, 6), ...
                'genderRecognition:InvalidFoldNumber');
            testCase.verifyError(@() run_lda_analysis(fixture.Folder, struct(), config, 0), ...
                'genderRecognition:InvalidFoldNumber');
        end
    end
end

function [X, y] = fullRankFeatures()
X = [-1 0 2; 1 0 2; 0 -1 2; 0 1 2; 3 1 3; 3 1 5; 3 1 7];
y = [0; 0; 0; 0; 1; 1; 1];
end

function [X, y] = wideFeatures()
X = [-1 -1 0 4 0 0; 1 1 0 4 0 0; 1 0 3 4 2 0; 3 2 3 4 2 0];
y = [0; 0; 1; 1];
end

function [R, delta] = referenceResiduals(X, y)
mu0 = mean(X(y == 0, :), 1);
mu1 = mean(X(y == 1, :), 1);
R = [X(y == 0, :) - mu0; X(y == 1, :) - mu1];
delta = (mu1 - mu0).';
end

function [direction, alpha, R, delta] = denseReference(X, y, rho)
[R, delta] = referenceResiduals(X, y);
SW = R.' * R;
alpha = rho * trace(SW) / size(X, 2);
direction = (SW + alpha * eye(size(SW))) \ delta;
direction = direction / norm(direction);
if direction.' * delta < 0
    direction = -direction;
end
end

function relative = systemResidual(direction, R, delta, alpha)
lhs = R.' * (R * direction) + alpha * direction;
scale = (direction.' * lhs) / (direction.' * delta);
relative = norm(lhs - scale * delta) / (scale * norm(delta));
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
            pixels = syntheticPixels(label, subject, imageNumber);
            imwrite(reshape(pixels, 2, 2), fullfile(folder, filename));
        end
    end
end
end
