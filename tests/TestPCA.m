classdef TestPCA < matlab.unittest.TestCase
    methods (Test)
        function agreesWithSampleCovariance(testCase)
            X = tallFeatures();
            model = fitPCA(X, struct('NumComponents', 2));
            Z = transformPCA(model, X);
            referenceEigenvalues = sort(eig(cov(X)), 'descend');
            referenceRatios = referenceEigenvalues / trace(cov(X));

            testCase.verifyEqual(model.Mean, mean(X, 1), 'AbsTol', 1e-14);
            testCase.verifyEqual(mean(X - model.Mean, 1), zeros(1, 3), 'AbsTol', 1e-14);
            testCase.verifyEqual(model.Components.' * model.Components, eye(2), 'AbsTol', 1e-14);
            testCase.verifyGreaterThanOrEqual(model.Eigenvalues, zeros(3, 1));
            testCase.verifyLessThanOrEqual(diff(model.Eigenvalues), zeros(2, 1));
            testCase.verifyEqual(model.Eigenvalues, referenceEigenvalues, 'AbsTol', 1e-13);
            testCase.verifyEqual(model.ExplainedVarianceRatio, referenceRatios, 'AbsTol', 1e-14);
            testCase.verifyEqual(sum(model.ExplainedVarianceRatio), 1, 'AbsTol', 1e-14);
            testCase.verifyGreaterThanOrEqual(diff(model.CumulativeExplainedVariance), zeros(2, 1));
            testCase.verifyEqual(model.CumulativeExplainedVariance, cumsum(referenceRatios), 'AbsTol', 1e-14);
            testCase.verifyEqual(model.RetainedExplainedVariance, sum(referenceRatios(1:2)), 'AbsTol', 1e-14);
            testCase.verifyEqual(mean(Z, 1), zeros(1, 2), 'AbsTol', 1e-14);
            testCase.verifyEqual(cov(Z), diag(referenceEigenvalues(1:2)), 'AbsTol', 1e-13);
            testCase.verifyEqual(model.NumericalRank, 3);
            testCase.verifyEqual(model.NumTrainingSamples, 6);
            testCase.verifyEqual(model.NumInputFeatures, 3);
            testCase.verifyEqual(model.NumFeatures, 2);
            testCase.verifyEqual(model.NumFittedParameters, 9);
        end

        function selectsMinimumCountForVarianceThreshold(testCase)
            X = tallFeatures();
            thresholds = [0.5, 9/14, 0.90, 13/14, 13/14 + 64 * eps(1), 0.95, 0.99, 1];
            expected = [1 1 2 2 3 3 3 3];
            for k = 1:numel(thresholds)
                model = fitPCA(X, struct('ExplainedVarianceThreshold', thresholds(k)));
                testCase.verifyEqual(model.NumFeatures, expected(k));
                testCase.verifyEqual(model.Options.SelectionMode, "explainedVariance");
                testCase.verifyEmpty(model.Options.NumComponents);
            end
        end

        function projectsWithFixedTrainingState(testCase)
            X = tallFeatures();
            model = fitPCA(X, struct('NumComponents', 2));
            before = model;
            heldOut = [60 7 -2; 61 6 13; 74 3 8];
            Z = transformPCA(model, heldOut);
            expected = heldOut * model.Components - model.Mean * model.Components;

            testCase.verifySize(Z, [3 2]);
            testCase.verifyEqual(Z, expected, 'AbsTol', 3e-14);
            testCase.verifyEqual(transformPCA(model, heldOut(2, :)), Z(2, :), 'AbsTol', 3e-14);
            testCase.verifyEqual(transformPCA(model, heldOut([2 1 3], :)), Z([2 1 3], :), 'AbsTol', 3e-14);
            testCase.verifyEqual(model, before);
            testCase.verifySize(transformPCA(model, zeros(0, 3)), [0 2]);
        end

        function reconstructsCenteredDataWithNonincreasingError(testCase)
            X = tallFeatures();
            centered = X - mean(X, 1);
            errors = zeros(1, 3);
            referenceEigenvalues = sort(eig(cov(X)), 'descend');
            for k = 1:3
                model = fitPCA(X, struct('NumComponents', k));
                reconstructed = transformPCA(model, X) * model.Components.';
                errors(k) = norm(centered - reconstructed, 'fro');
                testCase.verifyEqual(errors(k)^2, (size(X, 1) - 1) * sum(referenceEigenvalues(k+1:end)), ...
                    'AbsTol', 1e-12);
            end
            testCase.verifyLessThanOrEqual(diff(errors), [0 0]);
            testCase.verifyLessThan(errors(end), 1e-13);
        end

        function handlesRepeatedEigenvaluesByTheirSubspace(testCase)
            X = [eye(2), zeros(2, 1); -eye(2), zeros(2, 1)];
            model = fitPCA(X);
            testCase.verifyEqual(model.Eigenvalues, [2/3; 2/3; 0], 'AbsTol', 1e-14);
            testCase.verifyEqual(model.Components * model.Components.', diag([1 1 0]), 'AbsTol', 1e-14);
            testCase.verifyEqual(transformPCA(model, X) * model.Components.', X, 'AbsTol', 1e-14);
        end

        function detectsRankDeficiencyInWideData(testCase)
            t = [-2; -1; -1; 0; 1; 2];
            u = [1; 0; 0; -1; 2; -2];
            X = [t, t, u, repmat(0.1, 6, 1), t + u, -t, 2*u, zeros(6, 1)];
            model = fitPCA(X);
            testCase.verifyEqual(model.NumericalRank, 2);
            testCase.verifyEqual(model.NumFeatures, 2);
            testCase.verifyEqual(model.Mean(4), 0.1);
            testCase.verifyEqual(model.Options.NumComponents, 40);
            testCase.verifyEqual(model.Options.SelectionMode, "default");
            testCase.verifyGreaterThan(model.RankTolerance, 0);
            testCase.verifySize(model.Eigenvalues, [6 1]);
            testCase.verifyEqual(transformPCA(model, X) * model.Components.', X - model.Mean, 'AbsTol', 1e-14);
            testCase.verifyError(@() fitPCA(X, struct('NumComponents', 3)), ...
                'genderRecognition:PCAComponentsExceedRank');
            testCase.verifyError(@() fitPCA(X, struct('NumComponents', 40)), ...
                'genderRecognition:PCAComponentsExceedRank');
            allVariance = fitPCA(X, struct('ExplainedVarianceThreshold', 1));
            testCase.verifyEqual(allVariance.NumFeatures, 2);
        end

        function handlesMinimumSampleAndFeatureCounts(testCase)
            % A rounding residual from the mean cannot increase centered rank.
            model = fitPCA([1, 0; 1 + eps(1), eps(1)]);
            testCase.verifyEqual(model.NumericalRank, 1);
            testCase.verifySize(transformPCA(model, zeros(0, 2)), [0 1]);

            X = single([1; 3]);
            model = fitPCA(X);
            Z = transformPCA(model, X);
            testCase.verifyClass(Z, 'double');
            testCase.verifyEqual(model.Mean, 2);
            testCase.verifyEqual(model.Eigenvalues, 2, 'AbsTol', 1e-14);
            testCase.verifyEqual(abs(Z), [1; 1]);
        end

        function handlesVerySmallAndVeryLargeScales(testCase)
            X = [1 0; -1 0; 0 0.5; 0 -0.5];
            for scale = [1e-155 1e154]
                model = fitPCA(scale * X);
                testCase.verifyEqual(model.NumericalRank, 2);
                testCase.verifyEqual(model.Eigenvalues, (scale / sqrt(3)).^2 * [2; 0.5], 'RelTol', 1e-12);
                testCase.verifyEqual(model.ExplainedVarianceRatio, [0.8; 0.2], 'AbsTol', 1e-14);
                Z = transformPCA(model, scale * X);
                testCase.verifyEqual((Z / scale) * model.Components.', X, 'AbsTol', 1e-14);
            end
        end

        function fitsSyntheticProjectDimensions(testCase)
            stream = RandStream('mt19937ar', 'Seed', 23);
            X = randn(stream, 160, 4096);
            model = fitPCA(X, struct('NumComponents', 40));
            testCase.verifySize(model.Mean, [1 4096]);
            testCase.verifySize(model.Components, [4096 40]);
            testCase.verifySize(model.Eigenvalues, [160 1]);
            testCase.verifyEqual(model.NumericalRank, 159);
            testCase.verifyEqual(model.NumFittedParameters, 4096 * 41);
            testCase.verifySize(transformPCA(model, X), [160 40]);
        end

        function rejectsInvalidFeaturesAndVariance(testCase)
            invalid = {[], zeros(0, 3), zeros(3, 0), [1 2 3], zeros(2, 2, 2), ...
                [1; NaN], [1; Inf], [1; 1i], sparse([0; 1]), uint8([0; 1]), [true; false], ["1"; "2"]};
            for k = 1:numel(invalid)
                testCase.verifyError(@() fitPCA(invalid{k}), 'genderRecognition:InvalidPCAFeatures');
            end
            for X = {zeros(3, 2), repmat([0.1 0.7 realmax], 7, 1)}
                testCase.verifyError(@() fitPCA(X{1}), 'genderRecognition:ZeroPCAVariance');
            end
            for X = {[realmax; realmax; -realmax], [-1e160; 1e160], [-1e-170; 1e-170]}
                testCase.verifyError(@() fitPCA(X{1}), 'genderRecognition:PCANumericalRange');
            end
        end

        function rejectsInvalidConfiguration(testCase)
            X = tallFeatures();
            invalid = {[], "fixed", repmat(struct(), 2, 1), struct('Unknown', 2), ...
                struct('NumComponents', 2, 'ExplainedVarianceThreshold', 0.9)};
            for value = {[], 0, -1, 1.5, Inf, NaN, 1i, [1 2], true, "2", sparse(2), 2 * flintmax}
                invalid{end + 1} = struct('NumComponents', value{1}); %#ok<AGROW>
            end
            for value = {[], 0, -0.5, 1.1, Inf, NaN, 1i, [0.5 0.9], true, "0.9", sparse(0.9)}
                invalid{end + 1} = struct('ExplainedVarianceThreshold', value{1}); %#ok<AGROW>
            end
            for k = 1:numel(invalid)
                testCase.verifyError(@() fitPCA(X, invalid{k}), 'genderRecognition:InvalidConfig');
            end
        end

        function rejectsMalformedModelsAndTransformInputs(testCase)
            X = tallFeatures();
            model = fitPCA(X);
            invalidModels = {[], struct(), repmat(model, 2, 1), rmfield(model, 'Mean'), ...
                replaceField(model, 'Mean', model.Mean.'), replaceField(model, 'Mean', [0 0 Inf]), ...
                replaceField(model, 'Components', single(model.Components)), ...
                replaceField(model, 'Components', complex(model.Components, 1)), ...
                replaceField(model, 'Components', nan(3)), replaceField(model, 'Components', zeros(3)), ...
                replaceField(model, 'NumFeatures', 1), ...
                replaceField(model, 'NumInputFeatures', 2), replaceField(model, 'NumericalRank', 4), ...
                replaceField(model, 'NumTrainingSamples', 1), replaceField(model, 'NumFittedParameters', 1)};
            for k = 1:numel(invalidModels)
                testCase.verifyError(@() transformPCA(invalidModels{k}, X), 'genderRecognition:InvalidPCAModel');
            end
            for value = {zeros(3, 2), []}
                testCase.verifyError(@() transformPCA(model, value{1}), ...
                    'genderRecognition:PCAFeatureDimensionMismatch');
            end
            for value = {nan(1, 3), [0 Inf 0], [1 2i 3], sparse(X), uint16(X), zeros(2, 3, 2)}
                testCase.verifyError(@() transformPCA(model, value{1}), 'genderRecognition:InvalidPCAFeatures');
            end

            model = fitPCA([-2 -2; 2 2]);
            testCase.verifyError(@() transformPCA(model, [realmax realmax]), 'genderRecognition:PCANumericalRange');
        end

        function integratesGroupedFoldsWithoutHeldOutLeakage(testCase)
            fixture = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            folder = fixture.Folder;
            writeSyntheticImages(folder);
            [images, y, subjectIDs, metadata] = loadDataset(folder);
            [folds, partitionInfo] = createFolds(y, subjectIDs);
            trainIdx = folds(1).TrainIndices;
            valIdx = folds(1).ValidationIndices;
            testCase.verifyEqual(trainIdx, setdiff((1:numel(y)).', valIdx));
            testCase.verifyEqual(partitionInfo.SampleToFold(valIdx), ones(4, 1));
            testCase.verifyEmpty(intersect(subjectIDs(trainIdx), subjectIDs(valIdx)));
            testCase.verifyEqual(y, metadata.Label);
            testCase.verifyEqual(subjectIDs, metadata.SubjectID);
            [filenameLabels, filenameIDs] = arrayfun(@parseImageFilename, metadata.Filename);
            testCase.verifyEqual(y, filenameLabels);
            testCase.verifyEqual(subjectIDs, filenameIDs);

            preprocessingConfig = struct('ImageSize', []);
            pcaConfig = struct('NumComponents', 2);
            preprocessor = fitPreprocessor(images(trainIdx), preprocessingConfig);
            XTrain = transformImages(preprocessor, images(trainIdx));
            XValidation = transformImages(preprocessor, images(valIdx));
            subjectNumber = str2double(extractAfter(subjectIDs, "_")) - 900000;
            imageNumber = str2double(metadata.ImageID);
            expectedPixels = syntheticPixels(y, subjectNumber, imageNumber);
            testCase.verifyEqual(XTrain, expectedPixels(trainIdx, :) / 255, 'AbsTol', 1e-14);
            testCase.verifyEqual(XValidation, expectedPixels(valIdx, :) / 255, 'AbsTol', 1e-14);

            model = fitPCA(XTrain, pcaConfig);
            ZTrain = transformPCA(model, XTrain);
            ZValidation = transformPCA(model, XValidation);
            testCase.verifySize(ZTrain, [16 2]);
            testCase.verifySize(ZValidation, [4 2]);
            testCase.verifyEqual(model.Mean, mean(expectedPixels(trainIdx, :) / 255, 1), 'AbsTol', 1e-14);
            testCase.verifyEqual(ZValidation, (expectedPixels(valIdx, :) / 255 - model.Mean) * model.Components, ...
                'AbsTol', 1e-14);

            [report, stats] = evalc('run_pca_analysis(folder, pcaConfig, preprocessingConfig, 1)');
            testCase.verifyEqual(stats.TrainFeatureSize, [16 4]);
            testCase.verifyEqual(stats.ValidationFeatureSize, [4 4]);
            testCase.verifyEqual(stats.ReducedTrainSize, [16 2]);
            testCase.verifyEqual(stats.ReducedValidationSize, [4 2]);
            testCase.verifyEqual(stats.NumericalRank, model.NumericalRank);
            testCase.verifyEqual(stats.RetainedExplainedVariance, model.RetainedExplainedVariance);
            testCase.verifyEqual(stats.SubjectOverlapCount, 0);
            testCase.verifyFalse(contains(report, [".png", "man_", "woman_", string(folder)]));
            testCase.verifyFalse(any(isfield(stats, {'Mean', 'Components', 'TrainIndices', 'ValidationIndices'})));

            for idx = valIdx.'
                imwrite(255 - images{idx}, fullfile(folder, metadata.Filename(idx)));
            end
            [changedImages, changedY, changedIDs] = loadDataset(folder);
            testCase.verifyEqual(changedY, y);
            testCase.verifyEqual(changedIDs, subjectIDs);
            testCase.verifyEqual(createFolds(changedY, changedIDs), folds);
            refittedPreprocessor = fitPreprocessor(changedImages(trainIdx), preprocessingConfig);
            changedTrain = transformImages(refittedPreprocessor, changedImages(trainIdx));
            changedValidation = transformImages(refittedPreprocessor, changedImages(valIdx));
            refitted = fitPCA(changedTrain, pcaConfig);
            testCase.verifyEqual(refittedPreprocessor, preprocessor);
            testCase.verifyEqual(refitted, model, 'AbsTol', 1e-14);
            testCase.verifyEqual(transformPCA(refitted, changedTrain), ZTrain, 'AbsTol', 1e-14);
            testCase.verifyGreaterThan(norm(transformPCA(refitted, changedValidation) - ZValidation, 'fro'), 0.1);
            [~, changedStats] = evalc('run_pca_analysis(folder, pcaConfig, preprocessingConfig, 1)');
            testCase.verifyEqual(changedStats.RetainedExplainedVariance, stats.RetainedExplainedVariance, 'AbsTol', 1e-14);
        end
    end
end

function X = tallFeatures()
X = [3 0 0; -3 0 0; 0 2 0; 0 -2 0; 0 0 1; 0 0 -1] * ...
    [0.6 0.8 0; -0.8 0.6 0; 0 0 1] + [10 -3 5];
end

function model = replaceField(model, field, value)
model.(field) = value;
end

function pixels = syntheticPixels(label, subject, imageNumber)
pixels = [40 + 90*label + 4*subject, 20 + 10*subject + 2*imageNumber, ...
    80 + 5*imageNumber + 3*subject, 60 + 7*subject.^2 - 20*label];
end

function writeSyntheticImages(folder)
prefixes = ["man" "woman"];
for label = 0:1
    for subject = 1:5
        for imageNumber = 1:2
            pixels = syntheticPixels(label, subject, imageNumber);
            filename = sprintf('%s_%d_%d.png', prefixes(label + 1), 900000 + subject, imageNumber);
            imwrite(reshape(uint8(pixels), 2, 2), fullfile(folder, filename));
        end
    end
end
end
