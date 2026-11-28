classdef TestFinalDatasetIntegrity < matlab.unittest.TestCase
    methods (Test)
        function decodedDuplicatesAndConflictingLabels(testCase)
            fixture = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            path = string(fixture.Folder);
            pixels = uint8(magic(8));
            a = fullfile(path, "man_1_1.png");
            b = fullfile(path, "man_2_1.png");
            imwrite(pixels, a);
            imwrite(pixels, b, 'Description', 'Distinct encoding of identical decoded pixels.');
            imwrite(pixels, fullfile(path, "woman_1_1.png"));
            testCase.verifyNotEqual(finalInternal.checksum(a, true), finalInternal.checksum(b, true));
            [images, index, integrity] = finalInternal.loadImages(path);
            testCase.verifyEqual(numel(unique(index.Hashes)), 1);
            testCase.verifyEqual(integrity.DuplicateSamplePairs, 3);
            testCase.verifyEqual(integrity.DuplicateCrossSubjectPairs, 3);
            testCase.verifyEqual(integrity.ConflictingLabelPairs, 2);
            report = finalInternal.trainingDiagnostics(integrity, images, index, [1 1 2]);
            testCase.verifyEqual(report.CrossOuterFoldDuplicatePairs, 2);
            testCase.verifyTrue(report.MaterialDefect);
            testCase.verifyError(@() finalInternal.checkDatasetProfile(integrity, "training", false), ...
                'genderRecognition:UnexpectedDatasetProfile');
        end

        function ambiguityAndFingerprintMismatch(testCase)
            fixture = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            path = string(fixture.Folder);
            testing = fullfile(path, "holdout");
            training = fullfile(path, "training");
            mkdir(testing);
            mkdir(training);
            pixels = uint8(magic(8));
            imwrite(pixels, fullfile(training, "man_1_1.png"));
            imwrite(pixels + 1, fullfile(training, "woman_1_1.png"));
            imwrite(pixels + 2, fullfile(testing, "man_1_1.png"));
            imwrite(pixels + 3, fullfile(testing, "woman_1_1.png"));
            [~, reference, a] = finalInternal.loadImages(training);
            [~, holdout, b] = finalInternal.loadImages(testing);
            report = finalInternal.testOverlap(b, reference, holdout);
            testCase.verifyNotEqual(a.DecodedSHA256, b.DecodedSHA256);
            testCase.verifyEqual(report.TrainingTestDuplicatePairs, 0);
            testCase.verifyEqual(report.SharedClassQualifiedSubjectIDs, 2);
            testCase.verifyEqual(report.SharedNumericSubjectIDs, 1);
            testCase.verifyError(@() finalInternal.checkDatasetProfile(report, "testing", false), ...
                'genderRecognition:UnexpectedDatasetProfile');
            imwrite(pixels, fullfile(testing, "woman_1_1.png"));
            [~, changed, info] = finalInternal.loadImages(testing);
            overlap = finalInternal.testOverlap(info, reference, changed);
            testCase.verifyEqual(overlap.TrainingTestDuplicatePairs, 1);
            testCase.verifyEqual(overlap.TrainingTestConflictingLabelPairs, 1);

            baseline = fullfile(path, "prior-baseline");
            rngBefore = rng;
            [images, labels, subjects] = makeCVFixture();
            config = crossValidationConfig(struct('Mode', "baseline", 'OutputDirectory', baseline, ...
                'GeneratePlots', false, 'Benchmark', struct('WarmupRuns', 0, 'NumRepeats', 1)));
            config.Pipeline.PCA.NumComponents = 2;
            config.Pipeline.MLP.MaxEpochs = 1;
            results = crossValidatePipelines(images, labels, subjects, config);
            [results.Summary, results.MeanROC] = summarizeCrossValidation(results.Records, ...
                results.Methods, numel(results.Folds));
            results.Metadata = struct('ExperimentSchemaVersion', 1, 'CompletedRecords', 20, ...
                'ExpectedRecords', 20, 'Status', "complete", 'Dataset', struct('DecodedSHA256', a.DecodedSHA256));
            cvDirectory = saveCrossValidationRun(results);
            options = struct('Synthetic', false, 'NestedRun', "", 'BaselineRun', cvDirectory);
            testCase.verifyError(@() finalInternal.resolveRecommendations(options, b.DecodedSHA256), ...
                'genderRecognition:TrainingFingerprintMismatch');
            testCase.verifyEqual(rng, rngBefore);
        end
    end
end

function [images, labels, subjects] = makeCVFixture()
stream = RandStream('mt19937ar', 'Seed', 191);
labels = repelem([0; 1], 8);
subjects = string((1:16).');
images = arrayfun(@(~) uint8(randi(stream, [0 255], 8, 8)), labels, 'UniformOutput', false);
end
