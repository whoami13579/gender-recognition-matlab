classdef TestDatasetPartitioning < matlab.unittest.TestCase
    methods (Test)
        function createsFiveBalancedSubjectGroupedFolds(testCase)
            [y, subjectIDs] = groupedSamples(repmat(2, 50, 1), repmat(2, 50, 1));
            [folds, info] = createFolds(y, subjectIDs);

            testCase.verifyEqual(numel(y), 200);
            testCase.verifyEqual(numel(unique(subjectIDs)), 100);
            testCase.verifySize(folds, [5 1]);
            testCase.verifyEqual(info.NumFolds, 5);
            testCase.verifyEqual(info.RandomSeed, 42);
            testCase.verifyTrue(info.GroupBySubject);
            testCase.verifyTrue(info.Stratified);
            testCase.verifyEqual(info.TrainSampleCounts, repmat(160, 5, 1));
            testCase.verifyEqual(info.ValidationSampleCounts, repmat(40, 5, 1));
            testCase.verifyEqual(info.TrainClassCounts, repmat([80 80], 5, 1));
            testCase.verifyEqual(info.ValidationClassCounts, repmat([20 20], 5, 1));
            testCase.verifyEqual(info.TrainSubjectCounts, repmat(80, 5, 1));
            testCase.verifyEqual(info.ValidationSubjectCounts, repmat(20, 5, 1));
            testCase.verifyEqual(info.SubjectOverlapCounts, zeros(5, 1));
            testCase.verifyPartition(folds, info, y, subjectIDs);
        end

        function balancesImagesWithUnequalSubjectSizes(testCase)
            % Equal image counts require one fold to get more male subjects.
            [y, subjectIDs] = groupedSamples([8 7 1 1 1 1 1], [7 7 1 1 1 1 1 1 1 1]);
            [folds, info] = createFolds(y, subjectIDs, struct('NumFolds', 2, 'RandomSeed', 17));

            testCase.verifyEqual(info.NumFolds, 2);
            testCase.verifyEqual(info.ValidationClassCounts, [10 11; 10 11]);
            testCase.verifyEqual(sort(info.ValidationSubjectCounts), [8; 9]);
            testCase.verifyPartition(folds, info, y, subjectIDs);
        end

        function handlesImbalanceAndIndivisibleSampleCounts(testCase)
            [y, subjectIDs] = groupedSamples(ones(1, 17), ones(1, 8));
            [folds, info] = createFolds(y, subjectIDs);

            testCase.verifyEqual(info.ValidationSampleCounts, repmat(5, 5, 1));
            testCase.verifyEqual(sort(info.ValidationClassCounts(:, 1)), [3; 3; 3; 4; 4]);
            testCase.verifyEqual(sort(info.ValidationClassCounts(:, 2)), [1; 1; 2; 2; 2]);
            testCase.verifyPartition(folds, info, y, subjectIDs);

            [y, subjectIDs] = groupedSamples(ones(1, 18), ones(1, 8));
            [folds, info] = createFolds(y, subjectIDs);
            testCase.verifyEqual(sort(info.ValidationSampleCounts), [5; 5; 5; 5; 6]);
            testCase.verifyPartition(folds, info, y, subjectIDs);
        end

        function reproducesAssignmentsWithoutChangingGlobalRandomState(testCase)
            [y, subjectIDs] = groupedSamples(repmat(2, 20, 1), repmat(2, 20, 1));
            callerState = rng;
            testCase.addTeardown(@() rng(callerState));
            rng(1234, 'twister');
            firstState = rng;
            config = struct('RandomSeed', 0);
            [folds, info] = createFolds(y, subjectIDs, config);
            testCase.verifyEqual(rng, firstState);

            rng(5678, 'twister');
            secondState = rng;
            [repeated, repeatedInfo] = createFolds(y, subjectIDs, config);
            [different, differentInfo] = createFolds(y, subjectIDs, ...
                struct('RandomSeed', intmax('uint32')));
            testCase.verifyEqual(rng, secondState);
            testCase.verifyEqual(repeated, folds);
            testCase.verifyEqual(repeatedInfo, info);
            testCase.verifyNotEqual(differentInfo.SampleToFold, info.SampleToFold);
            testCase.verifyPartition(folds, info, y, subjectIDs);
            testCase.verifyPartition(repeated, repeatedInfo, y, subjectIDs);
            testCase.verifyPartition(different, differentInfo, y, subjectIDs);
        end

        function explicitlyStratifiesSamplesWithoutSubjectIDs(testCase)
            y = [zeros(16, 1); ones(7, 1)];
            [folds, info] = createFolds(y, [], struct('GroupBySubject', false));

            testCase.verifyFalse(info.GroupBySubject);
            testCase.verifyTrue(info.Stratified);
            testCase.verifyEmpty(info.TrainSubjectCounts);
            testCase.verifyEmpty(info.ValidationSubjectCounts);
            testCase.verifyEmpty(info.SubjectOverlapCounts);
            testCase.verifyEqual(sort(info.ValidationSampleCounts), [4; 4; 5; 5; 5]);
            testCase.verifyEqual(sort(info.ValidationClassCounts(:, 1)), [3; 3; 3; 3; 4]);
            testCase.verifyEqual(sort(info.ValidationClassCounts(:, 2)), [1; 1; 1; 2; 2]);
            testCase.verifyPartition(folds, info, y, []);
        end

        function canDisableGroupingWithSubjectIDsPresent(testCase)
            [y, subjectIDs] = groupedSamples(5, 5);
            [folds, info] = createFolds(y, subjectIDs, struct('GroupBySubject', false));

            testCase.verifyEqual(info.TrainSubjectCounts, repmat(2, 5, 1));
            testCase.verifyEqual(info.ValidationSubjectCounts, repmat(2, 5, 1));
            testCase.verifyEqual(info.SubjectOverlapCounts, repmat(2, 5, 1));
            testCase.verifyPartition(folds, info, y, subjectIDs);
        end

        function rejectsConflictingLabelsWithinSubject(testCase)
            [y, subjectIDs] = groupedSamples(ones(1, 6), ones(1, 6));
            subjectIDs(end) = subjectIDs(1);
            testCase.verifyError(@() createFolds(y, subjectIDs), ...
                'genderRecognition:ConflictingSubjectLabels');
        end

        function rejectsMissingAndMalformedSubjectIDs(testCase)
            [y, subjectIDs] = groupedSamples(ones(1, 5), ones(1, 5));
            missingID = subjectIDs;
            missingID(3) = missing;
            blankID = subjectIDs;
            blankID(3) = "   ";
            emptyID = subjectIDs;
            emptyID(3) = "";
            invalid = {[], strings(0, 1), subjectIDs(2:end), subjectIDs.', ...
                cellstr(subjectIDs), (1:numel(y)).', missingID, blankID, emptyID};
            testCase.verifyError(@() createFolds(y), 'genderRecognition:InvalidSubjectIDs');
            for k = 1:numel(invalid)
                testCase.verifyError(@() createFolds(y, invalid{k}), 'genderRecognition:InvalidSubjectIDs');
            end
            testCase.verifyError(@() createFolds(y, subjectIDs(2:end), ...
                struct('GroupBySubject', false)), 'genderRecognition:InvalidSubjectIDs');
        end

        function rejectsImpossibleTwoClassValidation(testCase)
            [y, subjectIDs] = groupedSamples(repmat(2, 8, 1), [4 4 4 4]);
            testCase.verifyError(@() createFolds(y, subjectIDs), ...
                'genderRecognition:InsufficientClassGroups');
            sampleConfig = struct('GroupBySubject', false);
            for labels = {[0; 1], [zeros(6, 1); ones(4, 1)]}
                testCase.verifyError(@() createFolds(labels{1}, [], sampleConfig), ...
                    'genderRecognition:InsufficientClassGroups');
            end
        end

        function rejectsInvalidLabelsAndDimensions(testCase)
            invalid = {[], zeros(0, 1), [0 1], [0 1; 1 0], zeros(2, 1, 2), ...
                [0; 2], [0; NaN], [1; Inf], [0; 1i], single([0; 1]), sparse([0; 1]), ["0"; "1"]};
            for k = 1:numel(invalid)
                testCase.verifyError(@() createFolds(invalid{k}), 'genderRecognition:InvalidLabels');
            end
            testCase.verifyError(@() createFolds(zeros(10, 1)), 'genderRecognition:MissingClass');
            testCase.verifyError(@() createFolds(ones(10, 1)), 'genderRecognition:MissingClass');
        end

        function rejectsInvalidConfiguration(testCase)
            [y, subjectIDs] = groupedSamples(ones(1, 5), ones(1, 5));
            for config = {[], "five", repmat(struct('NumFolds', 5), 2, 1), ...
                    struct('UnknownOption', 5), struct('GroupBySubject', 1), ...
                    struct('GroupBySubject', [true false])}
                testCase.verifyError(@() createFolds(y, subjectIDs, config{1}), ...
                    'genderRecognition:InvalidConfig');
            end
            for value = {0, 1, -3, 2.5, Inf, NaN, [], 5i, "5", [2 3], true, 2 * flintmax}
                testCase.verifyError(@() createFolds(y, subjectIDs, struct('NumFolds', value{1})), ...
                    'genderRecognition:InvalidNumFolds');
            end
            for value = {-1, 5.5, 2^32, Inf, NaN, [], 7i, "42", [0 1], true}
                testCase.verifyError(@() createFolds(y, subjectIDs, struct('RandomSeed', value{1})), ...
                    'genderRecognition:InvalidRandomSeed');
            end
        end

        function preservesAlignmentThroughLoadingAndPreprocessing(testCase)
            fixture = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            prefixes = ["man" "woman"];
            for label = 0:1
                for subject = 1:5
                    for imageID = 1:2
                        value = 100 * label + 10 * subject + imageID;
                        filename = sprintf('%s_%d_%d.png', prefixes(label + 1), 900000 + subject, imageID);
                        imwrite(uint8([value value + 1; value + 2 value + 3]), ...
                            fullfile(fixture.Folder, filename));
                    end
                end
            end

            [images, y, subjectIDs, metadata] = loadDataset(fixture.Folder);
            [folds, info] = createFolds(y, subjectIDs);
            testCase.verifyPartition(folds, info, y, subjectIDs);
            for k = 1:numel(folds)
                train = folds(k).TrainIndices;
                validation = folds(k).ValidationIndices;
                preprocessor = fitPreprocessor(images(train), struct('ImageSize', []));
                for split = {train, validation}
                    idx = split{1};
                    rows = metadata(idx, :);
                    [labels, ids, imageIDs, sampleIDs] = arrayfun(@parseImageFilename, rows.Filename);
                    testCase.verifyEqual(rows.Label, y(idx));
                    testCase.verifyEqual(rows.SubjectID, subjectIDs(idx));
                    testCase.verifyEqual(labels, rows.Label);
                    testCase.verifyEqual(ids, rows.SubjectID);
                    testCase.verifyEqual(imageIDs, rows.ImageID);
                    testCase.verifyEqual(sampleIDs, rows.SampleID);
                    subjectNumber = str2double(extractAfter(ids, "_")) - 900000;
                    basePixel = 100 * labels + 10 * subjectNumber + str2double(imageIDs);
                    expected = (basePixel + [0 2 1 3]) / 255;
                    testCase.verifyEqual(transformImages(preprocessor, images(idx)), expected, ...
                        'AbsTol', 1e-14);
                end
            end

            [report, stats] = evalc("run_partition_analysis(fixture.Folder)");
            testCase.verifyEqual(stats.NumSamples, 20);
            testCase.verifyEqual(stats.ImagesPerSubjectDistribution, [2 10]);
            testCase.verifyEqual(stats.ValidationSubjectClassCounts, ones(5, 2));
            testCase.verifyEqual(stats.NumValidatedExactlyOnce, 20);
            testCase.verifyTrue(stats.PartitionIntegrityPassed);
            testCase.verifyTrue(stats.Reproducible);
            testCase.verifyTrue(stats.GlobalRandomStatePreserved);
            testCase.verifyFalse(isfield(stats, 'SampleToFold'));
            testCase.verifyFalse(contains(report, [".png", "man_", "woman_", string(fixture.Folder)]));

            [~, sampleStats] = evalc("run_partition_analysis(fixture.Folder, " + ...
                "struct('NumFolds', 2, 'RandomSeed', 17, 'GroupBySubject', false))");
            testCase.verifyEqual(sampleStats.NumFolds, 2);
            testCase.verifyEqual(sampleStats.RandomSeed, 17);
            testCase.verifyFalse(sampleStats.GroupBySubject);
            testCase.verifyEqual(sampleStats.ValidationClassCounts, [5 5; 5 5]);
        end

        function analysisOmitsFilenamesFromLoaderErrors(testCase)
            fixture = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            filename = 'synthetic_invalid_name.png';
            imwrite(uint8(10), fullfile(fixture.Folder, filename));
            try
                run_partition_analysis(fixture.Folder);
            catch cause
                testCase.verifyEqual(cause.identifier, 'genderRecognition:InvalidFilename');
                testCase.verifyFalse(contains(cause.message, [string(filename), string(fixture.Folder)]));
                return;
            end
            testCase.verifyFail('Expected an invalid filename error.');
        end
    end

    methods (Access = private)
        function verifyPartition(testCase, folds, info, y, subjectIDs)
            allIndices = (1:numel(y)).';
            validationVisits = zeros(size(y));
            trainVisits = zeros(size(y));
            testCase.verifySize(info.SampleToFold, size(y));
            for k = 1:numel(folds)
                train = folds(k).TrainIndices;
                validation = folds(k).ValidationIndices;
                testCase.verifyNotEmpty(validation);
                testCase.verifyEqual(train, unique(train, 'sorted'));
                testCase.verifyEqual(validation, unique(validation, 'sorted'));
                testCase.verifyClass(train, 'double');
                testCase.verifyClass(validation, 'double');
                testCase.verifyEqual(train, setdiff(allIndices, validation));
                testCase.verifyEmpty(intersect(train, validation));
                testCase.assertEqual(sort([train; validation]), allIndices);
                testCase.verifyEqual(info.SampleToFold(validation), repmat(k, numel(validation), 1));
                testCase.verifyEqual(info.TrainClassCounts(k, :), [nnz(y(train) == 0), nnz(y(train) == 1)]);
                testCase.verifyEqual(info.ValidationClassCounts(k, :), ...
                    [nnz(y(validation) == 0), nnz(y(validation) == 1)]);
                testCase.verifyEqual(unique(y(validation)), [0; 1]);
                validationVisits(validation) = validationVisits(validation) + 1;
                trainVisits(train) = trainVisits(train) + 1;
                if info.GroupBySubject
                    testCase.verifyEmpty(intersect(subjectIDs(train), subjectIDs(validation)));
                end
            end
            testCase.verifyEqual(validationVisits, ones(size(y)));
            testCase.verifyEqual(trainVisits, repmat(numel(folds) - 1, size(y)));
        end
    end
end

function [y, subjectIDs] = groupedSamples(maleSizes, femaleSizes)
y = [zeros(sum(maleSizes), 1); ones(sum(femaleSizes), 1)];
maleIDs = "man_" + (900000 + (1:numel(maleSizes)).');
femaleIDs = "woman_" + (900000 + (1:numel(femaleSizes)).');
maleIDs = repelem(maleIDs, maleSizes(:));
femaleIDs = repelem(femaleIDs, femaleSizes(:));
subjectIDs = [maleIDs(:); femaleIDs(:)];
end
