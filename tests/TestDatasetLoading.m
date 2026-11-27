classdef TestDatasetLoading < matlab.unittest.TestCase
    properties
        DataRoot
    end

    methods (TestMethodSetup)
        function createTemporaryDataset(testCase)
            fixture = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            testCase.DataRoot = fixture.Folder;
        end
    end

    methods (Test)
        function extractsLabelsAndClassQualifiedIdentifiers(testCase)
            [label, subjectID, imageID, sampleID] = parseImageFilename('man_0900001_0002.png');
            testCase.verifyEqual(label, 0);
            testCase.verifyEqual(subjectID, "man_900001");
            testCase.verifyEqual(imageID, "2");
            testCase.verifyEqual(sampleID, "man_900001_2");

            [label, subjectID, imageID, sampleID] = parseImageFilename("WOMAN_900001_2.PNG");
            testCase.verifyEqual(label, 1);
            testCase.verifyEqual(subjectID, "woman_900001");
            testCase.verifyEqual(imageID, "2");
            testCase.verifyEqual(sampleID, "woman_900001_2");
        end

        function loadsDeterministicallyWithAlignedMetadata(testCase)
            first = uint8([10 20; 30 40]);
            second = uint16([1000 2000; 3000 4000]);
            third = cat(3, first, fliplr(first), flipud(first));
            imwrite(third, fullfile(testCase.DataRoot, 'woman_9000010_1.png'));
            imwrite(second, fullfile(testCase.DataRoot, 'man_900002_1.png'));
            imwrite(first, fullfile(testCase.DataRoot, 'man_9000010_1.png'));

            [images, y, subjects, metadata] = loadDataset(string(testCase.DataRoot));
            testCase.verifyEqual(images, {first; second; third});
            testCase.verifyEqual(y, [0; 0; 1]);
            testCase.verifyEqual(subjects, ["man_9000010"; "man_900002"; "woman_9000010"]);
            testCase.verifyEqual(metadata.Label, y);
            testCase.verifyEqual(metadata.SubjectID, subjects);
            testCase.verifyEqual(metadata.SampleID, ["man_9000010_1"; "man_900002_1"; "woman_9000010_1"]);
            testCase.verifyEqual(metadata.ImageID, ["1"; "1"; "1"]);
            testCase.verifyEqual(metadata.Filename, ["man_9000010_1.png"; "man_900002_1.png"; "woman_9000010_1.png"]);
            testCase.verifyEqual(metadata{:, {'Height', 'Width', 'Channels'}}, [2 2 1; 2 2 1; 2 2 3]);
            testCase.verifyEqual(metadata.PixelType, ["uint8"; "uint16"; "uint8"]);
        end

        function rejectsInvalidDatasetPaths(testCase)
            testCase.verifyError(@() loadDataset(fullfile(testCase.DataRoot, 'missing')), ...
                'genderRecognition:InvalidDatasetPath');
            testCase.verifyError(@() loadDataset(["path1" "path2"]), ...
                'genderRecognition:InvalidDatasetPath');
            testCase.verifyError(@() loadDataset(""), 'genderRecognition:InvalidDatasetPath');
            path = fullfile(testCase.DataRoot, 'man_900001_1.png');
            imwrite(uint8(10), path);
            testCase.verifyError(@() loadDataset(path), 'genderRecognition:InvalidDatasetPath');
        end

        function rejectsEmptyDataset(testCase)
            testCase.verifyError(@() loadDataset(testCase.DataRoot), 'genderRecognition:EmptyDataset');
        end

        function rejectsUnexpectedSubdirectories(testCase)
            mkdir(testCase.DataRoot, 'nested');
            testCase.verifyError(@() loadDataset(testCase.DataRoot), ...
                'genderRecognition:UnexpectedSubdirectory');
        end

        function rejectsUnsupportedOrAmbiguousFilenames(testCase)
            badNames = ["unknown_1_1.png", "man_900001.png", "woman_900001_2_3.png", ...
                "man_900001_2.jpg", "man_900001_2.png.bak", "folder/man_900001_2.png"];
            for name = badNames
                testCase.verifyError(@() parseImageFilename(name), 'genderRecognition:InvalidFilename');
            end

            imwrite(uint8(10), fullfile(testCase.DataRoot, 'man_900001_1.png'));
            imwrite(uint8(20), fullfile(testCase.DataRoot, 'unknown_1_1.png'));
            testCase.verifyError(@() loadDataset(testCase.DataRoot), 'genderRecognition:InvalidFilename');
        end

        function rejectsDuplicateSampleIdentifiers(testCase)
            imwrite(uint8(10), fullfile(testCase.DataRoot, 'man_900001_1.png'));
            imwrite(uint8(20), fullfile(testCase.DataRoot, 'man_0900001_001.png'));
            testCase.verifyError(@() loadDataset(testCase.DataRoot), 'genderRecognition:DuplicateSampleID');
        end

        function rejectsCorruptedPng(testCase)
            writelines("This is not PNG image data.", fullfile(testCase.DataRoot, 'man_900001_1.png'));
            testCase.verifyError(@() loadDataset(testCase.DataRoot), 'genderRecognition:UnreadableImage');
        end

        function rejectsIndexedPng(testCase)
            indices = uint8([0 1; 1 0]);
            colorMap = [0 0 0; 1 0 0];
            imwrite(indices, colorMap, fullfile(testCase.DataRoot, 'woman_900001_1.png'));
            testCase.verifyError(@() loadDataset(testCase.DataRoot), 'genderRecognition:UnsupportedImage');
        end

        function rejectsPngWithTransparency(testCase)
            pixels = uint8([10 20; 30 40]);
            alpha = uint8([255 0; 255 255]);
            imwrite(pixels, fullfile(testCase.DataRoot, 'woman_900001_1.png'), 'Alpha', alpha);
            testCase.verifyError(@() loadDataset(testCase.DataRoot), 'genderRecognition:UnsupportedImage');
        end

        function loadsAndPreprocessesSyntheticTrainingDataEndToEnd(testCase)
            imwrite(repmat(uint8(255), 2, 3), fullfile(testCase.DataRoot, 'woman_900001_1.png'));
            imwrite(repmat(uint8(51), 2, 3), fullfile(testCase.DataRoot, 'man_900001_2.png'));
            imwrite(zeros(2, 3, 'uint8'), fullfile(testCase.DataRoot, 'man_900001_1.png'));

            [images, y, subjects, metadata] = loadDataset(testCase.DataRoot);
            X = transformImages(fitPreprocessor(images), images);
            testCase.verifySize(X, [3 4096]);
            testCase.verifySize(y, [3 1]);
            testCase.verifySize(subjects, [3 1]);
            testCase.verifyEqual(y, [0; 0; 1]);
            testCase.verifyEqual(subjects, ["man_900001"; "man_900001"; "woman_900001"]);
            testCase.verifyEqual(X(:, 1), [0; 0.2; 1], 'AbsTol', 1e-14);

            [reloaded, labelsAgain, subjectsAgain, metadataAgain] = loadDataset(testCase.DataRoot);
            XAgain = transformImages(fitPreprocessor(reloaded), reloaded);
            testCase.verifyEqual(XAgain, X);
            testCase.verifyEqual(labelsAgain, y);
            testCase.verifyEqual(subjectsAgain, subjects);
            testCase.verifyEqual(metadataAgain, metadata);

            report = evalc('stats = run_dataset_analysis(testCase.DataRoot);');
            testCase.verifyEqual(stats.FeatureMatrixSize, [3 4096]);
            testCase.verifyEqual(stats.LabelVectorSize, [3 1]);
            testCase.verifyEqual(stats.ClassCounts, [2 1]);
            testCase.verifyEqual(stats.NumSubjects, 2);
            testCase.verifyEqual(stats.SubjectCounts, [1 1]);
            testCase.verifyEqual(stats.CrossClassQualifiedIDCollisions, 0);
            testCase.verifyTrue(contains(report, ...
                'Cross-class qualified ID collisions: 0 (filename-derived IDs)'));
            testCase.verifyEqual(stats.ImagesPerSubjectDistribution, [1 1; 2 1]);
            testCase.verifyFalse(stats.MatchesOfficialTrainingCounts);
            testCase.verifyFalse(contains(report, '.png'));
        end
    end
end
