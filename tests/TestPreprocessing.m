classdef TestPreprocessing < matlab.unittest.TestCase
    methods (Test)
        function convertsRgbToGrayscale(testCase)
            image = zeros(1, 3, 3, 'uint8');
            image(1, 1, 1) = 255;
            image(1, 2, 2) = 255;
            image(1, 3, 3) = 255;
            model = fitPreprocessor({image}, struct('ImageSize', []));
            X = transformImages(model, {image});
            testCase.verifyEqual(X, [0.299 0.587 0.114], 'AbsTol', 0.0002);
            testCase.verifyEqual(model.OutputImageSize, [1 3 1]);
        end

        function usesColumnMajorVectorizationForGrayscale(testCase)
            image = uint8([0 51 102; 153 204 255]);
            model = fitPreprocessor({image}, struct('ImageSize', []));
            X = transformImages(model, {image});
            testCase.verifyEqual(X, [0 153 51 204 102 255] / 255);
            testCase.verifySize(X, [1 6]);
            testCase.verifyClass(X, 'double');
        end

        function resizesToConfiguredHeightAndWidth(testCase)
            image = repmat(uint8(128), 5, 7);
            model = fitPreprocessor({image}, struct('ImageSize', [2 3]));
            processed = transformImages(model, {image}, "images");
            X = transformImages(model, {image});
            testCase.verifySize(processed, [1 1]);
            testCase.verifySize(processed{1}, [2 3]);
            testCase.verifyEqual(X, repmat(128 / 255, 1, 6), 'AbsTol', 1e-14);
        end

        function preservesOriginalResolutionAndRgbChannelsWhenRequested(testCase)
            image = cat(3, [0 0.25; 0.125 0.375], [0.5 0.75; 0.625 0.875], ones(2));
            config = struct('ImageSize', [], 'Grayscale', false);
            model = fitPreprocessor({image}, config);
            X = transformImages(model, {image});
            processed = transformImages(model, {image}, "images");
            testCase.verifyEqual(model.OutputImageSize, [2 2 3]);
            testCase.verifyEqual(processed, {image});
            testCase.verifyEqual(X, [0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1 1 1 1]);
        end

        function scalesMixedInputTypesToUnitRange(testCase)
            images = {logical([0 0 1]); uint8([0 128 255]); uint16([0 32768 65535]); ...
                single([0 0.5 1]); double([0 0.5 1])};
            quantizationTolerance = [0.5/255; 0.5/65535; 0; 0];
            for outputType = ["single", "double"]
                config = struct('ImageSize', [], 'OutputType', outputType, 'IntensityScaling', "unit");
                model = fitPreprocessor(images, config);
                X = transformImages(model, images);
                testCase.verifyClass(X, char(outputType));
                testCase.verifyEqual(X(:, [1 3]), cast(repmat([0 1], numel(images), 1), outputType));

                % Logical inputs have no intermediate intensity.
                midpointError = abs(double(X(2:end, 2)) - 0.5);
                tolerance = quantizationTolerance + double(eps(char(outputType)));
                testCase.verifyLessThanOrEqual(midpointError, tolerance);
                testCase.verifyTrue(all(X >= 0 & X <= 1, 'all'));
            end
        end

        function keepsUnitScaleAfterRgbConversionAndResizing(testCase)
            black = zeros(4, 4, 3, 'uint8');
            white = repmat(uint16(65535), 4, 4, 3);
            images = {black; white};
            model = fitPreprocessor(images, struct('ImageSize', [2 2]));
            X = transformImages(model, images);
            testCase.verifyEqual(X, [zeros(1, 4); ones(1, 4)], 'AbsTol', 1e-14);
            testCase.verifyTrue(all(X >= 0 & X <= 1, 'all'));
        end

        function rejectsUnsupportedIntensityScaling(testCase)
            images = {uint8([0 255])};
            for scaling = ["native", "standardize"]
                config = struct('IntensityScaling', scaling);
                testCase.verifyError(@() fitPreprocessor(images, config), ...
                    'genderRecognition:InvalidConfig');
            end

            legacyModel = fitPreprocessor(images);
            legacyModel.Options.IntensityScaling = "native";
            testCase.verifyError(@() transformImages(legacyModel, images), ...
                'genderRecognition:InvalidConfig');
        end

        function appliesFittedRepresentationIndependentlyOfBatch(testCase)
            training = {zeros(3, 4, 'uint8')};
            heldOut = {repmat(uint8(255), 2, 3, 3); repmat(uint8(51), 7, 5)};
            model = fitPreprocessor(training);
            features = transformImages(model, heldOut);
            testCase.verifyEqual(model.NumFittedParameters, 0);
            testCase.verifyEqual(model.NumFeatures, 4096);
            testCase.verifySize(features, [2 4096]);
            testCase.verifyEqual(features(1, :), ones(1, 4096), 'AbsTol', 1e-14);
            testCase.verifyEqual(features(2, :), transformImages(model, heldOut(2)));
            testCase.verifyEqual(transformImages(model, heldOut), features);
            testCase.verifySize(transformImages(model, cell(0, 1)), [0 4096]);
        end

        function rejectsInvalidOptions(testCase)
            invalid = {[], struct('UnknownOption', 1), struct('Grayscale', 1), ...
                struct('ImageSize', [0 64]), struct('ImageSize', [64; 64]), ...
                struct('ImageSize', [Inf 4]), struct('ImageSize', [2 2.5]), ...
                struct('OutputType', 'uint8')};
            for k = 1:numel(invalid)
                testCase.verifyError(@() fitPreprocessor({uint8(1)}, invalid{k}), ...
                    'genderRecognition:InvalidConfig');
            end
        end

        function rejectsInvalidImageRepresentations(testCase)
            unsupported = {ones(2, 2, 4), ones(2, 2, 3, 2), int16(1), sparse(1), ...
                complex(1, 1), []};
            model = fitPreprocessor({uint8(1)});
            for k = 1:numel(unsupported)
                testCase.verifyError(@() transformImages(model, unsupported(k)), ...
                    'genderRecognition:UnsupportedImage');
            end
            for value = [NaN Inf -0.1 1.1]
                testCase.verifyError(@() fitPreprocessor({value}), 'genderRecognition:InvalidIntensity');
            end
            testCase.verifyError(@() fitPreprocessor({}), 'genderRecognition:InvalidImageBatch');
            testCase.verifyError(@() transformImages(model, uint8(1)), ...
                'genderRecognition:InvalidImageBatch');
        end

        function rejectsIncompatibleOutputDimensions(testCase)
            first = zeros(2, 3);
            second = zeros(3, 2);
            options = struct('ImageSize', []);
            testCase.verifyError(@() fitPreprocessor({first; second}, options), ...
                'genderRecognition:InconsistentImageSize');
            model = fitPreprocessor({first}, options);
            testCase.verifyError(@() transformImages(model, {second}), ...
                'genderRecognition:InconsistentImageSize');

            options = struct('Grayscale', false);
            model = fitPreprocessor({first}, options);
            testCase.verifyError(@() transformImages(model, {zeros(2, 3, 3)}), ...
                'genderRecognition:InconsistentImageSize');
        end
    end
end
