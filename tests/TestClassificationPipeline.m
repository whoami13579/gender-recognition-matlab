classdef TestClassificationPipeline < matlab.unittest.TestCase
    methods (Test)
        function allMethodsMatchExistingModules(testCase)
            [images, labels, ~, config] = cvFixture();
            train = [1:10 13:22];
            validation = [24 12 11 23];
            methods = ["pca-bayes", "lda-bayes", "pca-mlp", "lda-mlp"];
            expectedCounts = [2 108 0 11 119; 1 72 0 5 77; ...
                2 108 4 13 125; 1 72 2 10 84];
            for k = 1:numel(methods)
                method = methods(k);
                pipeline = fitPipeline(images(train), labels(train), method, config.Pipeline);
                oldState = pipeline;
                [predicted, scores] = predictPipeline(pipeline, images(validation));
                [reverse, reverseScores] = predictPipeline(pipeline, images(flip(validation)));
                testCase.verifyEqual(reverse, flip(predicted));
                testCase.verifyEqual(reverseScores, flip(scores), 'AbsTol', 1e-12);
                testCase.verifyEqual(pipeline, oldState);
                testCase.verifySize(scores, [4 1]);
                testCase.verifyEqual(predicted, double(scores > 0));
                testCase.verifyEqual(pipeline.NumTrainingSamples, numel(train));
                testCase.verifyEqual(pipeline.Method, method);
                p = pipeline.Parameters;
                testCase.verifyEqual(p.NumOriginalFeatures, 36);
                testCase.verifyEqual([p.NumFeatures, p.NumProjectionParameters, ...
                    p.NumNormalizationParameters, p.NumClassifierParameters, p.CombinedParameterCount], ...
                    expectedCounts(k, :));
                X = transformImages(pipeline.Preprocessor, images(validation));
                if startsWith(method, "pca")
                    Z = transformPCA(pipeline.Projection, X);
                else
                    Z = transformLDA(pipeline.Projection, X);
                end
                if endsWith(method, "bayes")
                    [direct, directScores] = predictBayes(pipeline.Classifier, Z);
                else
                    [direct, directScores] = predictMLP(pipeline.Classifier, Z);
                end
                testCase.verifyEqual(predicted, direct);
                testCase.verifyEqual(scores, directScores);
            end
        end

        function rejectsInvalidInputs(testCase)
            [images, labels, ~, config] = cvFixture();
            testCase.verifyError(@() fitPipeline(images, labels, "other", config.Pipeline), ...
                'genderRecognition:InvalidPipelineMethod');
            testCase.verifyError(@() fitPipeline(images, labels, "pca-bayes", struct('Typo', 2)), ...
                'genderRecognition:InvalidPipelineConfig');
            wrong = config.Pipeline;
            wrong.PCA.NumComponents = 40;
            testCase.verifyError(@() fitPipeline(images, labels, "pca-bayes", wrong), ...
                'genderRecognition:PCAComponentsExceedRank');
        end
    end
end
