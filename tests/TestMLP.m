classdef TestMLP < matlab.unittest.TestCase
    methods (Test)
        function scalarForwardMatchesReferenceAndBreaksTies(testCase)
            model = fixedModel([1 2 1], {[1 -2], [3; 4]}, {[0 0], 0});
            Z = [-2; 0; 1; -1];
            reference = 3 * tanh(Z) + 4 * tanh(-2 * Z);
            [y, scores] = predictMLP(model, Z);
            testCase.verifyEqual(scores, reference, 'AbsTol', 1e-14);
            testCase.verifyEqual(y, [1; 0; 0; 1]);
            testCase.verifyEqual(scores(2), 0);
        end

        function multilayerForwardMatchesReferenceAndPreservesState(testCase)
            model = fixedModel([2 2 1 1], {[1 -1; 2 0.5], [0.7; -1.4], -2}, ...
                {[0.1 -0.3], 0.5, 0.2});
            original = model;
            Z = single([2 0; -1 2; 0 0; 0 -1]);
            x = double(Z(:, 1));
            z = double(Z(:, 2));
            hidden1 = tanh(x + 2 * z + 0.1);
            hidden2 = tanh(-x + 0.5 * z - 0.3);
            reference = -2 * tanh(0.7 * hidden1 - 1.4 * hidden2 + 0.5) + 0.2;
            before = rng;
            [y, scores] = predictMLP(model, Z);
            testCase.verifyEqual(scores, reference, 'AbsTol', 1e-14);
            testCase.verifyEqual(y, double(reference > 0));
            for row = 1:size(Z, 1)
                [label, score] = predictMLP(model, Z(row, :));
                testCase.verifyEqual(label, y(row));
                testCase.verifyEqual(score, scores(row), 'AbsTol', 1e-14);
            end
            [emptyLabels, emptyScores] = predictMLP(model, zeros(0, 2, 'single'));
            testCase.verifyEqual(emptyLabels, zeros(0, 1));
            testCase.verifyEqual(emptyScores, zeros(0, 1));
            testCase.verifyEqual(model, original);
            testCase.verifyEqual(rng, before);
        end

        function logitsRetainRankingAfterSigmoidSaturates(testCase)
            model = fixedModel([1 1 1], {1, 2000}, {0, 0});
            reference = [1100; -1200; 1000; 0; -1000; 1200];
            [y, scores] = predictMLP(model, atanh(reference / 2000));
            testCase.verifyEqual(scores, reference, 'AbsTol', 1e-12);
            testCase.verifyEqual(numel(unique(scores)), numel(reference));
            testCase.verifyEqual(y, double(reference > 0));
        end

        function wideMarginLossAndDerivativeRemainMeaningful(testCase)
            % All four label/logit combinations, including correct saturation.
            cases = [1000 1 0 0; -1000 0 0 0; 1000 0 1000 1; -1000 1 1000 -1];
            for row = 1:size(cases, 1)
                [loss, objective, dW, db] = mlpInternal.batchObjective( ...
                    0, cases(row, 2), {1, 0}, {0, cases(row, 1)}, 0);
                testCase.verifyEqual(loss, cases(row, 3));
                testCase.verifyEqual(objective, loss);
                testCase.verifyEqual(dW, {0, 0});
                testCase.verifyEqual(db, {0, cases(row, 4)});
            end
            % A correctly classified positive logit of 40 still has a
            % representable loss and derivative even though sigmoid(40) == 1.
            tail = exp(-40);
            [loss, ~, ~, db] = mlpInternal.batchObjective(0, 1, {1, 0}, {0, 40}, 0);
            testCase.verifyEqual(loss, log1p(tail), 'RelTol', 1e-14);
            testCase.verifyEqual(db{2}, -tail / (1 + tail), 'RelTol', 1e-14);
        end

        function handCalculatedBatchAverageAndWeightOnlyPenalty(testCase)
            Z = [-1; 0; 2];
            y = [0; 1; 0];
            W = {0.45, -0.55};
            b = {0.1, 0.2};
            lambda = 0.125;
            [loss, objective, dW, db] = mlpInternal.batchObjective(Z, y, W, b, lambda);
            h = tanh(0.45 * Z + 0.1);
            p = 1 ./ (1 + exp(-(-0.55 * h + 0.2)));
            residual = p - y;
            hiddenError = -0.55 * residual .* (1 - h.^2);
            referenceLoss = -mean(y .* log(p) + (1 - y) .* log(1 - p));
            testCase.verifyEqual(loss, referenceLoss, 'AbsTol', 1e-14);
            testCase.verifyEqual(objective, referenceLoss + lambda / 2 * (0.45^2 + 0.55^2), 'AbsTol', 1e-14);
            testCase.verifyEqual([dW{:}], [mean(Z .* hiddenError) + lambda * 0.45, ...
                mean(h .* residual) - lambda * 0.55], 'AbsTol', 1e-14);
            testCase.verifyEqual([db{:}], [mean(hiddenError), mean(residual)], 'AbsTol', 1e-14);
            [~, ~, repeatedW, repeatedB] = mlpInternal.batchObjective( ...
                repmat(Z, 3, 1), repmat(y, 3, 1), W, b, lambda);
            testCase.verifyEqual([repeatedW{:}], [dW{:}], 'AbsTol', 1e-14);
            testCase.verifyEqual([repeatedB{:}], [db{:}], 'AbsTol', 1e-14);
        end

        function centralDifferencesCheckEveryWeightAndBias(testCase)
            Z = [-0.8 0.4; 1.1 -0.7; 0.2 0.9; -0.3 -0.5];
            y = [0; 1; 0; 1];
            W = {[0.24 -0.39; 0.61 0.15], [0.55 -0.28; -0.42 0.66], [0.72; -0.51]};
            b = {[-0.1 0.23], [0.09 -0.16], 0.31};
            lambda = 0.13;
            step = 1e-5;
            [~, ~, dW, db] = mlpInternal.batchObjective(Z, y, W, b, lambda);
            relativeError = zeros(numel(W), 2);
            for layer = 1:numel(W)
                referenceW = zeros(size(W{layer}));
                referenceB = zeros(size(b{layer}));
                for index = 1:numel(referenceW)
                    referenceW(index) = centralDifference(Z, y, W, b, lambda, layer, index, step, true);
                end
                for index = 1:numel(referenceB)
                    referenceB(index) = centralDifference(Z, y, W, b, lambda, layer, index, step, false);
                end
                relativeError(layer, 1) = max(abs(referenceW - dW{layer}) ./ ...
                    max(1e-6, abs(referenceW) + abs(dW{layer})), [], 'all');
                relativeError(layer, 2) = max(abs(referenceB - db{layer}) ./ ...
                    max(1e-6, abs(referenceB) + abs(db{layer})), [], 'all');
            end
            testCase.verifyLessThan(relativeError, 1e-5);
            fprintf('\nMLP finite differences, max relative error by layer [weights, bias]\n');
            fprintf('%.3g %.3g\n', relativeError.');
        end

        function seedControlsGlorotInitializationAndTraining(testCase)
            layers = [2 4 3 1];
            seed = 42;
            before = rng;
            [W, b] = mlpInternal.initialize(layers, RandStream('mt19937ar', 'Seed', seed));
            [repeatW, repeatB] = mlpInternal.initialize(layers, RandStream('mt19937ar', 'Seed', seed));
            [otherW, ~] = mlpInternal.initialize(layers, RandStream('mt19937ar', 'Seed', seed + 1));
            testCase.verifyEqual(repeatW, W);
            testCase.verifyEqual(repeatB, b);
            testCase.verifyNotEqual(otherW, W);
            for layer = 1:numel(W)
                testCase.verifySize(W{layer}, layers(layer:layer+1));
                testCase.verifyLessThanOrEqual(abs(W{layer}), sqrt(6 / sum(layers(layer:layer+1))));
                testCase.verifyEqual(b{layer}, zeros(1, layers(layer+1)));
            end
            Z = [-1 0; -2 1; 0 0; 1 0; 2 -1];
            y = [0; 0; 1; 1; 1];
            config = struct('HiddenLayerSizes', [4; 3], 'MaxEpochs', 5, 'BatchSize', 2, 'RandomSeed', seed);
            model = fitMLP(Z, y, config);
            testCase.verifyEqual(fitMLP(Z, y, config), model);
            testCase.verifyEqual(model.LayerSizes, layers);
            testCase.verifyEqual(model.Options.RandomSeed, seed);
            testCase.verifyEqual(model.Options.Initialization, "glorotUniform");
            testCase.verifyEqual(model.Options.RandomStream, "mt19937ar");
            testCase.verifyEqual(rng, before);
        end

        function matchesIndependentShuffledMomentumUpdates(testCase)
            Z = [-1.4; 0.2; -0.6; 0.8; 1.9];
            y = [0; 1; 0; 1; 1];
            for beta = [0 0.6]
                config = struct('HiddenLayerSizes', 1, 'LearningRate', 0.07, ...
                    'Momentum', beta, 'BatchSize', 2, 'MaxEpochs', 2, ...
                    'L2Regularization', 0.03, 'Standardize', false, 'RandomSeed', 7);
                stream = RandStream('mt19937ar', 'Seed', config.RandomSeed);
                bound = sqrt(3);
                theta = [(2 * rand(stream) - 1) * bound, 0, (2 * rand(stream) - 1) * bound, 0];
                velocity = zeros(1, 4);
                expectedLoss = zeros(3, 1);
                expectedObjective = zeros(3, 1);
                for epoch = 0:2
                    p = 1 ./ (1 + exp(-(theta(3) * tanh(Z * theta(1) + theta(2)) + theta(4))));
                    expectedLoss(epoch+1) = -mean(y .* log(p) + (1 - y) .* log(1 - p));
                    expectedObjective(epoch+1) = expectedLoss(epoch+1) + ...
                        config.L2Regularization / 2 * (theta(1)^2 + theta(3)^2);
                    if epoch == 2
                        break
                    end
                    order = randperm(stream, 5);
                    for rows = {order(1:2), order(3:4), order(5)}
                        gradient = zeros(1, 4);
                        for index = rows{1}
                            hidden = tanh(Z(index) * theta(1) + theta(2));
                            p = 1 / (1 + exp(-(theta(3) * hidden + theta(4))));
                            residual = p - y(index);
                            hiddenDerivative = theta(3) * (1 - hidden^2);
                            gradient = gradient + residual * [hiddenDerivative * Z(index), hiddenDerivative, hidden, 1];
                        end
                        gradient = gradient / numel(rows{1}) + config.L2Regularization * theta .* [1 0 1 0];
                        velocity = beta * velocity - config.LearningRate * gradient;
                        theta = theta + velocity;
                    end
                end
                model = fitMLP(Z, y, config);
                testCase.verifyEqual([model.Weights{1}, model.Biases{1}, model.Weights{2}, model.Biases{2}], ...
                    theta, 'AbsTol', 5e-15);
                testCase.verifyEqual(model.TrainingLossHistory, expectedLoss, 'AbsTol', 5e-15);
                testCase.verifyEqual(model.TrainingObjectiveHistory, expectedObjective, 'AbsTol', 5e-15);
            end
        end

        function learnsLinearSeparations(testCase)
            config = struct('HiddenLayerSizes', 6, 'LearningRate', 0.06, 'BatchSize', 3, 'MaxEpochs', 100);
            y = [0; 0; 0; 1; 1; 1];
            Z = [-3; -2; -1; 1; 2; 3];
            verifyLearning(testCase, Z, y, [-1.5; 2.5], [0; 1], config, '1D linear');
            Z = [Z [0; 1; -1; 2; -1; 0.5]];
            verifyLearning(testCase, Z, y, [-2 3; 2 -2], [0; 1], config, '2D linear');
        end

        function learnsXorWithNonlinearHiddenLayer(testCase)
            Z = [-1 -1; -1 1; 1 -1; 1 1];
            y = [0; 1; 1; 0];
            config = struct('HiddenLayerSizes', 8, 'LearningRate', 0.08, ...
                'Momentum', 0.9, 'BatchSize', 4, 'MaxEpochs', 240, 'RandomSeed', 42);
            verifyLearning(testCase, Z, y, 0.8 * Z, y, config, 'XOR');
        end

        function countsActualParametersIndependently(testCase)
            counts = [49 609; 673 1857];
            dimensions = [1 40];
            hiddenLayers = {16, [32 16]};
            for input = 1:2
                K = dimensions(input);
                Z = reshape(sin(1:4*K), 4, K);
                for architecture = 1:2
                    config = struct('HiddenLayerSizes', hiddenLayers{architecture}, 'MaxEpochs', 1);
                    model = fitMLP(Z, [0; 1; 0; 1], config);
                    layers = model.LayerSizes;
                    formula = sum((layers(1:end-1) + 1) .* layers(2:end));
                    arrays = sum(cellfun(@numel, model.Weights)) + sum(cellfun(@numel, model.Biases));
                    testCase.verifyEqual([model.NumClassifierParameters, formula, arrays], ...
                        repmat(counts(input, architecture), 1, 3));
                    testCase.verifyEqual(model.NumNormalizationParameters, 2 * K);
                end
            end
        end

        function usesTrainingSampleStatisticsAndPreservesSmallScales(testCase)
            Z = [-2 9 -2e-200; 0 9 0; 2 9 2e-200];
            y = [0; 0; 1];
            model = fitMLP(Z, y, struct('MaxEpochs', 3));
            testCase.verifyEqual(model.StandardizationMean, [0 9 0]);
            testCase.verifyEqual(model.StandardizationScale, [2 1 2e-200], 'RelTol', 1e-14);
            % Evaluate the independently normalized rows with the same fixed weights.
            bare = model;
            bare.StandardizationMean = [];
            bare.StandardizationScale = [];
            ZVal = [2 9 1e-200; 0 10 -2e-200];
            [~, scores] = predictMLP(model, ZVal);
            [~, reference] = predictMLP(bare, [1 0 0.5; 0 1 -1]);
            testCase.verifyEqual(scores, reference, 'AbsTol', 1e-14);
            testCase.verifyEqual(model.StandardizationMean, [0 9 0]);
            [emptyLabels, emptyScores] = predictMLP(model, zeros(0, 3));
            testCase.verifyEqual(emptyLabels, zeros(0, 1));
            testCase.verifyEqual(emptyScores, zeros(0, 1));
        end

        function trainsSinglePrecisionConstantFeaturesWithoutOptionalTerms(testCase)
            config = struct('MaxEpochs', 2, 'BatchSize', 12, 'Momentum', 0, ...
                'L2Regularization', 0, 'Standardize', false);
            Z = single([2; 2; 2; 2]);
            y = [0; 0; 1; 1];
            model = fitMLP(Z, y, config);
            config.BatchSize = 4;
            fullBatch = fitMLP(double(Z), y, config);
            testCase.verifyEqual(model.Weights, fullBatch.Weights);
            testCase.verifyEqual(model.Biases, fullBatch.Biases);
            testCase.verifyEmpty(model.StandardizationMean);
            testCase.verifyEmpty(model.StandardizationScale);
            testCase.verifyEqual(model.NumNormalizationParameters, 0);
            testCase.verifyEqual(model.TrainingObjectiveHistory, model.TrainingLossHistory);
            [labels, scores] = predictMLP(model, Z);
            testCase.verifySize(labels, [4 1]);
            testCase.verifyClass(scores, 'double');
            testCase.verifyTrue(all(isfinite(scores)));
            testCase.verifyEqual(scores, repmat(scores(1), 4, 1));
        end

        function rejectsInvalidFeaturesAndLabels(testCase)
            y = [0; 1];
            for bad = {zeros(0, 1), zeros(2, 0), ones(2, 2, 2), uint8([1; 2]), ...
                    sparse([1; 2]), [1; 2i], [1; Inf], [1; NaN]}
                testCase.verifyError(@() fitMLP(bad{1}, y), 'genderRecognition:InvalidMLPFeatures');
            end
            for bad = {[0 1], [0; 1; 1], [0; 2], [0; NaN], [0; Inf], [0; 1i], ...
                    single(y), logical(y), sparse(y)}
                testCase.verifyError(@() fitMLP([1; 2], bad{1}), 'genderRecognition:InvalidMLPLabels');
            end
            testCase.verifyError(@() fitMLP([1; 2], [1; 1]), 'genderRecognition:MissingMLPClass');
        end

        function rejectsInvalidOptions(testCase)
            bad = {[], struct([]), struct('Unknown', 1), struct('Activation', "relu"), ...
                struct('Activation', string(missing)), struct('HiddenLayerSizes', []), ...
                struct('HiddenLayerSizes', [1 0]), struct('HiddenLayerSizes', [2 1.5]), ...
                struct('HiddenLayerSizes', Inf), struct('HiddenLayerSizes', ones(2)), ...
                struct('LearningRate', 0), struct('LearningRate', Inf), struct('LearningRate', sparse(1)), ...
                struct('Momentum', -0.1), struct('Momentum', 1), struct('Momentum', NaN), ...
                struct('MaxEpochs', 0), struct('MaxEpochs', 1.5), struct('MaxEpochs', Inf), ...
                struct('BatchSize', 0), struct('BatchSize', [1 2]), struct('BatchSize', 1i), ...
                struct('L2Regularization', -1), struct('L2Regularization', NaN), ...
                struct('RandomSeed', -1), struct('RandomSeed', 2^32), struct('RandomSeed', 1.5), ...
                struct('RandomSeed', Inf), struct('Standardize', 1), struct('Standardize', sparse(true))};
            for i = 1:numel(bad)
                testCase.verifyError(@() fitMLP([1; 2], [0; 1], bad{i}), 'genderRecognition:InvalidMLPConfig');
            end
        end

        function rejectsMalformedPredictionModelsAndFeatures(testCase)
            model = fixedModel([1 2 1], {[1 -1], [2; 3]}, {[0 0], 0});
            invalid = 'genderRecognition:InvalidMLPModel';
            testCase.verifyError(@() predictMLP(rmfield(model, 'Biases'), 0), invalid);
            testCase.verifyError(@() predictMLP(repmat(model, 2, 1), 0), invalid);
            bad = {'ClassLabels', [1 0]; 'NumFeatures', 2; 'Activation', "sigmoid"; ...
                'LayerSizes', [1 2 2]; 'Weights', {[1; -1], [2; 3]}; ...
                'Weights', {[1 -1], [2; Inf]}; 'Weights', {[1 -1], sparse([2; 3])}; ...
                'Biases', {[0; 0], 0}; 'Biases', {[0 0], NaN}; ...
                'StandardizationMean', 0; 'StandardizationScale', 0};
            for row = 1:size(bad, 1)
                malformed = model;
                malformed.(bad{row, 1}) = bad{row, 2};
                testCase.verifyError(@() predictMLP(malformed, 0), invalid);
            end
            for Z = {[0 1], zeros(0, 2), NaN, sparse(2), int8(2), 1i}
                testCase.verifyError(@() predictMLP(model, Z{1}), 'genderRecognition:InvalidMLPFeatures');
            end
        end

        function rejectsNumericalOverflow(testCase)
            id = 'genderRecognition:MLPNumericalRange';
            y = [0; 1];
            testCase.verifyError(@() fitMLP([-realmax; realmax], y), id);
            config = struct('HiddenLayerSizes', 1, 'LearningRate', realmax, ...
                'L2Regularization', 4, 'MaxEpochs', 1);
            testCase.verifyError(@() fitMLP([-1; 1], y, config), id);
            model = fixedModel([1 1 1], {2, 1}, {0, 0});
            testCase.verifyError(@() predictMLP(model, realmax), id);
            model.StandardizationMean = 0;
            model.StandardizationScale = realmin;
            testCase.verifyError(@() predictMLP(model, realmax), id);
        end

        function integratesPcaAndLdaWithoutHeldOutLeakage(testCase)
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            writeSyntheticImages(folder);
            [images, y, subjects, metadata] = loadDataset(folder);
            testCase.verifyEqual(metadata.Label, y);
            testCase.verifyEqual(metadata.SubjectID, subjects);
            [folds, partition] = createFolds(y, subjects);
            train = folds(1).TrainIndices;
            validation = folds(1).ValidationIndices;
            testCase.verifyEmpty(intersect(subjects(train), subjects(validation)));
            testCase.verifyEqual(partition.SubjectOverlapCounts(1), 0);
            prepConfig = struct('ImageSize', []);
            prep = fitPreprocessor(images(train), prepConfig);
            X = transformImages(prep, images(train));
            XVal = transformImages(prep, images(validation));
            verifyRows(testCase, X, train, metadata, y, subjects);
            verifyRows(testCase, XVal, validation, metadata, y, subjects);

            for index = validation.'
                imwrite(255 - images{index}, fullfile(folder, metadata.Filename(index)));
            end
            [changed, changedY, changedSubjects] = loadDataset(folder);
            testCase.verifyEqual(changedY, y);
            testCase.verifyEqual(changedSubjects, subjects);
            testCase.verifyEqual(createFolds(changedY, changedSubjects), folds);
            changedPrep = fitPreprocessor(changed(train), prepConfig);
            testCase.verifyEqual(changedPrep, prep);
            changedX = transformImages(changedPrep, changed(train));
            changedVal = transformImages(changedPrep, changed(validation));
            testCase.verifyEqual(changedX, X);
            testCase.verifyNotEqual(changedVal, XVal);
            config = struct('MaxEpochs', 12);
            for method = ["pca" "lda"]
                if method == "pca"
                    featConfig = struct('NumComponents', 2);
                    projection = fitPCA(X, featConfig);
                    refitProjection = fitPCA(changedX, featConfig);
                    transform = @transformPCA;
                    K = 2;
                else
                    projection = fitLDA(X, y(train));
                    refitProjection = fitLDA(changedX, changedY(train));
                    transform = @transformLDA;
                    K = 1;
                end
                testCase.verifyEqual(refitProjection, projection);
                ZTrain = transform(projection, X);
                ZVal = transform(projection, XVal);
                testCase.verifySize(ZTrain, [16 K]);
                testCase.verifySize(ZVal, [4 K]);
                testCase.verifyEqual(ZTrain, vertcat(transform(projection, X(1, :)), ...
                    transform(projection, X(2:end, :))), 'AbsTol', 1e-14);
                model = fitMLP(ZTrain, y(train), config);
                refit = fitMLP(transform(refitProjection, changedX), changedY(train), config);
                testCase.verifyEqual(refit, model);
                testCase.verifyEqual(model.StandardizationMean, mean(ZTrain, 1));
                testCase.verifyEqual(model.StandardizationScale, std(ZTrain, 0, 1));
                [yPred, scores] = predictMLP(model, ZVal);
                [changedPred, changedScores] = predictMLP(refit, transform(refitProjection, changedVal));
                testCase.verifyEqual(yPred, double(scores > 0));
                testCase.verifyEqual(changedPred, double(changedScores > 0));
                testCase.verifySize(yPred, [4 1]);
                testCase.verifySize(scores, [4 1]);
                testCase.verifyTrue(all(isfinite([scores; changedScores])));
                testCase.verifyNotEqual(changedScores, scores);
                testCase.verifyEqual(model, refit);
            end
        end

        function analysisReturnsAggregatesWithConfigurableFold(testCase)
            folder = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            writeSyntheticImages(folder);
            prep = struct('ImageSize', []);
            config = struct('MaxEpochs', 3, 'HiddenLayerSizes', [4 2]);
            for method = ["pca" "lda"]
                feature = struct();
                K = 1;
                if method == "pca"
                    feature = struct('NumComponents', 2);
                    K = 2;
                end
                [report, stats] = evalc('run_mlp_analysis(folder, method, config, feature, prep, 2)');
                testCase.verifyEqual(stats.FoldNumber, 2);
                testCase.verifyEqual(stats.TrainClassCounts, [8 8]);
                testCase.verifyEqual(stats.ValidationClassCounts, [2 2]);
                testCase.verifyEqual(stats.SubjectOverlapCount, 0);
                testCase.verifyEqual(stats.ReducedTrainSize, [16 K]);
                testCase.verifyEqual(stats.LayerSizes, [K 4 2 1]);
                testCase.verifyEqual(stats.NumClassifierParameters, 4 * (K + 1) + 13);
                testCase.verifyEqual(stats.NumNormalizationParameters, 2 * K);
                testCase.verifyEqual(stats.ScoreSize, [4 1]);
                testCase.verifyEqual(stats.NumFiniteScores, 4);
                testCase.verifyEqual(stats.EpochsCompleted, 3);
                testCase.verifyEqual(stats.StoppingReason, "MaxEpochs");
                testCase.verifyFalse(any(contains(report, [".png", "man_", "woman_", string(folder)])));
                testCase.verifyFalse(any(isfield(stats, {'Weights', 'Biases', 'StandardizationMean', ...
                    'Scores', 'Predictions', 'TrainIndices', 'ValidationIndices'})));
            end
            testCase.verifyError(@() run_mlp_analysis(folder, "other"), 'genderRecognition:InvalidMLPAnalysisMethod');
            testCase.verifyError(@() run_mlp_analysis(folder, "lda", config, feature, prep, 6), ...
                'genderRecognition:InvalidFoldNumber');
            testCase.verifyError(@() run_mlp_analysis(folder, "pca", config, struct(), prep, 1, 1), ...
                'genderRecognition:InvalidMLPAnalysisPlot');
        end
    end
end

function model = fixedModel(layers, weights, biases)
model = struct('ClassLabels', [0 1], 'NumFeatures', layers(1), ...
    'LayerSizes', layers, 'Weights', {weights}, 'Biases', {biases}, ...
    'Activation', "tanh", 'StandardizationMean', [], 'StandardizationScale', []);
end

function gradient = centralDifference(Z, y, W, b, lambda, layer, index, step, isWeight)
plusW = W;
minusW = W;
plusB = b;
minusB = b;
if isWeight
    plusW{layer}(index) = W{layer}(index) + step;
    minusW{layer}(index) = W{layer}(index) - step;
else
    plusB{layer}(index) = b{layer}(index) + step;
    minusB{layer}(index) = b{layer}(index) - step;
end
[~, plus] = mlpInternal.batchObjective(Z, y, plusW, plusB, lambda);
[~, minus] = mlpInternal.batchObjective(Z, y, minusW, minusB, lambda);
gradient = (plus - minus) / (2 * step);
end

function verifyLearning(testCase, Z, y, probes, probeLabels, config, description)
model = fitMLP(Z, y, config);
testCase.verifyEqual(model.EpochsCompleted, config.MaxEpochs);
testCase.verifyEqual(model.StoppingReason, "MaxEpochs");
testCase.verifyLessThan(model.FinalTrainingObjective, model.InitialTrainingObjective);
testCase.verifyLessThan(model.FinalTrainingLoss, model.InitialTrainingLoss);
testCase.verifyTrue(all(isfinite([model.TrainingLossHistory model.TrainingObjectiveHistory]), 'all'));
testCase.verifyTrue(all(cellfun(@(values) all(isfinite(values), 'all'), [model.Weights model.Biases])));
[yPred, scores] = predictMLP(model, Z);
[probePred, probeScores] = predictMLP(model, probes);
testCase.verifyEqual(yPred, y);
testCase.verifyEqual(probePred, probeLabels);
testCase.verifyTrue(all(isfinite([scores; probeScores])));
fprintf('\nMLP synthetic %s: BCE %.9g -> %.9g, objective %.9g -> %.9g, training %d/%d, probes %d/%d\n', ...
    description, model.InitialTrainingLoss, model.FinalTrainingLoss, ...
    model.InitialTrainingObjective, model.FinalTrainingObjective, ...
    nnz(yPred == y), numel(y), nnz(probePred == probeLabels), numel(probeLabels));
end

function verifyRows(testCase, X, indices, metadata, y, subjects)
for row = 1:numel(indices)
    index = indices(row);
    [label, subject, imageNumber] = parseImageFilename(metadata.Filename(index));
    testCase.verifyEqual(label, y(index));
    testCase.verifyEqual(subject, subjects(index));
    pixels = syntheticPixels(label, str2double(extractAfter(subject, "_")), str2double(imageNumber));
    testCase.verifyEqual(X(row, :), double(pixels) / 255, 'AbsTol', eps);
end
end

function pixels = syntheticPixels(label, subject, imageNumber)
pixels = uint8([15 + 140 * label + 2 * subject, 25 + 12 * imageNumber, ...
    70 + subject * imageNumber, 90 + 20 * label + subject ^ 2]);
end

function writeSyntheticImages(folder)
prefixes = ["man", "woman"];
for label = 0:1
    for subject = 1:5
        for imageNumber = 1:2
            filename = sprintf('%s_%d_%d.png', prefixes(label + 1), subject, imageNumber);
            image = reshape(syntheticPixels(label, subject, imageNumber), 2, 2);
            imwrite(image, fullfile(folder, filename));
        end
    end
end
end
