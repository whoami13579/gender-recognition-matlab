function mlpModel = fitMLP(ZTrain, yTrain, config)
%FITMLP Train a fully connected binary classifier on the supplied rows.
%   See docs/en/mlp-classifier.md for options and fitted model fields.
arguments
    ZTrain
    yTrain
    config = struct()
end

ZTrain = validateMLPFeatures(ZTrain);
[N, K] = size(ZTrain);
if N == 0
    error('genderRecognition:InvalidMLPFeatures', 'Training requires at least one feature row.');
end
if ~isa(yTrain, 'double') || ~isequal(size(yTrain), [N 1]) || ...
        ~isreal(yTrain) || issparse(yTrain) || any(yTrain ~= 0 & yTrain ~= 1)
    error('genderRecognition:InvalidMLPLabels', 'Labels must be an aligned double column of 0s and 1s.');
end
classCounts = [nnz(yTrain == 0) nnz(yTrain == 1)];
if any(classCounts == 0)
    error('genderRecognition:MissingMLPClass', 'Training requires samples from both classes.');
end
options = mlpOptions(config);
featureMean = [];
featureScale = [];
if options.Standardize
    featureMean = mean(ZTrain, 1);
    featureScale = std(ZTrain, 0, 1);
    constant = all(ZTrain == ZTrain(1, :), 1);
    featureMean(constant) = ZTrain(1, constant);
    featureScale(constant) = 1;
    % Rescale residuals if squaring them exceeded the normal numeric range.
    rescale = ~constant & (featureScale == 0 | ~isfinite(featureScale));
    if any(rescale)
        residuals = ZTrain(:, rescale) - featureMean(rescale);
        largest = max(abs(residuals), [], 1);
        featureScale(rescale) = largest .* sqrt(sum((residuals ./ largest).^2, 1) / (N - 1));
    end
    if any(~isfinite(featureMean)) || any(~isfinite(featureScale)) || any(featureScale <= 0)
        error('genderRecognition:MLPNumericalRange', ...
            'A training mean or positive sample standard deviation is outside the double precision range.');
    end
    ZTrain = applyMLPStandardization(ZTrain, featureMean, featureScale);
end

layers = [K options.HiddenLayerSizes 1];
stream = RandStream(char(options.RandomStream), 'Seed', options.RandomSeed);
[weights, biases] = mlpInternal.initialize(layers, stream);
weightVelocity = cellfun(@(W) zeros(size(W)), weights, 'UniformOutput', false);
biasVelocity = cellfun(@(b) zeros(size(b)), biases, 'UniformOutput', false);
lossHistory = zeros(options.MaxEpochs + 1, 1);
objectiveHistory = zeros(size(lossHistory));
lambda = options.L2Regularization;
[lossHistory(1), objectiveHistory(1)] = mlpInternal.batchObjective(ZTrain, yTrain, weights, biases, lambda);
batchSize = min(options.BatchSize, N);
for epoch = 1:options.MaxEpochs
    order = randperm(stream, N);
    for first = 1:batchSize:N
        batch = order(first:min(first + batchSize - 1, N));
        [~, ~, weightGradients, biasGradients] = mlpInternal.batchObjective( ...
            ZTrain(batch, :), yTrain(batch), weights, biases, lambda);
        % All gradients were evaluated at the same pre-update parameters.
        for layer = 1:numel(weights)
            weightVelocity{layer} = options.Momentum * weightVelocity{layer} ...
                - options.LearningRate * weightGradients{layer};
            biasVelocity{layer} = options.Momentum * biasVelocity{layer} ...
                - options.LearningRate * biasGradients{layer};
            weights{layer} = weights{layer} + weightVelocity{layer};
            biases{layer} = biases{layer} + biasVelocity{layer};
            if any(~isfinite(weights{layer}), 'all') || any(~isfinite(biases{layer}), 'all')
                error('genderRecognition:MLPNumericalRange', 'A training update exceeded the double precision range.');
            end
        end
    end
    [lossHistory(epoch + 1), objectiveHistory(epoch + 1)] = ...
        mlpInternal.batchObjective(ZTrain, yTrain, weights, biases, lambda);
end

numParameters = sum(cellfun(@numel, weights)) + sum(cellfun(@numel, biases));
assert(numParameters == sum((layers(1:end-1) + 1) .* layers(2:end)), ...
    'genderRecognition:InvalidMLPModel', 'Stored parameters disagree with the network widths.');
mlpModel = struct('ClassLabels', [0 1], 'ClassCounts', classCounts, ...
    'NumFeatures', K, 'LayerSizes', layers, 'Weights', {weights}, 'Biases', {biases}, ...
    'Activation', options.Activation, 'StandardizationMean', featureMean, ...
    'StandardizationScale', featureScale, 'NumNormalizationParameters', numel(featureMean) + numel(featureScale), ...
    'NumClassifierParameters', numParameters, 'NumTrainingSamples', N, ...
    'EpochsCompleted', options.MaxEpochs, 'StoppingReason', "MaxEpochs", ...
    'InitialTrainingLoss', lossHistory(1), 'FinalTrainingLoss', lossHistory(end), ...
    'InitialTrainingObjective', objectiveHistory(1), 'FinalTrainingObjective', objectiveHistory(end), ...
    'TrainingLossHistory', lossHistory, 'TrainingObjectiveHistory', objectiveHistory, 'Options', options);
end

function options = mlpOptions(config)
options = struct('HiddenLayerSizes', 16, 'Activation', "tanh", ...
    'LearningRate', 0.01, 'Momentum', 0.9, 'MaxEpochs', 300, 'BatchSize', 16, ...
    'L2Regularization', 1e-4, 'Standardize', true, 'RandomSeed', 42);
if ~isstruct(config) || ~isscalar(config) || ~all(ismember(fieldnames(config), fieldnames(options)))
    error('genderRecognition:InvalidMLPConfig', 'Expected a scalar struct of supported MLP options.');
end
for name = string(fieldnames(config)).'
    options.(name) = config.(name);
end
widths = options.HiddenLayerSizes;
if ~isnumeric(widths) || ~isvector(widths) || isempty(widths) || ...
        ~isreal(widths) || issparse(widths) || any(~isfinite(widths)) || ...
        any(widths < 1 | widths ~= fix(widths) | widths >= flintmax)
    error('genderRecognition:InvalidMLPConfig', 'HiddenLayerSizes must contain positive integer widths.');
end
options.HiddenLayerSizes = double(widths(:).');
activation = options.Activation;
if ischar(activation) && isrow(activation)
    activation = string(activation);
end
if ~isequal(activation, "tanh")
    error('genderRecognition:InvalidMLPConfig', 'Activation must be "tanh".');
end
options.Activation = activation;
if ~islogical(options.Standardize) || ~isscalar(options.Standardize) || issparse(options.Standardize)
    error('genderRecognition:InvalidMLPConfig', 'Standardize must be a logical scalar.');
end

options.LearningRate = numericOption(options.LearningRate, 'LearningRate', 0, Inf, false);
options.Momentum = numericOption(options.Momentum, 'Momentum', 0, 1, false);
options.MaxEpochs = numericOption(options.MaxEpochs, 'MaxEpochs', 1, flintmax, true);
options.BatchSize = numericOption(options.BatchSize, 'BatchSize', 1, flintmax, true);
options.L2Regularization = numericOption(options.L2Regularization, 'L2Regularization', 0, Inf, false);
options.RandomSeed = numericOption(options.RandomSeed, 'RandomSeed', 0, 2^32, true);
if options.LearningRate == 0
    error('genderRecognition:InvalidMLPConfig', 'LearningRate must be strictly positive.');
end
options.Initialization = "glorotUniform";
options.RandomStream = "mt19937ar";
end

function value = numericOption(value, name, lower, upper, requireInteger)
if ~isnumeric(value) || ~isscalar(value) || ~isreal(value) || issparse(value) || ...
        ~isfinite(value) || value < lower || value >= upper || (requireInteger && value ~= fix(value))
    error('genderRecognition:InvalidMLPConfig', 'Invalid value for %s.', name);
end
value = double(value);
end
