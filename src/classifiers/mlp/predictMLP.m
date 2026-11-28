function [yPred, scores] = predictMLP(mlpModel, Z)
%PREDICTMLP Apply a fixed binary network to rows of projected features.
%   SCORES are female-class logits. Scores strictly above zero predict 1.
Z = validateMLPFeatures(Z);
validateNetwork(mlpModel);
if size(Z, 2) ~= mlpModel.NumFeatures
    error('genderRecognition:InvalidMLPFeatures', 'Prediction requires %d feature columns.', mlpModel.NumFeatures);
end
Z = applyMLPStandardization(Z, mlpModel.StandardizationMean, mlpModel.StandardizationScale);
scores = mlpInternal.forward(Z, mlpModel.Weights, mlpModel.Biases);
yPred = double(scores > 0);
end

function validateNetwork(model)
required = {'ClassLabels', 'NumFeatures', 'LayerSizes', 'Weights', 'Biases', ...
    'Activation', 'StandardizationMean', 'StandardizationScale'};
id = 'genderRecognition:InvalidMLPModel';
if ~isstruct(model) || ~isscalar(model) || ~all(isfield(model, required)) || ...
        ~isequal(model.ClassLabels, [0 1]) || ~isequal(model.Activation, "tanh")
    error(id, 'Expected a fitted binary tanh MLP model.');
end
layers = model.LayerSizes;
if ~isa(layers, 'double') || ~isrow(layers) || ~isreal(layers) || issparse(layers) || ...
        numel(layers) < 3 || any(~isfinite(layers) | layers < 1 | layers ~= fix(layers)) || ...
        layers(end) ~= 1 || ~isequal(model.NumFeatures, layers(1))
    error(id, 'The input, hidden, and output layer widths are inconsistent.');
end
if ~iscell(model.Weights) || ~iscell(model.Biases) || ...
        numel(model.Weights) ~= numel(layers) - 1 || numel(model.Biases) ~= numel(model.Weights)
    error(id, 'Expected one weight matrix and one bias row per layer.');
end
for layer = 1:numel(model.Weights)
    if ~validArray(model.Weights{layer}, layers(layer:layer + 1)) || ...
            ~validArray(model.Biases{layer}, [1 layers(layer + 1)])
        error(id, 'Invalid weight or bias array at layer %d.', layer);
    end
end
mu = model.StandardizationMean;
scale = model.StandardizationScale;
if ~isequal(mu, []) || ~isequal(scale, [])
    if ~validArray(mu, [1 layers(1)]) || ~validArray(scale, [1 layers(1)]) || any(scale <= 0)
        error(id, 'Expected finite normalization rows and strictly positive scales.');
    end
end
end

function valid = validArray(value, shape)
valid = isa(value, 'double') && isreal(value) && ~issparse(value) && ...
    isequal(size(value), shape) && all(isfinite(value), 'all');
end
