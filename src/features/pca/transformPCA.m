function Z = transformPCA(pcaModel, X)
%TRANSFORMPCA Apply a fitted training mean and PCA basis to feature rows.
%   X can have zero rows, provided it has NumInputFeatures columns.

validateProjection(pcaModel);
validatePCAFeatures(X);
if size(X, 2) ~= pcaModel.NumInputFeatures
    error('genderRecognition:PCAFeatureDimensionMismatch', ...
        'Expected %d input features, but received %d.', pcaModel.NumInputFeatures, size(X, 2));
end

centered = double(X) - pcaModel.Mean;
Z = centered * pcaModel.Components;
if any(~isfinite(centered), 'all') || any(~isfinite(Z), 'all')
    error('genderRecognition:PCANumericalRange', ...
        'Centering or projecting features exceeded the double precision range.');
end
end

function validateProjection(model)
if ~isstruct(model) || ~isscalar(model) || ...
        ~all(isfield(model, {'Mean', 'Components', 'NumInputFeatures', ...
        'NumFeatures', 'NumTrainingSamples', 'NumericalRank', 'NumFittedParameters'}))
    error('genderRecognition:InvalidPCAModel', 'Expected a scalar fitted PCA model.');
end

[numInputFeatures, numFeatures] = size(model.Components);
if ~isFiniteDoubleMatrix(model.Mean) || ~isFiniteDoubleMatrix(model.Components) || ...
        ~isequal(size(model.Mean), [1 numInputFeatures]) || ...
        ~isequal(model.NumInputFeatures, numInputFeatures) || ...
        ~isequal(model.NumFeatures, numFeatures) || numFeatures == 0 || ...
        ~isequal(model.NumFittedParameters, numInputFeatures * (1 + numFeatures)) || ...
        ~isPositiveInteger(model.NumTrainingSamples) || ~isPositiveInteger(model.NumericalRank) || ...
        model.NumericalRank > min(model.NumTrainingSamples - 1, numInputFeatures) || ...
        numFeatures > model.NumericalRank
    error('genderRecognition:InvalidPCAModel', 'Fitted PCA coefficients or dimensions are invalid.');
end
% Check each column's scale without forming a component Gram matrix at inference.
if any(abs(sum(model.Components.^2, 1) - 1) > 8 * numInputFeatures * eps(1))
    error('genderRecognition:InvalidPCAModel', 'Fitted PCA directions must have unit norm.');
end
end

function valid = isFiniteDoubleMatrix(value)
valid = isa(value, 'double') && ismatrix(value) && isreal(value) && ...
    ~issparse(value) && all(isfinite(value), 'all');
end

function valid = isPositiveInteger(value)
valid = isFiniteDoubleMatrix(value) && isscalar(value) && value > 0 && value == fix(value);
end
