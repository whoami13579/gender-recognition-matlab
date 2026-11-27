function Z = transformLDA(ldaModel, X)
%TRANSFORMLDA Apply the fitted training mean and Fisher direction to rows.
%   Returns an N-by-1 double column. A 0-by-D input gives a 0-by-1 result.

validateProjection(ldaModel);
validateLDAFeatures(X);
if size(X, 2) ~= ldaModel.NumInputFeatures
    error('genderRecognition:LDAFeatureDimensionMismatch', ...
        'Expected %d input features, but received %d.', ldaModel.NumInputFeatures, size(X, 2));
end

centered = double(X) - ldaModel.Mean;
Z = centered * ldaModel.Components;
if any(~isfinite(centered), 'all') || any(~isfinite(Z), 'all')
    error('genderRecognition:LDANumericalRange', ...
        'Centering or projecting features exceeded the double precision range.');
end
end

function validateProjection(model)
if ~isstruct(model) || ~isscalar(model) || ...
        ~all(isfield(model, {'Mean', 'Components', 'NumInputFeatures', 'NumFeatures'}))
    error('genderRecognition:InvalidLDAModel', 'Expected a scalar fitted LDA model.');
end
[numInputFeatures, numFeatures] = size(model.Components);
if ~isFiniteDoubleMatrix(model.Mean) || ~isFiniteDoubleMatrix(model.Components) || ...
        ~isequal(size(model.Mean), [1 numInputFeatures]) || numInputFeatures == 0 || ...
        ~isequal(model.NumInputFeatures, numInputFeatures) || numFeatures ~= 1 || ...
        ~isequal(model.NumFeatures, 1) || abs(norm(model.Components) - 1) > 8 * numInputFeatures * eps(1)
    error('genderRecognition:InvalidLDAModel', 'Fitted LDA coefficients, norm, or dimensions are invalid.');
end
end

function valid = isFiniteDoubleMatrix(value)
valid = isa(value, 'double') && ismatrix(value) && isreal(value) && ...
    ~issparse(value) && all(isfinite(value), 'all');
end
