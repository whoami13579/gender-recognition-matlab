function [yPred, scores] = predictBayes(bayesModel, Z)
%PREDICTBAYES Score feature rows with a fitted Gaussian Bayes model.
%   SCORES are log posterior odds of class 1 versus class 0. Exact ties
%   predict class 0. A 0-by-K input returns two 0-by-1 double columns.
[factorIndex, diagonal] = validateModel(bayesModel);
validateBayesFeatures(Z);
if size(Z, 2) ~= bayesModel.NumFeatures
    error('genderRecognition:BayesFeatureDimensionMismatch', ...
        'Expected %d input features, but received %d.', bayesModel.NumFeatures, size(Z, 2));
end

logPriorOdds = diff(log(bayesModel.Priors));
factors = bayesModel.CovarianceFactors;
logDeterminants = bayesModel.LogDeterminants(factorIndex);
if isequal(bayesModel.ClassMeans(1, :), bayesModel.ClassMeans(2, :)) && ...
        isequal(factors(:, :, factorIndex(1)), factors(:, :, factorIndex(2))) && ...
        logDeterminants(1) == logDeterminants(2)
    % Identical likelihoods cancel everywhere, including far from the mean.
    scores = repmat(logPriorOdds, size(Z, 1), 1);
else
    distances = zeros(size(Z, 1), 2);
    Z = double(Z);
    for c = 1:2
        centered = Z - bayesModel.ClassMeans(c, :);
        if any(~isfinite(centered), 'all')
            error('genderRecognition:BayesNumericalRange', 'Centering exceeded the double precision range.');
        end
        factor = factors(:, :, factorIndex(c));
        if diagonal
            whitened = centered ./ factor;
        else
            whitened = centered / factor.';
        end
        distances(:, c) = sum(whitened.^2, 2);
        if any(~isfinite(distances(:, c))) || ...
                any(distances(:, c) == 0 & any(centered ~= 0, 2))
            error('genderRecognition:BayesNumericalRange', ...
                'A squared Mahalanobis distance overflowed or underflowed.');
        end
    end
    % Compute g1 - g0 term by term. The common K*log(2*pi) term cancels,
    % and large common distances cannot erase a smaller prior term.
    scores = -0.5 * diff(distances, 1, 2) - 0.5 * diff(logDeterminants) + logPriorOdds;
    if any(~isfinite(scores))
        error('genderRecognition:BayesNumericalRange', 'The log odds exceeded the double precision range.');
    end
end
yPred = double(scores > 0);
end

function [factorIndex, diagonal] = validateModel(model)
required = {'ClassLabels', 'NumFeatures', 'ClassMeans', 'Priors', 'CovarianceType', ...
    'CovarianceFactors', 'LogDeterminants'};
if ~isstruct(model) || ~isscalar(model) || ~all(isfield(model, required))
    error('genderRecognition:InvalidBayesModel', 'Expected a scalar fitted Gaussian Bayes model.');
end
K = size(model.ClassMeans, 2);
if K == 0 || ~isequal(model.NumFeatures, K) || ~isequal(model.ClassLabels, [0 1]) || ...
        ~isFiniteDoubleArray(model.ClassMeans) || ~isequal(size(model.ClassMeans), [2 K]) || ...
        ~isFiniteDoubleArray(model.Priors) || ~isequal(size(model.Priors), [1 2]) || ...
        any(model.Priors <= 0) || abs(sum(model.Priors) - 1) > 2 * eps(1) || ...
        ~isstring(model.CovarianceType) || ~isscalar(model.CovarianceType) || ...
        ismissing(model.CovarianceType) || ~any(model.CovarianceType == ["full" "shared" "diagonal"])
    error('genderRecognition:InvalidBayesModel', 'Fitted Bayes classes, means, priors, or covariance type are invalid.');
end

factorIndex = [1 2];
if model.CovarianceType == "shared"
    factorIndex = [1 1];
end
numFactors = factorIndex(2);
diagonal = model.CovarianceType == "diagonal";
factorRows = K;
if diagonal
    factorRows = 1;
end
L = model.CovarianceFactors;
logDeterminants = model.LogDeterminants;
if ~isFiniteDoubleArray(L) || ndims(L) > 3 || ...
        size(L, 1) ~= factorRows || size(L, 2) ~= K || size(L, 3) ~= numFactors || ...
        ~isFiniteDoubleArray(logDeterminants) || ~isequal(size(logDeterminants), [1 numFactors])
    error('genderRecognition:InvalidBayesModel', 'Fitted Bayes covariance factors or log determinants are invalid.');
end
for c = 1:numFactors
    factor = L(:, :, c);
    if diagonal
        factorDiagonal = factor;
    else
        if ~istril(factor)
            error('genderRecognition:InvalidBayesModel', 'Full covariance factors must be lower triangular.');
        end
        factorDiagonal = diag(factor);
    end
    if any(factorDiagonal <= 0)
        error('genderRecognition:InvalidBayesModel', 'Covariance factor diagonals must be positive.');
    end
    expected = 2 * sum(log(factorDiagonal));
    tolerance = 8 * K * eps(max(1, abs(expected)));
    if abs(logDeterminants(c) - expected) > tolerance
        error('genderRecognition:InvalidBayesModel', 'Log determinants must agree with the covariance factors.');
    end
end
end

function valid = isFiniteDoubleArray(value)
valid = isa(value, 'double') && isreal(value) && ~issparse(value) && all(isfinite(value), 'all');
end
