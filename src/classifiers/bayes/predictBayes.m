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
L0 = factors(:, :, factorIndex(1));
L1 = factors(:, :, factorIndex(2));
mu0 = bayesModel.ClassMeans(1, :);
mu1 = bayesModel.ClassMeans(2, :);
Z = double(Z);
if isempty(Z)
    scores = zeros(0, 1);
elseif isequal(L0, L1)
    if isequal(mu0, mu1)
        % Identical likelihoods cancel everywhere, including far from the mean.
        scores = repmat(logPriorOdds, size(Z, 1), 1);
    else
        scores = affineLogOdds(Z, mu0, mu1, L0, diagonal) + logPriorOdds;
    end
else
    logDeterminants = bayesModel.LogDeterminants(factorIndex);
    scores = quadraticLogOdds(Z, mu0, mu1, L0, L1, diagonal) - ...
        0.5 * diff(logDeterminants) + logPriorOdds;
end
checkFinite(scores, 'Log odds');
yPred = double(scores > 0);
end

function scores = affineLogOdds(Z, mu0, mu1, L, diagonal)
% Equal covariance cancels the common quadratic and log determinant terms.
delta = mu1 - mu0;
deltaScale = 1;
if any(~isfinite(delta))
    % Solve for half the direction if the mean difference would overflow.
    delta = mu1 / 2 - mu0 / 2;
    deltaScale = 2;
end
direction = whiten(delta, L, diagonal);
if diagonal
    direction = direction ./ L;
else
    direction = direction / L;
end
direction = direction * deltaScale;
checkFinite(direction, 'Affine direction');
active = direction ~= 0;
if ~any(active) || (diagonal && any(~active & delta ~= 0))
    error('genderRecognition:BayesNumericalRange', 'The affine direction underflowed.');
end

mu0 = mu0(active);
mu1 = mu1(active);
midpoint = (mu0 + mu1) / 2;
overflowed = ~isfinite(midpoint);
midpoint(overflowed) = mu0(overflowed) / 2 + mu1(overflowed) / 2;
centered = Z(:, active) - midpoint;

% Midpoint rounding can consume half the significant bits of a small gap.
% Keep the rounding error separately so centering preserves that gap.
halfGap = (deltaScale / 2) * delta(active);
closeMeans = halfGap ~= 0 & abs(halfGap) <= (sqrt(eps(1)) / 2) * max(abs(mu0), abs(mu1));
if any(closeMeans)
    midpointRoundoff = (midpoint(closeMeans) - mu0(closeMeans)) - halfGap(closeMeans);
    centered(:, closeMeans) = centered(:, closeMeans) + midpointRoundoff;
end
checkFinite(centered, 'Affine centering');
scores = centered * direction(active).';
end

function scores = quadraticLogOdds(Z, mu0, mu1, L0, L1, diagonal)
w0 = whiten(Z - mu0, L0, diagonal);
w1 = whiten(Z - mu1, L1, diagonal);
difference = w0 - w1;
magnitude = max(abs(w0), abs(w1));
% Outside this range the subtraction amplifies relative errors by at most four.
recover = magnitude > 0 & abs(difference) <= magnitude / 2;
rows = any(recover, 2);
if any(rows)
    % (w0-w1)*L1' = w0*(L1-L0)' + (mu1-mu0). Recover small differences
    % from the factors, including mean shifts rounded away in w0 or w1.
    delta = mu1 - mu0;
    if diagonal
        residual = w0(rows, :) .* (L1 - L0) + delta;
    else
        residual = w0(rows, :) * (L1 - L0).' + delta;
    end
    recovered = whiten(residual, L1, diagonal);
    difference(recover) = recovered(recover(rows, :));
end
average = 0.5 * w0 + 0.5 * w1;
scores = sum(difference .* average, 2);
end

function values = whiten(rows, L, diagonal)
checkFinite(rows, 'Whitening input');
if diagonal
    values = rows ./ L;
else
    values = rows / L.';
end
checkFinite(values, 'Whitened features');
end

function checkFinite(values, operation)
if any(~isfinite(values), 'all')
    error('genderRecognition:BayesNumericalRange', '%s exceeded the double precision range.', operation);
end
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
