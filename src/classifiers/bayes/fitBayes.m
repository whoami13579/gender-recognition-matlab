function bayesModel = fitBayes(ZTrain, yTrain, config)
%FITBAYES Fit a regularized binary Gaussian classifier to training rows.
%   CONFIG fields are CovarianceType ("full", "shared", "diagonal"),
%   PriorType ("empirical", "equal"), and positive Regularization.
%   Defaults are "full", "empirical", and 0.01, respectively.
arguments
    ZTrain
    yTrain
    config = struct()
end

validateBayesFeatures(ZTrain);
[N, K] = size(ZTrain);
if N == 0
    error('genderRecognition:InvalidBayesFeatures', 'Training requires at least one feature row.');
end
if ~isa(yTrain, 'double') || ~isequal(size(yTrain), [N 1]) || ...
        ~isreal(yTrain) || issparse(yTrain) || any(yTrain ~= 0 & yTrain ~= 1)
    error('genderRecognition:InvalidBayesLabels', 'Labels must be an aligned double column of 0s and 1s.');
end
options = bayesOptions(config);
classLabels = [0 1];
classCounts = [nnz(yTrain == 0) nnz(yTrain == 1)];
if any(classCounts == 0)
    error('genderRecognition:MissingBayesClass', 'Training requires samples from both classes.');
end

classWeights = classCounts / N;
priors = classWeights;
if options.PriorType == "equal"
    priors = [0.5 0.5];
end

classMeans = zeros(2, K);
residuals = cell(1, 2);
for c = 1:2
    Zc = double(ZTrain(yTrain == classLabels(c), :));
    % Preserve exact constants and avoid overflow in sums of large inputs.
    columnScale = max(abs(Zc), [], 1);
    columnScale(columnScale == 0) = 1;
    classMeans(c, :) = mean(Zc ./ columnScale, 1) .* columnScale;
    residuals{c} = Zc - classMeans(c, :);
    if any(~isfinite(residuals{c}), 'all')
        error('genderRecognition:BayesNumericalRange', 'Centering exceeded the double precision range.');
    end
end

[pooledVariances, pooled] = columnVariances(vertcat(residuals{:}), N);
scaleSource = "withinClass";
if any(pooledVariances)
    largest = max(pooledVariances);
    covarianceScale = largest * mean(pooledVariances / largest);
elseif any(classMeans(1, :) ~= classMeans(2, :))
    % Mean per-feature variance of the two class means, weighted by counts.
    weightedDifference = (classMeans(2, :) - classMeans(1, :)) * ...
        (sqrt(prod(classWeights)) / sqrt(K));
    covarianceScale = sum(weightedDifference.^2);
    scaleSource = "betweenClass";
else
    % Identical constant rows give identical likelihoods and prior-only odds.
    covarianceScale = 1;
    scaleSource = "constant";
end
alpha = options.Regularization * covarianceScale;
if ~isfinite(alpha) || alpha <= 0
    error('genderRecognition:BayesNumericalRange', ...
        'The covariance scale or ridge is outside the positive double precision range.');
end

switch options.CovarianceType
    case "full"
        numCovariances = 2;
        covarianceRows = K;
        numCovarianceParameters = K * (K + 1);
    case "shared"
        numCovariances = 1;
        covarianceRows = K;
        numCovarianceParameters = K * (K + 1) / 2;
    case "diagonal"
        numCovariances = 2;
        covarianceRows = 1;
        numCovarianceParameters = 2 * K;
end
covariances = zeros(covarianceRows, K, numCovariances);
factors = zeros(size(covariances));
logDeterminants = zeros(1, numCovariances);
covarianceRanks = zeros(1, numCovariances);
diagonal = options.CovarianceType == "diagonal";
for c = 1:numCovariances
    if options.CovarianceType == "shared"
        scaled = pooled;
    else
        [variances, scaled] = columnVariances(residuals{c}, classCounts(c));
    end
    if diagonal
        covarianceRanks(c) = nnz(variances > K * eps(max(variances)));
        sigma = variances + alpha;
        factor = sqrt(sigma);
        factorDiagonal = factor;
    else
        sigma = scaled.' * scaled;
        % Symmetrize without adding two potentially large matrix entries.
        sigma = tril(sigma) + tril(sigma, -1).';
        if any(~isfinite(sigma), 'all')
            error('genderRecognition:BayesNumericalRange', 'Covariance overflowed.');
        end
        largest = max(abs(sigma), [], 'all');
        if largest > 0
            covarianceRanks(c) = rank(sigma / largest);
        end
        diagonalIndices = 1:K+1:K*K;
        sigma(diagonalIndices) = sigma(diagonalIndices) + alpha;
        if any(~isfinite(sigma), 'all')
            error('genderRecognition:BayesNumericalRange', 'Regularized covariance overflowed.');
        end
        [factor, status] = chol(sigma, 'lower');
        if status ~= 0
            error('genderRecognition:BayesCovarianceFactorization', ...
                'The regularized covariance is not positive definite at this precision.');
        end
        factorDiagonal = diag(factor);
    end
    logDeterminant = 2 * sum(log(factorDiagonal));
    if any(~isfinite(factor), 'all') || any(factorDiagonal <= 0) || ~isfinite(logDeterminant)
        error('genderRecognition:BayesNumericalRange', ...
            'A regularized covariance factor or log determinant is outside the double precision range.');
    end
    covariances(:, :, c) = sigma;
    factors(:, :, c) = factor;
    logDeterminants(c) = logDeterminant;
end

bayesModel = struct('ClassLabels', classLabels, 'ClassCounts', classCounts, ...
    'Priors', priors, 'ClassMeans', classMeans, 'CovarianceType', options.CovarianceType, ...
    'Covariances', covariances, 'CovarianceFactors', factors, 'LogDeterminants', logDeterminants, ...
    'RegularizationCoefficient', options.Regularization, 'EffectiveRegularization', alpha, ...
    'CovarianceScale', covarianceScale, 'CovarianceScaleSource', scaleSource, ...
    'CovarianceRanks', covarianceRanks, 'NumTrainingSamples', N, 'NumFeatures', K, ...
    'NumClassifierParameters', 2 * K + numCovarianceParameters + double(options.PriorType == "empirical"), ...
    'Options', options);
end

function [variances, scaledResiduals] = columnVariances(residuals, count)
% Scale before accumulating the Gram matrix to avoid scatter overflow.
scaledResiduals = residuals ./ sqrt(count);
variances = sum(scaledResiduals.^2, 1);
if any(~isfinite(variances)) || any(variances == 0 & any(residuals ~= 0, 1))
    error('genderRecognition:BayesNumericalRange', 'A positive training variance overflowed or underflowed.');
end
end

function options = bayesOptions(config)
options = struct('CovarianceType', "full", 'PriorType', "empirical", 'Regularization', 1e-2);
if ~isstruct(config) || ~isscalar(config) || ~all(ismember(fieldnames(config), fieldnames(options)))
    error('genderRecognition:InvalidBayesConfig', 'Expected a scalar struct of supported Bayes options.');
end
for name = string(fieldnames(config)).'
    options.(name) = config.(name);
end
options.CovarianceType = optionValue(options.CovarianceType, ["full" "shared" "diagonal"], 'CovarianceType');
options.PriorType = optionValue(options.PriorType, ["empirical" "equal"], 'PriorType');
rho = options.Regularization;
if ~(isa(rho, 'double') || isa(rho, 'single')) || ~isscalar(rho) || ...
        ~isreal(rho) || issparse(rho) || ~isfinite(rho) || rho <= 0
    error('genderRecognition:InvalidBayesConfig', 'Regularization must be a positive finite single or double scalar.');
end
options.Regularization = double(rho);
end

function value = optionValue(value, choices, name)
if ischar(value) && isrow(value)
    value = string(value);
end
if ~isstring(value) || ~isscalar(value) || ismissing(value) || ~any(value == choices)
    error('genderRecognition:InvalidBayesConfig', '%s must be one of: %s.', name, strjoin(choices, ', '));
end
end
