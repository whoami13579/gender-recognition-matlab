function pcaModel = fitPCA(XTrain, config)
%FITPCA Learn a mean-centered PCA projection using economy SVD.
%   Rows of XTRAIN are training samples. CONFIG optionally supplies either
%   NumComponents or ExplainedVarianceThreshold (a fraction in (0, 1]).
%   The default retains up to 40 components, limited by numerical rank.

arguments
    XTrain
    config = struct()
end

validatePCAFeatures(XTrain);
[numSamples, numInputFeatures] = size(XTrain);
if numSamples < 2 || numInputFeatures < 1
    error('genderRecognition:InvalidPCAFeatures', ...
        'Fitting PCA requires at least two samples and one feature.');
end
options = pcaOptions(config);

XTrain = double(XTrain);
featureMean = mean(XTrain, 1);
% Preserve exactly constant columns, even if summation rounded their mean.
constant = all(XTrain == XTrain(1, :), 1);
featureMean(constant) = XTrain(1, constant);
% Scale only columns whose sum overflowed during mean calculation.
overflow = ~isfinite(featureMean);
if any(overflow)
    scale = max(abs(XTrain(:, overflow)), [], 1);
    featureMean(overflow) = mean(XTrain(:, overflow) ./ scale, 1) .* scale;
end
centered = XTrain - featureMean;
if any(~isfinite(centered), 'all')
    error('genderRecognition:PCANumericalRange', ...
        'Centered training features exceed the double precision range.');
end

[~, singularValues, directions] = svd(centered, 'econ', 'vector');
if any(~isfinite(singularValues)) || any(~isfinite(directions), 'all')
    error('genderRecognition:PCANumericalRange', 'Economy SVD produced nonfinite values.');
end
rankTolerance = max(size(centered)) * eps(singularValues(1));
numericalRank = min([nnz(singularValues > rankTolerance), numSamples - 1, numInputFeatures]);
if numericalRank == 0
    error('genderRecognition:ZeroPCAVariance', 'Training features have no usable PCA variance.');
end

% Divide before squaring so intermediate squared values need not fit.
eigenvalues = (singularValues / sqrt(numSamples - 1)).^2;
if any(~isfinite(eigenvalues)) || eigenvalues(numericalRank) == 0
    error('genderRecognition:PCANumericalRange', ...
        'A usable covariance eigenvalue is outside the double precision range.');
end
% All economy singular values contribute to total variance. Normalizing
% before squaring keeps the fraction calculation in a safe numeric range.
weights = (singularValues / singularValues(1)).^2;
varianceRatio = weights / sum(weights);
cumulativeVariance = min(cumsum(varianceRatio), 1);
cumulativeVariance(end) = 1;

switch options.SelectionMode
    case "default"
        numFeatures = min(options.NumComponents, numericalRank);
    case "fixed"
        numFeatures = options.NumComponents;
        if numFeatures > numericalRank
            error('genderRecognition:PCAComponentsExceedRank', ...
                'Requested %d PCA components, but only %d are usable.', numFeatures, numericalRank);
        end
    case "explainedVariance"
        numFeatures = find(cumulativeVariance(1:numericalRank) + 8 * eps(1) >= ...
            options.ExplainedVarianceThreshold, 1);
        if isempty(numFeatures)
            error('genderRecognition:PCAThresholdExceedsRank', ...
                'The requested explained variance is unreachable within numerical rank.');
        end
end

pcaModel = struct('Mean', featureMean, 'Components', directions(:, 1:numFeatures), ...
    'Eigenvalues', eigenvalues, 'ExplainedVarianceRatio', varianceRatio, ...
    'CumulativeExplainedVariance', cumulativeVariance, 'NumericalRank', numericalRank, ...
    'RankTolerance', rankTolerance, 'NumTrainingSamples', numSamples, ...
    'NumInputFeatures', numInputFeatures, 'NumFeatures', numFeatures, ...
    'NumFittedParameters', numInputFeatures * (1 + numFeatures), ...
    'RetainedExplainedVariance', cumulativeVariance(numFeatures), 'Options', options);
end

function options = pcaOptions(config)
if ~isstruct(config) || ~isscalar(config) || ...
        ~all(ismember(fieldnames(config), {'NumComponents', 'ExplainedVarianceThreshold'}))
    error('genderRecognition:InvalidConfig', ...
        'PCA configuration must be a scalar struct with supported option names.');
end
hasCount = isfield(config, 'NumComponents');
hasThreshold = isfield(config, 'ExplainedVarianceThreshold');
if hasCount && hasThreshold
    error('genderRecognition:InvalidConfig', ...
        'Specify either NumComponents or ExplainedVarianceThreshold for PCA.');
end

options = struct('SelectionMode', "default", 'NumComponents', 40, 'ExplainedVarianceThreshold', []);
if hasCount
    value = config.NumComponents;
    if ~isFiniteScalar(value) || value < 1 || value > flintmax || value ~= fix(value)
        error('genderRecognition:InvalidConfig', 'NumComponents must be a positive integer up to flintmax.');
    end
    options.SelectionMode = "fixed";
    options.NumComponents = double(value);
elseif hasThreshold
    value = config.ExplainedVarianceThreshold;
    if ~isFiniteScalar(value) || value <= 0 || value > 1
        error('genderRecognition:InvalidConfig', 'ExplainedVarianceThreshold must be a fraction in (0, 1].');
    end
    options.SelectionMode = "explainedVariance";
    options.NumComponents = [];
    options.ExplainedVarianceThreshold = double(value);
end
end

function valid = isFiniteScalar(value)
valid = isnumeric(value) && isscalar(value) && isreal(value) && ...
    ~issparse(value) && isfinite(value);
end
