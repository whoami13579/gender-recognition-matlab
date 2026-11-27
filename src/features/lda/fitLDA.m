function ldaModel = fitLDA(XTrain, yTrain, config)
%FITLDA Learn one regularized Fisher direction from labeled training rows.
%   YTRAIN is a double column with 0 for male and 1 for female. CONFIG
%   optionally supplies a positive Regularization coefficient (default 0.01).
%   TRANSFORMLDA reuses the global training mean and unit direction.

arguments
    XTrain
    yTrain
    config = struct()
end

validateLDAFeatures(XTrain);
[numSamples, numInputFeatures] = size(XTrain);
if isempty(XTrain)
    error('genderRecognition:InvalidLDAFeatures', 'Fitting LDA requires a nonempty feature matrix.');
end
if ~isa(yTrain, 'double') || ~isreal(yTrain) || issparse(yTrain) || ...
        ~isequal(size(yTrain), [numSamples 1]) || any(yTrain ~= 0 & yTrain ~= 1)
    error('genderRecognition:InvalidLabels', ...
        'Labels must be an N-by-1 double column of zeros and ones aligned with the training rows.');
end
class0 = yTrain == 0;
classCounts = [nnz(class0) nnz(~class0)];
if any(classCounts == 0)
    error('genderRecognition:MissingClass', 'Both label classes are required.');
end
options = ldaOptions(config);

XTrain = double(XTrain);
X0 = XTrain(class0, :);
X1 = XTrain(~class0, :);
classMeans = [trainingMean(X0); trainingMean(X1)];
featureMean = trainingMean(XTrain);
delta = (classMeans(2, :) - classMeans(1, :)).';
if ~any(delta)
    error('genderRecognition:IdenticalLDAClassMeans', ...
        'Fisher LDA requires distinct training-class means.');
end
deltaNorm = norm(delta);
R = [X0 - classMeans(1, :); X1 - classMeans(2, :)];
if ~isfinite(deltaNorm) || deltaNorm == 0 || any(~isfinite(R), 'all')
    error('genderRecognition:LDANumericalRange', ...
        'The class mean difference or within-class residuals exceed the reliable double precision range.');
end
unitDelta = delta ./ deltaNorm;
residualNorm = norm(R, 'fro');
scatterTrace = residualNorm^2;
zeroScatter = ~any(R, 'all');
ridgeNorm = residualNorm;
if zeroScatter
    ridgeNorm = deltaNorm;
end
% Apply the dimension and coefficient before squaring to keep the ridge in range.
alpha = ((ridgeNorm / sqrt(numInputFeatures)) * sqrt(options.Regularization))^2;
if ~isfinite(scatterTrace) || (~zeroScatter && scatterTrace == 0) || ...
        ~isfinite(alpha) || alpha < realmin
    error('genderRecognition:LDANumericalRange', ...
        'The scatter trace or effective ridge is outside the supported double precision range.');
end

if zeroScatter
    direction = unitDelta / norm(unitDelta);
    rankTolerance = max(size(R)) * eps(0);
    withinClassRank = 0;
    relativeResidual = 0;
else
    [direction, singularValues, relativeResidual] = solveFisher(R, unitDelta, residualNorm, alpha);
    rankTolerance = max(size(R)) * eps(singularValues(1));
    withinClassRank = min(nnz(singularValues > rankTolerance), min(numSamples - 2, numInputFeatures));
end

if direction.' * unitDelta < 0
    direction = -direction;
end
separation = direction.' * delta;
projectedMeans = (classMeans - featureMean) * direction;
projectedResidualNorm = norm(R * direction);
projectedScatter = projectedResidualNorm^2;
criterion = (separation / hypot(projectedResidualNorm, sqrt(alpha)))^2;
if any(~isfinite([projectedMeans; separation; projectedScatter; criterion])) || ...
        separation <= 0 || projectedMeans(2) <= projectedMeans(1) || criterion <= 0
    error('genderRecognition:LDANumericalRange', ...
        'The fitted class separation or Fisher criterion cannot be represented reliably.');
end

ldaModel = struct('Mean', featureMean, 'Components', direction, ...
    'ClassMeans', classMeans, 'ClassCounts', classCounts, ...
    'WithinClassRank', withinClassRank, 'RankTolerance', rankTolerance, ...
    'RegularizationCoefficient', options.Regularization, 'EffectiveRegularization', alpha, ...
    'NumTrainingSamples', numSamples, 'NumInputFeatures', numInputFeatures, 'NumFeatures', 1, ...
    'NumFittedParameters', 2 * numInputFeatures, 'Options', options, ...
    'MeanDifferenceNorm', deltaNorm, 'WithinClassScatterTrace', scatterTrace, ...
    'ProjectedClassMeans', projectedMeans.', 'ProjectedWithinClassScatter', projectedScatter, ...
    'ProjectedMeanSeparation', separation, 'RegularizedFisherCriterion', criterion, ...
    'RelativeResidual', relativeResidual, 'UsedZeroScatterFallback', zeroScatter);
end

function options = ldaOptions(config)
options = struct('Regularization', 1e-2);
if ~isstruct(config) || ~isscalar(config) || ...
        ~all(strcmp(fieldnames(config), 'Regularization'))
    error('genderRecognition:InvalidConfig', ...
        'LDA configuration must be a scalar struct with an optional Regularization field.');
end
if isfield(config, 'Regularization')
    value = config.Regularization;
    if ~isnumeric(value) || ~isreal(value) || issparse(value) || ...
            ~isscalar(value) || ~isfinite(value) || value <= 0
        error('genderRecognition:InvalidConfig', 'Regularization must be a positive finite numeric scalar.');
    end
    options.Regularization = double(value);
end
end

function center = trainingMean(X)
center = mean(X, 1);
% Preserve exact constancy and avoid overflow from summing large values.
constant = all(X == X(1, :), 1);
center(constant) = X(1, constant);
overflow = ~isfinite(center);
if any(overflow)
    scale = max(abs(X(:, overflow)), [], 1);
    center(overflow) = mean(X(:, overflow) ./ scale, 1) .* scale;
end
end

function [direction, singularValues, relativeResidual] = solveFisher(R, unitDelta, residualNorm, alpha)
% Scale the system before squaring the singular values.
systemScale = max(residualNorm, sqrt(alpha));
A = R ./ systemScale;
beta = (sqrt(alpha) / systemScale)^2;
if beta < realmin
    error('genderRecognition:LDANumericalRange', 'The scaled ridge is below the normal double precision range.');
end

[numSamples, numInputFeatures] = size(A);
if numInputFeatures > numSamples
    % The solution lies in span([unitDelta, A.']). A thin QR captures
    % this span, including the part of delta in the within-class null
    % space, without subtracting two nearly equal feature-space vectors.
    [Q, T] = qr([unitDelta A.'], 'econ');
    rhs = T(:, 1);
    reducedResidual = T(:, 2:end).';
    % One zero row gives economy SVD every right direction of the
    % compressed problem, including the extra null direction.
    [~, s, V] = svd([reducedResidual; zeros(1, size(T, 1))], 'econ', 'vector');
    scaledSolution = Q * (V * ((beta ./ (s.^2 + beta)) .* (V.' * rhs)));
else
    [~, s, V] = svd(A, 'econ', 'vector');
    scaledSolution = V * ((beta ./ (s.^2 + beta)) .* (V.' * unitDelta));
end
singularValues = s * systemScale;

solutionNorm = norm(scaledSolution);
defect = A.' * (A * scaledSolution) + beta * (scaledSolution - unitDelta);
relativeResidual = norm(defect) / (beta * norm(unitDelta));
if any(~isfinite(singularValues)) || ~isfinite(solutionNorm) || solutionNorm == 0 || ...
        ~isfinite(relativeResidual) || relativeResidual > 1e-6
    error('genderRecognition:LDANumericalRange', ...
        'The regularized Fisher solution failed its finite-range or relative-residual check.');
end
direction = scaledSolution / solutionNorm;
end
