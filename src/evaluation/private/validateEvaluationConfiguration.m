function validateEvaluationConfiguration(config, method)
%VALIDATEEVALUATIONCONFIGURATION Limit exports to known, path-free fit options.
id = 'genderRecognition:InvalidEvaluationResult';
if ~hasEvaluationFields(config, {'Preprocessing', 'Projection', 'Classifier'})
    error(id, 'Expected preprocessing, projection, and classifier configuration.');
end
prep = config.Preprocessing;
if ~isequal(prep, [])
    if ~hasEvaluationFields(prep, {'Grayscale', 'ImageSize', 'OutputType', 'IntensityScaling'}) || ...
            ~islogical(prep.Grayscale) || ~isscalar(prep.Grayscale) || issparse(prep.Grayscale) || ...
            ~(isequal(prep.ImageSize, []) || isIntegerRow(prep.ImageSize, 2)) || ...
            ~isChoice(prep.OutputType, ["double", "single"]) || ~isChoice(prep.IntensityScaling, "unit")
        error(id, 'Expected supported preprocessing options.');
    end
end

projection = config.Projection;
if startsWith(method, "pca-")
    if ~hasEvaluationFields(projection, {'SelectionMode', 'NumComponents', 'ExplainedVarianceThreshold'}) || ...
            ~isChoice(projection.SelectionMode, ["default", "fixed", "explainedVariance"])
        error(id, 'Expected supported PCA options.');
    end
    if projection.SelectionMode == "explainedVariance"
        valid = isequal(projection.NumComponents, []) && isPositive(projection.ExplainedVarianceThreshold) && ...
            projection.ExplainedVarianceThreshold <= 1;
    else
        valid = isEvaluationCount(projection.NumComponents, 1) && isequal(projection.ExplainedVarianceThreshold, []);
    end
    if ~valid
        error(id, 'PCA settings must match the selected retention mode.');
    end
elseif ~hasEvaluationFields(projection, {'Regularization'}) || ~isPositive(projection.Regularization)
    error(id, 'Expected supported Fisher LDA options.');
end

classifier = config.Classifier;
if endsWith(method, "-bayes")
    if ~hasEvaluationFields(classifier, {'CovarianceType', 'PriorType', 'Regularization'}) || ...
            ~isChoice(classifier.CovarianceType, ["full", "shared", "diagonal"]) || ...
            ~isChoice(classifier.PriorType, ["empirical", "equal"]) || ~isPositive(classifier.Regularization)
        error(id, 'Expected supported Bayes options.');
    end
elseif ~hasEvaluationFields(classifier, {'HiddenLayerSizes', 'Activation', 'LearningRate', ...
        'Momentum', 'MaxEpochs', 'BatchSize', 'L2Regularization', 'Standardize', 'RandomSeed', ...
        'Initialization', 'RandomStream'}) || ...
        ~isIntegerRow(classifier.HiddenLayerSizes, numel(classifier.HiddenLayerSizes)) || ...
        ~isChoice(classifier.Activation, "tanh") || ~isPositive(classifier.LearningRate) || ...
        ~isChoice(classifier.Initialization, "glorotUniform") || ~isChoice(classifier.RandomStream, "mt19937ar") || ...
        ~isNonnegative(classifier.Momentum) || classifier.Momentum >= 1 || ...
        ~isEvaluationCount(classifier.MaxEpochs, 1) || ~isEvaluationCount(classifier.BatchSize, 1) || ...
        ~isNonnegative(classifier.L2Regularization) || ~islogical(classifier.Standardize) || ...
        ~isscalar(classifier.Standardize) || issparse(classifier.Standardize) || ...
        ~isEvaluationCount(classifier.RandomSeed, 0) || classifier.RandomSeed > double(intmax('uint32'))
    error(id, 'Expected supported MLP options.');
end
end

function valid = isChoice(value, choices)
valid = isstring(value) && isscalar(value) && ~ismissing(value) && any(value == choices);
end

function valid = isIntegerRow(value, count)
valid = isnumeric(value) && isreal(value) && ~issparse(value) && isrow(value) && count > 0 && ...
    numel(value) == count && all(isfinite(value) & value >= 1 & value <= flintmax & value == fix(value));
end

function valid = isPositive(value)
valid = isNonnegative(value) && value > 0;
end

function valid = isNonnegative(value)
valid = isa(value, 'double') && isscalar(value) && isreal(value) && ~issparse(value) && ...
    isfinite(value) && value >= 0;
end
