function summary = summarizeModelParameters(preprocessor, projection, classifier)
%SUMMARIZEMODELPARAMETERS Read dimensions and fitted counts from model metadata.
%   Supply [] when no preprocessing model is used. Bayes has no extra
%   normalization field. MLP must report its count even when disabled.
numOriginal = readCount(projection, 'NumInputFeatures', 1);
numFeatures = readCount(projection, 'NumFeatures', 1);
if numFeatures > numOriginal || numFeatures ~= readCount(classifier, 'NumFeatures', 1)
    error('genderRecognition:InvalidEvaluationModel', 'Projection and classifier dimensions are inconsistent.');
end
numPreprocessing = 0;
if ~isequal(preprocessor, [])
    if numOriginal ~= readCount(preprocessor, 'NumFeatures', 1)
        error('genderRecognition:InvalidEvaluationModel', 'Preprocessor and projection dimensions differ.');
    end
    numPreprocessing = readCount(preprocessor, 'NumFittedParameters', 0);
end
numProjection = readCount(projection, 'NumFittedParameters', 0);
numClassifier = readCount(classifier, 'NumClassifierParameters', 1);
numNormalization = 0;
if isfield(classifier, 'NumNormalizationParameters') || isfield(classifier, 'LayerSizes')
    numNormalization = readCount(classifier, 'NumNormalizationParameters', 0);
end
counts = [numPreprocessing, numProjection, numNormalization, numClassifier];
% Reject a combined count beyond the exact integer range even when the sum rounds.
remaining = flintmax;
for count = counts
    if count > remaining
        error('genderRecognition:InvalidEvaluationModel', 'Combined parameter count exceeds flintmax.');
    end
    remaining = remaining - count;
end
summary = struct('NumOriginalFeatures', numOriginal, 'NumFeatures', numFeatures, ...
    'DimensionalityReductionFraction', 1 - numFeatures / numOriginal, ...
    'NumPreprocessingParameters', numPreprocessing, 'NumProjectionParameters', numProjection, ...
    'NumNormalizationParameters', numNormalization, 'NumClassifierParameters', numClassifier, ...
    'CombinedParameterCount', sum(counts));
end

function count = readCount(model, field, minimum)
if ~isstruct(model) || ~isscalar(model) || ~isfield(model, field) || ~isEvaluationCount(model.(field), minimum)
    error('genderRecognition:InvalidEvaluationModel', 'Expected fitted model metadata with valid %s.', field);
end
count = model.(field);
end
