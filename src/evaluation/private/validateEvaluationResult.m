function validateEvaluationResult(result)
%VALIDATEEVALUATIONRESULT Validate one schema-v1 record before use or export.
id = 'genderRecognition:InvalidEvaluationResult';
dimensions = {'NumOriginalFeatures', 'NumFeatures', 'DimensionalityReductionFraction'};
counts = {'NumPreprocessingParameters', 'NumProjectionParameters', ...
    'NumNormalizationParameters', 'NumClassifierParameters', 'CombinedParameterCount'};
fields = [{'SchemaVersion', 'Method', 'EvaluationStage', 'FoldNumber', ...
    'NumTrainingSamples', 'NumEvaluationSamples', 'Accuracy', 'NumCorrect', 'ConfusionMatrix', ...
    'ROC', 'AUC', 'Timing', 'Configuration'}, dimensions, counts];
if ~hasEvaluationFields(result, fields) || ~isequal(result.SchemaVersion, 1) || ...
        ~isChoice(result.Method, ["pca-bayes", "lda-bayes", "pca-mlp", "lda-mlp"]) || ...
        ~isChoice(result.EvaluationStage, ["synthetic", "cross-validation", "independent-test"])
    error(id, 'Expected one supported schema-v1 evaluation record.');
end
if result.EvaluationStage == "cross-validation"
    validFold = isEvaluationCount(result.FoldNumber, 1);
else
    validFold = isequal(result.FoldNumber, []);
end
if ~validFold || ~isEvaluationCount(result.NumTrainingSamples, 1) || ~isEvaluationCount(result.NumEvaluationSamples, 1)
    error(id, 'Expected valid sample counts and a fold number only for cross-validation.');
end

matrix = result.ConfusionMatrix;
if ~isRealDouble(matrix) || ~isequal(size(matrix), [2 2]) || ...
        ~all(arrayfun(@(value) isEvaluationCount(value, 0), matrix(:))) || ...
        sum(matrix, 'all') ~= result.NumEvaluationSamples || ...
        ~isequal(result.NumCorrect, trace(matrix)) || ~isequal(result.Accuracy, trace(matrix) / sum(matrix, 'all'))
    error(id, 'Confusion matrix, accuracy and sample counts are inconsistent.');
end
validateROC(result.ROC, result.AUC, sum(matrix, 2));

if ~isEvaluationCount(result.NumOriginalFeatures, 1) || ~isEvaluationCount(result.NumFeatures, 1) || ...
        result.NumFeatures > result.NumOriginalFeatures || ...
        ~isequal(result.DimensionalityReductionFraction, 1 - result.NumFeatures / result.NumOriginalFeatures)
    error(id, 'Feature dimensions and reduction fraction are inconsistent.');
end
remaining = flintmax;
for idx = 1:4
    count = result.(counts{idx});
    if ~isEvaluationCount(count, double(idx == 4)) || count > remaining
        error(id, 'Expected supported model parameter counts.');
    end
    remaining = remaining - count;
end
if ~isequal(result.CombinedParameterCount, flintmax - remaining)
    error(id, 'Combined parameter count does not match component counts.');
end

validateTiming(result.Timing);
validateEvaluationConfiguration(result.Configuration, result.Method);
end

function validateROC(roc, auc, counts)
id = 'genderRecognition:InvalidEvaluationResult';
if ~hasEvaluationFields(roc, {'FPR', 'TPR', 'Thresholds', 'NumPositiveSamples', ...
        'NumNegativeSamples', 'NumDistinctScores', 'Valid', 'Status'}) || ...
        ~isequal(roc.NumNegativeSamples, counts(1)) || ~isequal(roc.NumPositiveSamples, counts(2)) || ...
        ~isEvaluationCount(roc.NumDistinctScores, 1) || roc.NumDistinctScores > sum(counts) || ...
        ~islogical(roc.Valid) || ~isscalar(roc.Valid) || issparse(roc.Valid) || ...
        roc.Valid ~= all(counts > 0)
    error(id, 'ROC metadata does not match the evaluation sample counts.');
end
if ~roc.Valid
    status = "MissingPositiveClass";
    if counts(1) == 0
        status = "MissingNegativeClass";
    end
    if ~isequaln(auc, NaN) || ~isequal(roc.Status, status) || ...
            ~isequal(roc.FPR, zeros(0, 1)) || ~isequal(roc.TPR, zeros(0, 1)) || ...
            ~isequal(roc.Thresholds, zeros(0, 1))
        error(id, 'Single-class records require an undefined AUC, empty ROC vectors, and a diagnostic.');
    end
    return;
end
expectedSize = [roc.NumDistinctScores + 1, 1];
if ~isequal(roc.Status, "OK") || ~isequal(size(roc.FPR), expectedSize) || ...
        ~isequal(size(roc.TPR), expectedSize) || ~isequal(size(roc.Thresholds), expectedSize) || ...
        ~isRealDouble(roc.FPR) || ~isRealDouble(roc.TPR) || ~isRealDouble(roc.Thresholds) || ...
        ~all(isfinite(roc.FPR)) || ~all(isfinite(roc.TPR)) || ...
        roc.FPR(1) ~= 0 || roc.TPR(1) ~= 0 || roc.FPR(end) ~= 1 || roc.TPR(end) ~= 1 || ...
        ~issorted(roc.FPR) || ~issorted(roc.TPR) || any(diff(roc.FPR) == 0 & diff(roc.TPR) == 0) || ...
        roc.Thresholds(1) ~= Inf || ~all(isfinite(roc.Thresholds(2:end))) || ...
        ~issorted(roc.Thresholds, 'strictdescend') || ~isequal(auc, trapz(roc.FPR, roc.TPR))
    error(id, 'Expected consistent ROC coordinates, thresholds and AUC.');
end
end

function validateTiming(timing)
id = 'genderRecognition:InvalidEvaluationResult';
scopes = evaluationTimingFields();
if ~hasEvaluationFields(timing, {'Units', 'Training', 'Inference', 'Stages'}) || ...
        ~isequal(timing.Units, "seconds") || ~hasEvaluationFields(timing.Training, {'Classifier', 'Pipeline'}) || ...
        ~hasEvaluationFields(timing.Inference, {'Classifier', 'Pipeline'}) || ...
        ~hasEvaluationFields(timing.Stages, scopes(5:end))
    error(id, 'Expected training, inference, and stage timings in seconds.');
end
validateMeasurement(timing.Training.Classifier, false);
validateMeasurement(timing.Training.Pipeline, false);
validateMeasurement(timing.Inference.Classifier, true);
validateMeasurement(timing.Inference.Pipeline, true);
if ~isempty(timing.Inference.Classifier) && ~isempty(timing.Inference.Pipeline) && ...
        timing.Inference.Classifier.BatchSize ~= timing.Inference.Pipeline.BatchSize
    error(id, 'Classifier and pipeline inference must use the same batch size.');
end
for idx = 5:numel(scopes)
    validateMeasurement(timing.Stages.(scopes{idx}), false);
end
end

function validateMeasurement(value, requiresBatch)
if isequal(value, [])
    return;
end
id = 'genderRecognition:InvalidEvaluationResult';
fields = {'SamplesSeconds', 'WarmupRuns', 'NumRepeats', 'MedianSeconds', 'MeanSeconds', ...
    'StdSeconds', 'MinSeconds', 'MaxSeconds', 'BatchSize', 'SecondsPerSample', 'MeanSecondsPerSample'};
if ~hasEvaluationFields(value, fields) || ~isEvaluationCount(value.WarmupRuns, 0) || ...
        ~isRealDouble(value.SamplesSeconds) || ~iscolumn(value.SamplesSeconds) || isempty(value.SamplesSeconds) || ...
        any(~isfinite(value.SamplesSeconds) | value.SamplesSeconds < 0)
    error(id, 'Expected finite, nonnegative benchmark measurements.');
end
if requiresBatch
    validBatch = isEvaluationCount(value.BatchSize, 1);
else
    validBatch = isequal(value.BatchSize, []);
end
if ~validBatch
    error(id, 'Batch size does not match the measurement scope.');
end
expected = summarizeTimingSamples(value.SamplesSeconds, value.WarmupRuns, value.BatchSize);
if ~isfinite(expected.MeanSeconds) || ~isfinite(expected.StdSeconds) || ~isequaln(value, expected)
    error(id, 'Benchmark summaries or batch size do not match the measurement scope.');
end
end

function valid = isRealDouble(value)
valid = isa(value, 'double') && isreal(value) && ~issparse(value);
end

function valid = isChoice(value, choices)
valid = isstring(value) && isscalar(value) && ~ismissing(value) && any(value == choices);
end
