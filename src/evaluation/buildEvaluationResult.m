function result = buildEvaluationResult(context, metrics, preprocessor, projection, classifier, timings)
%BUILDEVALUATIONRESULT Assemble a schema-v1 row for one fitted method and batch.
%   CONTEXT has Method, EvaluationStage and optionally FoldNumber. METRICS
%   comes from evaluateClassification. TIMINGS maps named scopes to the
%   results of benchmarkOperation. Omitted scopes are recorded as [].
if nargin < 6
    timings = struct();
end
if isstruct(context) && isscalar(context) && ~isfield(context, 'FoldNumber')
    context.FoldNumber = [];
end
if ~hasEvaluationFields(context, {'Method', 'EvaluationStage', 'FoldNumber'}) || ...
        ~isTextScalar(context.Method) || ~isTextScalar(context.EvaluationStage) || ...
        ~hasEvaluationFields(metrics, {'NumEvaluationSamples', 'Accuracy', 'NumCorrect', ...
        'ConfusionMatrix', 'ROC', 'AUC'})
    error('genderRecognition:InvalidEvaluationResult', 'Expected an evaluation context and classification metrics.');
end
names = evaluationTimingFields();
if ~isstruct(timings) || ~isscalar(timings) || ~all(ismember(fieldnames(timings), names))
    error('genderRecognition:InvalidEvaluationResult', 'Expected a scalar struct of supported timing scopes.');
end
for idx = 1:numel(names)
    if ~isfield(timings, names{idx})
        timings.(names{idx}) = [];
    end
end
timing = struct('Units', "seconds", ...
    'Training', struct('Classifier', timings.ClassifierFit, 'Pipeline', timings.PipelineFit), ...
    'Inference', struct('Classifier', timings.ClassifierInference, 'Pipeline', timings.PipelineInference), ...
    'Stages', rmfield(timings, names(1:4)));

parameters = summarizeModelParameters(preprocessor, projection, classifier);
if ~isfield(classifier, 'NumTrainingSamples') || ~isfield(projection, 'NumTrainingSamples') || ...
        ~isequal(classifier.NumTrainingSamples, projection.NumTrainingSamples)
    error('genderRecognition:InvalidEvaluationModel', 'Projection and classifier training counts must agree.');
end
config = struct('Preprocessing', [], 'Projection', readOptions(projection), ...
    'Classifier', readOptions(classifier));
if ~isequal(preprocessor, [])
    config.Preprocessing = readOptions(preprocessor);
end

result = struct('SchemaVersion', 1, 'Method', string(context.Method), ...
    'EvaluationStage', string(context.EvaluationStage), 'FoldNumber', context.FoldNumber, ...
    'NumTrainingSamples', classifier.NumTrainingSamples);
for field = [string(fieldnames(metrics)); string(fieldnames(parameters))].'
    if isfield(metrics, field)
        result.(field) = metrics.(field);
    else
        result.(field) = parameters.(field);
    end
end
result.Timing = timing;
result.Configuration = config;
validateEvaluationResult(result);
end

function valid = isTextScalar(value)
valid = (isstring(value) && isscalar(value) && ~ismissing(value)) || (ischar(value) && isrow(value));
end

function options = readOptions(model)
if ~isfield(model, 'Options') || ~isstruct(model.Options) || ~isscalar(model.Options)
    error('genderRecognition:InvalidEvaluationModel', 'Expected effective Options on the fitted model.');
end
options = model.Options;
end
