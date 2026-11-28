function config = mergePipelineConfig(config, overrides)
% Module-specific validation is performed by the existing fit functions.
if ~isstruct(overrides) || ~isscalar(overrides) || ...
        ~all(ismember(fieldnames(overrides), fieldnames(config)))
    error('genderRecognition:InvalidPipelineConfig', 'Expected Preprocessing, PCA, LDA, Bayes or MLP options.');
end
for group = string(fieldnames(overrides)).'
    options = overrides.(group);
    allowed = fieldnames(config.(group));
    if group == "PCA"
        allowed = {'NumComponents'; 'ExplainedVarianceThreshold'};
    end
    if ~isstruct(options) || ~isscalar(options) || ~all(ismember(fieldnames(options), allowed))
        error('genderRecognition:InvalidPipelineConfig', 'Unsupported field in %s options.', group);
    end
    if group == "PCA" && isfield(options, 'ExplainedVarianceThreshold')
        config.PCA = options;
    else
        for field = string(fieldnames(options)).'
            config.(group).(field) = options.(field);
        end
    end
end
end
