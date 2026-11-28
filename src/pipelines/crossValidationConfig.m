function config = crossValidationConfig(overrides)
%CROSSVALIDATIONCONFIG Resolve experiment options. Paths are handled by the script.
arguments
    overrides = struct()
end
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
config = struct('TrainingPath', "train", ...
    'Pipeline', defaultPipelineConfig(), 'Partition', struct('NumFolds', 5, ...
    'RandomSeed', 42, 'GroupBySubject', true), ...
    'Benchmark', struct('WarmupRuns', 2, 'NumRepeats', 5), ...
    'Mode', "baseline", 'Search', struct('NumFolds', 4, 'RandomSeed', 42, ...
    'Candidates', struct(), 'RecommendFinal', true), ...
    'OutputDirectory', string(fullfile(root, 'results', 'cross-validation')), ...
    'GeneratePlots', true, 'PlotFoldROC', false, 'ExportResults', true);

if ~isstruct(overrides) || ~isscalar(overrides) || ...
        ~all(ismember(fieldnames(overrides), fieldnames(config)))
    error('genderRecognition:InvalidCVConfig', 'Expected a scalar struct of supported cross-validation options.');
end
for field = string(fieldnames(overrides)).'
    value = overrides.(field);
    if field == "Pipeline"
        config.Pipeline = mergePipelineConfig(config.Pipeline, value);
    elseif any(field == ["Partition", "Benchmark", "Search"])
        if ~isstruct(value) || ~isscalar(value) || ...
                ~all(ismember(fieldnames(value), fieldnames(config.(field))))
            error('genderRecognition:InvalidCVConfig', 'Invalid %s options.', field);
        end
        for option = string(fieldnames(value)).'
            config.(field).(option) = value.(option);
        end
    else
        config.(field) = value;
    end
end
config.Mode = textOption(config.Mode, 'Mode');
config.TrainingPath = textOption(config.TrainingPath, 'TrainingPath');
config.OutputDirectory = textOption(config.OutputDirectory, 'OutputDirectory');
if ~any(config.Mode == ["baseline", "nested"]) || ~isequal(config.Partition.NumFolds, 5)
    error('genderRecognition:InvalidCVConfig', 'Use baseline or nested mode and exactly five outer folds.');
end
if ~islogical(config.GeneratePlots) || ~isscalar(config.GeneratePlots) || ...
        ~islogical(config.PlotFoldROC) || ~isscalar(config.PlotFoldROC) || ...
        ~islogical(config.ExportResults) || ~isscalar(config.ExportResults) || ...
        ~islogical(config.Search.RecommendFinal) || ~isscalar(config.Search.RecommendFinal)
    error('genderRecognition:InvalidCVConfig', 'Plot, export and recommendation flags must be scalar logicals.');
end
keys = replace(pipelineMethods(), "-", "_");
if ~isstruct(config.Search.Candidates) || ~isscalar(config.Search.Candidates) || ...
        ~all(ismember(fieldnames(config.Search.Candidates), keys))
    error('genderRecognition:InvalidCVConfig', 'Specify candidate lists using method keys such as pca_bayes.');
end
end

function value = textOption(value, name)
if ischar(value) && isrow(value)
    value = string(value);
end
if ~isstring(value) || ~isscalar(value) || ismissing(value) || strlength(strip(value)) == 0
    error('genderRecognition:InvalidCVConfig', '%s must be nonempty scalar text.', name);
end
end
