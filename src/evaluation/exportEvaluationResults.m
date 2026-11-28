function files = exportEvaluationResults(results, outputDirectory)
%EXPORTEVALUATIONRESULTS Write full JSON and a scalar CSV summary.
%   Existing evaluation-results.json and evaluation-summary.csv are replaced.
%   Keep any real-data results local: finite ROC thresholds retain scores.
if ~isstruct(results) || ~isvector(results) || isempty(results)
    error('genderRecognition:InvalidEvaluationResult', 'Expected a nonempty vector of evaluation records.');
end
if ischar(outputDirectory) && isrow(outputDirectory)
    outputDirectory = string(outputDirectory);
end
if ~isstring(outputDirectory) || ~isscalar(outputDirectory) || ismissing(outputDirectory) || ...
        strlength(strtrim(outputDirectory)) == 0
    error('genderRecognition:InvalidEvaluationOutput', 'Specify a nonempty output directory.');
end
for idx = 1:numel(results)
    validateEvaluationResult(results(idx));
end

json = encodeResults(results);
summary = buildSummary(results);
if ~isfolder(outputDirectory)
    [ok, message] = mkdir(outputDirectory);
    if ~ok
        error('genderRecognition:EvaluationExportFailed', '%s', message);
    end
end
files = struct('JSON', fullfile(outputDirectory, "evaluation-results.json"), ...
    'CSV', fullfile(outputDirectory, "evaluation-summary.csv"));
if isfolder(files.JSON) || isfolder(files.CSV)
    error('genderRecognition:EvaluationExportFailed', 'An output filename names an existing directory.');
end
jsonTemp = string(tempname(outputDirectory)) + ".json";
csvTemp = string(tempname(outputDirectory)) + ".csv";
cleanup = onCleanup(@() deleteTemporaryFiles(jsonTemp, csvTemp));
writeJSON(jsonTemp, json);
writetable(summary, csvTemp);
replaceFile(jsonTemp, files.JSON);
replaceFile(csvTemp, files.CSV);
end

function json = encodeResults(results)
rows = num2cell(results(:));
for idx = 1:numel(rows)
    roc = rows{idx}.ROC;
    roc.InitialThreshold = [];
    if roc.Valid
        roc.InitialThreshold = "positive-infinity";
        roc.Thresholds = roc.Thresholds(2:end);
    end
    % A singleton threshold or timing sample still represents an array.
    roc.Thresholds = num2cell(roc.Thresholds);
    rows{idx}.ROC = roc;
    groups = ["Training", "Inference", "Stages"];
    for group = groups
        values = rows{idx}.Timing.(group);
        for field = string(fieldnames(values)).'
            if ~isempty(values.(field))
                values.(field).SamplesSeconds = num2cell(values.(field).SamplesSeconds);
            end
        end
        rows{idx}.Timing.(group) = values;
    end
end
% The validated schema permits NaN only for an undefined AUC.
json = jsonencode(rows, 'ConvertInfAndNaN', true);
end

function summary = buildSummary(results)
fields = {'SchemaVersion', 'Method', 'EvaluationStage', 'FoldNumber', ...
    'NumTrainingSamples', 'NumEvaluationSamples', 'Accuracy', 'AUC', ...
    'NumOriginalFeatures', 'NumFeatures', 'NumPreprocessingParameters', ...
    'NumProjectionParameters', 'NumNormalizationParameters', ...
    'NumClassifierParameters', 'CombinedParameterCount'};
rows = rmfield(results(:), setdiff(fieldnames(results), fields));
rows = orderfields(rows, fields);
for idx = 1:numel(rows)
    result = results(idx);
    timing = result.Timing;
    if isempty(rows(idx).FoldNumber)
        rows(idx).FoldNumber = NaN;
    end
    rows(idx).ROCValid = result.ROC.Valid;
    rows(idx).ROCStatus = result.ROC.Status;
    rows(idx).ClassifierFitSeconds = timingValue(timing.Training.Classifier, 'MedianSeconds');
    rows(idx).PipelineFitSeconds = timingValue(timing.Training.Pipeline, 'MedianSeconds');
    rows(idx).ClassifierInferenceSeconds = timingValue(timing.Inference.Classifier, 'MedianSeconds');
    rows(idx).PipelineInferenceSeconds = timingValue(timing.Inference.Pipeline, 'MedianSeconds');
    rows(idx).InferenceBatchSize = timingValue(timing.Inference.Pipeline, 'BatchSize');
    if isempty(timing.Inference.Pipeline)
        rows(idx).InferenceBatchSize = timingValue(timing.Inference.Classifier, 'BatchSize');
    end
    rows(idx).PipelineInferenceSecondsPerSample = timingValue(timing.Inference.Pipeline, 'SecondsPerSample');
end
summary = struct2table(rows, 'AsArray', true);
for field = string(summary.Properties.VariableNames)
    values = summary.(field);
    if isnumeric(values) && any(isnan(values))
        cells = num2cell(values);
        cells(isnan(values)) = {[]};
        summary.(field) = cells;
    end
end
end

function value = timingValue(measurement, field)
value = NaN;
if ~isempty(measurement)
    value = measurement.(field);
end
end

function writeJSON(path, json)
[file, message] = fopen(path, 'w', 'n', 'UTF-8');
if file < 0
    error('genderRecognition:EvaluationExportFailed', '%s', message);
end
closeFile = onCleanup(@() fclose(file));
fprintf(file, '%s\n', json);
[message, code] = ferror(file);
if code ~= 0
    error('genderRecognition:EvaluationExportFailed', '%s', message);
end
end

function replaceFile(source, target)
[ok, message] = movefile(source, target, 'f');
if ~ok
    error('genderRecognition:EvaluationExportFailed', '%s', message);
end
end

function deleteTemporaryFiles(varargin)
for idx = 1:nargin
    if isfile(varargin{idx})
        delete(varargin{idx});
    end
end
end
