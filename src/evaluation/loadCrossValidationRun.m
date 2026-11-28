function results = loadCrossValidationRun(directory)
%LOADCROSSVALIDATIONRUN Rebuild summaries and plot inputs without fitting models.
directory = strip(string(directory));
if endsWith(strip(replace(directory, "\", "/"), 'right', '/'), ".incomplete")
    error('genderRecognition:CVReloadFailed', 'The directory contains an uncommitted export.');
end
metadata = readJSON(directory, 'experiment-metadata.json');
if metadata.ExperimentSchemaVersion ~= 1 || ~metadata.ResultsExported
    error('genderRecognition:CVReloadFailed', 'Expected a supported run directory with exported results.');
end
config = readJSON(directory, 'experiment-config.json');
foldData = readJSON(directory, 'folds.json');
methods = reshape(string(metadata.Methods), 1, []);
records = struct([]);
if metadata.CompletedRecords > 0
    records = loadEvaluationResults(fullfile(directory, 'evaluation-results.json'));
end
[summary, curves] = summarizeCrossValidation(records, methods, numel(foldData.Folds));
status = string(metadata.Status);
if numel(records) ~= metadata.CompletedRecords || ...
        numel(methods) * numel(foldData.Folds) ~= metadata.ExpectedRecords || ...
        string(metadata.Mode) ~= string(config.Mode) || ~any(status == ["complete", "incomplete"]) || ...
        (status == "complete" && any(summary.ValidAUCFolds ~= summary.ExpectedFolds))
    error('genderRecognition:CVReloadFailed', 'The manifest and evaluation records disagree.');
end
results = struct('Config', config, 'Metadata', metadata, 'Status', status, ...
    'Methods', methods, 'Folds', foldData.Folds, 'Partition', foldData.Partition, ...
    'Records', records, 'Summary', summary, 'MeanROC', curves, 'OutputDirectory', directory);
end

function value = readJSON(directory, name)
value = jsondecode(fileread(fullfile(directory, name)));
end
