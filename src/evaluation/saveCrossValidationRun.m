function directory = saveCrossValidationRun(results)
%SAVECROSSVALIDATIONRUN Publish one local run directory after all writes succeed.
config = results.Config;
modeDirectory = fullfile(config.OutputDirectory, config.Mode);
ensureDirectory(modeDirectory);
[~, token] = fileparts(tempname(modeDirectory));
directory = fullfile(modeDirectory, "run-" + token);
staging = directory + ".incomplete";
ensureDirectory(staging);

if config.ExportResults
    if ~isempty(results.Records)
        exportEvaluationResults(results.Records, staging);
        writeFoldDetails(results.Records, fullfile(staging, 'fold-details.csv'));
    end
    writetable(results.Summary, fullfile(staging, 'method-summary.csv'), 'Encoding', 'UTF-8');
    writeRunJSON(staging, 'experiment-config.json', config);
    writeRunJSON(staging, 'folds.json', struct('Folds', results.Folds, 'Partition', results.Partition));
    writeRunJSON(staging, 'configuration-recommendations.json', results.Recommendations);
    if config.Mode == "nested"
        writeRunJSON(staging, 'hyperparameter-search.json', struct('Methods', results.Methods, ...
            'Inner', {results.Inner}, 'Searches', {results.Searches}, ...
            'SearchSeconds', results.SearchSeconds, 'FinalSearch', results.FinalSearch));
    end
    predictions = results.Predictions;
    save(fullfile(staging, 'predictions.mat'), 'predictions');
end
if config.GeneratePlots
    plotCrossValidationRun(results, fullfile(staging, 'figures'), config.PlotFoldROC);
end
metadata = results.Metadata;
metadata.Mode = config.Mode;
metadata.Methods = results.Methods;
metadata.Failures = results.Failures;
metadata.ResultsExported = config.ExportResults;
% The manifest is the last file written; staging names are never loadable runs.
writeRunJSON(staging, 'experiment-metadata.json', metadata);
[ok, message] = movefile(staging, directory);
if ~ok
    error('genderRecognition:CVExportFailed', '%s', message);
end
end

function writeFoldDetails(records, path)
rows = struct2table(rmfield(records, {'ROC', 'Timing', 'Configuration', 'ConfusionMatrix'}), 'AsArray', true);
for idx = 1:numel(records)
    row = records(idx);
    rows.NegativeSamples(idx) = row.ROC.NumNegativeSamples;
    rows.PositiveSamples(idx) = row.ROC.NumPositiveSamples;
    rows.TN(idx) = row.ConfusionMatrix(1, 1);
    rows.FP(idx) = row.ConfusionMatrix(1, 2);
    rows.FN(idx) = row.ConfusionMatrix(2, 1);
    rows.TP(idx) = row.ConfusionMatrix(2, 2);
    rows.ClassifierFitSeconds(idx) = row.Timing.Training.Classifier.MedianSeconds;
    rows.PipelineFitSeconds(idx) = row.Timing.Training.Pipeline.MedianSeconds;
    rows.ClassifierInferenceSeconds(idx) = row.Timing.Inference.Classifier.MedianSeconds;
    rows.PipelineInferenceSeconds(idx) = row.Timing.Inference.Pipeline.MedianSeconds;
    rows.ClassifierSecondsPerImage(idx) = row.Timing.Inference.Classifier.SecondsPerSample;
    rows.PipelineSecondsPerImage(idx) = row.Timing.Inference.Pipeline.SecondsPerSample;
    rows.EffectiveConfiguration(idx) = string(jsonencode(row.Configuration));
end
writetable(rows, path, 'Encoding', 'UTF-8');
end

function ensureDirectory(directory)
[ok, message] = mkdir(directory);
if ~ok
    error('genderRecognition:CVExportFailed', '%s', message);
end
end

function writeRunJSON(directory, name, value)
[file, message] = fopen(fullfile(directory, name), 'w', 'n', 'UTF-8');
if file < 0
    error('genderRecognition:CVExportFailed', '%s', message);
end
cleanup = onCleanup(@() fclose(file));
fprintf(file, '%s\n', jsonencode(value, 'PrettyPrint', true));
[message, code] = ferror(file);
if code ~= 0
    error('genderRecognition:CVExportFailed', '%s', message);
end
end
