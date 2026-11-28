function result = demoIndependent(bundle, options, evaluate)
context = "Official independent testing";
if bundle.Synthetic
    context = "Synthetic independent testing";
end
[directory, metadata] = selectAttempt(bundle, options);
if ~isempty(metadata) && string(metadata.Status) == "complete"
    result = completed(directory, context);
    return
end

if evaluate
    destination = fullfile(finalInternal.resolvePath(options.OutputDirectory), "independent-test");
    run = run_independent_testing(bundle.OutputDirectory, options.TestingPath, ...
        OutputDirectory=destination, GeneratePlots=false);
    directory = run.OutputDirectory;
    if run.Status == "complete"
        result = completed(directory, context);
        return
    end
    metadata = readMetadata(directory);
    status = run.Status;
elseif bundle.TrainingIntegrityDefect
    status = "blocked";
elseif ~isempty(metadata) && any(string(metadata.Status) == ["blocked", "incomplete"])
    status = string(metadata.Status);
else
    testPath = finalInternal.resolvePath(options.TestingPath);
    status = "ready";
    if ~isfolder(testPath) && ~isfile(testPath)
        status = "unavailable";
    end
end
result = struct('Context', context, 'Status', status, 'OutputDirectory', directory, 'Records', struct([]));
fprintf('\n%s | %s\n', context, status);
if ~isempty(metadata)
    fprintf('Saved local attempt: %s | %s | %s\n', directory, metadata.Status, metadata.CompletedUTC);
    if string(metadata.Status) ~= "unavailable" && strlength(string(metadata.Reason)) > 0
        fprintf('%s\n', metadata.Reason);
    end
end
switch status
    case "unavailable"
        fprintf('%s dataset unavailable.\n', context);
        fprintf('Final models are trained and ready.\n');
        fprintf('Independent accuracy, ROC, AUC, and inference time remain pending.\n');
    case "ready"
        fprintf('Configured dataset path exists. Run the independent action to validate and evaluate it.\n');
    case "blocked"
        if bundle.TrainingIntegrityDefect
            fprintf('The training audit recorded a material integrity defect.\n');
        end
end
end

function [directory, metadata] = selectAttempt(bundle, options)
directory = "";
metadata = [];
if strlength(options.IndependentAttempt) > 0
    directory = finalInternal.resolvePath(options.IndependentAttempt);
    assert(isfile(fullfile(directory, 'test-metadata.json')), 'genderRecognition:DemoMissingArtifact', ...
        'IndependentAttempt must point to a local attempt with test-metadata.json.');
    metadata = readMetadata(directory);
    assert(matchesBundle(metadata, bundle), 'genderRecognition:DemoIndependentMismatch', ...
        'The selected independent attempt belongs to a different frozen bundle or data scope.');
    return
end

output = finalInternal.resolvePath(options.OutputDirectory);
parents = unique([fullfile(bundle.OutputDirectory, "independent-test"), ...
    fullfile(output, "independent-test")]);
complete = strings(0, 1);
unavailable = strings(0, 1);
for parent = parents
    entries = dir(fullfile(parent, 'attempt-*', 'test-metadata.json'));
    for entry = entries.'
        if endsWith(entry.folder, ".incomplete")
            continue
        end
        try
            candidate = readMetadata(entry.folder);
            matches = matchesBundle(candidate, bundle);
        catch cause
            [~, name] = fileparts(entry.folder);
            fprintf('Skipping invalid independent attempt %s (%s).\n', name, cause.identifier);
            continue
        end
        if matches
            if string(candidate.Status) == "complete"
                complete(end + 1, 1) = string(entry.folder); %#ok<AGROW>
            elseif string(candidate.Status) == "unavailable"
                unavailable(end + 1, 1) = string(entry.folder); %#ok<AGROW>
            end
        end
    end
end
if numel(complete) > 1
    error('genderRecognition:DemoAmbiguousRun', ...
        'Multiple completed independent attempts for this bundle. Set IndependentAttempt to one of:\n%s', ...
        strjoin(complete, newline));
elseif isscalar(complete)
    directory = complete(1);
elseif isscalar(unavailable)
    directory = unavailable(1);
end
if strlength(directory) > 0
    metadata = readMetadata(directory);
end
end

function metadata = readMetadata(directory)
metadata = jsondecode(fileread(fullfile(directory, 'test-metadata.json')));
assert(isequal(metadata.SchemaVersion, 1) && ...
    any(string(metadata.Status) == ["complete", "unavailable", "blocked", "incomplete"]) && ...
    metadata.OfficialEvaluation == (string(metadata.Status) == "complete" && ~metadata.Synthetic), ...
    'genderRecognition:FinalArtifactInvalid', 'Invalid independent attempt metadata.');
end

function matches = matchesBundle(metadata, bundle)
matches = metadata.Synthetic == bundle.Synthetic && ...
    string(metadata.TrainingRevision) == bundle.Revision && ...
    string(metadata.TrainingConfigurationSHA256) == bundle.ConfigurationSHA256 && ...
    string(metadata.FrozenUTC) == bundle.FrozenUTC && ...
    string(metadata.TrainingCompletedUTC) == bundle.CompletedUTC;
if isfield(metadata, 'ModelManifestSHA256')
    matches = matches && string(metadata.ModelManifestSHA256) == bundle.ManifestSHA256;
end
end

function result = completed(directory, context)
[records, metadata] = loadIndependentTestRun(directory);
fprintf('\n%s | complete | %s | frozen revision %s\nLocal attempt: %s\n', ...
    context, metadata.CompletedUTC, metadata.TrainingRevision, directory);
fprintf('In-memory inference medians in ms for a batch of %d images.\n', metadata.BatchSize);
summary = table([records.Method].', 100*[records.Accuracy].', [records.AUC].', ...
    1000*arrayfun(@(r) r.Timing.Inference.Classifier.MedianSeconds, records(:)), ...
    1000*arrayfun(@(r) r.Timing.Inference.Pipeline.MedianSeconds, records(:)), ...
    'VariableNames', {'Method','AccuracyPct','AUC','ClassifierBatchMs','PipelineBatchMs'});
disp(summary);
fprintf('Confusion matrices use actual labels on rows and predicted labels on columns [0, 1].\n');
for record = records(:).'
    fprintf('%s\n', record.Method);
    disp(record.ConfusionMatrix);
end
result = struct('Context', context, 'Status', "complete", 'OutputDirectory', directory, ...
    'Records', records, 'Summary', summary);
end
