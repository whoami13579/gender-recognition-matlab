function results = crossValidatePipelines(images, y, subjectIDs, config)
%CROSSVALIDATEPIPELINES Evaluate four methods on one five-fold training partition.
arguments
    images
    y
    subjectIDs
    config = struct()
end
config = crossValidationConfig(config);
if ~iscell(images) || ~isvector(images) || numel(images) ~= numel(y)
    error('genderRecognition:InvalidImageBatch', 'Supply one image cell per label.');
end
images = images(:);
[folds, partition] = createFolds(y, subjectIDs, config.Partition);
methods = pipelineMethods();
records = struct([]);
predictions = struct([]);
failures = struct('Method', {}, 'FoldNumber', {}, 'Identifier', {}, 'Reason', {});
inner = cell(numel(folds), 1);
searches = cell(numel(folds), numel(methods));
recommendations = cell(1, numel(methods));
finalSearch = [];
searchSeconds = 0;
nested = config.Mode == "nested";
candidateLists = cell(size(methods));
innerOptions = struct('NumFolds', config.Search.NumFolds, 'RandomSeed', config.Search.RandomSeed, ...
    'GroupBySubject', partition.GroupBySubject);
if nested
    for m = 1:numel(methods)
        candidateLists{m} = pipelineCandidates(methods(m), config.Pipeline, config.Search);
    end
end

for fold = 1:numel(folds)
    fit = folds(fold).TrainIndices;
    validation = folds(fold).ValidationIndices;
    imagesTrain = images(fit);
    yTrain = y(fit);
    if nested
        subjects = subjectIDs;
        if ~isempty(subjects)
            subjects = subjects(fit);
        end
        started = tic;
        [innerFolds, innerPartition] = createFolds(yTrain, subjects, innerOptions);
        inner{fold} = struct('Folds', innerFolds, 'Partition', innerPartition);
        searchSeconds = searchSeconds + toc(started);
    end
    % Rotate the first method across folds, then benchmark serially.
    order = mod((0:numel(methods)-1) + fold - 1, numel(methods)) + 1;
    for m = order
        method = methods(m);
        fitConfig = config.Pipeline;
        if nested
            selection = selectPipelineConfig(imagesTrain, yTrain, innerFolds, method, candidateLists{m});
            searches{fold, m} = selection;
            searchSeconds = searchSeconds + selection.Seconds;
            if isempty(selection.SelectedIndex)
                failures(end + 1) = newFailure(method, fold, "genderRecognition:NoValidCandidate", ...
                    selection.SelectionReason); %#ok<AGROW>
                continue
            end
            fitConfig = selection.SelectedConfiguration;
        end
        try
            [record, prediction] = evaluatePipelineFold(imagesTrain, yTrain, ...
                images(validation), y(validation), method, fold, fitConfig, config.Benchmark);
            records = appendRow(records, record);
            predictions = appendRow(predictions, prediction);
        catch cause
            if ~startsWith(cause.identifier, 'genderRecognition:')
                rethrow(cause);
            end
            failures(end + 1) = newFailure(method, fold, cause.identifier, cause.message); %#ok<AGROW>
        end
    end
end

if ~nested
    for m = 1:numel(methods)
        recommendations{m} = config.Pipeline;
    end
elseif isempty(failures) && config.Search.RecommendFinal
    started = tic;
    [finalFolds, finalPartition] = createFolds(y, subjectIDs, innerOptions);
    finalSearch = struct('Folds', finalFolds, 'Partition', finalPartition, 'Selections', []);
    for m = 1:numel(methods)
        selection = selectPipelineConfig(images, y, finalFolds, methods(m), candidateLists{m});
        finalSearch.Selections = appendRow(finalSearch.Selections, selection);
        recommendations{m} = selection.SelectedConfiguration;
    end
    finalSearch.Seconds = toc(started);
    finalSearch.Status = "complete";
    if any(cellfun(@isempty, recommendations))
        finalSearch.Status = "incomplete";
    end
end
handoff = struct('Method', cellstr(methods), 'Configuration', recommendations);
status = "complete";
if ~isempty(failures) || any(arrayfun(@(record) ~record.ROC.Valid, records))
    status = "incomplete";
end
results = struct('Config', config, 'Methods', methods, 'Folds', folds, 'Partition', partition, ...
    'Records', records, 'Predictions', predictions, 'Failures', failures, 'Inner', {inner}, ...
    'Searches', {searches}, 'SearchSeconds', searchSeconds, 'FinalSearch', finalSearch, ...
    'Recommendations', handoff, 'Status', status);
end

function failure = newFailure(method, fold, identifier, reason)
failure = struct('Method', method, 'FoldNumber', fold, 'Identifier', string(identifier), ...
    'Reason', string(reason));
end

function rows = appendRow(rows, row)
if isempty(rows)
    rows = row;
else
    rows(end + 1, 1) = row;
end
end
