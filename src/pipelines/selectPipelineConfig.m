function selection = selectPipelineConfig(imagesTrain, yTrain, innerFolds, method, candidates)
%SELECTPIPELINECONFIG Rank training-only candidates on one shared inner partition.
%   CANDIDATES is an ordered cell vector of pipeline configurations.
pipelineOperations(method);
if ~iscell(candidates) || isempty(candidates) || ~isvector(candidates) || numel(innerFolds) < 2
    error('genderRecognition:InvalidCVConfig', 'Selection needs a candidate list and multiple inner folds.');
end
indices = (1:numel(yTrain)).';
visits = zeros(size(indices));
for fold = 1:numel(innerFolds)
    fit = innerFolds(fold).TrainIndices;
    validation = innerFolds(fold).ValidationIndices;
    if isempty(fit) || isempty(validation) || ~isequal(sort([fit(:); validation(:)]), indices)
        error('genderRecognition:InvalidPartition', 'Inner fold %d does not partition the supplied training batch.', fold);
    end
    visits(validation) = visits(validation) + 1;
end
if any(visits ~= 1)
    error('genderRecognition:InvalidPartition', 'Inner validation must cover every training sample once.');
end

tolerance = 1e-12;
rule = "Highest mean inner AUC, then mean inner accuracy within 1e-12 of each maximum; " + ...
    "then fewest mean combined parameters; then earliest candidate.";
started = tic;
for idx = numel(candidates):-1:1
    rows(idx, 1) = struct('Configuration', candidates{idx}, 'Folds', [], 'Valid', false, ...
        'MeanAUC', NaN, 'MeanAccuracy', NaN, 'MeanCombinedParameters', NaN, 'Seconds', NaN);
end
for idx = 1:numel(candidates)
    candidateTimer = tic;
    outcomes = repmat(struct('Accuracy', NaN, 'AUC', NaN, 'CombinedParameters', NaN, ...
        'NumTrainingSamples', 0, 'NumValidationSamples', 0, 'CovarianceScale', NaN, ...
        'MedianAbsoluteScore', NaN, 'FailureID', "", 'FailureReason', ""), numel(innerFolds), 1);
    for fold = 1:numel(innerFolds)
        fit = innerFolds(fold).TrainIndices;
        validation = innerFolds(fold).ValidationIndices;
        outcomes(fold).NumTrainingSamples = numel(fit);
        outcomes(fold).NumValidationSamples = numel(validation);
        try
            pipeline = fitPipeline(imagesTrain(fit), yTrain(fit), method, candidates{idx});
            [labels, scores] = predictPipeline(pipeline, imagesTrain(validation));
            metrics = evaluateClassification(yTrain(validation), labels, scores);
            if ~metrics.ROC.Valid
                error('genderRecognition:UndefinedInnerAUC', 'Inner ROC is undefined (%s).', metrics.ROC.Status);
            end
            outcomes(fold).Accuracy = metrics.Accuracy;
            outcomes(fold).AUC = metrics.AUC;
            outcomes(fold).CombinedParameters = pipeline.Parameters.CombinedParameterCount;
            if string(method) == "lda-bayes"
                outcomes(fold).CovarianceScale = pipeline.Classifier.CovarianceScale;
                outcomes(fold).MedianAbsoluteScore = median(abs(scores));
            end
        catch cause
            if ~startsWith(cause.identifier, 'genderRecognition:')
                rethrow(cause);
            end
            outcomes(fold).FailureID = string(cause.identifier);
            outcomes(fold).FailureReason = string(cause.message);
        end
    end
    rows(idx).Folds = outcomes;
    rows(idx).Seconds = toc(candidateTimer);
    rows(idx).Valid = all([outcomes.FailureID] == "");
    if rows(idx).Valid
        rows(idx).MeanAUC = mean([outcomes.AUC]);
        rows(idx).MeanAccuracy = mean([outcomes.Accuracy]);
        rows(idx).MeanCombinedParameters = mean([outcomes.CombinedParameters]);
    end
end

eligible = find([rows.Valid]);
selected = [];
configuration = [];
reason = "No candidate completed every inner fold.";
if ~isempty(eligible)
    auc = [rows(eligible).MeanAUC];
    eligible = eligible(max(auc) - auc <= tolerance);
    accuracy = [rows(eligible).MeanAccuracy];
    eligible = eligible(max(accuracy) - accuracy <= tolerance);
    parameters = [rows(eligible).MeanCombinedParameters];
    eligible = eligible(parameters == min(parameters));
    selected = eligible(1);
    configuration = candidates{selected};
    reason = string(sprintf('Candidate %d selected. %s', selected, rule));
end
selection = struct('Method', string(method), 'Candidates', rows, 'SelectedIndex', selected, ...
    'SelectedConfiguration', configuration, 'SelectionRule', rule, 'SelectionReason', reason, ...
    'Seconds', toc(started));
end
