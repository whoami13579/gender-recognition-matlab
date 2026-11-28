function [summary, meanROC] = summarizeCrossValidation(records, methods, numFolds)
%SUMMARIZECROSSVALIDATION Summarize complete/partial CV records by method.
%   Missing or undefined folds stay visible and leave complete-fold AUC undefined.
arguments
    records struct
    methods (1,:) string = ["pca-bayes", "lda-bayes", "pca-mlp", "lda-mlp"]
    numFolds (1,1) double {mustBeInteger, mustBePositive} = 5
end
for k = 1:numel(records)
    validateEvaluationResult(records(k));
    if records(k).EvaluationStage ~= "cross-validation" || ~any(records(k).Method == methods) || ...
            records(k).FoldNumber > numFolds
        error('genderRecognition:InvalidCVRecords', 'Unexpected method, evaluation stage or fold number.');
    end
end
for m = 1:numel(methods)
    rows = records;
    if ~isempty(records)
        rows = records([records.Method] == methods(m));
    end
    folds = zeros(1, 0);
    if ~isempty(rows)
        folds = [rows.FoldNumber];
    end
    if numel(unique(folds)) ~= numel(folds)
        error('genderRecognition:InvalidCVRecords', 'Found duplicate fold records for %s.', methods(m));
    end
    accuracy = nan(1, numFolds);
    auc = accuracy;
    matrix = zeros(2);
    blankROC = struct('FPR', [], 'TPR', [], 'Valid', false);
    curves = repmat(blankROC, 1, numFolds);
    counts = nan(numFolds, 5);
    medians = nan(numFolds, 6);
    for k = 1:numel(rows)
        row = rows(k);
        fold = row.FoldNumber;
        accuracy(fold) = row.Accuracy;
        auc(fold) = row.AUC;
        matrix = matrix + row.ConfusionMatrix;
        curves(fold) = struct('FPR', row.ROC.FPR, 'TPR', row.ROC.TPR, 'Valid', row.ROC.Valid);
        t = row.Timing;
        counts(fold, :) = [row.NumFeatures, row.NumProjectionParameters, ...
            row.NumNormalizationParameters, row.NumClassifierParameters, row.CombinedParameterCount];
        medians(fold, :) = [latency(t.Training.Pipeline, 'MedianSeconds'), ...
            latency(t.Inference.Pipeline, 'MedianSeconds'), latency(t.Training.Classifier, 'MedianSeconds'), ...
            latency(t.Inference.Classifier, 'MedianSeconds'), latency(t.Inference.Pipeline, 'SecondsPerSample'), ...
            latency(t.Inference.Classifier, 'SecondsPerSample')];
    end
    meanROC(m, 1) = aggregateROCCurves(curves); %#ok<AGROW>
    total = sum(matrix, 'all');
    rowSummary = struct('Method', methods(m), 'CompletedFolds', numel(folds), ...
        'ExpectedFolds', numFolds, 'ValidAUCFolds', nnz(isfinite(auc)), ...
        'MissingFolds', string(mat2str(setdiff(1:numFolds, folds))), ...
        'UndefinedAUCFolds', string(mat2str(folds(isnan(auc(folds))))), ...
        'FoldAccuracy', accuracy, 'FoldAUC', auc, ...
        'MeanAccuracy', mean(accuracy), 'StdAccuracy', std(accuracy, 0), ...
        'MinAccuracy', min(accuracy, [], 'includemissing'), 'MaxAccuracy', max(accuracy, [], 'includemissing'), ...
        'MeanAUC', mean(auc), 'StdAUC', std(auc, 0), ...
        'MinAUC', min(auc, [], 'includemissing'), 'MaxAUC', max(auc, [], 'includemissing'), ...
        'MeanROCArea', meanROC(m).Area, 'TotalCorrect', trace(matrix), 'TotalEvaluated', total, ...
        'WeightedAccuracy', trace(matrix) / total, 'TN', matrix(1, 1), 'FP', matrix(1, 2), ...
        'FN', matrix(2, 1), 'TP', matrix(2, 2));
    names = ["Features", "ProjectionParameters", "NormalizationParameters", "ClassifierParameters", "CombinedParameters"];
    timingNames = ["PipelineFit", "PipelineInference", "ClassifierFit", "ClassifierInference", ...
        "PipelineInferencePerImage", "ClassifierInferencePerImage"];
    for k = 1:numel(names)
        data = counts(:, k).';
        rowSummary.("Fold" + names(k)) = data;
        rowSummary.(names(k) + "Range") = displayRange(data);
    end
    for k = 1:numel(timingNames)
        data = medians(:, k).';
        rowSummary.("FoldMedian" + timingNames(k) + "Seconds") = data;
        rowSummary.("MeanMedian" + timingNames(k) + "Seconds") = mean(data);
        rowSummary.("StdMedian" + timingNames(k) + "Seconds") = std(data, 0);
    end
    summaries(m, 1) = rowSummary; %#ok<AGROW>
end
summary = struct2table(summaries, 'AsArray', true);
end

function seconds = latency(measurement, field)
seconds = NaN;
if ~isempty(measurement)
    seconds = measurement.(field);
end
end

function value = displayRange(data)
present = data(isfinite(data));
value = "";
if ~isempty(present)
    value = string(min(present));
    if min(present) ~= max(present)
        value = value + "–" + string(max(present));
    end
end
end
