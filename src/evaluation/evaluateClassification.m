function metrics = evaluateClassification(yTrue, yPred, scores)
%EVALUATECLASSIFICATION Measure supplied predictions and continuous scores.
%   Confusion matrix rows are actual [0, 1], columns are predicted [0, 1].
yPred = validateEvaluationVector(yPred, 'yPred', true);
if numel(yTrue) ~= numel(yPred)
    error('genderRecognition:InvalidEvaluationInput', 'yTrue and yPred must have equal lengths.');
end
[roc, auc] = computeROC(yTrue, scores);
numSamples = numel(yTrue);
numCorrect = nnz(yTrue == yPred);
tn = nnz(yTrue == 0 & yPred == 0);
tp = nnz(yTrue == 1 & yPred == 1);
metrics = struct('NumEvaluationSamples', numSamples, 'Accuracy', numCorrect / numSamples, ...
    'NumCorrect', numCorrect, 'ConfusionMatrix', ...
    [tn, roc.NumNegativeSamples - tn; roc.NumPositiveSamples - tp, tp], 'ROC', roc, 'AUC', auc);
end
