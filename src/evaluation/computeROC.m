function [roc, auc] = computeROC(yTrue, scores)
%COMPUTEROC Sweep distinct scores for positive class 1 using score >= threshold.
%   ROC starts at threshold Inf. Both classes are required for ROC and AUC.
yTrue = validateEvaluationVector(yTrue, 'yTrue', true);
scores = validateEvaluationVector(scores, 'scores', false);
if numel(yTrue) ~= numel(scores)
    error('genderRecognition:InvalidEvaluationInput', 'yTrue and scores must have equal lengths.');
end

[sortedScores, order] = sort(scores, 'descend');
lastInGroup = [sortedScores(1:end-1) ~= sortedScores(2:end); true];
numPositive = nnz(yTrue);
numNegative = numel(yTrue) - numPositive;
roc = struct('FPR', zeros(0, 1), 'TPR', zeros(0, 1), 'Thresholds', zeros(0, 1), ...
    'NumPositiveSamples', numPositive, 'NumNegativeSamples', numNegative, ...
    'NumDistinctScores', nnz(lastInGroup), 'Valid', false, 'Status', "MissingPositiveClass");
auc = NaN;
if numPositive == 0
    return;
end
if numNegative == 0
    roc.Status = "MissingNegativeClass";
    return;
end

groupEnds = find(lastInGroup);
truePositives = cumsum(yTrue(order));
truePositives = truePositives(groupEnds);
roc.FPR = [0; (groupEnds - truePositives) / numNegative];
roc.TPR = [0; truePositives / numPositive];
roc.Thresholds = [Inf; sortedScores(groupEnds)];
roc.Valid = true;
roc.Status = "OK";
auc = trapz(roc.FPR, roc.TPR);
end
