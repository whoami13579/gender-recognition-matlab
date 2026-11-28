function ax = plotROCCurves(results, ax)
%PLOTROCCURVES Plot precomputed ROC coordinates and a random-ranking reference.
if ~isstruct(results) || ~isvector(results) || isempty(results)
    error('genderRecognition:InvalidEvaluationResult', 'Expected a nonempty vector of evaluation records.');
end
for idx = 1:numel(results)
    validateEvaluationResult(results(idx));
end
if nargin < 2
    ax = axes('Parent', figure('Name', 'ROC curves'));
elseif ~isscalar(ax) || ~isgraphics(ax, 'axes')
    error('genderRecognition:InvalidROCAxes', 'Expected one axes handle.');
end
previousNextPlot = ax.NextPlot;
restoreNextPlot = onCleanup(@() set(ax, 'NextPlot', previousNextPlot));
hold(ax, 'on');
plot(ax, [0 1], [0 1], 'k--', 'DisplayName', 'Random ranking');
for idx = 1:numel(results)
    result = results(idx);
    if ~result.ROC.Valid
        warning('genderRecognition:UndefinedROC', 'Skipping %s ROC: %s.', result.Method, result.ROC.Status);
        continue;
    end
    name = upper(replace(result.Method, "-", " + "));
    if result.EvaluationStage == "cross-validation"
        name = name + " fold " + result.FoldNumber;
    end
    plot(ax, result.ROC.FPR, result.ROC.TPR, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('%s (AUC %.3f)', name, result.AUC));
end
xlim(ax, [0 1]);
ylim(ax, [0 1]);
xlabel(ax, 'False-positive rate');
ylabel(ax, 'True-positive rate');
legend(ax, 'Location', 'southeast');
end
