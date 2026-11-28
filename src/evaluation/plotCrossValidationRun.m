function plotCrossValidationRun(results, directory, foldROC, context)
%PLOTCROSSVALIDATIONRUN Export comparisons from in-memory or reloaded records.
arguments
    results struct
    directory
    foldROC (1,1) logical = false
    context (1,1) string = ""
end
makeDirectory(directory);
titlePrefix = upperFirst(string(results.Config.Mode));
if string(results.Status) ~= "complete"
    titlePrefix = titlePrefix + " (incomplete)";
end
if strlength(context) > 0
    titlePrefix = context;
end
names = replace(upper(replace(results.Methods, "-", " + ")), "BAYES", "Bayes");
colors = [0.00 0.35 0.64; 0.78 0.31 0.07; 0.05 0.48 0.37; 0.50 0.27 0.64];
styles = ["-", "--", "-.", ":"];

fig = figure('Visible', 'off', 'Color', 'white', 'Position', [100 100 720 580]);
cleanup = onCleanup(@() close(fig));
ax = axes('Parent', fig, 'FontSize', 11);
hold(ax, 'on');
plot(ax, [0 1], [0 1], '--', 'Color', [0.65 0.65 0.65], 'DisplayName', 'Random ranking');
for k = 1:numel(names)
    curve = results.MeanROC(k);
    if curve.NumValidCurves > 0
        label = sprintf('%s (mean AUC %.3f)', names(k), results.Summary.MeanAUC(k));
        if isnan(results.Summary.MeanAUC(k))
            label = sprintf('%s (%d valid folds)', names(k), curve.NumValidCurves);
        end
        plot(ax, curve.FPR, curve.TPR, 'Color', colors(k, :), 'LineStyle', styles(k), ...
            'LineWidth', 2, 'DisplayName', label);
    end
end
axis(ax, 'square');
xlim(ax, [0 1]);
ylim(ax, [0 1]);
xlabel(ax, 'False-positive rate');
ylabel(ax, 'True-positive rate');
title(ax, titlePrefix + " mean ROC");
if all([results.MeanROC.NumValidCurves] > 0) && all(arrayfun(@(curve) ...
        isequal(curve.FPR, results.MeanROC(1).FPR) && isequal(curve.TPR, results.MeanROC(1).TPR), results.MeanROC))
    subtitle(ax, 'All four mean curves coincide');
end
grid(ax, 'on');
legend(ax, 'Location', 'southeast');
saveCVFigure(fig, directory, 'mean-roc');
clear cleanup

if foldROC && ~isempty(results.Records)
    fig = figure('Visible', 'off', 'Color', 'white', 'Position', [100 100 1000 820]);
    cleanup = onCleanup(@() close(fig));
    layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact');
    for k = 1:numel(names)
        ax = nexttile(layout);
        rows = results.Records([results.Records.Method] == results.Methods(k));
        if ~isempty(rows)
            [~, order] = sort([rows.FoldNumber]);
            plotROCCurves(rows(order), ax);
            grid(ax, 'on');
        end
        title(ax, names(k));
    end
    title(layout, titlePrefix + " fold ROC curves");
    saveCVFigure(fig, directory, 'fold-roc');
    clear cleanup
end

fig = figure('Visible', 'off', 'Color', 'white', 'Position', [100 100 1000 410]);
cleanup = onCleanup(@() close(fig));
layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact');
summary = results.Summary;
metricPanel(nexttile(layout), names, summary.MeanAccuracy, summary.StdAccuracy, 'Accuracy');
metricPanel(nexttile(layout), names, summary.MeanAUC, summary.StdAUC, 'AUC');
title(layout, titlePrefix + " accuracy and AUC (mean ± fold SD)");
saveCVFigure(fig, directory, 'accuracy-auc');
clear cleanup

fig = figure('Visible', 'off', 'Color', 'white', 'Position', [100 100 1280 720]);
cleanup = onCleanup(@() close(fig));
layout = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact');
timings = ["PipelineFit", "ClassifierFit", "PipelineInference", "ClassifierInference"];
headings = ["Pipeline fit", "Classifier fit", "Pipeline batch inference", "Classifier batch inference"];
tiles = [1 2 4 5];
for k = 1:numel(timings)
    ax = nexttile(layout, tiles(k));
    metricPanel(ax, names, summary.("MeanMedian" + timings(k) + "Seconds"), ...
        summary.("StdMedian" + timings(k) + "Seconds"), 'Seconds');
    title(ax, headings(k));
end
countPanel(nexttile(layout, 3), names, summary.FoldClassifierParameters, 'Classifier parameters');
countPanel(nexttile(layout, 6), names, summary.FoldCombinedParameters, 'Combined parameters');
title(layout, titlePrefix + " computational cost");
subtitle(layout, 'Timings show the mean and SD of fold medians. Parameter markers show the fold range.');
saveCVFigure(fig, directory, 'computational-cost');
end

function metricPanel(ax, names, values, deviations, label)
positions = 1:numel(names);
hold(ax, 'on');
bar(ax, positions, values, 'FaceColor', [0.15 0.40 0.62], 'EdgeColor', 'none');
% Latency error ranges start at zero when a large SD crosses the origin.
errorbar(ax, positions, values, min(deviations, values), deviations, ...
    'k.', 'LineWidth', 1.1, 'CapSize', 8);
formatPanel(ax, names, label);
if any(string(label) == ["Accuracy", "AUC"])
    ylim(ax, [0 max([1; values + deviations], [], 'omitmissing')]);
end
end

function countPanel(ax, names, counts, heading)
hold(ax, 'on');
for k = 1:numel(names)
    values = counts(k, isfinite(counts(k, :)));
    if ~isempty(values)
        plot(ax, [k k], [min(values) max(values)], '-', 'Color', [0.15 0.40 0.62], 'LineWidth', 2);
        plot(ax, k, median(values), 'o', 'MarkerFaceColor', [0.15 0.40 0.62], ...
            'MarkerEdgeColor', 'none', 'MarkerSize', 7);
    end
end
set(ax, 'YScale', 'log');
present = counts(isfinite(counts));
if ~isempty(present)
    ylim(ax, [min(present) / 1.4, max(present) * 1.4]);
end
formatPanel(ax, names, 'Count (log scale)');
title(ax, heading);
end

function formatPanel(ax, names, yLabel)
set(ax, 'XTick', 1:numel(names), 'XTickLabel', names, 'XTickLabelRotation', 20, 'FontSize', 10);
xlim(ax, [0.5 numel(names) + 0.5]);
ylabel(ax, yLabel);
grid(ax, 'on');
end

function value = upperFirst(value)
value = upper(extractBetween(value, 1, 1)) + extractAfter(value, 1);
end

function makeDirectory(directory)
[ok, message] = mkdir(directory);
if ~ok
    error('genderRecognition:CVExportFailed', '%s', message);
end
end
