function plotIndependentTestFigures(records, comparison, directory, synthetic)
%PLOTINDEPENDENTTESTFIGURES Plot four saved method records and optional CV summaries.
methods = ["pca-bayes", "lda-bayes", "pca-mlp", "lda-mlp"];
assert(isequal([records.Method], methods), ...
    'genderRecognition:FinalArtifactInvalid', 'Expected all four method records in order.');
names = ["PCA + Bayes", "LDA + Bayes", "PCA + MLP", "LDA + MLP"];
tag = "Independent test";
if synthetic
    tag = "Synthetic test";
end
[ok, message] = mkdir(directory);
assert(ok, 'genderRecognition:FinalExportFailed', '%s', message);
fig = newFigure([680 530]);
cleanup = onCleanup(@() close(fig));
ax = axes(fig);
plotROCCurves(records, ax);
grid(ax, 'on');
title(ax, tag + " ROC comparison");
if overlappingROC(records)
    subtitle(ax, "Some or all curves coincide at the same coordinates");
end
saveCVFigure(fig, directory, "test-roc");
clear cleanup

fig = newFigure([900 440]);
cleanup = onCleanup(@() close(fig));
layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact');
plotMetric(nexttile(layout), names, [records.Accuracy].', "Accuracy", [0.15 0.40 0.62]);
plotMetric(nexttile(layout), names, [records.AUC].', "AUC", [0.13 0.52 0.45]);
title(layout, tag + " accuracy and AUC");
saveCVFigure(fig, directory, "accuracy-auc");
clear cleanup

fig = newFigure([1260 485]);
cleanup = onCleanup(@() close(fig));
layout = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact');
timings = [records.Timing];
training = [timings.Training];
inference = [timings.Inference];
plotCosts(nexttile(layout), names, timingMedians(training), "Fit (ms)", ["Classifier", "Full pipeline"]);
plotCosts(nexttile(layout), names, timingMedians(inference), "Inference (ms)", ["Classifier", "Full pipeline"]);
plotCosts(nexttile(layout), names, [[records.NumClassifierParameters].' [records.CombinedParameterCount].'], ...
    "Fitted parameters", ["Classifier", "Combined"]);
title(layout, sprintf('%s computational cost (%d-image inference batch, median latency)', ...
    tag, records(1).NumEvaluationSamples));
saveCVFigure(fig, directory, "computational-cost");
clear cleanup

if isempty(comparison.Baseline.MeanAccuracy) || isempty(comparison.Nested.MeanAccuracy)
    return
end
fig = newFigure([1100 520]);
cleanup = onCleanup(@() close(fig));
layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact');
colors = [0.15 0.40 0.62; 0.74 0.36 0.16; 0.13 0.52 0.45];
for metric = ["Accuracy", "AUC"]
    ax = nexttile(layout);
    values = [comparison.Baseline.("Mean" + metric)(:), comparison.Nested.("Mean" + metric)(:), ...
        [records.(metric)].'];
    plotMetric(ax, names, values, metric, colors);
end
legend(ax, ["Fixed baseline CV mean", "Nested outer CV mean", tag], 'Location', 'southoutside');
title(layout, "Cross-validation and " + lower(tag));
subtitle(layout, "Final PCA + MLP uses 20 components. Nested outer PCA + MLP uses 10.");
saveCVFigure(fig, directory, "cross-validation-vs-independent");
end

function fig = newFigure(dimensions)
fig = figure('Visible', 'off', 'Color', 'white', 'Position', [100 100 dimensions]);
end

function found = overlappingROC(records)
found = false;
curves = arrayfun(@(record) curveCorners(record.ROC), records, 'UniformOutput', false);
for k = 1:numel(curves) - 1
    for j = k + 1:numel(records)
        found = found || isequal(curves{k}, curves{j});
    end
end
end

function corners = curveCorners(roc)
coordinates = [roc.FPR(:), roc.TPR(:)];
steps = diff(coordinates);
turn = steps(1:end-1, 1) .* steps(2:end, 2) - steps(1:end-1, 2) .* steps(2:end, 1);
corners = coordinates([true; abs(turn) > 8 * eps; true], :);
end

function values = timingMedians(scope)
classifier = [scope.Classifier];
pipeline = [scope.Pipeline];
values = 1e3 * [[classifier.MedianSeconds].' [pipeline.MedianSeconds].'];
end

function plotMetric(ax, names, values, metric, colors)
drawBars(ax, names, values, colors);
ylim(ax, [0 1.05]);
yticks(ax, 0:0.2:1);
ylabel(ax, metric);
if iscolumn(values)
    text(ax, 1:numel(names), values + 0.017, compose("%.3f", values), ...
        'HorizontalAlignment', 'center');
end
end

function plotCosts(ax, names, values, metric, scopes)
drawBars(ax, names, values, [0.15 0.40 0.62; 0.13 0.52 0.45]);
if all(values(:) > 0)
    ax.YScale = 'log';
end
ylabel(ax, metric);
legend(ax, scopes, 'Location', 'southoutside');
end

function drawBars(ax, names, values, colors)
bars = bar(ax, values, 'EdgeColor', 'none');
for k = 1:numel(bars)
    bars(k).FaceColor = colors(k, :);
end
set(ax, 'XTick', 1:numel(names), 'XTickLabel', names, 'XTickLabelRotation', 18, 'FontSize', 11);
grid(ax, 'on');
end
