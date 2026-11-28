function plotCrossValidationComparison(baseline, nested, directory)
%PLOTCROSSVALIDATIONCOMPARISON Compare complete runs using identical folds.
if string(baseline.Config.Mode) ~= "baseline" || string(nested.Config.Mode) ~= "nested" || ...
        string(baseline.Status) ~= "complete" || string(nested.Status) ~= "complete" || ...
        ~isequal(baseline.Methods, nested.Methods) || ...
        ~isequal(baseline.Partition.SampleToFold(:), nested.Partition.SampleToFold(:)) || ...
        string(baseline.Metadata.Dataset.DecodedSHA256) ~= string(nested.Metadata.Dataset.DecodedSHA256)
    error('genderRecognition:IncompatibleCVRuns', 'Compare complete baseline and nested runs on the same dataset and folds.');
end
[ok, message] = mkdir(directory);
if ~ok
    error('genderRecognition:CVExportFailed', '%s', message);
end
fig = figure('Visible', 'off', 'Color', 'white', 'Position', [100 100 1100 480]);
cleanup = onCleanup(@() close(fig));
layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact');
names = replace(upper(replace(baseline.Methods, "-", " + ")), "BAYES", "Bayes");
for metric = ["Accuracy", "AUC"]
    ax = nexttile(layout);
    means = [baseline.Summary.("Mean" + metric), nested.Summary.("Mean" + metric)];
    deviations = [baseline.Summary.("Std" + metric), nested.Summary.("Std" + metric)];
    bars = bar(ax, means, 'EdgeColor', 'none');
    bars(1).FaceColor = [0.15 0.40 0.62];
    bars(2).FaceColor = [0.73 0.34 0.13];
    hold(ax, 'on');
    for k = 1:numel(bars)
        errorbar(ax, bars(k).XEndPoints, means(:, k), deviations(:, k), 'k.', ...
            'HandleVisibility', 'off');
    end
    set(ax, 'XTick', 1:numel(names), 'XTickLabel', names, 'XTickLabelRotation', 20);
    ylim(ax, [0 max([1; means(:) + deviations(:)])]);
    ylabel(ax, metric);
    grid(ax, 'on');
    legend(ax, bars, {'Fixed baseline', 'Nested search'}, 'Location', 'southoutside', ...
        'Orientation', 'horizontal');
end
title(layout, 'Baseline and nested outer results (mean ± fold SD)');
saveCVFigure(fig, directory, 'baseline-vs-nested');
end
