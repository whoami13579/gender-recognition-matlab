function directory = demoFigures(options)
baseline = demoCV("baseline", options, false);
nested = demoCV("nested", options, false);
hasBaseline = strlength(baseline.OutputDirectory) > 0;
hasNested = strlength(nested.OutputDirectory) > 0;
hasAttempt = strlength(options.IndependentAttempt) > 0;
assert(hasBaseline || hasNested || hasAttempt, 'genderRecognition:DemoMissingArtifact', ...
    'No plots to generate. Set BaselineRun, NestedRun or IndependentAttempt.');
if hasAttempt
    bundle = demoModels(options, false);
    independent = demoIndependent(bundle, options, false);
    hasAttempt = independent.Status == "complete";
    assert(hasBaseline || hasNested || hasAttempt, 'genderRecognition:DemoMissingArtifact', ...
        'No completed records are available for plotting.');
end

parent = finalInternal.resolvePath(options.OutputDirectory);
[directory, staging] = finalInternal.newRun(parent, "figures-");
for run = {baseline, nested}
    current = run{1};
    if strlength(current.OutputDirectory) == 0
        continue
    end
    mode = string(current.Config.Mode);
    fprintf('%s, revision %s\nLocal run: %s\n', ...
        current.Context, current.Metadata.RepositoryCommit, current.OutputDirectory);
    plotCrossValidationRun(current, fullfile(staging, mode), options.FoldROC, current.Context);
end
if hasBaseline && hasNested
    try
        context = "Course-profile training";
        if options.Synthetic
            context = "Synthetic fixture";
        end
        plotCrossValidationComparison(baseline, nested, staging, context);
    catch cause
        if ~strcmp(cause.identifier, 'genderRecognition:IncompatibleCVRuns')
            rethrow(cause);
        end
        fprintf('Paired comparison omitted because the runs use different data or partitions.\n');
    end
end
if hasAttempt
    plotIndependentTestRun(independent.OutputDirectory, fullfile(staging, "independent"));
end
finalInternal.moveArtifact(staging, directory);
fprintf('Local figures: %s\n', directory);
if options.ShowFigures
    files = dir(fullfile(directory, '**', '*.png'));
    selected = ["mean-roc.png", "test-roc.png"];
    if options.FoldROC
        selected(end + 1) = "fold-roc.png";
    end
    for file = files(ismember(string({files.name}), selected)).'
        [~, folder] = fileparts(file.folder);
        fig = figure('Name', char(string(folder) + " " + string(file.name)), 'NumberTitle', 'off');
        imshow(imread(fullfile(file.folder, file.name)), 'Parent', axes('Parent', fig));
    end
end
end
