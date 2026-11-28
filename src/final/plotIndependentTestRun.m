function plotIndependentTestRun(directory, figureDirectory)
%PLOTINDEPENDENTTESTRUN Rebuild figures from a completed saved attempt only.
arguments
    directory (1,1) string
    figureDirectory (1,1) string
end
directory = finalInternal.resolvePath(directory);
[records, metadata] = loadIndependentTestRun(directory);
comparison = jsondecode(fileread(fullfile(directory, 'cv-comparison.json')));
plotIndependentTestFigures(records, comparison, finalInternal.resolvePath(figureDirectory), metadata.Synthetic);
end
