function plotIndependentTestRun(directory, figureDirectory)
%PLOTINDEPENDENTTESTRUN Rebuild figures from a completed saved attempt only.
arguments
    directory (1,1) string
    figureDirectory (1,1) string
end
directory = finalInternal.resolvePath(directory);
assert(~endsWith(directory, ".incomplete"), 'genderRecognition:IncompleteIndependentTest', ...
    'The independent test export is incomplete.');
metadata = jsondecode(fileread(fullfile(directory, 'test-metadata.json')));
assert(isequal(metadata.SchemaVersion, 1) && string(metadata.Status) == "complete", ...
    'genderRecognition:IncompleteIndependentTest', 'Expected a completed test attempt.');
records = loadEvaluationResults(fullfile(directory, 'evaluation-results.json'));
stage = "independent-test";
if metadata.Synthetic
    stage = "synthetic";
end
assert(numel(records) == 4 && all([records.EvaluationStage] == stage) && ...
    all([records.NumEvaluationSamples] == metadata.BatchSize) && ...
    metadata.CompletedMethods == 4 && metadata.OfficialEvaluation == ~metadata.Synthetic, ...
    'genderRecognition:FinalArtifactInvalid', 'The independent test records disagree with their metadata.');
comparison = jsondecode(fileread(fullfile(directory, 'cv-comparison.json')));
plotIndependentTestFigures(records, comparison, finalInternal.resolvePath(figureDirectory), metadata.Synthetic);
end
