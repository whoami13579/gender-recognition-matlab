function [records, metadata] = loadIndependentTestRun(directory)
%LOADINDEPENDENTTESTRUN Read validated records from a completed saved attempt.
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
methods = ["pca-bayes", "lda-bayes", "pca-mlp", "lda-mlp"];
assert(isequal([records.Method], methods) && all([records.EvaluationStage] == stage) && ...
    all([records.NumEvaluationSamples] == metadata.BatchSize) && ...
    metadata.CompletedMethods == numel(methods) && metadata.OfficialEvaluation == ~metadata.Synthetic && ...
    (metadata.Synthetic || metadata.BatchSize == 100), ...
    'genderRecognition:FinalArtifactInvalid', 'The independent test records disagree with their metadata.');
end
