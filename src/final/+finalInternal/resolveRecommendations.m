function [rows, provenance, outerAssignment, comparison] = resolveRecommendations(options, fingerprint)
rows = finalInternal.documentedConfigurations();
rule = "Highest mean inner AUC, then mean inner accuracy within 1e-12 of each maximum; " + ...
    "then fewest mean combined parameters; then earliest candidate.";
provenance = struct('Source', "documented-issue-11-handoff", 'Issue', 11, 'PullRequest', 22, ...
    'SelectionFolds', 4, 'SelectionSeed', 42, 'SelectionRule', rule, ...
    'ExperimentRevision', "", 'RecommendationFileSHA256', "");
outerAssignment = [];
empty = struct('SourceRevision', "", 'NumFolds', [], 'MeanAccuracy', [], 'StdAccuracy', [], ...
    'MeanAUC', [], 'StdAUC', []);
comparison = struct('Baseline', empty, 'Nested', empty);
if options.Synthetic
    provenance.Source = "synthetic-fixture-with-documented-configurations";
    return
end
nestedPath = findRun(options.NestedRun, "nested", fingerprint, [], "");
if strlength(nestedPath) > 0
    nested = checkedRun(nestedPath, "nested", fingerprint);
    search = nested.Config.Search;
    assert(search.RecommendFinal && search.NumFolds == 4 && search.RandomSeed == 42 && ...
        string(nested.Metadata.FinalSelectionStatus) == "complete", ...
        'genderRecognition:InvalidRecommendation', 'Expected completed four-fold training-only final selection.');
    recommendationPath = fullfile(nestedPath, "configuration-recommendations.json");
    if isfile(recommendationPath)
        candidateRows = jsondecode(fileread(recommendationPath));
        evidence = jsondecode(fileread(fullfile(nestedPath, "hyperparameter-search.json")));
        verifySearch(evidence.FinalSearch, candidateRows, rows, rule);
        rows = candidateRows;
        for k = 1:numel(rows)
            rows(k).Configuration.Preprocessing.ImageSize = reshape( ...
                rows(k).Configuration.Preprocessing.ImageSize, 1, []);
        end
        provenance.Source = "verified-issue-11-full-training-selection";
        provenance.ExperimentRevision = string(nested.Metadata.RepositoryCommit);
        provenance.RecommendationFileSHA256 = finalInternal.checksum(recommendationPath, true);
    end
    comparison.Nested = aggregateCV(nested);
    outerAssignment = nested.Partition.SampleToFold;
end
baselinePath = findRun(options.BaselineRun, "baseline", fingerprint, outerAssignment, ...
    comparison.Nested.SourceRevision);
if strlength(baselinePath) > 0
    baseline = checkedRun(baselinePath, "baseline", fingerprint);
    assert(isempty(outerAssignment) || isequal(outerAssignment(:), baseline.Partition.SampleToFold(:)), ...
        'genderRecognition:IncompatibleCVRuns', 'Baseline and nested outer partitions differ.');
    outerAssignment = baseline.Partition.SampleToFold;
    comparison.Baseline = aggregateCV(baseline);
end
end

function directory = findRun(explicit, mode, fingerprint, assignment, revision)
directory = string(explicit);
if strlength(directory) > 0
    directory = finalInternal.resolvePath(directory);
    return
end
parent = fullfile(finalInternal.root(), "results", "cross-validation", mode);
entries = dir(fullfile(parent, 'run-*'));
for entry = reshape(entries, 1, [])
    candidate = string(fullfile(entry.folder, entry.name));
    metadataPath = fullfile(candidate, "experiment-metadata.json");
    if ~entry.isdir || endsWith(entry.name, '.incomplete') || ~isfile(metadataPath)
        continue
    end
    metadata = jsondecode(fileread(metadataPath));
    if string(metadata.Mode) ~= mode || string(metadata.Status) ~= "complete" || ...
            string(metadata.Dataset.DecodedSHA256) ~= fingerprint || ...
            (mode == "nested" && string(metadata.FinalSelectionStatus) ~= "complete") || ...
            (strlength(revision) > 0 && string(metadata.RepositoryCommit) ~= revision)
        continue
    end
    if ~isempty(assignment)
        folds = jsondecode(fileread(fullfile(candidate, "folds.json")));
        if ~isequal(folds.Partition.SampleToFold(:), assignment(:))
            continue
        end
    end
    assert(strlength(directory) == 0, 'genderRecognition:AmbiguousCVRun', ...
        'Multiple completed %s runs match. Specify the intended run explicitly.', mode);
    directory = candidate;
end
end

function run = checkedRun(directory, mode, fingerprint)
run = loadCrossValidationRun(directory);
assert(string(run.Metadata.Dataset.DecodedSHA256) == fingerprint, ...
    'genderRecognition:TrainingFingerprintMismatch', 'Cross-validation and training decoded fingerprints differ.');
assert(string(run.Config.Mode) == mode && run.Status == "complete" && ...
    isequal(run.Methods, ["pca-bayes", "lda-bayes", "pca-mlp", "lda-mlp"]) && ...
    numel(run.Folds) == 5 && numel(run.Partition.SampleToFold) == 200 && ...
    isequal(sort(unique(run.Partition.SampleToFold(:))).', 1:5) && ...
    all(run.Partition.SubjectOverlapCounts == 0), ...
    'genderRecognition:InvalidRecommendation', 'Expected a complete Issue #11 experiment.');
end

function verifySearch(search, rows, documented, rule)
sameConfig = @(a, b) isequal(jsondecode(jsonencode(a)), jsondecode(jsonencode(b)));
assert(numel(rows) == 4 && numel(search.Selections) == 4 && string(search.Status) == "complete" && ...
    search.Partition.GroupBySubject && search.Partition.Stratified && ...
    search.Partition.NumFolds == 4 && search.Partition.RandomSeed == 42 && ...
    all(search.Partition.SubjectOverlapCounts == 0) && all(search.Partition.TrainSampleCounts == 150) && ...
    all(search.Partition.ValidationSampleCounts == 50) && numel(search.Folds) == 4 && ...
    numel(search.Partition.SampleToFold) == 200, ...
    'genderRecognition:InvalidRecommendation', 'The recorded final selection does not match the documented protocol.');
for k = 1:4
    selection = search.Selections(k);
    candidates = selection.Candidates;
    eligible = find([candidates.Valid]);
    auc = [candidates(eligible).MeanAUC];
    eligible = eligible(max(auc) - auc <= 1e-12);
    accuracy = [candidates(eligible).MeanAccuracy];
    eligible = eligible(max(accuracy) - accuracy <= 1e-12);
    parameters = [candidates(eligible).MeanCombinedParameters];
    eligible = eligible(parameters == min(parameters));
    assert(~isempty(eligible) && isequal(eligible(1), selection.SelectedIndex) && ...
        strcmp(rows(k).Method, documented(k).Method) && strcmp(selection.Method, rows(k).Method) && ...
        strcmp(selection.SelectionRule, rule) && sameConfig(rows(k).Configuration, selection.SelectedConfiguration) && ...
        sameConfig(rows(k).Configuration, candidates(eligible(1)).Configuration) && ...
        sameConfig(rows(k).Configuration, documented(k).Configuration), ...
        'genderRecognition:InvalidRecommendation', 'The saved training-only recommendation disagrees with its evidence.');
end
end

function aggregate = aggregateCV(run)
summary = run.Summary;
aggregate = struct('SourceRevision', string(run.Metadata.RepositoryCommit), 'NumFolds', numel(run.Folds), ...
    'MeanAccuracy', summary.MeanAccuracy, 'StdAccuracy', summary.StdAccuracy, ...
    'MeanAUC', summary.MeanAUC, 'StdAUC', summary.StdAUC);
end
