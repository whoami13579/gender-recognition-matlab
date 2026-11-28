function run = run_final_training(trainingPath, options)
%RUN_FINAL_TRAINING Freeze Issue #11 settings and fit four full-data models.
%   Relative paths resolve from the repository root. Synthetic explicitly
%   allows small fixtures. OutputDirectory should be local and Git-ignored.
arguments
    trainingPath (1,1) string = "train"
    options.OutputDirectory (1,1) string = "results/final-evaluation"
    options.NestedRun (1,1) string = ""
    options.BaselineRun (1,1) string = ""
    options.Synthetic (1,1) logical = false
    options.Benchmark (1,1) struct = struct('WarmupRuns', 2, 'NumRepeats', 5)
end
assert(all(ismember(fieldnames(options.Benchmark), {'WarmupRuns', 'NumRepeats'})), ...
    'genderRecognition:InvalidBenchmarkConfig', 'Specify only warm-up and measured repeat counts.');
rejectTestPath(trainingPath);
trainingPath = finalInternal.resolvePath(trainingPath);
rejectTestPath(trainingPath);
[images, index, integrity] = finalInternal.loadImages(trainingPath);
finalInternal.checkDatasetProfile(integrity, "training", options.Synthetic);
[configs, provenance, outerAssignment, comparison] = ...
    finalInternal.resolveRecommendations(options, integrity.DecodedSHA256);
integrity = finalInternal.trainingDiagnostics(integrity, images, index, outerAssignment);
if integrity.MaterialDefect
    warning('genderRecognition:TrainingIntegrityDefect', ...
        'Training integrity findings compromise the Issue #11 evidence. Official independent testing is blocked.');
end
root = finalInternal.root();
[status, revision] = system(sprintf('git -C "%s" rev-parse HEAD', root));
assert(status == 0, 'genderRecognition:FinalRevisionUnavailable', 'Could not determine source revision.');
[status, tree] = system(sprintf('git -C "%s" status --porcelain --untracked-files=no', root));
[finalDirectory, staging] = finalInternal.newRun(finalInternal.resolvePath(options.OutputDirectory), "run-");
mkdir(fullfile(staging, 'models'));
effective = arrayfun(@finalInternal.effectiveConfig, configs, 'UniformOutput', false);
frozen = struct('SchemaVersion', 1, 'FrozenUTC', finalInternal.utcNow(), ...
    'TrainingDecodedSHA256', integrity.DecodedSHA256, 'TrainingSamples', numel(images), ...
    'RepositoryRevision', string(strtrim(revision)), 'MATLABVersion', string(version), ...
    'SelectionProvenance', provenance, 'Synthetic', options.Synthetic, ...
    'Threshold', 0, 'DecisionRule', "score > 0", 'PositiveClass', 1, ...
    'ConfigurationSHA256', finalInternal.checksum(configs, false), 'Configurations', configs, ...
    'EffectiveConfigurations', {effective});
frozenPath = fullfile(staging, "frozen-configurations.json");
finalInternal.writeJSON(frozenPath, frozen);
finalInternal.writeJSON(fullfile(staging, "dataset-integrity.json"), integrity);
finalInternal.writeJSON(fullfile(staging, "cv-comparison.json"), comparison);

initialRNG = rng;
verificationIndices = unique(round(linspace(1, numel(images), min(numel(images), 16))));
for k = numel(configs):-1:1
    models(k) = struct('Method', string(configs(k).Method), 'File', "", 'SHA256', "", 'Bytes', 0);
end
for k = 1:numel(configs)
    method = string(configs(k).Method);
    [pipeline, timings] = benchmarkPipelineFit(images, index.Labels, method, ...
        configs(k).Configuration, options.Benchmark);
    assert(pipeline.NumTrainingSamples == numel(images) && ...
        strcmp(jsonencode(pipeline.Configuration), jsonencode(effective{k})), ...
        'genderRecognition:FrozenConfigurationMismatch', ...
        'The fitted %s pipeline disagrees with the frozen settings or sample count.', method);
    file = lower(method) + ".mat";
    modelPath = fullfile(staging, "models", file);
    verification(k) = finalInternal.verifyPersistence(pipeline, images(verificationIndices), modelPath); %#ok<AGROW>
    fileInfo = dir(modelPath);
    models(k) = struct('Method', method, 'File', file, ...
        'SHA256', finalInternal.checksum(modelPath, true), 'Bytes', fileInfo.bytes);
    fitResults(k) = struct('Method', method, 'NumTrainingSamples', pipeline.NumTrainingSamples, ...
        'Parameters', pipeline.Parameters, 'Timing', timings); %#ok<AGROW>
end
assert(isequal(initialRNG, rng), 'genderRecognition:FinalRNGChanged', 'Final training changed the global RNG.');
verificationPath = fullfile(staging, "persistence-verification.json");
finalInternal.writeJSON(verificationPath, verification);
indexPath = fullfile(staging, "training-content-index.mat");
save(indexPath, 'index', '-v7');
rows = [fitResults.Parameters].';
summary = struct2table(rows);
summary = addvars(summary, [fitResults.Method].', [fitResults.NumTrainingSamples].', ...
    'Before', 1, 'NewVariableNames', {'Method', 'NumTrainingSamples'});
summary.ClassifierFitSeconds = arrayfun(@(x) x.Timing.ClassifierFit.MedianSeconds, fitResults).';
summary.PipelineFitSeconds = arrayfun(@(x) x.Timing.PipelineFit.MedianSeconds, fitResults).';
summary.ModelBytes = [models.Bytes].';
writetable(summary, fullfile(staging, "training-summary.csv"));
metadata = struct('SchemaVersion', 1, 'Status', "complete", 'PersistenceStatus', "verified", ...
    'InitialIndependentTestStatus', "pending", 'CompletedUTC', finalInternal.utcNow(), ...
    'TrainingPath', trainingPath, 'Synthetic', options.Synthetic, 'MaterialIntegrityDefect', integrity.MaterialDefect, ...
    'TrackedTreeClean', status == 0 && isempty(strtrim(tree)), 'ThreadLimit', maxNumCompThreads, ...
    'OperatingSystem', string(system_dependent('getos')), 'Benchmark', options.Benchmark, ...
    'TimingUnits', "seconds", 'Results', fitResults);
metadataPath = fullfile(staging, "training-metadata.json");
finalInternal.writeJSON(metadataPath, metadata);
manifest = struct('SchemaVersion', 1, 'ModelFormatVersion', 1, 'Models', models, ...
    'TrainingDecodedSHA256', frozen.TrainingDecodedSHA256, 'RepositoryRevision', frozen.RepositoryRevision, ...
    'MATLABVersion', frozen.MATLABVersion, 'SelectionProvenance', provenance, ...
    'FrozenFileSHA256', finalInternal.checksum(frozenPath, true), ...
    'VerificationFileSHA256', finalInternal.checksum(verificationPath, true), ...
    'TrainingMetadataSHA256', finalInternal.checksum(metadataPath, true), ...
    'CVComparisonSHA256', finalInternal.checksum(fullfile(staging, "cv-comparison.json"), true), ...
    'TrainingIndexSHA256', finalInternal.checksum(indexPath, true));
finalInternal.writeJSON(fullfile(staging, "model-manifest.json"), manifest);
finalInternal.moveArtifact(staging, finalDirectory);
run = struct('OutputDirectory', finalDirectory, 'Status', "complete", ...
    'Summary', summary, 'Integrity', integrity, 'Verification', verification);
fprintf('Final training complete. Four saved models verified. Local run: %s\n', finalDirectory);
end

function rejectTestPath(path)
assert(~any(strcmpi(split(replace(path, "\", "/"), "/"), "test")), ...
    'genderRecognition:IndependentTestPath', 'The final training entry point accepts training folders only.');
end
