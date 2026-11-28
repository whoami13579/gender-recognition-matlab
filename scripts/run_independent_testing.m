function run = run_independent_testing(trainingRunDirectory, testingPath, options)
%RUN_INDEPENDENT_TESTING Evaluate only models reloaded from a verified bundle.
%   Each attempt has a new local directory, including unavailable/blocked attempts.
arguments
    trainingRunDirectory (1,1) string
    testingPath (1,1) string = "test"
    options.GeneratePlots (1,1) logical = true
    options.OutputDirectory (1,1) string = ""
end
trainingRunDirectory = finalInternal.resolvePath(trainingRunDirectory);
% This is the only gate to the test directory, including existence checks.
[pipelines, ~, frozen, training, trainingIndex, comparison] = finalInternal.readBundle(trainingRunDirectory);
verifiedUTC = finalInternal.utcNow();
destination = fullfile(trainingRunDirectory, "independent-test");
if strlength(options.OutputDirectory) > 0
    destination = finalInternal.resolvePath(options.OutputDirectory);
end
[directory, staging] = finalInternal.newRun(destination, "attempt-");
metadata = struct('SchemaVersion', 1, 'Status', "incomplete", 'Synthetic', frozen.Synthetic, ...
    'OfficialEvaluation', false, 'TrainingRevision', frozen.RepositoryRevision, ...
    'FrozenUTC', frozen.FrozenUTC, 'TrainingCompletedUTC', training.CompletedUTC, ...
    'BundleVerifiedUTC', verifiedUTC, 'TestingPath', testingPath, 'DatasetLoadedUTC', "", ...
    'TimingUnits', "seconds", 'MATLABVersion', string(version), 'ThreadLimit', maxNumCompThreads, ...
    'TrainingConfigurationSHA256', frozen.ConfigurationSHA256, 'Benchmark', training.Benchmark, ...
    'ModelManifestSHA256', finalInternal.checksum(fullfile(trainingRunDirectory, "model-manifest.json"), true), ...
    'CompletedMethods', 0, 'BatchSize', [], 'Reason', "", 'ErrorID', "");
run = struct('OutputDirectory', staging, 'Status', "incomplete", 'Synthetic', frozen.Synthetic, 'Records', struct([]));
try
    if training.MaterialIntegrityDefect
        metadata.Status = "blocked";
        metadata.Reason = "Material training integrity defect affecting the Issue #11 evidence.";
    else
        metadata.TestingPath = finalInternal.resolvePath(testingPath);
        trainPath = finalInternal.resolvePath(string(training.TrainingPath));
        sameDirectory = strcmp(trainPath, metadata.TestingPath);
        if ispc
            sameDirectory = strcmpi(trainPath, metadata.TestingPath);
        end
        if sameDirectory
            metadata.Status = "blocked";
            metadata.Reason = "The canonical training and testing directories are identical.";
        elseif ~isfolder(metadata.TestingPath) && ~isfile(metadata.TestingPath)
            metadata.Status = "unavailable";
            metadata.Reason = "Configured testing directory is absent. Official measurements are pending.";
        else
            metadata.DatasetLoadStartedUTC = finalInternal.utcNow();
            [images, testIndex, integrity] = finalInternal.loadImages(metadata.TestingPath);
            metadata.DatasetLoadedUTC = finalInternal.utcNow();
            metadata.BatchSize = numel(images);
            integrity = finalInternal.testOverlap(integrity, trainingIndex, testIndex);
            finalInternal.writeJSON(fullfile(staging, "dataset-integrity.json"), integrity);
            if integrity.TrainingTestDuplicatePairs > 0 || integrity.ConflictingLabelPairs > 0
                metadata.Status = "blocked";
                metadata.Reason = "Exact decoded-pixel overlap or conflicting labels invalidate this test set.";
            else
                finalInternal.checkDatasetProfile(integrity, "testing", frozen.Synthetic);
                initialRNG = rng;
                for k = 1:numel(pipelines)
                    pipeline = pipelines(k);
                    [yPred, scores, inference] = benchmarkPipelineInference(pipeline, images, training.Benchmark);
                    metrics = evaluateClassification(testIndex.Labels, yPred, scores);
                    timings = training.Results(k).Timing;
                    timings.ClassifierInference = inference.ClassifierInference;
                    timings.PipelineInference = inference.PipelineInference;
                    stage = "independent-test";
                    if frozen.Synthetic
                        stage = "synthetic";
                    end
                    context = struct('Method', pipeline.Method, 'EvaluationStage', stage);
                    record = buildEvaluationResult(context, metrics, pipeline.Preprocessor, ...
                        pipeline.Projection, pipeline.Classifier, timings);
                    records(k) = record; %#ok<AGROW>
                    predictions(k) = struct('Method', pipeline.Method, 'Labels', testIndex.Labels, ...
                        'Predictions', yPred, 'Scores', scores); %#ok<AGROW>
                    metadata.CompletedMethods = k;
                end
                assert(isequal(initialRNG, rng), 'genderRecognition:FinalRNGChanged', ...
                    'Independent testing changed the global RNG.');
                exportEvaluationResults(records, staging);
                save(fullfile(staging, "predictions.mat"), 'predictions', '-v7');
                finalInternal.writeJSON(fullfile(staging, "cv-comparison.json"), comparison);
                if options.GeneratePlots
                    plotIndependentTestFigures(records, comparison, fullfile(staging, "figures"), frozen.Synthetic);
                end
                metadata.Status = "complete";
                metadata.OfficialEvaluation = ~frozen.Synthetic;
                run.Records = records;
            end
        end
    end
catch cause
    metadata.ErrorID = string(cause.identifier);
    metadata.Reason = string(cause.message);
end
metadata.CompletedUTC = finalInternal.utcNow();
finalInternal.writeJSON(fullfile(staging, "test-metadata.json"), metadata);
if metadata.Status ~= "incomplete"
    finalInternal.moveArtifact(staging, directory);
    run.OutputDirectory = directory;
end
run.Status = metadata.Status;
run.Reason = metadata.Reason;
run.ErrorID = metadata.ErrorID;
fprintf('Independent testing: %s. Local attempt: %s\n', run.Status, run.OutputDirectory);
end
