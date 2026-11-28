function run = demoCV(mode, options, displayResults)
selector = "BaselineRun";
if mode == "nested"
    selector = "NestedRun";
end
parents = [fullfile("results", "cross-validation", mode), ...
    fullfile(options.OutputDirectory, "cross-validation", mode)];
[run, directory] = demoSavedRun(options.(selector), parents, ...
    "run-*", "experiment-metadata.json", mode + " cross-validation", selector, ...
    @(path) loadCompatible(path, mode, options.Synthetic));
if isempty(run)
    run = struct('OutputDirectory', "");
    if displayResults
        fprintf('No compatible %s CV run. Set %s to its local directory.\n', mode, selector);
    end
    return
end
if ~displayResults
    return
end
fprintf('\n%s | %s | revision %s | %s\nLocal run: %s\n', ...
    run.Context, run.Status, run.Metadata.RepositoryCommit, run.Metadata.TimestampUTC, directory);
fprintf(['Accuracy and AUC use unweighted fold means and sample SD. ' ...
    'SD describes fold variation.\n']);
s = run.Summary;
disp(table(s.Method, 100*s.MeanAccuracy, 100*s.StdAccuracy, s.MeanAUC, s.StdAUC, ...
    'VariableNames', {'Method','AccuracyPct','AccuracySDPct','AUC','AUCSD'}));
disp(table(s.Method, s.FeaturesRange, s.ClassifierParametersRange, s.ProjectionParametersRange, ...
    s.NormalizationParametersRange, s.CombinedParametersRange, ...
    'VariableNames', {'Method','ReducedFeatures','Classifier','Projection','Normalization','Combined'}));
rows = run.Records;
batch = arrayfun(@(row) row.NumEvaluationSamples, rows);
fprintf('Mean fold medians in ms. Inference is in memory (batch sizes: %s).\n', ...
    strjoin(string(unique(batch)), ', '));
disp(table(s.Method, 1000*s.MeanMedianClassifierFitSeconds, 1000*s.MeanMedianPipelineFitSeconds, ...
    1000*s.MeanMedianClassifierInferenceSeconds, 1000*s.MeanMedianPipelineInferenceSeconds, ...
    'VariableNames', {'Method','ClassifierFitMs','PipelineFitMs','ClassifierBatchMs','PipelineBatchMs'}));
curves = run.MeanROC;
if all(arrayfun(@(curve) isequal(curve.FPR, curves(1).FPR) && ...
        isequal(curve.TPR, curves(1).TPR), curves))
    fprintf('The four mean ROC curves coincide at their measured coordinates.\n');
end
if options.FoldDetails
    disp(foldTable(rows));
end
end

function run = loadCompatible(path, mode, synthetic)
run = loadCrossValidationRun(path);
methods = ["pca-bayes", "lda-bayes", "pca-mlp", "lda-mlp"];
assert(string(run.Config.Mode) == mode && run.Status == "complete" && ...
    isequal(run.Methods, methods) && numel(run.Folds) == 5 && ...
    all([run.Records.EvaluationStage] == "cross-validation"), ...
    'genderRecognition:DemoIncompatibleRun', 'Expected completed four-method five-fold %s CV.', mode);
assert(run.Metadata.Dataset.MatchesExpectedProfile == ~synthetic, ...
    'genderRecognition:DemoDataScope', 'The CV dataset profile differs from the selected demonstration scope.');
run.Context = "Course-profile training | " + mode + " CV";
if synthetic
    run.Context = "Synthetic fixture | " + mode + " CV";
end
end

function details = foldTable(rows)
details = table([rows.Method].', [rows.FoldNumber].', [rows.NumEvaluationSamples].', ...
    100*[rows.Accuracy].', [rows.AUC].', ...
    'VariableNames', {'Method','Fold','BatchSize','AccuracyPct','AUC'});
for name = ["NumFeatures","NumClassifierParameters","NumProjectionParameters", ...
        "NumNormalizationParameters","CombinedParameterCount"]
    details.(name) = [rows.(name)].';
end
for scope = ["Classifier","Pipeline"]
    details.(scope + "FitMs") = 1000 * arrayfun(@(r) r.Timing.Training.(scope).MedianSeconds, rows(:));
    details.(scope + "BatchMs") = 1000 * arrayfun(@(r) r.Timing.Inference.(scope).MedianSeconds, rows(:));
end
end
