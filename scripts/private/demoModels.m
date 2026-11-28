function [info, pipelines] = demoModels(options, displayResults)
parents = ["results/final-evaluation", fullfile(options.OutputDirectory, "final-evaluation")];
[bundle, directory] = demoSavedRun(options.FinalRun, parents, ...
    "run-*", "model-manifest.json", "final training", "FinalRun", ...
    @(path) loadCompatible(path, options.Synthetic));
assert(~isempty(bundle), 'genderRecognition:DemoMissingModels', ...
    ['No compatible verified final models. Set FinalRun to a completed bundle, ' ...
    'or prepare one with run_final_training before the presentation.']);
info = bundle.Info;
info.OutputDirectory = directory;
pipelines = bundle.Pipelines;
fprintf('\n%s | verified | revision %s | trained %s | %d images\nLocal bundle: %s\n', ...
    info.Context, info.Revision, info.CompletedUTC, info.NumTrainingSamples, directory);
fprintf('Selection provenance: %s\n', info.Selection);
if displayResults
    disp(info.Summary);
    fprintf('Fit time is the median of %d repeats after %d warm-ups (in-memory fit).\n', ...
        info.Benchmark.NumRepeats, info.Benchmark.WarmupRuns);
    fprintf('Combined = projection + normalization + classifier scalars. MAT bytes measure serialized files.\n');
    for k = 1:numel(pipelines)
        fprintf('%s frozen settings\n', pipelines(k).Method);
        optionsBlock = pipelines(k).Configuration;
        for section = string(fieldnames(optionsBlock)).'
            fprintf('  %s: %s\n', section, jsonencode(optionsBlock.(section)));
        end
    end
    fprintf('Baseline, nested outer fits, and final full-training settings can differ.\n');
end
end

function bundle = loadCompatible(directory, synthetic)
[pipelines, manifest, frozen, training] = loadFinalModels(directory);
assert(frozen.Synthetic == synthetic, 'genderRecognition:DemoDataScope', ...
    'The final model bundle differs from the selected demonstration scope.');
rows = [training.Results.Parameters].';
summary = struct2table(rows);
summary = addvars(summary, [pipelines.Method].', 'Before', 1, 'NewVariableNames', 'Method');
summary.ClassifierFitMs = 1000 * arrayfun(@(r) r.Timing.ClassifierFit.MedianSeconds, training.Results(:));
summary.PipelineFitMs = 1000 * arrayfun(@(r) r.Timing.PipelineFit.MedianSeconds, training.Results(:));
summary.ModelBytes = [manifest.Models.Bytes].';
info = struct('Context', "Course final full-training models", 'Synthetic', synthetic, ...
    'Revision', string(frozen.RepositoryRevision), 'FrozenUTC', string(frozen.FrozenUTC), ...
    'CompletedUTC', string(training.CompletedUTC), 'ConfigurationSHA256', string(frozen.ConfigurationSHA256), ...
    'ManifestSHA256', finalInternal.checksum(fullfile(directory, "model-manifest.json"), true), ...
    'NumTrainingSamples', frozen.TrainingSamples, 'Summary', summary, ...
    'TrainingIntegrityDefect', training.MaterialIntegrityDefect, ...
    'Benchmark', training.Benchmark, 'Selection', string(frozen.SelectionProvenance.Source));
if synthetic
    info.Context = "Synthetic fixture final models";
end
bundle = struct('Info', info, 'Pipelines', pipelines);
end
