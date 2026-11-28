function results = run_cross_validation(trainingPath, overrides)
%RUN_CROSS_VALIDATION Run a baseline or nested experiment using training PNGs.
arguments
    trainingPath = []
    overrides = struct()
end
root = fileparts(fileparts(mfilename('fullpath')));
config = crossValidationConfig(overrides);
if ~isequal(trainingPath, [])
    config.TrainingPath = trainingPath;
    config = crossValidationConfig(config);
end
trainingPath = resolveTrainingPath(root, config.TrainingPath);
config.TrainingPath = trainingPath;
started = tic;
metadata = describeRun(root);
[images, y, subjects, samples] = loadDataset(trainingPath);
metadata.Dataset = describeDataset(root, trainingPath, images, y, subjects, samples);
if metadata.Dataset.RepositoryTrainingPath && ~metadata.Dataset.MatchesExpectedProfile
    warning('genderRecognition:UnexpectedTrainingProfile', ...
        'The repository training folder differs from the documented official profile. Review Dataset metadata.');
end
results = crossValidatePipelines(images, y, subjects, config);
metadata.Status = results.Status;
metadata.ExpectedRecords = numel(results.Methods) * numel(results.Folds);
metadata.CompletedRecords = numel(results.Records);
metadata.SearchSeconds = results.SearchSeconds;
metadata.FinalSelectionSeconds = [];
metadata.FinalSelectionStatus = "not-run";
if ~isempty(results.FinalSearch)
    metadata.FinalSelectionSeconds = results.FinalSearch.Seconds;
    metadata.FinalSelectionStatus = results.FinalSearch.Status;
end
metadata.ElapsedSecondsBeforeExport = toc(started);
results.Metadata = metadata;
[results.Summary, results.MeanROC] = summarizeCrossValidation(results.Records, results.Methods, numel(results.Folds));
results.OutputDirectory = "";
if config.ExportResults || config.GeneratePlots
    results.OutputDirectory = saveCrossValidationRun(results);
end
fprintf('%s: %s (%d/%d method-fold records).\n', config.Mode, results.Status, ...
    metadata.CompletedRecords, metadata.ExpectedRecords);
if strlength(results.OutputDirectory) > 0
    fprintf('Saved locally to %s\n', results.OutputDirectory);
end
end

function path = resolveTrainingPath(root, path)
argument = java.io.File(char(path));
if ~argument.isAbsolute()
    path = string(fullfile(root, path));
end
% Check the lexical path before resolving links, and check the real path as well.
lexical = java.io.File(char(path)).toPath().toAbsolutePath().normalize();
rejectTestingPath(string(lexical.toString()));
if ~isfolder(path)
    error('genderRecognition:InvalidDatasetPath', 'The training directory does not exist.');
end
path = realDirectory(path);
rejectTestingPath(path);
end

function rejectTestingPath(path)
if any(strcmpi(split(replace(path, "\", "/"), "/"), "test"))
    error('genderRecognition:IndependentTestPath', 'This entry point accepts training folders only. Test folders are reserved.');
end
end

function path = realDirectory(path)
options = javaArray('java.nio.file.LinkOption', 0);
file = java.io.File(char(path));
path = string(file.toPath().toRealPath(options).toString());
end

function metadata = describeRun(root)
[status, revision] = system(sprintf('git -C "%s" rev-parse HEAD', root));
if status ~= 0
    revision = 'unavailable';
end
[status, tree] = system(sprintf('git -C "%s" status --porcelain --untracked-files=no', root));
processor = string(getenv('PROCESSOR_IDENTIFIER'));
if ispc
    processor = string(winqueryreg('HKEY_LOCAL_MACHINE', ...
        'HARDWARE\DESCRIPTION\System\CentralProcessor\0', 'ProcessorNameString'));
end
metadata = struct('ExperimentSchemaVersion', 1, 'TimestampUTC', string(datetime('now', ...
    'TimeZone', 'UTC', 'Format', "yyyy-MM-dd'T'HH:mm:ss'Z'")), ...
    'RepositoryCommit', string(strtrim(revision)), 'TrackedTreeClean', status == 0 && isempty(strtrim(tree)), ...
    'MATLABVersion', string(version), 'MATLABRelease', string(version('-release')), ...
    'OperatingSystem', string(system_dependent('getos')), 'Architecture', string(computer), ...
    'Processor', strtrim(processor), 'ThreadLimit', maxNumCompThreads, ...
    'ExecutionOrder', "Serial ascending folds, fixed method list cyclically rotated by fold; " + ...
    "complete fit before training stages; complete inference before inference stages.");
end

function dataset = describeDataset(root, path, images, y, subjects, samples)
[~, first, groups] = unique(subjects, 'stable');
imageCounts = accumarray(groups, 1);
classCounts = [nnz(y == 0), nnz(y == 1)];
subjectCounts = [nnz(y(first) == 0), nnz(y(first) == 1)];
shapes = unique([samples.Height samples.Width samples.Channels], 'rows');
official = isequal(classCounts, [100 100]) && isequal(subjectCounts, [50 50]) && ...
    all(imageCounts == 2) && isequal(shapes, [640 640 3]);
digest = java.security.MessageDigest.getInstance('SHA-256');
for idx = 1:numel(images)
    % Length is implied by shape and type. Pixel bytes use little-endian order.
    header = sprintf('%s\n%d %d %d %s\n', samples.Filename(idx), ...
        samples.Height(idx), samples.Width(idx), samples.Channels(idx), samples.PixelType(idx));
    digest.update(typecast(unicode2native(header, 'UTF-8'), 'int8'));
    pixels = images{idx}(:);
    [~, ~, endian] = computer;
    if endian == 'B' && ~islogical(pixels)
        pixels = swapbytes(pixels);
    end
    if islogical(pixels)
        pixels = uint8(pixels);
    end
    digest.update(typecast(pixels, 'int8'));
end
fingerprint = lower(string(reshape(dec2hex(typecast(digest.digest(), 'uint8'), 2).', 1, [])));
repoTraining = fullfile(root, 'train');
isRepositoryPath = isfolder(repoTraining) && strcmp(path, realDirectory(repoTraining));
dataset = struct('RepositoryTrainingPath', isRepositoryPath, 'MatchesExpectedProfile', official, ...
    'NumImages', numel(images), 'ClassCounts', classCounts, 'SubjectCounts', subjectCounts, ...
    'ImagesPerSubjectValues', unique(imageCounts).', 'ImageShapes', shapes, ...
    'PixelTypes', unique(samples.PixelType).', 'DecodedSHA256', fingerprint);
end
