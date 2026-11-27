function stats = run_partition_analysis(trainingPath, config)
%RUN_PARTITION_ANALYSIS Print and return aggregate fold statistics.
%   Call SETUP first. The default training folder is train/ at the project
%   root. CONFIG accepts the options documented by CREATEFOLDS.

if nargin < 1
    root = fileparts(fileparts(mfilename('fullpath')));
    trainingPath = fullfile(root, 'train');
end
if nargin < 2
    config = struct();
end

try
    [~, y, subjectIDs] = loadDataset(trainingPath);
catch cause
    % Loader diagnostics may contain private filenames.
    if startsWith(cause.identifier, 'genderRecognition:')
        error(cause.identifier, ...
            'Training data could not be loaded (%s). Check the training folder locally.', ...
            cause.identifier);
    end
    rethrow(cause);
end

randomState = rng;
[folds, info] = createFolds(y, subjectIDs, config);
[repeatedFolds, repeatedInfo] = createFolds(y, subjectIDs, config);
stats = rmfield(info, 'SampleToFold');
stats.Reproducible = isequal(folds, repeatedFolds) && isequal(info, repeatedInfo);
stats.GlobalRandomStatePreserved = isequal(randomState, rng);

stats.NumSamples = numel(y);
stats.ClassCounts = [nnz(y == 0), nnz(y == 1)];
[subjects, ~, groups] = unique(subjectIDs);
stats.NumSubjects = numel(subjects);
stats.SubjectClassCounts = [numel(unique(groups(y == 0))), numel(unique(groups(y == 1)))];
imagesPerSubject = accumarray(groups, 1);
[imageCounts, ~, countGroups] = unique(imagesPerSubject);
stats.ImagesPerSubjectDistribution = [imageCounts, accumarray(countGroups, 1)];

stats.ValidationSubjectClassCounts = zeros(info.NumFolds, 2);
validationVisits = zeros(size(y));
for k = 1:info.NumFolds
    idx = folds(k).ValidationIndices;
    validationVisits(idx) = validationVisits(idx) + 1;
    validationLabels = y(idx);
    validationGroups = groups(idx);
    stats.ValidationSubjectClassCounts(k, :) = ...
        [numel(unique(validationGroups(validationLabels == 0))), ...
        numel(unique(validationGroups(validationLabels == 1)))];
end
stats.NumValidatedExactlyOnce = nnz(validationVisits == 1);
assert(stats.NumValidatedExactlyOnce == stats.NumSamples, ...
    'genderRecognition:InvalidPartition', 'Validation coverage check failed.');
assert(stats.Reproducible && stats.GlobalRandomStatePreserved, ...
    'genderRecognition:PartitionReproducibility', ...
    'Repeated partitioning must give identical results and preserve the random state.');
% CREATEFOLDS validates membership, class coverage, and grouped separation.
stats.PartitionIntegrityPassed = true;

fprintf('Training images: %d (%d male, %d female)\n', stats.NumSamples, stats.ClassCounts);
fprintf('Filename-derived subject IDs: %d (%d male, %d female)\n', ...
    stats.NumSubjects, stats.SubjectClassCounts);
for row = 1:size(stats.ImagesPerSubjectDistribution, 1)
    fprintf('Images per subject: %d (%d subject IDs)\n', stats.ImagesPerSubjectDistribution(row, :));
end
fprintf('\nFolds: %d | Seed: %d | Subject grouping: %s | Stratification: %s\n', ...
    info.NumFolds, info.RandomSeed, string(info.GroupBySubject), string(info.Stratified));
fprintf('Fold | Train (M F) | Validation (M F) | Train IDs | Validation IDs (M F) | ID overlap\n');
for k = 1:info.NumFolds
    fprintf('%4d | %5d (%2d %2d) | %10d (%2d %2d) | %9d | %14d (%2d %2d) | %10d\n', ...
        k, info.TrainSampleCounts(k), info.TrainClassCounts(k, :), ...
        info.ValidationSampleCounts(k), info.ValidationClassCounts(k, :), ...
        info.TrainSubjectCounts(k), info.ValidationSubjectCounts(k), ...
        stats.ValidationSubjectClassCounts(k, :), info.SubjectOverlapCounts(k));
end
fprintf('Validation coverage: %d/%d samples validated exactly once\n', ...
    stats.NumValidatedExactlyOnce, stats.NumSamples);
fprintf('Partition integrity: passed\n');
fprintf('Reproducibility: passed (identical folds and counts, global random state preserved)\n');
end
