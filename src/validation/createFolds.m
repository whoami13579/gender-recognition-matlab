function [folds, partitionInfo] = createFolds(y, subjectIDs, config)
%CREATEFOLDS Stratify labeled samples for cross-validation.
%   Y is a nonempty N-by-1 double column with labels 0 (male) and 1
%   (female). SUBJECTIDS is an aligned N-by-1 string column.
%   CONFIG fields are NumFolds (5), RandomSeed (42), GroupBySubject (true).
%   For sample-level stratification, explicitly set GroupBySubject to false
%   and pass [] when IDs are unavailable.
%   Fold indices are ascending one-based double columns into the input.
%   PartitionInfo counts use one row per fold and class columns [0 1].
%   Subject counts and overlap counts are empty when IDs are unavailable.

arguments
    y
    subjectIDs = []
    config = struct()
end

if ~isa(y, 'double') || ~isreal(y) || issparse(y) || ~iscolumn(y) ...
        || isempty(y) || any(y ~= 0 & y ~= 1)
    error('genderRecognition:InvalidLabels', ...
        'Labels must be a nonempty N-by-1 double column of zeros and ones.');
end
if ~any(y == 0) || ~any(y == 1)
    error('genderRecognition:MissingClass', 'Both label classes are required.');
end
options = partitionOptions(config);
numFolds = options.NumFolds;
numSamples = numel(y);

subjectGroups = [];
if options.GroupBySubject || ~isempty(subjectIDs)
    if ~isstring(subjectIDs) || ~isequal(size(subjectIDs), size(y)) ...
            || any(ismissing(subjectIDs)) || any(strlength(strip(subjectIDs)) == 0)
        error('genderRecognition:InvalidSubjectIDs', ...
            ['Supply an N-by-1 string column of nonblank subject IDs. ' ...
             'If IDs are unavailable, pass [] and explicitly disable GroupBySubject.']);
    end
    [~, ~, subjectGroups] = unique(subjectIDs, 'stable');
end

if options.GroupBySubject
    sampleGroups = subjectGroups;
    groupSizes = accumarray(sampleGroups, 1);
    femaleCounts = accumarray(sampleGroups, y);
    if any(femaleCounts > 0 & femaleCounts < groupSizes)
        error('genderRecognition:ConflictingSubjectLabels', ...
            'Each subject ID must have the same label for all its samples.');
    end
    groupLabels = double(femaleCounts > 0);
else
    sampleGroups = (1:numSamples).';
    groupSizes = ones(numSamples, 1);
    groupLabels = y;
end

if nnz(groupLabels == 0) < numFolds || nnz(groupLabels == 1) < numFolds
    error('genderRecognition:InsufficientClassGroups', ...
        ['Each class must have at least NumFolds distinct subject groups ' ...
         '(or samples in sample-level mode) to appear in every validation fold.']);
end

stream = RandStream('mt19937ar', 'Seed', options.RandomSeed);
foldOrder = randperm(stream, numFolds).';
groupFolds = zeros(size(groupLabels));
totalLoads = zeros(numFolds, 1);
for label = 0:1
    groups = find(groupLabels == label);
    groups = groups(randperm(stream, numel(groups)));
    % Stable descending sort preserves the seeded order among equal sizes.
    [~, sizeOrder] = sort(groupSizes(groups), 'descend');
    classLoads = zeros(numFolds, 1);
    for group = reshape(groups(sizeOrder), 1, [])
        % Prioritize per-class image counts, then total image counts.
        candidates = foldOrder(classLoads(foldOrder) == min(classLoads));
        [~, firstLightest] = min(totalLoads(candidates));
        fold = candidates(firstLightest);
        groupFolds(group) = fold;
        classLoads(fold) = classLoads(fold) + groupSizes(group);
        totalLoads(fold) = totalLoads(fold) + groupSizes(group);
    end
end

partitionInfo = options;
partitionInfo.Stratified = true;
partitionInfo.SampleToFold = groupFolds(sampleGroups);
[folds, partitionInfo] = buildAndValidateFolds(y, subjectGroups, partitionInfo);
end

function options = partitionOptions(config)
options = struct('NumFolds', 5, 'RandomSeed', 42, 'GroupBySubject', true);
if ~isstruct(config) || ~isscalar(config)
    error('genderRecognition:InvalidConfig', 'Partition options must be a scalar struct.');
end
fields = fieldnames(config);
for k = 1:numel(fields)
    field = fields{k};
    if ~isfield(options, field)
        error('genderRecognition:InvalidConfig', 'Unknown partition option "%s".', field);
    end
    options.(field) = config.(field);
end
if ~isIntegerScalar(options.NumFolds) || options.NumFolds < 2 ...
        || options.NumFolds > flintmax
    error('genderRecognition:InvalidNumFolds', 'NumFolds must be an integer from 2 to flintmax.');
end
if ~isIntegerScalar(options.RandomSeed) || options.RandomSeed < 0 ...
        || options.RandomSeed > double(intmax('uint32'))
    error('genderRecognition:InvalidRandomSeed', ...
        'RandomSeed must be an integer from 0 to 2^32 - 1.');
end
if ~islogical(options.GroupBySubject) || ~isscalar(options.GroupBySubject) ...
        || issparse(options.GroupBySubject)
    error('genderRecognition:InvalidConfig', 'GroupBySubject must be a full logical scalar.');
end
options.NumFolds = double(options.NumFolds);
options.RandomSeed = double(options.RandomSeed);
end

function valid = isIntegerScalar(value)
valid = isnumeric(value) && isreal(value) && ~issparse(value) && isscalar(value) ...
    && isfinite(value) && value == fix(value);
end

function [folds, info] = buildAndValidateFolds(y, subjectGroups, info)
numFolds = info.NumFolds;
allIndices = (1:numel(y)).';
validationVisits = zeros(size(y));
folds = repmat(struct('TrainIndices', [], 'ValidationIndices', []), numFolds, 1);
info.TrainSampleCounts = zeros(numFolds, 1);
info.ValidationSampleCounts = zeros(numFolds, 1);
info.TrainClassCounts = zeros(numFolds, 2);
info.ValidationClassCounts = zeros(numFolds, 2);
info.TrainSubjectCounts = [];
info.ValidationSubjectCounts = [];
info.SubjectOverlapCounts = [];
if ~isempty(subjectGroups)
    info.TrainSubjectCounts = zeros(numFolds, 1);
    info.ValidationSubjectCounts = zeros(numFolds, 1);
    info.SubjectOverlapCounts = zeros(numFolds, 1);
end

for k = 1:numFolds
    train = find(info.SampleToFold ~= k);
    validation = find(info.SampleToFold == k);
    % Exact membership checks range, uniqueness, separation, and completeness.
    if ~isequal(sort([train; validation]), allIndices)
        error('genderRecognition:InvalidPartition', 'Fold %d has invalid sample membership.', k);
    end
    validationCounts = [nnz(y(validation) == 0), nnz(y(validation) == 1)];
    if any(validationCounts == 0)
        error('genderRecognition:InvalidPartition', 'Fold %d must contain both classes.', k);
    end
    validationVisits(validation) = validationVisits(validation) + 1;
    folds(k).TrainIndices = train;
    folds(k).ValidationIndices = validation;
    info.TrainSampleCounts(k) = numel(train);
    info.ValidationSampleCounts(k) = numel(validation);
    info.TrainClassCounts(k, :) = [nnz(y(train) == 0), nnz(y(train) == 1)];
    info.ValidationClassCounts(k, :) = validationCounts;

    if ~isempty(subjectGroups)
        trainSubjects = unique(subjectGroups(train));
        validationSubjects = unique(subjectGroups(validation));
        info.TrainSubjectCounts(k) = numel(trainSubjects);
        info.ValidationSubjectCounts(k) = numel(validationSubjects);
        info.SubjectOverlapCounts(k) = numel(intersect(trainSubjects, validationSubjects));
        if info.GroupBySubject && info.SubjectOverlapCounts(k) > 0
            error('genderRecognition:InvalidPartition', 'Fold %d has subject overlap.', k);
        end
    end
end

% Complementary train membership implies K-1 training visits per sample.
if any(validationVisits ~= 1)
    error('genderRecognition:InvalidPartition', 'Every sample must be validated exactly once.');
end
end
