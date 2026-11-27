function stats = run_dataset_analysis(trainingPath, config)
%RUN_DATASET_ANALYSIS Print aggregate statistics for a local training folder.
%   Call SETUP first. The default folder is train/ at the project root.
%   STATS contains aggregate values only. No files are written.

if nargin < 1
    root = fileparts(fileparts(mfilename('fullpath')));
    trainingPath = fullfile(root, 'train');
end
if nargin < 2
    config = struct();
end

loadingTimer = tic;
[images, y, subjectIDs, metadata] = loadDataset(trainingPath);
stats.LoadingSeconds = toc(loadingTimer);

preprocessingTimer = tic;
preprocessor = fitPreprocessor(images, config);
X = transformImages(preprocessor, images);
stats.PreprocessingSeconds = toc(preprocessingTimer);

stats.NumImages = numel(y);
stats.ClassCounts = [nnz(y == 0), nnz(y == 1)];
stats.MatchesOfficialTrainingCounts = isequal(stats.ClassCounts, [100 100]);
maleSubjects = unique(subjectIDs(y == 0));
femaleSubjects = unique(subjectIDs(y == 1));
stats.SubjectCounts = [numel(maleSubjects), numel(femaleSubjects)];
% Prefixes encode the class, so this checks filename-derived IDs only.
stats.CrossClassQualifiedIDCollisions = numel(intersect(maleSubjects, femaleSubjects));
[subjects, ~, groups] = unique(subjectIDs);
stats.NumSubjects = numel(subjects);
subjectImageCounts = accumarray(groups, 1);
[imagesPerSubject, ~, countGroups] = unique(subjectImageCounts);
stats.ImagesPerSubjectDistribution = [imagesPerSubject, accumarray(countGroups, 1)];

[stats.OriginalImageSizes, ~, sizeGroups] = ...
    unique(metadata{:, {'Height', 'Width', 'Channels'}}, 'rows');
stats.OriginalSizeCounts = accumarray(sizeGroups, 1);
stats.OriginalPixelTypes = unique(metadata.PixelType);
stats.OriginalGrayscaleDimensions = prod(stats.OriginalImageSizes(:, 1:2), 2);
stats.OriginalNativeDimensions = prod(stats.OriginalImageSizes, 2);
stats.ProcessedImageSize = preprocessor.OutputImageSize;
stats.ProcessedFeatureDimension = preprocessor.NumFeatures;
stats.FeatureMatrixSize = size(X);
stats.LabelVectorSize = size(y);
stats.NumericType = string(class(X));
stats.IntensityRange = [min(X, [], 'all'), max(X, [], 'all')];
featureMatrix = whos('X');
stats.FeatureMatrixBytes = featureMatrix.bytes;

fprintf('Training images: %d\nMale images: %d\nFemale images: %d\n', ...
    stats.NumImages, stats.ClassCounts(1), stats.ClassCounts(2));
if stats.MatchesOfficialTrainingCounts
    fprintf('Official training counts: match (200 total, 100 per class)\n');
else
    fprintf('Official training counts: differ (expected 200 total, 100 per class)\n');
end
fprintf('\nUnique subject IDs: %d (%d male, %d female)\n', ...
    stats.NumSubjects, stats.SubjectCounts(1), stats.SubjectCounts(2));
fprintf('Cross-class qualified ID collisions: %d (filename-derived IDs)\n', ...
    stats.CrossClassQualifiedIDCollisions);
for k = 1:size(stats.ImagesPerSubjectDistribution, 1)
    fprintf('Images per subject: %d (%d subjects)\n', ...
        stats.ImagesPerSubjectDistribution(k, :));
end

fprintf('\nOriginal pixel type(s): %s\n', char(strjoin(stats.OriginalPixelTypes, ', ')));
for k = 1:size(stats.OriginalImageSizes, 1)
    fprintf('Original image size: %d x %d x %d (%d images)\n', ...
        stats.OriginalImageSizes(k, :), stats.OriginalSizeCounts(k));
    fprintf('Original grayscale feature dimension: %d\n', stats.OriginalGrayscaleDimensions(k));
    fprintf('Original native-channel feature dimension: %d\n', stats.OriginalNativeDimensions(k));
end
fprintf('Processed image size (H x W x C): %d x %d x %d\n', stats.ProcessedImageSize);
fprintf('Processed feature dimension: %d\n', stats.ProcessedFeatureDimension);
fprintf('Feature matrix X: %d x %d\n', stats.FeatureMatrixSize);
fprintf('Label vector y: %d x %d\n', stats.LabelVectorSize);
fprintf('Feature numeric type: %s\n', char(stats.NumericType));
fprintf('Observed intensity range: [%.6g, %.6g] (%s scaling)\n', ...
    stats.IntensityRange, char(preprocessor.Options.IntensityScaling));
fprintf('Approximate feature matrix memory: %d bytes (%.2f MiB)\n', ...
    stats.FeatureMatrixBytes, stats.FeatureMatrixBytes / 2^20);
fprintf('Dataset loading: %.3f s\nPreprocessor fit and transform: %.3f s\n', ...
    stats.LoadingSeconds, stats.PreprocessingSeconds);
end
