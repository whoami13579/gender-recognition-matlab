function stats = run_pca_analysis(trainingPath, pcaConfig, preprocessingConfig, foldNumber, showPlots)
%RUN_PCA_ANALYSIS Report aggregate PCA statistics for local training data.
%   Run SETUP first. FOLDNUMBER optionally selects a training fold from
%   CREATEFOLDS with its default settings. SHOWPLOTS defaults to false.
%   The function returns aggregate statistics and writes no files.

arguments
    trainingPath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'train')
    pcaConfig = struct()
    preprocessingConfig = struct()
    foldNumber = []
    showPlots = false
end

if ~isequal(foldNumber, []) && (~isnumeric(foldNumber) || ~isscalar(foldNumber) || ...
        ~isreal(foldNumber) || issparse(foldNumber) || ~isfinite(foldNumber) || ...
        foldNumber < 1 || foldNumber ~= fix(foldNumber))
    error('genderRecognition:InvalidFoldNumber', 'Fold number must be [] or a positive integer.');
end
if ~islogical(showPlots) || ~isscalar(showPlots)
    error('genderRecognition:InvalidConfig', 'Plot selection must be a logical scalar.');
end

try
    [images, y, subjectIDs, metadata] = loadDataset(trainingPath);
catch cause
    if startsWith(cause.identifier, 'genderRecognition:')
        error(cause.identifier, ...
            'Training data could not be loaded (%s). Check the training folder locally.', ...
            cause.identifier);
    end
    rethrow(cause);
end

stats.FoldNumber = foldNumber;
stats.FitScope = "complete training dataset";
trainIdx = (1:numel(y)).';
if ~isempty(foldNumber)
    [folds, partitionInfo] = createFolds(y, subjectIDs);
    if foldNumber > numel(folds)
        error('genderRecognition:InvalidFoldNumber', 'Select a fold from 1 through %d.', numel(folds));
    end
    trainIdx = folds(foldNumber).TrainIndices;
    validationIdx = folds(foldNumber).ValidationIndices;
    stats.FitScope = "training subset of subject-grouped fold " + string(foldNumber);
    stats.SubjectOverlapCount = partitionInfo.SubjectOverlapCounts(foldNumber);
end

preprocessor = fitPreprocessor(images(trainIdx), preprocessingConfig);
XTrain = transformImages(preprocessor, images(trainIdx));
fitTimer = tic;
pcaModel = fitPCA(XTrain, pcaConfig);
stats.FitSeconds = toc(fitTimer);
transformTimer = tic;
ZTrain = transformPCA(pcaModel, XTrain);
stats.TrainTransformSeconds = toc(transformTimer);

stats.NumTrainingSamples = pcaModel.NumTrainingSamples;
stats.TrainClassCounts = [nnz(y(trainIdx) == 0), nnz(y(trainIdx) == 1)];
stats.OriginalImageSizes = unique(metadata{trainIdx, {'Height', 'Width', 'Channels'}}, 'rows');
stats.ProcessedImageSize = preprocessor.OutputImageSize;
stats.NumericType = string(class(XTrain));
stats.TrainIntensityRange = [min(XTrain, [], 'all'), max(XTrain, [], 'all')];
stats.TrainFeatureSize = size(XTrain);
stats.ReducedTrainSize = size(ZTrain);
stats.MeanSize = size(pcaModel.Mean);
stats.ProjectionSize = size(pcaModel.Components);
stats.NumInputFeatures = pcaModel.NumInputFeatures;
stats.MaxTheoreticalRank = min(pcaModel.NumTrainingSamples - 1, pcaModel.NumInputFeatures);
stats.NumericalRank = pcaModel.NumericalRank;
stats.PCAOptions = pcaModel.Options;
stats.NumFeatures = pcaModel.NumFeatures;
stats.NumFittedParameters = pcaModel.NumFittedParameters;
stats.RetainedExplainedVariance = pcaModel.RetainedExplainedVariance;
stats.VarianceThresholds = [0.90 0.95 0.99];
cumulative = pcaModel.CumulativeExplainedVariance(1:pcaModel.NumericalRank);
stats.ComponentsAtThresholds = NaN(size(stats.VarianceThresholds));
for k = 1:numel(stats.VarianceThresholds)
    count = find(cumulative + 8 * eps(1) >= stats.VarianceThresholds(k), 1);
    if ~isempty(count)
        stats.ComponentsAtThresholds(k) = count;
    end
end
originalMatrix = whos('XTrain');
reducedMatrix = whos('ZTrain');
stats.TrainFeatureBytes = originalMatrix.bytes;
stats.ReducedTrainBytes = reducedMatrix.bytes;

fprintf('PCA fit scope: %s\n', stats.FitScope);
fprintf('Training samples: %d (%d male, %d female)\n', stats.NumTrainingSamples, stats.TrainClassCounts);
for k = 1:size(stats.OriginalImageSizes, 1)
    fprintf('Original training image size: %d x %d x %d\n', stats.OriginalImageSizes(k, :));
end
fprintf('Processed image size: %d x %d x %d\n', stats.ProcessedImageSize);
fprintf('Training features: %d x %d (%s), intensity range [%.6g, %.6g]\n', ...
    stats.TrainFeatureSize, stats.NumericType, stats.TrainIntensityRange);
fprintf('Numerical rank: %d (theoretical maximum: %d)\n', stats.NumericalRank, stats.MaxTheoreticalRank);
if stats.PCAOptions.SelectionMode == "explainedVariance"
    fprintf('Requested explained variance: %.8g%%\n', 100 * stats.PCAOptions.ExplainedVarianceThreshold);
else
    fprintf('Requested PCA dimension: %d (%s)\n', stats.PCAOptions.NumComponents, stats.PCAOptions.SelectionMode);
end
fprintf('Retained PCA dimension: %d, explained variance: %.6f%%\n', ...
    stats.NumFeatures, 100 * stats.RetainedExplainedVariance);
for k = 1:numel(stats.VarianceThresholds)
    fprintf('Components for %.0f%% variance: %g\n', ...
        100 * stats.VarianceThresholds(k), stats.ComponentsAtThresholds(k));
end
fprintf('PCA mean: %d x %d; projection: %d x %d\n', stats.MeanSize, stats.ProjectionSize);
fprintf('Stored fitted projection coefficients: %d\n', stats.NumFittedParameters);
fprintf('Reduced training features: %d x %d\n', stats.ReducedTrainSize);
fprintf('Training matrix storage: %d -> %d bytes\n', stats.TrainFeatureBytes, stats.ReducedTrainBytes);
fprintf('PCA fit: %.6f s; training transform: %.6f s\n', stats.FitSeconds, stats.TrainTransformSeconds);

if ~isempty(foldNumber)
    XValidation = transformImages(preprocessor, images(validationIdx));
    transformTimer = tic;
    ZValidation = transformPCA(pcaModel, XValidation);
    stats.ValidationTransformSeconds = toc(transformTimer);
    stats.ValidationClassCounts = [nnz(y(validationIdx) == 0), nnz(y(validationIdx) == 1)];
    stats.ValidationFeatureSize = size(XValidation);
    stats.ReducedValidationSize = size(ZValidation);
    originalMatrix = whos('XValidation');
    reducedMatrix = whos('ZValidation');
    stats.ValidationFeatureBytes = originalMatrix.bytes;
    stats.ReducedValidationBytes = reducedMatrix.bytes;
    fprintf('Validation samples: %d male, %d female; subject ID overlap: %d\n', ...
        stats.ValidationClassCounts, stats.SubjectOverlapCount);
    fprintf('Validation features: %d x %d -> %d x %d\n', ...
        stats.ValidationFeatureSize, stats.ReducedValidationSize);
    fprintf('Validation matrix storage: %d -> %d bytes; PCA transform: %.6f s\n', ...
        stats.ValidationFeatureBytes, stats.ReducedValidationBytes, stats.ValidationTransformSeconds);
end

if showPlots
    plotVariance(pcaModel, stats.VarianceThresholds);
end
end

function plotVariance(model, thresholds)
figure('Name', 'Training PCA explained variance');
layout = tiledlayout(2, 1);
individual = nexttile(layout);
plot(individual, model.ExplainedVarianceRatio);
ylabel(individual, 'Explained variance fraction');
xline(individual, model.NumFeatures, '--', 'Retained count');

cumulative = nexttile(layout);
plot(cumulative, model.CumulativeExplainedVariance);
xlabel(cumulative, 'Component index');
ylabel(cumulative, 'Cumulative fraction');
xline(cumulative, model.NumFeatures, '--');
yline(cumulative, thresholds, ':', compose('%.0f%%', 100 * thresholds));
ylim(cumulative, [0 1.02]);
end
