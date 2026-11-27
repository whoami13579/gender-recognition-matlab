function stats = run_lda_analysis(trainingPath, ldaConfig, preprocessingConfig, foldNumber)
%RUN_LDA_ANALYSIS Report aggregate Fisher LDA statistics for local training data.
%   Run SETUP first. FOLDNUMBER optionally selects a default subject-grouped
%   fold. The complete training fit is descriptive. The function returns
%   aggregate statistics and writes no files.

arguments
    trainingPath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'train')
    ldaConfig = struct()
    preprocessingConfig = struct()
    foldNumber = []
end

if ~isequal(foldNumber, []) && (~isnumeric(foldNumber) || ~isscalar(foldNumber) || ...
        ~isreal(foldNumber) || issparse(foldNumber) || ~isfinite(foldNumber) || ...
        foldNumber < 1 || foldNumber ~= fix(foldNumber))
    error('genderRecognition:InvalidFoldNumber', 'Fold number must be [] or a positive integer.');
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
stats.FitScope = "complete training dataset (descriptive)";
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
yTrain = y(trainIdx);
fitTimer = tic;
ldaModel = fitLDA(XTrain, yTrain, ldaConfig);
stats.FitSeconds = toc(fitTimer);
transformTimer = tic;
ZTrain = transformLDA(ldaModel, XTrain);
stats.TrainTransformSeconds = toc(transformTimer);

stats.NumTrainingSamples = ldaModel.NumTrainingSamples;
stats.TrainClassCounts = ldaModel.ClassCounts;
stats.OriginalImageSizes = unique(metadata{trainIdx, {'Height', 'Width', 'Channels'}}, 'rows');
stats.ProcessedImageSize = preprocessor.OutputImageSize;
stats.NumericType = string(class(XTrain));
stats.TrainIntensityRange = [min(XTrain, [], 'all'), max(XTrain, [], 'all')];
stats.NumInputFeatures = ldaModel.NumInputFeatures;
stats.TrainFeatureSize = size(XTrain);
stats.ReducedTrainSize = size(ZTrain);
stats.MeanSize = size(ldaModel.Mean);
stats.ClassMeansSize = size(ldaModel.ClassMeans);
stats.ResidualSize = stats.TrainFeatureSize;
stats.ProjectionSize = size(ldaModel.Components);
stats.NumClasses = numel(ldaModel.ClassCounts);
stats.MaxOutputDimension = min(stats.NumClasses - 1, stats.NumInputFeatures);
stats.NumFeatures = ldaModel.NumFeatures;
stats.MaxWithinClassRank = min(stats.NumTrainingSamples - stats.NumClasses, stats.NumInputFeatures);
stats.WithinClassRank = ldaModel.WithinClassRank;
stats.RankTolerance = ldaModel.RankTolerance;
stats.RegularizationCoefficient = ldaModel.RegularizationCoefficient;
stats.EffectiveRegularization = ldaModel.EffectiveRegularization;
stats.NumFittedParameters = ldaModel.NumFittedParameters;
[stats.TrainProjectedMeans, stats.TrainProjectedVariances] = projectedStats(ZTrain, yTrain);
stats.ProjectedMeanSeparation = ldaModel.ProjectedMeanSeparation;
stats.RegularizedFisherCriterion = ldaModel.RegularizedFisherCriterion;
stats.WithinClassScatterTrace = ldaModel.WithinClassScatterTrace;
stats.ProjectedWithinClassScatter = ldaModel.ProjectedWithinClassScatter;
stats.RelativeResidual = ldaModel.RelativeResidual;
stats.UsedZeroScatterFallback = ldaModel.UsedZeroScatterFallback;
originalMatrix = whos('XTrain');
reducedMatrix = whos('ZTrain');
stats.TrainFeatureBytes = originalMatrix.bytes;
stats.ReducedTrainBytes = reducedMatrix.bytes;

fprintf('LDA fit scope: %s\n', stats.FitScope);
fprintf('Training samples: %d (%d male, %d female)\n', stats.NumTrainingSamples, stats.TrainClassCounts);
for k = 1:size(stats.OriginalImageSizes, 1)
    fprintf('Original training image size: %d x %d x %d\n', stats.OriginalImageSizes(k, :));
end
fprintf('Processed image size: %d x %d x %d\n', stats.ProcessedImageSize);
fprintf('Training features: %d x %d (%s), intensity range [%.6g, %.6g]\n', ...
    stats.TrainFeatureSize, stats.NumericType, stats.TrainIntensityRange);
fprintf('Classes: %d; LDA features: %d (theoretical maximum: %d)\n', ...
    stats.NumClasses, stats.NumFeatures, stats.MaxOutputDimension);
fprintf('Within-class rank: %d (theoretical maximum: %d, tolerance: %.9g)\n', ...
    stats.WithinClassRank, stats.MaxWithinClassRank, stats.RankTolerance);
fprintf('Regularization: rho %.9g; effective alpha %.9g; zero-scatter fallback %d\n', ...
    stats.RegularizationCoefficient, stats.EffectiveRegularization, stats.UsedZeroScatterFallback);
fprintf('Global mean: %d x %d; class means: %d x %d; residuals: %d x %d; projection: %d x %d\n', ...
    stats.MeanSize, stats.ClassMeansSize, stats.ResidualSize, stats.ProjectionSize);
fprintf('Stored fitted projection coefficients: %d\n', stats.NumFittedParameters);
fprintf('Reduced training features: %d x %d\n', stats.ReducedTrainSize);
fprintf('Training projected means [male female]: %.9g %.9g; sample variances: %.9g %.9g\n', ...
    stats.TrainProjectedMeans, stats.TrainProjectedVariances);
fprintf('Training projected mean separation (female minus male): %.9g\n', stats.ProjectedMeanSeparation);
fprintf('Within-class scatter trace: %.9g; projected scatter: %.9g\n', ...
    stats.WithinClassScatterTrace, stats.ProjectedWithinClassScatter);
fprintf('Regularized Fisher criterion: %.9g; relative solver residual: %.9g\n', ...
    stats.RegularizedFisherCriterion, stats.RelativeResidual);
fprintf('Training matrix storage: %d -> %d bytes\n', stats.TrainFeatureBytes, stats.ReducedTrainBytes);
fprintf('LDA fit: %.6f s; training transform: %.6f s\n', stats.FitSeconds, stats.TrainTransformSeconds);

if ~isempty(foldNumber)
    XValidation = transformImages(preprocessor, images(validationIdx));
    transformTimer = tic;
    ZValidation = transformLDA(ldaModel, XValidation);
    stats.ValidationTransformSeconds = toc(transformTimer);
    yValidation = y(validationIdx);
    stats.ValidationClassCounts = [nnz(yValidation == 0) nnz(yValidation == 1)];
    stats.ValidationFeatureSize = size(XValidation);
    stats.ReducedValidationSize = size(ZValidation);
    [stats.ValidationProjectedMeans, stats.ValidationProjectedVariances] = projectedStats(ZValidation, yValidation);
    stats.ValidationProjectedMeanSeparation = diff(stats.ValidationProjectedMeans);
    originalMatrix = whos('XValidation');
    reducedMatrix = whos('ZValidation');
    stats.ValidationFeatureBytes = originalMatrix.bytes;
    stats.ReducedValidationBytes = reducedMatrix.bytes;
    fprintf('Validation samples: %d male, %d female; subject ID overlap: %d\n', ...
        stats.ValidationClassCounts, stats.SubjectOverlapCount);
    fprintf('Validation features: %d x %d -> %d x %d\n', ...
        stats.ValidationFeatureSize, stats.ReducedValidationSize);
    fprintf('Validation projected means [male female]: %.9g %.9g; sample variances: %.9g %.9g\n', ...
        stats.ValidationProjectedMeans, stats.ValidationProjectedVariances);
    fprintf('Validation projected mean separation (female minus male): %.9g\n', ...
        stats.ValidationProjectedMeanSeparation);
    fprintf('Validation matrix storage: %d -> %d bytes; LDA transform: %.6f s\n', ...
        stats.ValidationFeatureBytes, stats.ReducedValidationBytes, stats.ValidationTransformSeconds);
end
end

function [means, variances] = projectedStats(Z, y)
male = Z(y == 0);
female = Z(y == 1);
means = [mean(male) mean(female)];
variances = [var(male) var(female)];
end
