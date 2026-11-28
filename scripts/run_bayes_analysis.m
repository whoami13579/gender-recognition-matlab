function stats = run_bayes_analysis(trainingPath, method, bayesConfig, featureConfig, preprocessingConfig, foldNumber)
%RUN_BAYES_ANALYSIS Report aggregate Gaussian Bayes statistics for one fold.
%   Run SETUP first. METHOD defaults to "pca" and FOLDNUMBER to 1.
%   Feature and preprocessing configs are passed to their existing fitters.
%   The function returns aggregate statistics and writes no files.
arguments
    trainingPath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'train')
    method = "pca"
    bayesConfig = struct()
    featureConfig = struct()
    preprocessingConfig = struct()
    foldNumber = 1
end

if ischar(method) && isrow(method)
    method = string(method);
end
if ~isstring(method) || ~isscalar(method) || ismissing(method) || ~any(method == ["pca" "lda"])
    error('genderRecognition:InvalidBayesAnalysisMethod', 'Select "pca" or "lda".');
end
if ~isnumeric(foldNumber) || ~isscalar(foldNumber) || ~isreal(foldNumber) || ...
        issparse(foldNumber) || ~isfinite(foldNumber) || foldNumber < 1 || foldNumber ~= fix(foldNumber)
    error('genderRecognition:InvalidFoldNumber', 'Fold number must be a positive integer.');
end
try
    [images, y, subjectIDs, metadata] = loadDataset(trainingPath);
catch cause
    if startsWith(cause.identifier, 'genderRecognition:')
        error(cause.identifier, ...
            'Training data could not be loaded (%s). Check the training folder locally.', cause.identifier);
    end
    rethrow(cause);
end

[folds, partitionInfo] = createFolds(y, subjectIDs);
if foldNumber > numel(folds)
    error('genderRecognition:InvalidFoldNumber', 'Select a fold from 1 through %d.', numel(folds));
end
trainIdx = folds(foldNumber).TrainIndices;
validationIdx = folds(foldNumber).ValidationIndices;
yTrain = y(trainIdx);
preprocessor = fitPreprocessor(images(trainIdx), preprocessingConfig);
XTrain = transformImages(preprocessor, images(trainIdx));
if method == "pca"
    projection = fitPCA(XTrain, featureConfig);
    transform = @transformPCA;
else
    projection = fitLDA(XTrain, yTrain, featureConfig);
    transform = @transformLDA;
end
ZTrain = transform(projection, XTrain);

fitTimer = tic;
bayesModel = fitBayes(ZTrain, yTrain, bayesConfig);
fitSeconds = toc(fitTimer);
XValidation = transformImages(preprocessor, images(validationIdx));
ZValidation = transform(projection, XValidation);
predictTimer = tic;
[yPred, scores] = predictBayes(bayesModel, ZValidation);
predictSeconds = toc(predictTimer);

stats.Method = method;
stats.FoldNumber = foldNumber;
stats.DatasetClassCounts = [nnz(y == 0) nnz(y == 1)];
stats.NumTrainingSamples = bayesModel.NumTrainingSamples;
stats.TrainClassCounts = bayesModel.ClassCounts;
yValidation = y(validationIdx);
stats.ValidationClassCounts = [nnz(yValidation == 0) nnz(yValidation == 1)];
stats.SubjectOverlapCount = partitionInfo.SubjectOverlapCounts(foldNumber);
stats.OriginalImageSizes = unique(metadata{:, {'Height', 'Width', 'Channels'}}, 'rows');
stats.ProcessedImageSize = preprocessor.OutputImageSize;
stats.TrainFeatureSize = size(XTrain);
stats.ValidationFeatureSize = size(XValidation);
stats.NumericType = string(class(XTrain));
stats.TrainIntensityRange = [min(XTrain, [], 'all') max(XTrain, [], 'all')];
stats.ReducedTrainSize = size(ZTrain);
stats.ReducedValidationSize = size(ZValidation);
stats.ClassMeanSize = size(bayesModel.ClassMeans);
stats.CovarianceSize = [size(bayesModel.Covariances, [1 2]) size(bayesModel.Covariances, 3)];
stats.CovarianceType = bayesModel.CovarianceType;
stats.PriorType = bayesModel.Options.PriorType;
stats.Priors = bayesModel.Priors;
stats.CovarianceRanks = bayesModel.CovarianceRanks;
stats.CovarianceScale = bayesModel.CovarianceScale;
stats.CovarianceScaleSource = bayesModel.CovarianceScaleSource;
stats.RegularizationCoefficient = bayesModel.RegularizationCoefficient;
stats.EffectiveRegularization = bayesModel.EffectiveRegularization;
stats.NumClassifierParameters = bayesModel.NumClassifierParameters;
stats.NumProjectionParameters = projection.NumFittedParameters;
stats.PredictionSize = size(yPred);
stats.ScoreSize = size(scores);
stats.ScoreRange = [min(scores) max(scores)];
stats.NumFiniteScores = nnz(isfinite(scores));
stats.AllScoresFinite = all(isfinite(scores));
stats.NumDistinctScores = numel(unique(scores));
stats.PredictedClassCounts = [nnz(yPred == 0) nnz(yPred == 1)];
stats.FitSeconds = fitSeconds;
stats.PredictSeconds = predictSeconds;

fprintf('Bayes with %s features, subject-grouped fold %d\n', upper(method), foldNumber);
fprintf('Dataset counts [male female]: %d %d\n', stats.DatasetClassCounts);
fprintf('Training: %d male, %d female; validation: %d male, %d female; subject overlap: %d\n', ...
    stats.TrainClassCounts, stats.ValidationClassCounts, stats.SubjectOverlapCount);
fprintf('Preprocessed training shape: %d x %d (%s); intensity range [%.9g, %.9g]\n', ...
    stats.TrainFeatureSize, stats.NumericType, stats.TrainIntensityRange);
fprintf('Reduced training shape: %d x %d; reduced validation shape: %d x %d\n', ...
    stats.ReducedTrainSize, stats.ReducedValidationSize);
fprintf('Class means: %d x %d; %s covariance storage: %d x %d x %d\n', ...
    stats.ClassMeanSize, stats.CovarianceType, stats.CovarianceSize);
fprintf('Priors (%s) [male female]: %.9g %.9g\n', stats.PriorType, stats.Priors);
fprintf('Unregularized covariance ranks: %s; all %d covariance factorizations succeeded\n', ...
    strjoin(string(stats.CovarianceRanks), ', '), stats.CovarianceSize(3));
fprintf('Covariance scale %.9g (%s); rho %.9g; effective ridge %.9g\n', ...
    stats.CovarianceScale, stats.CovarianceScaleSource, ...
    stats.RegularizationCoefficient, stats.EffectiveRegularization);
fprintf('Gaussian classifier parameters: %d; fitted projection coefficients: %d\n', ...
    stats.NumClassifierParameters, stats.NumProjectionParameters);
fprintf('Prediction shape: %d x %d; score shape: %d x %d\n', stats.PredictionSize, stats.ScoreSize);
fprintf('Finite scores: %d/%d; distinct scores: %d; range [%.9g, %.9g]\n', ...
    stats.NumFiniteScores, numel(scores), stats.NumDistinctScores, stats.ScoreRange);
fprintf('Predicted counts [male female]: %d %d\n', stats.PredictedClassCounts);
fprintf('Classifier fit: %.6f s; batch prediction: %.6f s\n', stats.FitSeconds, stats.PredictSeconds);
end
