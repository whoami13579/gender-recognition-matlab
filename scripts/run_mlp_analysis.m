function stats = run_mlp_analysis(trainingPath, method, mlpConfig, featureConfig, preprocessingConfig, foldNumber, plotLearningCurve)
%RUN_MLP_ANALYSIS Report aggregate MLP diagnostics for one training fold.
%   Run SETUP first. METHOD defaults to "pca", FOLDNUMBER to 1, and
%   PLOTLEARNINGCURVE to false. This function writes no files.
arguments
    trainingPath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'train')
    method = "pca"
    mlpConfig = struct()
    featureConfig = struct()
    preprocessingConfig = struct()
    foldNumber = 1
    plotLearningCurve = false
end

if ischar(method) && isrow(method)
    method = string(method);
end
if ~isstring(method) || ~isscalar(method) || ismissing(method) || ~any(method == ["pca" "lda"])
    error('genderRecognition:InvalidMLPAnalysisMethod', 'Select "pca" or "lda".');
end
if ~isnumeric(foldNumber) || ~isreal(foldNumber) || ~isscalar(foldNumber) || ...
        issparse(foldNumber) || ~isfinite(foldNumber) || foldNumber < 1 || foldNumber ~= fix(foldNumber)
    error('genderRecognition:InvalidFoldNumber', 'Fold number must be a positive integer.');
end
if ~islogical(plotLearningCurve) || ~isscalar(plotLearningCurve) || issparse(plotLearningCurve)
    error('genderRecognition:InvalidMLPAnalysisPlot', 'The learning curve option must be a logical scalar.');
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

[folds, partition] = createFolds(y, subjectIDs);
if foldNumber > numel(folds)
    error('genderRecognition:InvalidFoldNumber', 'Select a fold from 1 through %d.', numel(folds));
end
train = folds(foldNumber).TrainIndices;
validation = folds(foldNumber).ValidationIndices;
yTrain = y(train);
prep = fitPreprocessor(images(train), preprocessingConfig);
XTrain = transformImages(prep, images(train));
if method == "pca"
    projection = fitPCA(XTrain, featureConfig);
    transform = @transformPCA;
else
    projection = fitLDA(XTrain, yTrain, featureConfig);
    transform = @transformLDA;
end
ZTrain = transform(projection, XTrain);

fitTimer = tic;
model = fitMLP(ZTrain, yTrain, mlpConfig);
fitSeconds = toc(fitTimer);
XValidation = transformImages(prep, images(validation));
ZValidation = transform(projection, XValidation);
predictTimer = tic;
[yPred, scores] = predictMLP(model, ZValidation);
predictSeconds = toc(predictTimer);

stats.Method = method;
stats.FoldNumber = foldNumber;
stats.DatasetClassCounts = [nnz(y == 0) nnz(y == 1)];
stats.NumTrainingSamples = model.NumTrainingSamples;
stats.NumValidationSamples = numel(validation);
stats.TrainClassCounts = model.ClassCounts;
stats.ValidationClassCounts = [nnz(y(validation) == 0) nnz(y(validation) == 1)];
stats.SubjectOverlapCount = partition.SubjectOverlapCounts(foldNumber);
stats.OriginalImageSizes = unique(metadata{:, {'Height', 'Width', 'Channels'}}, 'rows');
stats.ProcessedImageSize = prep.OutputImageSize;
stats.TrainFeatureSize = size(XTrain);
stats.ValidationFeatureSize = size(XValidation);
stats.NumericType = string(class(XTrain));
stats.TrainIntensityRange = [min(XTrain, [], 'all') max(XTrain, [], 'all')];
stats.ReducedTrainSize = size(ZTrain);
stats.ReducedValidationSize = size(ZValidation);
stats.LayerSizes = model.LayerSizes;
stats.Options = model.Options;
stats.EpochsCompleted = model.EpochsCompleted;
stats.StoppingReason = model.StoppingReason;
stats.InitialTrainingLoss = model.InitialTrainingLoss;
stats.FinalTrainingLoss = model.FinalTrainingLoss;
stats.InitialTrainingObjective = model.InitialTrainingObjective;
stats.FinalTrainingObjective = model.FinalTrainingObjective;
stats.NumClassifierParameters = model.NumClassifierParameters;
stats.NumNormalizationParameters = model.NumNormalizationParameters;
stats.NumProjectionParameters = projection.NumFittedParameters;
stats.PredictionSize = size(yPred);
stats.ScoreSize = size(scores);
stats.ScoreRange = [min(scores) max(scores)];
stats.NumFiniteScores = nnz(isfinite(scores));
stats.NumDistinctScores = numel(unique(scores));
stats.PredictedClassCounts = [nnz(yPred == 0) nnz(yPred == 1)];
stats.FitSeconds = fitSeconds;
stats.PredictSeconds = predictSeconds;

fprintf('MLP with %s features, subject-grouped fold %d\n', upper(method), foldNumber);
fprintf('Training %d [%d male, %d female]; validation %d [%d male, %d female]; subject overlap %d\n', ...
    stats.NumTrainingSamples, stats.TrainClassCounts, stats.NumValidationSamples, ...
    stats.ValidationClassCounts, stats.SubjectOverlapCount);
fprintf('Reduced training shape %d x %d; reduced validation shape %d x %d\n', ...
    stats.ReducedTrainSize, stats.ReducedValidationSize);
opts = stats.Options;
fprintf('Layers %s; activation %s; initialization %s; seed %d; standardization %d\n', ...
    strjoin(string(stats.LayerSizes), ' -> '), opts.Activation, opts.Initialization, opts.RandomSeed, opts.Standardize);
fprintf('Learning rate %.9g; momentum %.9g; batch size %d; L2 %.9g; epochs %d/%d (%s)\n', ...
    opts.LearningRate, opts.Momentum, opts.BatchSize, opts.L2Regularization, ...
    stats.EpochsCompleted, opts.MaxEpochs, stats.StoppingReason);
fprintf('Training BCE %.9g -> %.9g; regularized objective %.9g -> %.9g\n', ...
    stats.InitialTrainingLoss, stats.FinalTrainingLoss, stats.InitialTrainingObjective, stats.FinalTrainingObjective);
fprintf('Classifier parameters %d; normalization coefficients %d; projection coefficients %d\n', ...
    stats.NumClassifierParameters, stats.NumNormalizationParameters, stats.NumProjectionParameters);
fprintf('Prediction shape %d x %d; score shape %d x %d\n', stats.PredictionSize, stats.ScoreSize);
fprintf('Finite scores %d/%d; distinct scores %d; range [%.9g, %.9g]\n', ...
    stats.NumFiniteScores, numel(scores), stats.NumDistinctScores, stats.ScoreRange);
fprintf('Predicted counts [male female] %d %d\n', stats.PredictedClassCounts);
fprintf('Classifier fit %.6f s; batch prediction %.6f s\n', stats.FitSeconds, stats.PredictSeconds);

if plotLearningCurve
    figure('Name', 'MLP training learning curve');
    epochs = 0:model.EpochsCompleted;
    plot(epochs, [model.TrainingLossHistory model.TrainingObjectiveHistory]);
    xlabel('Epoch');
    ylabel('Training loss');
    legend('Binary cross-entropy', 'Regularized objective');
end
end
