function results = run_evaluation_analysis(outputDirectory, showPlot)
%RUN_EVALUATION_ANALYSIS Verify the common evaluation interface on synthetic data.
%   RESULTS = RUN_EVALUATION_ANALYSIS(DIRECTORY, true) also exports and plots.
%   By default, this demonstration writes no files and creates no figure.
if nargin < 1
    outputDirectory = "";
elseif ischar(outputDirectory) && isrow(outputDirectory)
    outputDirectory = string(outputDirectory);
end
if nargin < 2
    showPlot = false;
end
if ~isstring(outputDirectory) || ~isscalar(outputDirectory) || ismissing(outputDirectory) || ...
        ~islogical(showPlot) || ~isscalar(showPlot)
    error('genderRecognition:InvalidConfig', 'Supply a directory string and a logical plot flag.');
end

scores = [0; 1; 1; 2];
tied = evaluateClassification([0; 0; 1; 1], double(scores > 0), scores);
matrix = tied.ConfusionMatrix;
fprintf('Mixed ties: accuracy %.1f%%, TN %d, FP %d, FN %d, TP %d, ROC %s, AUC %.3f\n', ...
    100 * tied.Accuracy, matrix(1, 1), matrix(1, 2), matrix(2, 1), matrix(2, 2), tied.ROC.Status, tied.AUC);
singleClass = evaluateClassification([1; 1], [0; 1], [0; 2]);
fprintf('Single class: accuracy %.1f%%, ROC %s, AUC undefined\n', ...
    100 * singleClass.Accuracy, singleClass.ROC.Status);

stream = RandStream('mt19937ar', 'Seed', 10);
yTrain = [zeros(12, 1); ones(12, 1)];
yValidation = [zeros(4, 1); ones(4, 1)];
trainingImages = makeImages(stream, yTrain);
validationImages = makeImages(stream, yValidation);
methods = ["pca-bayes", "lda-bayes", "pca-mlp", "lda-mlp"];
records = cell(size(methods));
fprintf('\nSynthetic batches: %d training, %d validation samples\n', numel(yTrain), numel(yValidation));
for idx = 1:numel(methods)
    row = evaluateMethod(methods(idx), trainingImages, yTrain, validationImages, yValidation);
    records{idx} = row;
    fprintf(['%-9s accuracy %5.1f%%, AUC %.3f, features %d -> %d, ', ...
        'parameters (prep/projection/normalization/classifier/total) %d/%d/%d/%d/%d, ', ...
        'median pipeline fit %.6f s, batch inference %.6f s\n'], ...
        row.Method, 100 * row.Accuracy, row.AUC, row.NumOriginalFeatures, row.NumFeatures, ...
        row.NumPreprocessingParameters, row.NumProjectionParameters, row.NumNormalizationParameters, ...
        row.NumClassifierParameters, row.CombinedParameterCount, row.Timing.Training.Pipeline.MedianSeconds, ...
        row.Timing.Inference.Pipeline.MedianSeconds);
end
results = [records{:}];
if strlength(outputDirectory) > 0
    exportEvaluationResults(results, outputDirectory);
end
if showPlot
    plotROCCurves(results);
end
end

function images = makeImages(stream, labels)
images = cell(numel(labels), 1);
for idx = 1:numel(images)
    images{idx} = 0.5 * rand(stream, 3, 4) + 0.35 * labels(idx);
end
end

function result = evaluateMethod(method, trainingImages, yTrain, validationImages, yValidation)
prepConfig = struct('ImageSize', []);
if startsWith(method, "pca-")
    pcaConfig = struct('NumComponents', 3);
    fitProjection = @(X) fitPCA(X, pcaConfig);
    project = @transformPCA;
else
    fitProjection = @(X) fitLDA(X, yTrain);
    project = @transformLDA;
end
if endsWith(method, "-bayes")
    fitClassifier = @(Z) fitBayes(Z, yTrain);
    predict = @predictBayes;
else
    mlpConfig = struct('MaxEpochs', 20, 'RandomSeed', 11);
    fitClassifier = @(Z) fitMLP(Z, yTrain, mlpConfig);
    predict = @predictMLP;
end
fitPrep = @() fitPreprocessor(trainingImages, prepConfig);
trainPipeline = @() fitInMemory(fitPrep, trainingImages, fitProjection, project, fitClassifier);
[timings.PipelineFit, models] = benchmarkOperation(trainPipeline);

% Profile the component scopes separately around the first measured pipeline's stages.
[timings.PreprocessingFit, ~] = benchmarkOperation(fitPrep);
[timings.TrainingPreprocessing, XTrain] = benchmarkOperation(@() transformImages(models.Preprocessor, trainingImages));
[timings.ProjectionFit, ~] = benchmarkOperation(@() fitProjection(XTrain));
[timings.TrainingProjection, ZTrain] = benchmarkOperation(@() project(models.Projection, XTrain));
[timings.ClassifierFit, ~] = benchmarkOperation(@() fitClassifier(ZTrain));

batchConfig = struct('BatchSize', numel(yValidation));
predictPipeline = @() predictInMemory(models, validationImages, project, predict);
[timings.PipelineInference, yPred, scores] = benchmarkOperation(predictPipeline, batchConfig);
[timings.ValidationPreprocessing, XValidation] = ...
    benchmarkOperation(@() transformImages(models.Preprocessor, validationImages));
[timings.ValidationProjection, ZValidation] = benchmarkOperation(@() project(models.Projection, XValidation));
[timings.ClassifierInference, classifierLabels, classifierScores] = ...
    benchmarkOperation(@() predict(models.Classifier, ZValidation), batchConfig);
assert(isequal(yPred, classifierLabels) && isequal(scores, classifierScores), ...
    'genderRecognition:InconsistentEvaluationPrediction', 'The inference scopes returned different predictions.');

[timings.MetricCalculation, metrics] = benchmarkOperation(@() evaluateClassification(yValidation, yPred, scores));
context = struct('Method', method, 'EvaluationStage', "synthetic");
result = buildEvaluationResult(context, metrics, models.Preprocessor, models.Projection, models.Classifier, timings);
end

function models = fitInMemory(fitPrep, images, fitProjection, project, fitClassifier)
prep = fitPrep();
X = transformImages(prep, images);
projection = fitProjection(X);
Z = project(projection, X);
classifier = fitClassifier(Z);
models = struct('Preprocessor', prep, 'Projection', projection, 'Classifier', classifier);
end

function [yPred, scores] = predictInMemory(models, images, project, predict)
X = transformImages(models.Preprocessor, images);
Z = project(models.Projection, X);
[yPred, scores] = predict(models.Classifier, Z);
end
