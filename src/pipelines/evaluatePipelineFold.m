function [record, predictions, pipeline] = evaluatePipelineFold( ...
        imagesTrain, yTrain, imagesValidation, yValidation, method, foldNumber, config, benchmark)
%EVALUATEPIPELINEFOLD Profile a fresh fit and a frozen inference on explicit splits.
config = mergePipelineConfig(defaultPipelineConfig(), config);
[timings.PipelineFit, pipeline] = benchmarkOperation( ...
    @() fitPipeline(imagesTrain, yTrain, method, config), benchmark);
[project, predict, fitProjection, fitClassifier] = pipelineOperations(method, yTrain, config);
prep = pipeline.Preprocessor;
projection = pipeline.Projection;
classifier = pipeline.Classifier;
[timings.PreprocessingFit, ~] = benchmarkOperation( ...
    @() fitPreprocessor(imagesTrain, config.Preprocessing), benchmark);
[timings.TrainingPreprocessing, X] = benchmarkOperation(@() transformImages(prep, imagesTrain), benchmark);
[timings.ProjectionFit, ~] = benchmarkOperation(@() fitProjection(X), benchmark);
[timings.TrainingProjection, Z] = benchmarkOperation(@() project(projection, X), benchmark);
[timings.ClassifierFit, ~] = benchmarkOperation(@() fitClassifier(Z), benchmark);

batch = benchmark;
batch.BatchSize = numel(imagesValidation);
[timings.PipelineInference, yPred, scores] = benchmarkOperation( ...
    @() predictPipeline(pipeline, imagesValidation), batch);
[timings.ValidationPreprocessing, X] = benchmarkOperation( ...
    @() transformImages(prep, imagesValidation), benchmark);
[timings.ValidationProjection, Z] = benchmarkOperation(@() project(projection, X), benchmark);
[timings.ClassifierInference, stageLabels, stageScores] = benchmarkOperation( ...
    @() predict(classifier, Z), batch);
assert(isequal(yPred, stageLabels) && isequal(scores, stageScores), ...
    'genderRecognition:InconsistentEvaluationPrediction', 'The complete and staged predictions differ.');
metrics = evaluateClassification(yValidation, yPred, scores);
context = struct('Method', method, 'FoldNumber', foldNumber, 'EvaluationStage', "cross-validation");
record = buildEvaluationResult(context, metrics, prep, projection, classifier, timings);
predictions = struct('Method', method, 'FoldNumber', foldNumber, ...
    'Labels', yValidation, 'Predictions', yPred, 'Scores', scores);
end
