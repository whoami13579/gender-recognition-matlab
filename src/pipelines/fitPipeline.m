function pipeline = fitPipeline(imagesTrain, yTrain, method, config)
%FITPIPELINE Fit preprocessing, projection, and classification on one batch.
arguments
    imagesTrain
    yTrain
    method
    config = struct()
end
config = mergePipelineConfig(defaultPipelineConfig(), config);
[project, ~, fitProjection, fitClassifier] = pipelineOperations(method, yTrain, config);
prep = fitPreprocessor(imagesTrain, config.Preprocessing);
X = transformImages(prep, imagesTrain);
projection = fitProjection(X);
classifier = fitClassifier(project(projection, X));
parameters = summarizeModelParameters(prep, projection, classifier);
effective = struct('Preprocessing', prep.Options, 'Projection', projection.Options, ...
    'Classifier', classifier.Options);
pipeline = struct('Method', string(method), 'Preprocessor', prep, ...
    'Projection', projection, 'Classifier', classifier, 'Configuration', effective, ...
    'Parameters', parameters, 'NumTrainingSamples', classifier.NumTrainingSamples);
end
