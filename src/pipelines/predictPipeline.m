function [yPred, scores] = predictPipeline(pipeline, imagesEvaluation)
%PREDICTPIPELINE Apply a frozen pipeline in row order without evaluation labels.
[project, predict] = pipelineOperations(pipeline.Method);
X = transformImages(pipeline.Preprocessor, imagesEvaluation);
Z = project(pipeline.Projection, X);
[yPred, scores] = predict(pipeline.Classifier, Z);
end
