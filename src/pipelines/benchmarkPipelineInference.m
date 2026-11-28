function [labels, scores, timings] = benchmarkPipelineInference(pipeline, images, benchmark)
%BENCHMARKPIPELINEINFERENCE Time the fixed pipeline and the classifier in memory.
benchmark.BatchSize = numel(images);
[timings.PipelineInference, labels, scores] = benchmarkOperation( ...
    @() predictPipeline(pipeline, images), benchmark);
[project, predict] = pipelineOperations(pipeline.Method);
features = transformImages(pipeline.Preprocessor, images);
projected = project(pipeline.Projection, features);
[timings.ClassifierInference, stageLabels, stageScores] = benchmarkOperation( ...
    @() predict(pipeline.Classifier, projected), benchmark);
assert(isequal(labels, stageLabels) && isequal(scores, stageScores), ...
    'genderRecognition:InconsistentEvaluationPrediction', 'The complete and staged predictions differ.');
end
