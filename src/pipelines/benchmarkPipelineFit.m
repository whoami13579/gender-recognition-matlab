function [pipeline, timings] = benchmarkPipelineFit(images, labels, method, config, benchmark)
%BENCHMARKPIPELINEFIT Time fresh fits and retain the first measured full fit.
[timings.PipelineFit, pipeline] = benchmarkOperation( ...
    @() fitPipeline(images, labels, method, config), benchmark);
[project, ~, ~, fitClassifier] = pipelineOperations(method, labels, config);
features = transformImages(pipeline.Preprocessor, images);
projected = project(pipeline.Projection, features);
[timings.ClassifierFit, ~] = benchmarkOperation(@() fitClassifier(projected), benchmark);
end
