function validatePipeline(pipeline)
% Verify fitted dimensions, calculated counts, and internal option agreement.
try
    assert(isstruct(pipeline) && isscalar(pipeline) && ...
        isequal(sort(string(fieldnames(pipeline))), sort(["Method"; "Preprocessor"; ...
        "Projection"; "Classifier"; "Configuration"; "Parameters"; "NumTrainingSamples"])));
    prep = pipeline.Preprocessor;
    projection = pipeline.Projection;
    classifier = pipeline.Classifier;
    params = summarizeModelParameters(prep, projection, classifier);
    channels = prep.OutputImageSize(3);
    assert(isequal(params, pipeline.Parameters) && numel(prep.OutputImageSize) == 3 && ...
        all(isfinite(prep.OutputImageSize) & prep.OutputImageSize > 0 & ...
        prep.OutputImageSize == fix(prep.OutputImageSize)) && ...
        prod(prep.OutputImageSize) == params.NumOriginalFeatures && any(channels == [1 3]) && ...
        (~prep.Options.Grayscale || channels == 1) && ...
        (isempty(prep.Options.ImageSize) || isequal(prep.Options.ImageSize, prep.OutputImageSize(1:2))) && ...
        params.NumPreprocessingParameters == 0);
    N = pipeline.NumTrainingSamples;
    assert(isscalar(N) && isfinite(N) && N == fix(N) && N >= 2 && ...
        N == classifier.NumTrainingSamples && N == projection.NumTrainingSamples && ...
        isequal(size(classifier.ClassCounts), [1 2]) && all(classifier.ClassCounts >= 1) && ...
        all(classifier.ClassCounts == fix(classifier.ClassCounts)) && sum(classifier.ClassCounts) == N);
    config = struct('Preprocessing', prep.Options, 'Projection', projection.Options, ...
        'Classifier', classifier.Options);
    assert(isequal(config, pipeline.Configuration));
    method = pipeline.Method;
    assert(isscalar(method) && any(method == ["pca-bayes", "lda-bayes", "pca-mlp", "lda-mlp"]));
    assert(params.NumProjectionParameters == numel(projection.Mean) + numel(projection.Components));
    if startsWith(method, "pca-")
        assert(isfield(projection.Options, 'SelectionMode') && ...
            (projection.Options.SelectionMode ~= "fixed" || projection.Options.NumComponents == params.NumFeatures));
    else
        assert(isfield(projection.Options, 'Regularization') && ...
            isequal(projection.ClassCounts, classifier.ClassCounts));
    end
    if endsWith(method, "-mlp")
        assert(isequal(classifier.LayerSizes(2:end-1), classifier.Options.HiddenLayerSizes) && ...
            classifier.Activation == classifier.Options.Activation && ...
            params.NumNormalizationParameters == 2 * params.NumFeatures * classifier.Options.Standardize);
        count = sum(cellfun(@numel, classifier.Weights)) + sum(cellfun(@numel, classifier.Biases));
    else
        K = params.NumFeatures;
        assert(classifier.CovarianceType == classifier.Options.CovarianceType);
        switch classifier.CovarianceType
            case "full"
                covariance = K * (K + 1);
            case "shared"
                covariance = K * (K + 1) / 2;
            case "diagonal"
                covariance = 2 * K;
        end
        count = 2 * K + covariance + double(classifier.Options.PriorType == "empirical");
    end
    assert(params.NumClassifierParameters == count);
    % Invoke module validators through their usual prediction path with one
    % synthetic image carrying the expected input shape.
    image = zeros([prep.OutputImageSize(1:2), channels], 'uint8');
    [label, score] = predictPipeline(pipeline, {image});
    assert(isscalar(score) && isfinite(score) && isequal(label, double(score > 0)));
catch cause
    failure = MException('genderRecognition:InvalidPipelineArtifact', ...
        'The fitted pipeline has invalid state, dimensions, options, or parameter counts.');
    throwAsCaller(addCause(failure, cause));
end
end
