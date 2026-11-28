function [project, predict, fitProjection, fitClassifier] = pipelineOperations(method, y, config)
% Training and benchmarking share the same stage operations as prediction.
if ~((ischar(method) && isrow(method)) || (isstring(method) && isscalar(method))) || ...
        ismissing(string(method)) || ~any(string(method) == pipelineMethods())
    error('genderRecognition:InvalidPipelineMethod', 'Expected pca-bayes, lda-bayes, pca-mlp or lda-mlp.');
end
if startsWith(method, "pca-")
    project = @transformPCA;
else
    project = @transformLDA;
end
if endsWith(method, "-bayes")
    predict = @predictBayes;
else
    predict = @predictMLP;
end
if nargin == 1
    return
end
if startsWith(method, "pca-")
    fitProjection = @(X) fitPCA(X, config.PCA);
else
    fitProjection = @(X) fitLDA(X, y, config.LDA);
end
if endsWith(method, "-bayes")
    fitClassifier = @(Z) fitBayes(Z, y, config.Bayes);
else
    fitClassifier = @(Z) fitMLP(Z, y, config.MLP);
end
end
