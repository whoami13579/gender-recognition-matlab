function config = defaultPipelineConfig
%DEFAULTPIPELINECONFIG Fixed reference settings for all four methods.
config.Preprocessing = struct('Grayscale', true, 'ImageSize', [64 64], ...
    'OutputType', "double", 'IntensityScaling', "unit");
config.PCA = struct('NumComponents', 40);
config.LDA = struct('Regularization', 1e-2);
config.Bayes = struct('CovarianceType', "full", 'PriorType', "empirical", 'Regularization', 1e-2);
config.MLP = struct('HiddenLayerSizes', 16, 'Activation', "tanh", 'LearningRate', 0.01, ...
    'Momentum', 0.9, 'MaxEpochs', 300, 'BatchSize', 16, ...
    'L2Regularization', 1e-4, 'Standardize', true, 'RandomSeed', 42);
end
