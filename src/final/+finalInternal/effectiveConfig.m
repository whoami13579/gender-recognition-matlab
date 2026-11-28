function effective = effectiveConfig(row)
% Translate input options into the effective options stored by the fit modules.
config = row.Configuration;
if startsWith(row.Method, "pca-")
    projection = struct('SelectionMode', "fixed", ...
        'NumComponents', config.PCA.NumComponents, 'ExplainedVarianceThreshold', []);
else
    projection = config.LDA;
end
if endsWith(row.Method, "-bayes")
    classifier = config.Bayes;
else
    classifier = config.MLP;
    classifier.Initialization = "glorotUniform";
    classifier.RandomStream = "mt19937ar";
end
effective = struct('Preprocessing', config.Preprocessing, ...
    'Projection', projection, 'Classifier', classifier);
end
