function candidates = pipelineCandidates(method, config, search)
% A baseline followed by stable, one-factor-at-a-time alternatives.
key = replace(method, "-", "_");
if isfield(search.Candidates, key)
    changes = search.Candidates.(key);
    if ~iscell(changes) || (~isempty(changes) && ~isvector(changes))
        error('genderRecognition:InvalidCVConfig', 'Candidate overrides for %s must be a cell vector.', method);
    end
else
    if startsWith(method, "pca-")
        changes = variants("PCA", "NumComponents", {10, 20, 40});
    else
        changes = variants("LDA", "Regularization", {0.001, 0.01, 0.1, 1});
    end
    if endsWith(method, "-bayes")
        changes = [changes, variants("Bayes", "CovarianceType", {"full", "shared", "diagonal"}), ...
            variants("Bayes", "Regularization", {0.001, 0.01, 0.1})];
    else
        changes = [changes, variants("MLP", "HiddenLayerSizes", {8, 16, 32}), ...
            variants("MLP", "LearningRate", {0.005, 0.01, 0.02}), ...
            variants("MLP", "MaxEpochs", {100, 300, 500}), ...
            variants("MLP", "L2Regularization", {0.0001, 0.001, 0.01})];
    end
end
candidates = {config};
for change = changes(:).'
    base = config;
    % A fixed count replaces a variance-threshold retention rule.
    if isstruct(change{1}) && isscalar(change{1}) && isfield(change{1}, 'PCA') && ...
            isstruct(change{1}.PCA) && isfield(change{1}.PCA, 'NumComponents')
        base.PCA = struct('NumComponents', 40);
    end
    candidate = mergePipelineConfig(base, change{1});
    if ~isequaln(candidate.Preprocessing, config.Preprocessing)
        error('genderRecognition:InvalidCVConfig', 'Search candidates must share the same preprocessing options.');
    end
    if ~any(cellfun(@(prior) isequaln(prior, candidate), candidates))
        candidates{end + 1} = candidate; %#ok<AGROW>
    end
end
end

function changes = variants(group, field, values)
changes = cellfun(@(value) struct(group, struct(field, value)), values, 'UniformOutput', false);
end
