function [timing, varargout] = benchmarkOperation(operation, config)
%BENCHMARKOPERATION Time a zero-input operation, with consistent output arity.
%   [timing, outputs...] returns the first measured call's outputs.
%   Use an operation that performs a fresh fit or uses a fixed inference model.
%   Options are WarmupRuns (2), NumRepeats (5), and optional BatchSize ([]).
if nargin < 2
    config = struct();
end
options = struct('WarmupRuns', 2, 'NumRepeats', 5, 'BatchSize', []);
if ~isa(operation, 'function_handle') || ~isscalar(operation)
    error('genderRecognition:InvalidBenchmarkOperation', 'Expected a scalar function handle.');
end
if ~isstruct(config) || ~isscalar(config) || ~all(ismember(fieldnames(config), fieldnames(options)))
    error('genderRecognition:InvalidBenchmarkConfig', 'Expected a scalar struct of supported benchmark options.');
end
for field = string(fieldnames(config)).'
    options.(field) = config.(field);
end
if ~isEvaluationCount(options.WarmupRuns, 0) || ~isEvaluationCount(options.NumRepeats, 1) || ...
        ~(isequal(options.BatchSize, []) || isEvaluationCount(options.BatchSize, 1))
    error('genderRecognition:InvalidBenchmarkConfig', ...
        'WarmupRuns, NumRepeats and optional BatchSize must be valid integer counts.');
end

numOutputs = max(0, nargout - 1);
outputs = cell(1, numOutputs);
samples = zeros(options.NumRepeats, 1);
for run = 1:options.WarmupRuns
    [outputs{:}] = operation();
    outputs(:) = {[]};
end
for run = 1:options.NumRepeats
    started = tic;
    [outputs{:}] = operation();
    samples(run) = toc(started);
    if run == 1
        varargout = outputs;
    end
    outputs(:) = {[]};
end
timing = summarizeTimingSamples(samples, options.WarmupRuns, options.BatchSize);
end
