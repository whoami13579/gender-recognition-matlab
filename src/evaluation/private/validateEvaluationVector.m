function values = validateEvaluationVector(values, name, areLabels)
%VALIDATEEVALUATIONVECTOR Check the shared binary metric input contract.
supported = isa(values, 'double') || isa(values, 'single') || (areLabels && islogical(values));
if ~supported || isempty(values) || ~iscolumn(values) || ~isreal(values) || issparse(values)
    error('genderRecognition:InvalidEvaluationInput', ...
        '%s must be a nonempty, real, full column with a supported data type.', name);
end
if any(~isfinite(values)) || (areLabels && any(values ~= 0 & values ~= 1))
    error('genderRecognition:InvalidEvaluationInput', ...
        '%s must contain finite values and, for labels, only 0 and 1.', name);
end
values = double(values);
end
