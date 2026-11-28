function valid = hasEvaluationFields(value, fields)
%HASEVALUATIONFIELDS Check the exact fields of a scalar schema struct.
valid = isstruct(value) && isscalar(value) && ...
    numel(fieldnames(value)) == numel(fields) && all(isfield(value, fields));
end
