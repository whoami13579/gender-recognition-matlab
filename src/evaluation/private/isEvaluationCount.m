function valid = isEvaluationCount(value, minimum)
%ISEVALUATIONCOUNT Check for an exactly representable count.
valid = isa(value, 'double') && isscalar(value) && isreal(value) && ~issparse(value) && ...
    isfinite(value) && value >= minimum && value <= flintmax && value == fix(value);
end
