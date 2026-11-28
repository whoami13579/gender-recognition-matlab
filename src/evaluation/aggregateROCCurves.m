function average = aggregateROCCurves(curves)
%AGGREGATEROCCURVES Average valid fold ROC paths with equal weight.
%   Invalid folds are indexed explicitly. Coordinates have no thresholds.
if ~isstruct(curves) || (~isempty(curves) && (~isvector(curves) || ...
        ~all(isfield(curves, {'Valid', 'FPR', 'TPR'}))))
    error('genderRecognition:InvalidROCCurves', 'Supply a vector of fold ROC structs.');
end
valid = false(numel(curves), 1);
breakpoints = cell(size(valid));
incoming = breakpoints;
outgoing = breakpoints;
for k = 1:numel(curves)
    curve = curves(k);
    x = curve.FPR(:);
    y = curve.TPR(:);
    if ~isequal(curve.Valid, true) || numel(x) < 2 || numel(y) ~= numel(x) || ...
            ~isreal(x) || ~isreal(y) || any(~isfinite(x) | ~isfinite(y)) || ...
            x(1) ~= 0 || y(1) ~= 0 || x(end) ~= 1 || y(end) ~= 1 || ...
            ~issorted(x) || ~issorted(y)
        continue
    end
    first = [true; diff(x) > 0];
    last = [diff(x) > 0; true];
    breakpoints{k} = x(first);
    incoming{k} = y(first);
    outgoing{k} = y(last);
    valid(k) = true;
end
average = struct('FPR', zeros(0, 1), 'TPR', zeros(0, 1), 'Area', NaN, ...
    'NumValidCurves', nnz(valid), 'InvalidCurveIndices', find(~valid));
if ~any(valid)
    return
end
grid = unique(vertcat(breakpoints{valid}));
lo = zeros(numel(grid), nnz(valid));
hi = lo;
folds = find(valid);
for col = 1:numel(folds)
    k = folds(col);
    x = breakpoints{k};
    lower = incoming{k};
    upper = outgoing{k};
    [atVertex, vertex] = ismember(grid, x);
    lo(atVertex, col) = lower(vertex(atVertex));
    hi(atVertex, col) = upper(vertex(atVertex));
    between = ~atVertex;
    left = discretize(grid(between), x);
    fraction = (grid(between) - x(left)) ./ (x(left + 1) - x(left));
    value = upper(left) + fraction .* (lower(left + 1) - upper(left));
    lo(between, col) = value;
    hi(between, col) = value;
end
x = repelem(grid, 2);
y = reshape([mean(lo, 2), mean(hi, 2)].', [], 1);
keep = [true; diff(x) ~= 0 | diff(y) ~= 0];
average.FPR = x(keep);
average.TPR = y(keep);
average.Area = trapz(average.FPR, average.TPR);
end
