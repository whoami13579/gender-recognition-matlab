function Z = validateMLPFeatures(Z)
%VALIDATEMLPFEATURES Check the shared feature contract and convert to double.
if ~(isa(Z, 'double') || isa(Z, 'single')) || ~ismatrix(Z) || ...
        ~isreal(Z) || issparse(Z) || size(Z, 2) == 0 || any(~isfinite(Z), 'all')
    error('genderRecognition:InvalidMLPFeatures', ...
        'MLP features must be a full, real, finite double or single matrix with at least one column.');
end
Z = double(Z);
end
