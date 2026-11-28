function validateBayesFeatures(Z)
%VALIDATEBAYESFEATURES Check the shared fit and predict feature contract.
if ~(isa(Z, 'double') || isa(Z, 'single')) || ~ismatrix(Z) || ...
        ~isreal(Z) || issparse(Z) || size(Z, 2) == 0 || any(~isfinite(Z), 'all')
    error('genderRecognition:InvalidBayesFeatures', ...
        'Bayes features must be a full, real, finite double or single matrix with at least one column.');
end
end
