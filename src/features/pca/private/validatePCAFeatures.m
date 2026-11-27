function validatePCAFeatures(X)
%VALIDATEPCAFEATURES Check the common input contract for fit and transform.
if ~(isa(X, 'double') || isa(X, 'single')) || ~ismatrix(X) || ...
        ~isreal(X) || issparse(X) || any(~isfinite(X), 'all')
    error('genderRecognition:InvalidPCAFeatures', ...
        'PCA features must be a full, real, finite double or single matrix.');
end
end
