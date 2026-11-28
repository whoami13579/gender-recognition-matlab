function Z = applyMLPStandardization(Z, featureMean, featureScale)
%APPLYMLPSTANDARDIZATION Use supplied training coefficients, if enabled.
if ~isempty(featureMean)
    Z = (Z - featureMean) ./ featureScale;
    if any(~isfinite(Z), 'all')
        error('genderRecognition:MLPNumericalRange', 'Standardized features exceeded the double precision range.');
    end
end
end
