function report = trainingDiagnostics(report, images, index, outerAssignment)
matches = triu(index.Hashes == index.Hashes.', 1);
report.CrossOuterFoldDuplicatePairs = [];
if ~isempty(outerAssignment)
    report.CrossOuterFoldDuplicatePairs = nnz(matches & (outerAssignment(:) ~= outerAssignment(:).'));
end
options = struct('ImageSize', [16 16]);
thumbs = transformImages(fitPreprocessor(images, options), images);
threshold = 0.02;
nearPairs = 0;
nearCrossSubject = 0;
nearCrossFold = 0;
for k = 1:size(thumbs, 1) - 1
    candidates = (k + 1):size(thumbs, 1);
    distance = sqrt(mean((thumbs(candidates, :) - thumbs(k, :)).^2, 2));
    near = candidates(distance <= threshold & ~matches(k, candidates).');
    nearPairs = nearPairs + numel(near);
    nearCrossSubject = nearCrossSubject + nnz(index.Subjects(near) ~= index.Subjects(k));
    if ~isempty(outerAssignment)
        nearCrossFold = nearCrossFold + nnz(outerAssignment(near) ~= outerAssignment(k));
    end
end
if isempty(outerAssignment)
    nearCrossFold = [];
end
report.NearDuplicates = struct('Measure', "RMSE of 16-by-16 grayscale unit-scale thumbnails", ...
    'InclusiveThreshold', threshold, 'NonExactPairs', nearPairs, ...
    'CrossSubjectPairs', nearCrossSubject, 'CrossOuterFoldPairs', nearCrossFold);
report.MaterialDefect = report.ConflictingLabelPairs > 0 || any(report.CrossOuterFoldDuplicatePairs > 0);
end
