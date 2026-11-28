function report = testOverlap(report, training, testing)
matches = training.Hashes(:) == testing.Hashes(:).';
report.TrainingTestDuplicatePairs = nnz(matches);
report.TestImagesWithTrainingMatch = nnz(any(matches, 1));
report.TrainingTestConflictingLabelPairs = nnz(matches & (training.Labels(:) ~= testing.Labels(:).'));
report.SharedClassQualifiedSubjectIDs = numel(intersect(training.Subjects, testing.Subjects));
report.SharedNumericSubjectIDs = numel(intersect( ...
    extractAfter(string(training.Subjects), "_"), extractAfter(string(testing.Subjects), "_")));
end
