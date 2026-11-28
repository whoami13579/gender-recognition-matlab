function checkDatasetProfile(report, stage, synthetic)
assert(all(report.ClassCounts > 0), 'genderRecognition:UnexpectedDatasetProfile', ...
    'Both classes must be present in the %s dataset.', stage);
if synthetic
    return
end
if stage == "training"
    valid = isequal(report.ClassCounts, [100 100]) && isequal(report.ImageShapes, [640 640 3]) && ...
        isequal(report.PixelTypes, "uint8") && isequal(report.SubjectCounts, [50 50]) && ...
        isequal(report.ImagesPerSubjectValues, 2);
else
    valid = isequal(report.ClassCounts, [50 50]);
end
assert(valid, 'genderRecognition:UnexpectedDatasetProfile', ...
    'The %s dataset differs from the documented course profile.', stage);
end
