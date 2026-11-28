function savePipelineModel(pipeline, modelPath)
%SAVEPIPELINEMODEL Save a versioned fitted pipeline to a new local MAT file.
arguments
    pipeline struct
    modelPath (1,1) string
end
assert(endsWith(modelPath, ".mat", 'IgnoreCase', true), ...
    'genderRecognition:InvalidPipelineArtifact', 'The model path must end in .mat.');
finalInternal.validatePipeline(pipeline);
[parent, ~, ~] = fileparts(modelPath);
if strlength(parent) == 0
    parent = string(pwd);
end
temporary = string(tempname(parent)) + ".mat";
cleanup = onCleanup(@() removeTemporary(temporary));
artifact = struct('ModelFormatVersion', 1, 'Pipeline', pipeline);
save(temporary, 'artifact', '-v7');
finalInternal.moveArtifact(temporary, modelPath);
end

function removeTemporary(path)
if isfile(path)
    delete(path);
end
end
