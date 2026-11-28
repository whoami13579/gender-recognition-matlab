function pipeline = loadPipelineModel(modelPath)
%LOADPIPELINEMODEL Read one locally trusted versioned MAT file without fitting.
arguments
    modelPath (1,1) string
end
assert(isfile(modelPath), 'genderRecognition:MissingPipelineArtifact', 'The model MAT file is missing.');
try
    data = load(modelPath, 'artifact', '-mat');
    assert(isfield(data, 'artifact') && isstruct(data.artifact) && isscalar(data.artifact) && ...
        all(isfield(data.artifact, {'ModelFormatVersion', 'Pipeline'})), ...
        'genderRecognition:InvalidPipelineArtifact', 'Expected a versioned pipeline MAT artifact.');
catch cause
    failure = MException('genderRecognition:InvalidPipelineArtifact', 'The model MAT artifact could not be read.');
    throwAsCaller(addCause(failure, cause));
end
assert(isequal(data.artifact.ModelFormatVersion, 1), ...
    'genderRecognition:UnsupportedPipelineVersion', 'Unsupported pipeline MAT format version.');
pipeline = data.artifact.Pipeline;
finalInternal.validatePipeline(pipeline);
end
