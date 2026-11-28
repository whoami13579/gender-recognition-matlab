function moveArtifact(source, destination)
assert(~isfile(destination) && ~isfolder(destination), ...
    'genderRecognition:FinalExportExists', 'Refusing to replace an existing final artifact.');
try
    from = java.io.File(char(source)).toPath();
    to = java.io.File(char(destination)).toPath();
    java.nio.file.Files.move(from, to, javaArray('java.nio.file.CopyOption', 0));
catch cause
    failure = MException('genderRecognition:FinalExportFailed', 'Could not publish the new local artifact.');
    throwAsCaller(addCause(failure, cause));
end
end
