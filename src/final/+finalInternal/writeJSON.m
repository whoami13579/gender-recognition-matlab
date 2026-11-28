function writeJSON(path, value)
% Close the temporary file before publishing its contents.
temporary = string(tempname(fileparts(path))) + ".json";
[id, message] = fopen(temporary, 'w', 'n', 'UTF-8');
assert(id >= 0, 'genderRecognition:FinalExportFailed', '%s', message);
closeFile = onCleanup(@() fclose(id));
fprintf(id, '%s\n', jsonencode(value, 'PrettyPrint', true));
[message, status] = ferror(id);
assert(status == 0, 'genderRecognition:FinalExportFailed', '%s', message);
clear closeFile
finalInternal.moveArtifact(temporary, path);
end
