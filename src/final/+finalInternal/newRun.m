function [directory, staging] = newRun(parent, prefix)
[ok, message] = mkdir(parent);
assert(ok, 'genderRecognition:FinalExportFailed', '%s', message);
[~, token] = fileparts(tempname(parent));
directory = string(fullfile(parent, prefix + token));
staging = directory + ".incomplete";
[ok, message] = mkdir(staging);
assert(ok, 'genderRecognition:FinalExportFailed', '%s', message);
end
