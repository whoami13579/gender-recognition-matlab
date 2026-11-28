function path = resolvePath(path)
% Relative workflow paths are anchored at the project root.
assert(isscalar(string(path)) && ~ismissing(string(path)) && strlength(string(path)) > 0, ...
    'genderRecognition:InvalidDatasetPath', 'Specify one nonempty path.');
file = java.io.File(char(path));
if ~file.isAbsolute()
    path = fullfile(finalInternal.root(), path);
    file = java.io.File(char(path));
end
if isfolder(path)
    path = string(file.toPath().toRealPath(javaArray('java.nio.file.LinkOption', 0)).toString());
else
    path = string(file.getCanonicalPath());
end
end
