function [value, directory] = demoSavedRun(explicit, parents, pattern, marker, label, selector, loader)
% Automatically select only a unique compatible completed local run.
value = [];
directory = "";
if strlength(explicit) > 0
    directory = finalInternal.resolvePath(explicit);
    assert(isfile(fullfile(directory, marker)), 'genderRecognition:DemoMissingArtifact', ...
        '%s is missing %s. Select its completed directory with %s.', label, marker, selector);
    value = loader(directory);
    return
end
matches = strings(0, 1);
rejected = strings(0, 1);
for parent = unique(arrayfun(@finalInternal.resolvePath, parents))
    entries = dir(fullfile(parent, pattern));
    for entry = entries([entries.isdir]).'
        if endsWith(entry.name, ".incomplete")
            continue
        end
        candidate = string(fullfile(parent, entry.name));
        try
            loaded = loader(candidate);
            if isempty(matches)
                value = loaded;
            end
            matches(end + 1, 1) = candidate; %#ok<AGROW>
        catch cause
            rejected(end + 1, 1) = string(entry.name) + " (" + cause.identifier + ")"; %#ok<AGROW>
        end
    end
end
if numel(matches) > 1
    error('genderRecognition:DemoAmbiguousRun', ...
        'Multiple compatible %s runs. Set %s explicitly to one of:\n%s', ...
        label, selector, strjoin(matches, newline));
elseif isempty(matches)
    if ~isempty(rejected)
        fprintf('%s candidates did not validate: %s\n', label, strjoin(rejected, ', '));
    end
else
    directory = matches(1);
end
end
