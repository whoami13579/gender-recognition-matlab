function [label, subjectID, imageID, sampleID] = parseImageFilename(filename)
%PARSEIMAGEFILENAME Parse a local PNG basename into a label and stable IDs.
%   Label 0 is male and label 1 is female (the positive class). Subject and
%   image numbers are strings so parsing does not lose integer precision.

if ischar(filename) && isrow(filename)
    filename = string(filename);
end
if ~isstring(filename) || ~isscalar(filename) || ismissing(filename)
    error('genderRecognition:InvalidFilename', 'Expected one PNG filename.');
end

tokens = regexpi(char(filename), '^(man|woman)_([0-9]+)_([0-9]+)\.png$', ...
    'tokens', 'once');
if isempty(tokens)
    error('genderRecognition:InvalidFilename', ...
        'Unsupported filename "%s". Expected man_<person>_<image>.png or woman_<person>_<image>.png.', ...
        char(filename));
end

prefix = string(lower(tokens{1}));
% Equivalent padded and unpadded numbers identify the same sample.
person = string(regexprep(tokens{2}, '^0+(?=[0-9])', ''));
imageID = string(regexprep(tokens{3}, '^0+(?=[0-9])', ''));
label = double(prefix == "woman");
subjectID = prefix + "_" + person;
sampleID = subjectID + "_" + imageID;
end
