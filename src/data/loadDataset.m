function [images, y, subjectIDs, metadata] = loadDataset(datasetPath)
%LOADDATASET Read a flat directory of labeled PNG images in filename order.
%   IMAGES is an N-by-1 cell array. Y and SUBJECTIDS are N-by-1 columns.
%   Each row of METADATA describes the corresponding image. Keep it local.

if ischar(datasetPath) && isrow(datasetPath)
    datasetPath = string(datasetPath);
end
if ~isstring(datasetPath) || ~isscalar(datasetPath) || ismissing(datasetPath) ...
        || strlength(datasetPath) == 0 || ~isfolder(datasetPath)
    error('genderRecognition:InvalidDatasetPath', ...
        'The dataset path must identify an existing directory.');
end

entries = dir(fullfile(datasetPath, '*'));
entries = entries(~ismember({entries.name}, {'.', '..'}));
if isempty(entries)
    error('genderRecognition:EmptyDataset', 'The dataset directory is empty.');
end
if any([entries.isdir])
    error('genderRecognition:UnexpectedSubdirectory', ...
        'The dataset directory must be flat and contain only the labeled PNGs.');
end

[~, order] = sort(lower(string({entries.name})));
entries = entries(order);
filenames = reshape(string({entries.name}), [], 1);
numImages = numel(entries);
y = zeros(numImages, 1);
subjectIDs = strings(numImages, 1);
imageIDs = strings(numImages, 1);
sampleIDs = strings(numImages, 1);
for k = 1:numImages
    [y(k), subjectIDs(k), imageIDs(k), sampleIDs(k)] = parseImageFilename(filenames(k));
end

[~, ~, groups] = unique(sampleIDs);
counts = accumarray(groups, 1);
duplicate = find(counts(groups) > 1, 1);
if ~isempty(duplicate)
    error('genderRecognition:DuplicateSampleID', ...
        'Filename "%s" resolves to a sample identifier used by multiple files.', ...
        char(filenames(duplicate)));
end

images = cell(numImages, 1);
imageSizes = zeros(numImages, 3);
pixelTypes = strings(numImages, 1);
for k = 1:numImages
    path = fullfile(entries(k).folder, entries(k).name);
    try
        [pixels, colorMap, alpha] = imread(path, 'png');
    catch cause
        failure = MException('genderRecognition:UnreadableImage', ...
            'Could not decode PNG "%s".', entries(k).name);
        throwAsCaller(addCause(failure, cause));
    end
    if ~isempty(colorMap) || ~isempty(alpha)
        error('genderRecognition:UnsupportedImage', ...
            'PNG "%s" must be grayscale or truecolor RGB without transparency.', ...
            entries(k).name);
    end
    images{k} = pixels;
    imageSizes(k, :) = [size(pixels, 1), size(pixels, 2), size(pixels, 3)];
    pixelTypes(k) = string(class(pixels));
end

metadata = table(sampleIDs, subjectIDs, imageIDs, y, filenames, ...
    imageSizes(:, 1), imageSizes(:, 2), imageSizes(:, 3), pixelTypes, ...
    'VariableNames', {'SampleID', 'SubjectID', 'ImageID', 'Label', ...
    'Filename', 'Height', 'Width', 'Channels', 'PixelType'});
end
