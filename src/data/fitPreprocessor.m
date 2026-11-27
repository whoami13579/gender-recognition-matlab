function preprocessor = fitPreprocessor(trainingImages, config)
%FITPREPROCESSOR Validate options and fix the expected output image size.
%   CONFIG fields are Grayscale (true), ImageSize ([64 64], [] preserves
%   resolution), OutputType ("double"), and IntensityScaling ("unit").
%   The current operations are deterministic per image.

arguments
    trainingImages
    config = struct()
end

if ~iscell(trainingImages) || ~isvector(trainingImages) || isempty(trainingImages)
    error('genderRecognition:InvalidImageBatch', ...
        'Fitting requires a nonempty cell vector of training images.');
end

options = struct('Grayscale', true, 'ImageSize', [64 64], ...
    'OutputType', "double", 'IntensityScaling', "unit");
if ~isstruct(config) || ~isscalar(config)
    error('genderRecognition:InvalidConfig', 'Preprocessing options must be a scalar struct.');
end
fields = fieldnames(config);
for k = 1:numel(fields)
    field = fields{k};
    if ~isfield(options, field)
        error('genderRecognition:InvalidConfig', 'Unknown preprocessing option "%s".', field);
    end
    options.(field) = config.(field);
end

if ~islogical(options.Grayscale) || ~isscalar(options.Grayscale)
    error('genderRecognition:InvalidConfig', 'Grayscale must be a logical scalar.');
end

targetSize = options.ImageSize;
if ~isnumeric(targetSize) || ~isreal(targetSize) || issparse(targetSize) ...
        || ~(isequal(size(targetSize), [0 0]) || (isequal(size(targetSize), [1 2]) ...
        && all(isfinite(targetSize) & targetSize > 0 & targetSize == fix(targetSize))))
    error('genderRecognition:InvalidConfig', ...
        'ImageSize must be [] or a row vector of two positive integer dimensions.');
end
options.ImageSize = double(targetSize);
options.OutputType = menuValue(options.OutputType, ["double", "single"], 'OutputType');
options.IntensityScaling = menuValue(options.IntensityScaling, ["unit", "native"], 'IntensityScaling');

outputSize = preprocessedImageSize(trainingImages{1}, options, 1);
for k = 2:numel(trainingImages)
    if ~isequal(preprocessedImageSize(trainingImages{k}, options, k), outputSize)
        error('genderRecognition:InconsistentImageSize', ...
            'Training image %d would produce a different output size or channel count.', k);
    end
end

preprocessor = struct('Options', options, 'OutputImageSize', outputSize, ...
    'NumFeatures', prod(outputSize), 'NumFittedParameters', 0);
end

function value = menuValue(value, choices, optionName)
if ischar(value) && isrow(value)
    value = string(value);
end
if ~isstring(value) || ~isscalar(value) || ismissing(value) || ~ismember(value, choices)
    error('genderRecognition:InvalidConfig', '%s must be one of: %s.', ...
        optionName, char(strjoin(choices, ', ')));
end
end
