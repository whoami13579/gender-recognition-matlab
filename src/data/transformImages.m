function output = transformImages(preprocessor, images, outputFormat)
%TRANSFORMIMAGES Apply a fitted preprocessor in the supplied sample order.
%   The default result is an N-by-D feature matrix (one image per row).
%   Use "images" as the third argument to keep the N-by-1 processed cells.

arguments
    preprocessor (1, 1) struct
    images
    outputFormat (1, 1) string {mustBeMember(outputFormat, ["features", "images"])} = "features"
end

if ~iscell(images) || (~isvector(images) && ~isempty(images))
    error('genderRecognition:InvalidImageBatch', 'Expected a cell vector of images.');
end
if ~all(isfield(preprocessor, {'Options', 'OutputImageSize', 'NumFeatures'}))
    error('genderRecognition:InvalidPreprocessor', 'Use a model returned by fitPreprocessor.');
end

options = preprocessor.Options;
numImages = numel(images);
if outputFormat == "features"
    output = zeros(numImages, preprocessor.NumFeatures, options.OutputType);
else
    output = cell(numImages, 1);
end

for k = 1:numImages
    source = images{k};
    if ~isequal(preprocessedImageSize(source, options, k), preprocessor.OutputImageSize)
        error('genderRecognition:InconsistentImageSize', ...
            'Image %d does not match the fitted output size or channel count.', k);
    end

    pixels = cast(im2double(source), options.OutputType);
    if options.Grayscale && size(pixels, 3) == 3
        pixels = im2gray(pixels);
    end
    if ~isequal([size(pixels, 1), size(pixels, 2)], preprocessor.OutputImageSize(1:2))
        pixels = imresize(pixels, preprocessor.OutputImageSize(1:2), ...
            'bilinear', 'Antialiasing', true);
    end

    % Keep unit intensities in range even in the presence of roundoff.
    pixels = min(max(pixels, 0), 1);
    if options.IntensityScaling == "native" && isinteger(source)
        pixels = pixels .* cast(intmax(class(source)), options.OutputType);
    end

    if outputFormat == "features"
        output(k, :) = pixels(:).';
    else
        output{k} = pixels;
    end
end
end
