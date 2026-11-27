function outputSize = preprocessedImageSize(image, options, sampleIndex)
%PREPROCESSEDIMAGESIZE Validate one input image and determine its output size.
supportedTypes = {'logical', 'uint8', 'uint16', 'single', 'double'};
channels = size(image, 3);
if ~ismember(class(image), supportedTypes) || isempty(image) || ~isreal(image) ...
        || issparse(image) || ndims(image) > 3 || ~ismember(channels, [1 3])
    error('genderRecognition:UnsupportedImage', ...
        'Image %d must be a nonempty, real, full grayscale or RGB array of a supported type.', ...
        sampleIndex);
end
if isfloat(image) && ~all(image >= 0 & image <= 1, 'all')
    error('genderRecognition:InvalidIntensity', ...
        'Floating-point image %d must have finite values in [0, 1].', sampleIndex);
end

spatialSize = [size(image, 1), size(image, 2)];
if ~isempty(options.ImageSize)
    spatialSize = options.ImageSize;
end
if options.Grayscale
    channels = 1;
end
outputSize = [spatialSize, channels];
end
