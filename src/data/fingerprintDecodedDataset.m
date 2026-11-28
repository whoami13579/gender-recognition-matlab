function [fingerprint, imageHashes] = fingerprintDecodedDataset(images, samples)
%FINGERPRINTDECODEDDATASET Issue #11 SHA-256 and optional filename-free hashes.
datasetDigest = java.security.MessageDigest.getInstance('SHA-256');
imageHashes = strings(numel(images), 1);
[~, ~, endian] = computer;
for idx = 1:numel(images)
    shape = sprintf('%d %d %d %s\n', samples.Height(idx), samples.Width(idx), ...
        samples.Channels(idx), samples.PixelType(idx));
    header = sprintf('%s\n%s', samples.Filename(idx), shape);
    datasetDigest.update(typecast(unicode2native(header, 'UTF-8'), 'int8'));
    pixels = images{idx}(:);
    if endian == 'B' && ~islogical(pixels)
        pixels = swapbytes(pixels);
    end
    if islogical(pixels)
        pixels = uint8(pixels);
    end
    bytes = typecast(pixels, 'int8');
    datasetDigest.update(bytes);
    if nargout > 1
        imageDigest = java.security.MessageDigest.getInstance('SHA-256');
        imageDigest.update(typecast(unicode2native(shape, 'UTF-8'), 'int8'));
        imageDigest.update(bytes);
        imageHashes(idx) = toHex(imageDigest.digest());
    end
end
fingerprint = toHex(datasetDigest.digest());
end

function hex = toHex(bytes)
hex = lower(string(reshape(dec2hex(typecast(bytes, 'uint8'), 2).', 1, [])));
end
