function hex = checksum(value, isFile)
% SHA-256 over exact file bytes or UTF-8 JSON for a configuration value.
digest = java.security.MessageDigest.getInstance('SHA-256');
if isFile
    [id, message] = fopen(value, 'rb');
    assert(id >= 0, 'genderRecognition:FinalArtifactInvalid', '%s', message);
    closeFile = onCleanup(@() fclose(id));
    assert(fseek(id, 0, 'eof') == 0, 'genderRecognition:FinalArtifactInvalid', ...
        'Could not determine artifact size.');
    remaining = ftell(id);
    assert(remaining >= 0 && fseek(id, 0, 'bof') == 0, ...
        'genderRecognition:FinalArtifactInvalid', 'Could not rewind artifact.');
    while remaining > 0
        chunkSize = min(remaining, 1024 * 1024);
        [bytes, count] = fread(id, chunkSize, '*uint8');
        assert(count == chunkSize, 'genderRecognition:FinalArtifactInvalid', ...
            'Could not read the complete artifact.');
        digest.update(typecast(bytes, 'int8'));
        remaining = remaining - count;
    end
else
    digest.update(typecast(unicode2native(jsonencode(value), 'UTF-8'), 'int8'));
end
hex = lower(string(reshape(dec2hex(typecast(digest.digest(), 'uint8'), 2).', 1, [])));
end
