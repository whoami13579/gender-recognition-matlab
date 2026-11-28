function timestamp = utcNow
timestamp = string(datetime('now', 'TimeZone', 'UTC', 'Format', "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"));
end
