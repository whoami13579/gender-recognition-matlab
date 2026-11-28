function path = root
path = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
end
