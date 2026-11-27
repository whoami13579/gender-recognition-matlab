function setup
%SETUP Add this project's source and scripts to the MATLAB path.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'src', 'data'), fullfile(root, 'scripts'));
end
