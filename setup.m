function setup
%SETUP Add this project's source and scripts to the MATLAB path.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'src', 'data'), fullfile(root, 'src', 'validation'), ...
    fullfile(root, 'src', 'features', 'pca'), fullfile(root, 'scripts'));
end
