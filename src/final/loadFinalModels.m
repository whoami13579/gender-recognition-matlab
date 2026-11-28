function [pipelines, manifest] = loadFinalModels(directory)
%LOADFINALMODELS Load the four verified local pipelines for prediction or a demo.
[pipelines, manifest] = finalInternal.readBundle(directory);
end
