function [pipelines, manifest, frozen, training] = loadFinalModels(directory)
%LOADFINALMODELS Load the four verified local pipelines and training metadata.
[pipelines, manifest, frozen, training] = finalInternal.readBundle(directory);
end
