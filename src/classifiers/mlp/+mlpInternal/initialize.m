function [weights, biases] = initialize(layers, stream)
%INITIALIZE Glorot-uniform weights and zero biases using the supplied stream.
weights = cell(1, numel(layers) - 1);
biases = cell(size(weights));
for layer = 1:numel(weights)
    fanIn = layers(layer);
    fanOut = layers(layer + 1);
    limit = sqrt(6 / (fanIn + fanOut));
    weights{layer} = (2 * rand(stream, fanIn, fanOut) - 1) * limit;
    biases{layer} = zeros(1, fanOut);
end
end
