function [logits, activations] = forward(Z, weights, biases)
%FORWARD Compute logits, optionally retaining input and hidden activations.
numLayers = numel(weights);
activation = Z;
if nargout > 1
    activations = cell(1, numLayers);
end
for layer = 1:numLayers
    if nargout > 1
        activations{layer} = activation;
    end
    logits = activation * weights{layer} + biases{layer};
    if any(~isfinite(logits), 'all')
        error('genderRecognition:MLPNumericalRange', 'A layer pre-activation exceeded the double precision range.');
    end
    if layer < numLayers
        activation = tanh(logits);
    end
end
end
