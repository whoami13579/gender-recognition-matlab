function [loss, objective, weightGradients, biasGradients] = batchObjective(Z, labels, weights, biases, lambda)
%BATCHOBJECTIVE Evaluate stable BCE, weight-only L2, and optional gradients.
if nargout > 2
    [logits, activations] = mlpInternal.forward(Z, weights, biases);
else
    logits = mlpInternal.forward(Z, weights, biases);
end

% Softplus of the wrong-class margin, with no positive exponential inputs.
margin = (1 - 2 * labels) .* logits;
expNegativeAbs = exp(-abs(margin));
sampleLoss = max(margin, 0) + log1p(expNegativeAbs);
loss = mean(sampleLoss);
penalty = 0;
regularizationScale = sqrt(lambda) / sqrt(2);
for layer = 1:numel(weights)
    scaledWeights = regularizationScale * weights{layer};
    penalty = penalty + sum(scaledWeights.^2, 'all');
end
objective = loss + penalty;
if ~isfinite(objective)
    error('genderRecognition:MLPNumericalRange', 'The training objective exceeded the double precision range.');
end
if nargout <= 2
    return
end

% Evaluate sigmoid(t)-y without cancellation for confident positive samples.
probabilityWrong = expNegativeAbs ./ (1 + expNegativeAbs);
positiveMargins = margin > 0;
probabilityWrong(positiveMargins) = 1 ./ (1 + expNegativeAbs(positiveMargins));
delta = (1 - 2 * labels) .* probabilityWrong / size(Z, 1);
weightGradients = cell(size(weights));
biasGradients = cell(size(biases));
for layer = numel(weights):-1:1
    dW = activations{layer}.' * delta + lambda * weights{layer};
    db = sum(delta, 1);
    assert(isequal(size(dW), size(weights{layer})) && isequal(size(db), size(biases{layer})), ...
        'genderRecognition:InvalidMLPGradient', 'Gradient shapes must match their parameters.');
    if any(~isfinite(dW), 'all') || any(~isfinite(db), 'all')
        error('genderRecognition:MLPNumericalRange', 'A gradient exceeded the double precision range.');
    end
    weightGradients{layer} = dW;
    biasGradients{layer} = db;
    if layer > 1
        delta = (delta * weights{layer}.') .* (1 - activations{layer}.^2);
        if any(~isfinite(delta), 'all')
            error('genderRecognition:MLPNumericalRange', 'A backpropagated derivative exceeded the double precision range.');
        end
    end
end
end
