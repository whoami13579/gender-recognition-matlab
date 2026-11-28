function [images, labels, subjects, config] = cvFixture(directory)
% Balanced synthetic PNGs with two views per class-qualified subject.
stream = RandStream('mt19937ar', 'Seed', 17);
labels = repelem([0; 1], 12);
images = cell(size(labels));
subjects = strings(size(labels));
prefixes = ["man", "woman"];
for k = 1:numel(labels)
    subject = floor(mod(k - 1, 12) / 2) + 1;
    view = mod(k - 1, 2) + 1;
    subjects(k) = prefixes(labels(k) + 1) + "_" + subject;
    image = randi(stream, [10 170], [6 6]);
    image(1:3, :) = image(1:3, :) + 50 * labels(k);
    images{k} = uint8(image);
    if nargin > 0
        imwrite(images{k}, fullfile(directory, subjects(k) + "_" + view + ".png"));
    end
end
config = crossValidationConfig(struct('Pipeline', struct( ...
    'Preprocessing', struct('ImageSize', [6 6]), 'PCA', struct('NumComponents', 2), ...
    'MLP', struct('HiddenLayerSizes', 3, 'MaxEpochs', 4, 'BatchSize', 4)), ...
    'Benchmark', struct('WarmupRuns', 0, 'NumRepeats', 1), ...
    'Search', struct('NumFolds', 2, 'RecommendFinal', false), ...
    'GeneratePlots', false, 'ExportResults', false));
end
