function prediction = demoImage(pipelines, options)
assert(strlength(strip(options.ImagePath)) > 0, 'genderRecognition:DemoMissingImage', ...
    'Set ImagePath to one explicitly selected local RGB or grayscale image.');
path = finalInternal.resolvePath(options.ImagePath);
assert(isfile(path), 'genderRecognition:DemoMissingImage', 'The selected local image does not exist.');
context = "Local image demonstration";
folder = finalInternal.resolvePath(options.TrainingPath);
imageFolder = string(fileparts(path));
isTrainingImage = folder == imageFolder || (ispc && strcmpi(folder, imageFolder));
assert(options.ImageKind ~= "training" || isTrainingImage, 'genderRecognition:DemoImageSource', ...
    'A training example must be in the configured TrainingPath directory.');
if isTrainingImage
    context = "Training example | software demonstration";
end
if options.Synthetic
    context = "Synthetic fixture | " + context;
end
[pixels, map, alpha] = imread(path);
assert(isempty(map) && isempty(alpha), 'genderRecognition:DemoImageFormat', ...
    'Choose a direct RGB or grayscale image without an indexed palette or alpha channel.');
method = [pipelines.Method].';
for k = numel(method):-1:1
    prep = pipelines(k).Preprocessor;
    processedSize(k,:) = [prep.OutputImageSize(1:2), prep.NumFeatures / prod(prep.OutputImageSize(1:2))];
    rawFeatures(k,1) = prep.NumFeatures;
    reducedFeatures(k,1) = pipelines(k).Parameters.NumFeatures;
    [labels(k,1), scores(k,1)] = predictPipeline(pipelines(k), {pixels});
end
scoreType = repmat("MLP logit", size(method));
scoreType(endsWith(method, "-bayes")) = "Bayes log-posterior odds";
className = ["Male"; "Female"];
summary = table(method, processedSize, rawFeatures, reducedFeatures, labels, className(labels + 1), ...
    scores, scoreType, 'VariableNames', {'Method','PreprocessedHWC','OriginalFeatures', ...
    'ReducedFeatures','Label','CourseClass','Score','ScoreType'});
originalSize = [size(pixels,1) size(pixels,2) size(pixels,3)];
prediction = struct('Context', context, 'OriginalSize', originalSize, 'Summary', summary);
fprintf('\n%s | input %d x %d x %d\n', context, originalSize);
fprintf('Grayscale, resize, unit scale and flatten, then PCA or Fisher LDA, then Bayes or MLP.\n');
disp(summary);
fprintf(['Course labels: male = 0, female = 1 (positive). Score > 0 predicts 1. Zero predicts 0.\n' ...
    'Higher scores favor class 1. Score scales differ across methods. Facial appearance cannot establish gender identity.\n']);
if options.ShowFigures
    processed = transformImages(pipelines(1).Preprocessor, {pixels}, "images");
    fig = figure('Name', char(context), 'NumberTitle', 'off');
    layout = tiledlayout(fig, 1, 2);
    imshow(pixels, 'Parent', nexttile(layout));
    ax = nexttile(layout);
    imshow(processed{1}, 'Parent', ax);
    title(ax, 'Fitted preprocessing');
    title(layout, context);
end
end
