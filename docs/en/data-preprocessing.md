# Dataset loading and preprocessing

[繁體中文](../zh-TW/data-preprocessing.md) · [Project home](../../README.md)

## Setup and local data

Verified on Windows with MATLAB R2026a Update 5. Loading, preprocessing, and the synthetic tests require base MATLAB. Place the training PNGs directly in the local `train/` directory or pass another folder. Run the commands below from the repository root. The supplied training folder is the only dataset opened by the analysis.

```matlab
setup;
run_dataset_analysis;
stats = run_dataset_analysis("path/to/training-data", struct('ImageSize', [96 96]));
run_tests;
```

`run_dataset_analysis` prints and optionally returns aggregate counts, dimensions, observed intensity range, feature-matrix bytes, and timings. The analysis keeps its output in the console and the returned struct. Store any exported features, models, and private metadata in the ignored `results/` directory.

## Loading and row order

```matlab
[images, y, subjectIDs, metadata] = loadDataset("train");
[label, subjectID, imageID, sampleID] = parseImageFilename("man_900001_1.png");
```

- The loader expects a flat directory of labeled PNGs. It accepts opaque grayscale and RGB PNGs and reports errors for unexpected entries, malformed names, duplicate identifiers, unreadable images, indexed colors, and transparency. Counts specific to the official training set are checked separately by the analysis.
- The filename patterns are `man_<person>_<image>.png` and `woman_<person>_<image>.png`, with decimal digits for both numbers. The parser accepts letter case variants and removes redundant leading zeros from numeric IDs. The example above is synthetic. Class-qualified IDs such as `man_900001` and `woman_900001` refer to different subjects. The sample ID appends the image ID to the subject ID.
- Files are ordered lexicographically by case-insensitive filename. `images` is an N × 1 cell array. `y` is an N × 1 double column with male = `0` and female = `1`. Female is the positive class for subsequent scoring, ROC and AUC. `subjectIDs` is an N × 1 string column.
- Row `i` in `metadata` corresponds to `images{i}`, `y(i)`, `subjectIDs(i)` and eventually `X(i,:)`. The table contains `SampleID`, `SubjectID`, `ImageID`, `Label`, `Filename`, `Height`, `Width`, `Channels` and `PixelType`. Keep the table and the real identifiers local. Reliable subject grouping requires the dataset's filename convention. Unsupported names raise an error without inventing an identity.

## Fit and transform

```matlab
config = struct('Grayscale', true, 'ImageSize', [64 64], ...
    'OutputType', "double", 'IntensityScaling', "unit");
preprocessor = fitPreprocessor(images, config);
X = transformImages(preprocessor, images);
processedImages = transformImages(preprocessor, images, "images");
```

| Option | Default | Supported values |
|---|---|---|
| `Grayscale` | `true` | Logical scalar. `false` keeps each image's existing channel count. |
| `ImageSize` | `[64 64]` | A row vector `[height width]` of positive integers, or `[]` to keep the original resolution. |
| `OutputType` | `"double"` | `"double"` or `"single"`. |
| `IntensityScaling` | `"unit"` | `"unit"` scales unsigned inputs by the maximum for their type. `"native"` returns the float output on the source type's scale. |

Images must be nonempty, full, real grayscale or RGB arrays of type `logical`, `uint8`, `uint16`, `single` or `double`. Floating-point inputs must already be finite and in [0, 1]. Processing converts pixels to floating point, applies grayscale conversion when requested, and resizes with bilinear interpolation and antialiasing. `"native"` restores the unsigned input scale of [0, 255] or [0, 65535] after these operations. Logical and float inputs retain unit scaling. With no resizing, images must have the same height and width. With grayscale disabled, the batch must also have a common channel count. Held-out inputs must produce the same output shape as the fit inputs.

By default `transformImages` returns an N × D matrix in the order supplied, with one image per row. Use `"images"` to obtain the processed N × 1 cell array. For a processed H × W × C image, MATLAB column-major vectorization places pixel `(r,c,k)` at feature index `r + (c-1)*H + (k-1)*H*W`, using one-based indices. Channels are concatenated in image order.

For cross-validation, call `fitPreprocessor(trainingImages, config)` on the training split, then apply `transformImages(preprocessor, validationImages)` to held-out data. The current fit validates options, fixes the output representation, and stores `OutputImageSize`, `NumFeatures` and `NumFittedParameters` (zero). Transforms use deterministic per-image operations.

## Proposed downstream interfaces

```matlab
pcaModel = fitPCA(X, config);
Z = transformPCA(pcaModel, X);

ldaModel = fitLDA(X, y, config);
Z = transformLDA(ldaModel, X);

bayesModel = fitBayes(Z, y, config);
[yPred, scores] = predictBayes(bayesModel, Z);

mlpModel = fitMLP(Z, y, config);
[yPred, scores] = predictMLP(mlpModel, Z);
```

These signatures are proposals for the downstream issues. `Z` is N × K, and both `yPred` and continuous positive-class `scores` are N × 1. Higher scores indicate stronger evidence for female (`1`). A fitted model stores the configuration and state required to score unseen data. The proposed projection fields are `NumInputFeatures`, `NumFeatures` (output dimension) and `NumFittedParameters`. Proposed classifier fields are `NumFeatures` (input dimension) and `NumClassifierParameters`. Report classifier parameters and the fitted parameters of each transformation separately, with a documented counting convention.

## Verified training dataset

| Property | Local verification |
|---|---|
| Images | 200 PNGs, 100 male and 100 female |
| Original size and type | 640 × 640 × 3, RGB `uint8` |
| Subject IDs derived from names | 100, 50 per class, two images per subject, zero cross-class ID collisions |
| Invalid or unreadable images | 0 |
| Default processed representation | 64 × 64 grayscale, 4,096 features per image |
| Default output | `X` is 200 × 4,096 `double`, `y` is 200 × 1 |
| Feature-matrix memory | 6,553,600 bytes (6.25 MiB), excluding input images and temporary arrays |

Subject grouping relies on the filename convention. The 64 × 64 baseline and label encoding are project decisions. Original-resolution grayscale would retain 409,600 features per image and require 625 MiB for the same 200 images in a double feature matrix. Dimensions multiply with spatial resolution and channel count, so larger images increase storage and future fitting and inference costs.

## Tests and lecture references

Run `run_tests;` in MATLAB or `matlab -batch "run_tests;"` from the repository root. The suite generates synthetic images in temporary folders and cleans them up via MATLAB test fixtures. The tests exercise validation, metadata alignment, vectorization and reproducibility with synthetic inputs.

Sections 1.2 and 1.3 in Chapter 1 cover the recognition pipeline and feature representation. See combined lecture PDF pages 12 and 25–29. See the [lecture reference and system architecture](lecture-reference.md) and the [official project requirements](project-requirements.md).
