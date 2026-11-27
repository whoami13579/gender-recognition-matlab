# Five-fold dataset partitioning

[繁體中文](../zh-TW/dataset-partitioning.md) · [Project home](../../README.md)

## Why the project groups samples

Five-fold cross-validation rotates the held-out portion of the training dataset. Each sample is used for validation once and for training in the other four folds. Sharing the same split across PCA + Bayes, LDA + Bayes, PCA + MLP, and LDA + MLP gives all four methods the same validation samples.

- Ordinary cross-validation splits individual samples without considering their labels or subjects.
- Stratified cross-validation distributes samples by class to keep fold proportions close to the overall distribution.
- Subject-grouped stratified cross-validation assigns whole subjects and balances the number of images of each class. This keeps images with a shared subject ID out of the corresponding training split when that ID is held out.

The loader derives class-qualified IDs from filenames. Synthetic IDs such as `man_900001` and `woman_900001` stay distinct even when their numeric suffixes match. The separation guarantee applies to the supplied IDs. Filename-derived IDs alone cannot establish real-world identity uniqueness or detect someone represented under different IDs.

The [project requirements](project-requirements.md) specify five-fold cross-validation using the training dataset. Chapter 1 §1.4.5 and Chapter 6 §6.3.3 discuss validation, testing, generalization, and overfitting on combined lecture PDF pages 38 and 418–420. A complete five-fold procedure is outside the lecture coverage. The grouping, balancing, and tie-breaking rules below are implementation decisions. See the [lecture reference](lecture-reference.md).

## API and configuration

```matlab
setup;
[folds, partitionInfo] = createFolds(y, subjectIDs, config);
```

`y` is a full, real N × 1 double column with male = `0` and female = `1` (the positive class). `subjectIDs` is an aligned N × 1 string column with no missing, empty, or whitespace-only entries. IDs are matched exactly, including case, and the loader already normalizes its parsed IDs. The function works entirely from labels, IDs, and options.

`config` is an optional scalar struct. Omitted fields use these defaults.

| Option | Default | Supported values |
|---|---:|---|
| `NumFolds` | `5` | Integer ≥ 2 and ≤ `flintmax`. Each class must have at least this many assignable units. |
| `RandomSeed` | `42` | Integer in [0, 2³² − 1]. |
| `GroupBySubject` | `true` | Logical scalar. |

`folds` is a K × 1 struct array with `TrainIndices` and `ValidationIndices` for each fold. Both are ascending one-based double columns indexed into the original inputs. Apply them unchanged to images, labels, IDs, metadata rows, or features aligned to the same sample order.

| `partitionInfo` field | Meaning |
|---|---|
| `NumFolds`, `RandomSeed`, `GroupBySubject`, `Stratified` | Effective settings. Stratification is always enabled. |
| `SampleToFold` | N × 1 column giving the validation fold (1 through K) for each input row. Keep real assignments local. |
| `TrainSampleCounts`, `ValidationSampleCounts` | K × 1 sample counts. |
| `TrainClassCounts`, `ValidationClassCounts` | K × 2 sample counts with columns `[male female]`. |
| `TrainSubjectCounts`, `ValidationSubjectCounts`, `SubjectOverlapCounts` | K × 1 counts of distinct subject IDs in each split and in their intersection. Empty when IDs are omitted. |

For data without reliable IDs, explicitly use sample-level stratification.

```matlab
[folds, partitionInfo] = createFolds(y, [], struct('GroupBySubject', false));
```

Sample-level mode can split images of the same person across training and validation. You can supply a valid ID column with grouping disabled to measure the overlap. If IDs are incomplete, pass `[]` for the fallback. Grouped mode rejects missing or invalid IDs and any subject with conflicting labels.

## Balancing and reproducibility

1. Form assignable units in first-seen subject order, or use input sample order when grouping is disabled. Each unit has a single class and a weight equal to its image count.
2. Create a local `mt19937ar` stream from `RandomSeed`. Use it to permute fold priority and shuffle units separately for male, then female. Sort each class's units by descending image count, preserving the shuffled order on ties.
3. Assign each unit to a fold with the fewest images of that class so far. Break ties by the fewest total images so far, then by the seeded fold priority. Each class fills any empty class slots before a fold receives a second unit of that class.

The greedy scheme balances image counts even for unequal subject sizes. Large subject groups can force unequal fold sizes or class proportions, and the scheme does not search every possible assignment. Sample-level mode uses the same procedure with unit weights of one.

The function rejects empty, noncolumn, and unsupported label inputs, missing classes, invalid or unknown options, and wrong-length or malformed supplied IDs. At least K distinct groups per class in grouped mode or K samples per class in sample-level mode are required so every validation fold contains both classes for future ROC and AUC calculation. Before returning, the function verifies that each fold's training and validation indices are valid, unique, disjoint, and together contain every sample. It also checks exact-once validation coverage, both validation classes, and subject separation in grouped mode. Each sample therefore appears in training in the other K − 1 folds.

With the same input values, row order, and options, rerunning produces the same folds and leaves MATLAB's global random state unchanged. Retain the row order and options for reproduction, and give all four pipelines the returned `folds` struct from a single call.

## Running training partition analysis

Run from the repository root in MATLAB.

```matlab
setup;
run_partition_analysis;  % Reads local train/
stats = run_partition_analysis("path/to/training-data", ...
    struct('NumFolds', 5, 'RandomSeed', 42, 'GroupBySubject', true));
```

`run_partition_analysis(trainingPath, config)` uses the existing loader. It prints and returns aggregate counts, per-fold subject overlap, validation coverage, partition integrity, and same-seed reproducibility, including global random-state preservation. The report stays in the console and the optional returned struct. It uses the supplied training folder and performs no preprocessing or model fitting. Keep real fold assignments, filenames, and metadata local.

With MATLAB R2026a Update 5 on Windows, the local official training set had 200 images, 100 per class, and 100 filename-derived subject IDs, 50 per class and two images per ID. The default options produced these verified aggregates.

| Fold | Train images (male/female) | Validation images (male/female) | Train IDs | Validation IDs (male/female) | ID overlap |
|---:|---:|---:|---:|---:|---:|
| 1 | 160 (80/80) | 40 (20/20) | 80 | 20 (10/10) | 0 |
| 2 | 160 (80/80) | 40 (20/20) | 80 | 20 (10/10) | 0 |
| 3 | 160 (80/80) | 40 (20/20) | 80 | 20 (10/10) | 0 |
| 4 | 160 (80/80) | 40 (20/20) | 80 | 20 (10/10) | 0 |
| 5 | 160 (80/80) | 40 (20/20) | 80 | 20 (10/10) | 0 |

All 200 images appeared in validation exactly once. Both validation classes were present in all five folds, and the repeated run gave identical fold indices and counts.

## Using the shared folds in later pipelines

```matlab
[images, y, subjectIDs, metadata] = loadDataset(trainingPath);
config = struct('NumFolds', 5, 'RandomSeed', 42, 'GroupBySubject', true);
preprocessingConfig = struct();
[folds, partitionInfo] = createFolds(y, subjectIDs, config);

for k = 1:numel(folds)
    trainIdx = folds(k).TrainIndices;
    valIdx = folds(k).ValidationIndices;

    trainingImages = images(trainIdx);
    validationImages = images(valIdx);
    yTrain = y(trainIdx);
    yValidation = y(valIdx);

    preprocessor = fitPreprocessor(trainingImages, preprocessingConfig);
    XTrain = transformImages(preprocessor, trainingImages);
    XValidation = transformImages(preprocessor, validationImages);

    % PCA, LDA, and classifier fitting belong to subsequent issues.
end
```

`metadata(trainIdx,:)` and `metadata(valIdx,:)` follow the same order. `XTrain` and `XValidation` each have one row per input image and D feature columns. Each fold fits its preprocessing using only that fold's training images. See [dataset loading and preprocessing](data-preprocessing.md).

## Tests

Run `matlab -batch "run_tests;"` from the repository root, or `run_tests;` in MATLAB. Base MATLAB is sufficient. The suite covers balanced and imbalanced splits, unequal groups, fallback, invalid inputs, and reproducibility. Integration tests create temporary synthetic PNGs and verify image, label, metadata, and feature alignment. The independent test dataset is reserved for final model evaluation.
