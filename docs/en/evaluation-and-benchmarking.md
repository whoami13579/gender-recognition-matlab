# Evaluation and benchmarking

[繁體中文](../zh-TW/evaluation-and-benchmarking.md) · [Project requirements](project-requirements.md)

PCA + Bayes, LDA + Bayes, PCA + MLP, and LDA + MLP share the same accuracy, ROC, AUC, feature dimension, parameter count, training time, and inference time definitions. The analysis entry point uses synthetic data and base MATLAB. It was verified on Windows with MATLAB R2026a Update 5.

```matlab
setup;
results = run_evaluation_analysis;  % Console summary only
results = run_evaluation_analysis("results/evaluation-demo");  % JSON and CSV
run_evaluation_analysis("", true);  % Optional ROC plot
```

For headless plotting, set `groot`'s `DefaultFigureVisible` property to `'off'` before running the demo, or pass an axes in an invisible figure to `plotROCCurves(results, ax)`.

## Accuracy, ROC, and AUC

```matlab
[yPred, scores] = predictBayes(bayesModel, ZValidation);
% For MLP, use [yPred, scores] = predictMLP(mlpModel, ZValidation).
metrics = evaluateClassification(yValidation, yPred, scores);
[roc, auc] = computeROC(yValidation, scores);
```

Labels are 0 for male and 1 for female. Female (1) is the positive class. Accuracy is `nnz(yTrue == yPred) / numel(yTrue)`. The canonical value is a fraction. Multiply by 100 for percentage display. The confusion matrix is `[TN FP; FN TP]`, with actual class `[0, 1]` on rows and predicted class `[0, 1]` on columns. Both values use the supplied predictions. Bayes returns log-posterior odds and MLP returns raw logits. Both predict 1 only when `scores > 0`, so zero scores predict 0. ROC uses the raw continuous scores. A sigmoid or other probability conversion can saturate at floating-point 0 or 1 and turn distinguishable scores into ties.

ROC sweeps descending distinct score values and takes `score >= threshold` as positive at each point. At `+Inf`, every finite score is excluded and the curve starts at `(0,0)`. Each group of exactly equal scores enters together. At the smallest finite score both classes are fully included, giving `(1,1)`. At each threshold TPR is the selected positives divided by all actual positives, and FPR is the selected negatives divided by all actual negatives. The inclusive sweep can include zero scores at threshold zero even though the classifiers predict 0 there. Sorting and grouped cumulative counts take O(N log N) time.

AUC is `trapz(roc.FPR, roc.TPR)` on the ordered curve. Equivalently, for a randomly drawn positive and negative pair it is the probability the positive score is higher plus half the tie probability. Perfect ranking gives 1, reversed ranking gives 0, complete ties give 0.5, and labels `[0;0;1;1]` with scores `[0;1;1;2]` give 0.875. Exact ties receive half credit without reference to input order. Accuracy can change while AUC stays fixed because AUC is independent of the classifier's decision threshold.

Inputs must be nonempty, equal-length, real, full N × 1 columns. Labels accept double, single, or logical 0/1. Scores accept finite double or single, with exact single-to-double conversion. Unsupported types, shapes, label values, and nonfinite scores raise `genderRecognition:InvalidEvaluationInput`. With only one class, accuracy and confusion counts still apply. AUC is NaN, all three ROC vectors are empty, `ROC.Valid` is false, and `ROC.Status` is `MissingPositiveClass` or `MissingNegativeClass`. Valid records use `OK`.

## Fitted feature dimensions and parameter counts

```matlab
counts = summarizeModelParameters(preprocessor, projection, classifier);
```

The summary reads input dimension D and retained dimension K from the fitted projection and checks agreement with the preprocessing output and classifier input. `NumFeatures` is the required dimension and can vary across folds with variance-based PCA. `DimensionalityReductionFraction` is `1 - K/D`.

- Preprocessing and projection use their respective `NumFittedParameters`. Fixed grayscale conversion, resizing, and intensity scaling learn zero coefficients. Pass `[]` for a numerical workflow with no preprocessor. PCA stores D means and D × K loadings. Fisher LDA stores D means and D direction coefficients.
- The classifier uses `NumClassifierParameters` for the formal comparison. Gaussian Bayes uses 2K means, plus two sets of K(K+1)/2 symmetric covariance entries for full covariance, one such set for shared covariance, or 2K variances for diagonal covariance. Empirical binary priors add one independent parameter and equal fixed priors add zero. Derived factors and log determinants are excluded.
- MLP counts each layer's `(input width + 1) * output width` weights and biases. The default K → 16 → 1 architecture gives 16K + 33. Separately, `NumNormalizationParameters` counts the K stored standardization means and K scales when enabled, or zero when disabled. A fitted MLP must provide this field. Bayes has zero additional normalization coefficients.
- `CombinedParameterCount` sums all four components. It combines stored transform coefficients with independent statistical or trainable classifier parameters. It should be interpreted as a project-specific count. Serialized file size and mathematical degrees of freedom are separate quantities.

Default counts with zero learned preprocessing parameters

| Method | D → K | Projection | Normalization | Classifier | Combined |
|---|---:|---:|---:|---:|---:|
| PCA + Bayes | 4096 → 40 | 167936 | 0 | 1721 | 169657 |
| LDA + Bayes | 4096 → 1 | 8192 | 0 | 5 | 8197 |
| PCA + MLP | 4096 → 40 | 167936 | 80 | 673 | 168689 |
| LDA + MLP | 4096 → 1 | 8192 | 2 | 49 | 8243 |

## Timing protocol

```matlab
trainConfig = struct('WarmupRuns', 2, 'NumRepeats', 5);
[fitTiming, model] = benchmarkOperation(@() fitBayes(ZTrain, yTrain, bayesConfig), trainConfig);
inferenceConfig = trainConfig;
inferenceConfig.BatchSize = size(ZValidation, 1);
[predictionTiming, labels, scores] = ...
    benchmarkOperation(@() predictBayes(model, ZValidation), inferenceConfig);
```

Use one call per timed repetition. Prepare the zero-input function handle, inputs, and configuration before benchmarking. The utility runs two untimed warm-ups and five timed calls by default. It calls the operation with the same output arity every time and returns the outputs of the first measured call. It times each call with `tic/toc`, including function dispatch and output return. It reports elapsed samples, actual repetition and warm-up counts, median, mean, sample standard deviation (N−1, zero for one measurement), minimum, and maximum. Median is the primary latency. Errors propagate. Callbacks own their random behavior. Use a fresh fit with the same in-memory training data, hyperparameters, and seed on every call. The MLP fitter uses its own seeded stream. For evaluation use the first measured whole-pipeline fit when available. If timing the classifier alone as well, those separate fits serve only as measurements. Inference consistently uses the retained frozen model and identical input batch.

Provide named timing scopes to the result builder. The pipeline entries must come from directly timing the complete sequence in one operation. A sum of separate stage medians is a different statistic.

| Timing key passed to builder | Measured boundary |
|---|---|
| `DatasetLoading` | File discovery, disk reads and decoding, and loader validation |
| `PreprocessingFit` | Fit preprocessing on in-memory training images |
| `TrainingPreprocessing` | Apply the fitted preprocessor to training images |
| `ProjectionFit` | Fit PCA or LDA on preprocessed training features |
| `TrainingProjection` | Apply the fitted PCA or LDA model to training features |
| `ClassifierFit` | Fit Bayes or MLP on in-memory reduced features, including MLP standardization |
| `PipelineFit` | Direct wall-clock measurement from preprocessing fit through classifier fit, including both training transformations |
| `ValidationPreprocessing` | Apply frozen preprocessing to the evaluation batch already in memory |
| `ValidationProjection` | Apply the frozen projection to preprocessed evaluation features |
| `ClassifierInference` | Predict labels and compute scores on in-memory reduced features, including model validation and any fixed normalization |
| `PipelineInference` | Direct wall-clock measurement of frozen preprocessing, projection, and prediction together |
| `MetricCalculation` | Accuracy, confusion counts, ROC, and AUC from outputs already in memory |
| `ResultExport` | Serialize and write previously built results, measured separately if needed |

Classifier and pipeline fit/inference exclude disk and model I/O, metric calculation, serialization, plotting, and unrelated console output. Inference contains no fitting. The synthetic demo leaves disk loading and export timings unmeasured. To measure export for an existing record, benchmark a separate write to a designated local directory. The file used to measure export cannot contain the result of its own write.

All durations are in seconds. `BatchSize` is required for inference scopes, and both scopes use the same size when supplied in the same record. `SecondsPerSample` is median batch seconds divided by batch size, and `MeanSecondsPerSample` uses the batch mean. These measure batch-derived averages. Measure a batch of one for single-image latency. The timed batch size is tracked independently of the evaluation sample count. Use identical batches and timing settings across methods. Current plans use 40 validation samples in Issue #11 and 100 independent test samples in Issue #12. Benchmarks depend on hardware, MATLAB version, threading, cache state, and competing load. Record the environment and timing settings when comparing runs.

## Result schema and export

```matlab
context = struct('Method', "pca-bayes", 'EvaluationStage', "cross-validation", 'FoldNumber', 1);
timings = struct('ClassifierFit', fitTiming, 'ClassifierInference', predictionTiming);
result = buildEvaluationResult(context, metrics, preprocessor, projection, model, timings);
files = exportEvaluationResults(result, "results/evaluation-demo");
rows = jsondecode(fileread(files.JSON));
summary = readtable(files.CSV, 'TextType', 'string');
```

Set `Method` to `pca-bayes`, `lda-bayes`, `pca-mlp`, or `lda-mlp`. Set `EvaluationStage` to `synthetic`, `cross-validation`, or `independent-test`. A positive integer `FoldNumber` is required for cross-validation. It is `[]` for the other stages.

| Schema v1 fields | Meaning |
|---|---|
| `SchemaVersion`, `Method`, `EvaluationStage`, `FoldNumber` | Record identity, schema version 1, and optional fold |
| `NumTrainingSamples`, `NumEvaluationSamples`, `NumCorrect`, `Accuracy`, `ConfusionMatrix` | Fitted training count and evaluation counts with fractional accuracy |
| `AUC`, `ROC` | AUC and nested `FPR`, `TPR`, `Thresholds`, `NumPositiveSamples`, `NumNegativeSamples`, `NumDistinctScores`, `Valid`, and `Status` |
| `NumOriginalFeatures`, `NumFeatures`, `DimensionalityReductionFraction` | Fitted dimensions and reduction fraction |
| `NumPreprocessingParameters`, `NumProjectionParameters`, `NumNormalizationParameters`, `NumClassifierParameters`, `CombinedParameterCount` | Separate fitted counts and the sum |
| `Timing.Units`, `Timing.Training`, `Timing.Inference`, `Timing.Stages` | `seconds`, nested `Classifier` and `Pipeline` measurements, and named component scopes from the timing table |
| `Configuration` | Effective `Preprocessing`, `Projection`, and `Classifier` fit options from model metadata |

Each timing measurement has `SamplesSeconds`, `WarmupRuns`, `NumRepeats`, `MedianSeconds`, `MeanSeconds`, `StdSeconds`, `MinSeconds`, `MaxSeconds`, `BatchSize`, `SecondsPerSample`, and `MeanSecondsPerSample`. Unmeasured scopes and fields outside a scope are `[]`. The export validates records and accepts only known model options, so metadata fields for paths, fitted coefficients, or sample records are excluded.

Export writes `evaluation-results.json` and `evaluation-summary.csv` to the supplied directory and replaces those two files if they already exist. The JSON top level is an array of records. A valid curve starts with its origin in both coordinate arrays and stores the remaining thresholds as finite numbers. Prepend `Inf` to `ROC.Thresholds` when `ROC.InitialThreshold` is `"positive-infinity"`. An undefined curve has empty vectors and an empty `InitialThreshold`. Its `AUC` is JSON `null` (MATLAB decodes this scalar field as `[]`). These are excerpts for a valid complete-tie curve and an undefined curve.

```json
[
  {"AUC":0.5,"ROC":{"FPR":[0,1],"TPR":[0,1],"Thresholds":[0],"InitialThreshold":"positive-infinity","Valid":true,"Status":"OK"}},
  {"AUC":null,"ROC":{"FPR":[],"TPR":[],"Thresholds":[],"InitialThreshold":[],"Valid":false,"Status":"MissingNegativeClass"}}
]
```

The CSV has one row per record and scalar columns only. Training and inference latency columns contain medians in seconds. Missing values are blank. This is an excerpt of the scalar columns.

```csv
Method,EvaluationStage,FoldNumber,Accuracy,AUC,ROCValid,NumFeatures,NumClassifierParameters
pca-bayes,synthetic,,0.75,0.875,1,3,19
pca-bayes,synthetic,,0.5,,0,3,19
```

`plotROCCurves` reads in-memory coordinates, draws a random-ranking diagonal, and labels valid curves with method and AUC. An invalid curve is skipped with its diagnostic. Keep real-data result exports under the ignored `results/` directory or another private location. Finite ROC thresholds retain distinct raw score values.

## Lecture references and later integration

Chapter 2, §2.8.3 (combined lecture PDF pages 181–186) defines hit and false-alarm probabilities and illustrates threshold sweeping. Chapter 3, §3.7.2 (pages 266–274) discusses how dimension and sample size affect Gaussian parameter estimation and discrimination costs. Both ranges were checked against the locally available combined lecture PDF. Trapezoidal AUC, the grouped empirical sweep, timing settings, and serialization are project conventions. See the [lecture reference](lecture-reference.md).

Issue #11 can store one row per method and fold, compute the mean and standard deviation of per-fold accuracy and AUC, and build comparisons from the saved coordinates and scalar counts. Raw scores from independently fitted models can have different scales. A pooled ROC can differ from the per-fold AUC summary, and averaging thresholds across folds requires comparable score scales. Document and justify the aggregation strategy in Issue #11. Issue #12 can use the same schema with the frozen final model and independent evaluation batch.
