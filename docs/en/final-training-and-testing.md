# Final training and independent testing

[繁體中文](../zh-TW/final-training-and-testing.md) · [Project home](../../README.md)

## Course protocol

The course handout calls for final evaluation on 100 reserved images, 50 per class, with accuracy, ROC, AUC, and inference time for all four methods. Test-set AUC accounts for 10% of the project grade. The local combined lecture PDF was checked at Chapter 1 §§1.4.4–1.4.5 (PDF pp. 37–38) for training, evaluation and overfitting, and Chapter 6 §6.3.3 (PDF pp. 418–420) for learning curves and the roles of validation and testing. The selection rule, checksums, integrity diagnostics, persistence checks and timing protocol are project decisions. [Course requirements](project-requirements.md) · [Lecture reference](lecture-reference.md)

## Frozen configurations

Issue #11 used a separate stratified and subject-grouped four-fold search over the full training set (seed 42, with 150 fit and 50 validation images per split). It ranked eligible candidates by mean AUC, then mean accuracy, using a tolerance of `1e-12` at each comparison, then by fewer mean combined parameters and predefined candidate order. The final training entry verifies the saved completed nested run against the decoded training fingerprint, the recorded final search and the documented configurations. A unique matching local run can be discovered under `results/cross-validation/nested/`. Set `NestedRun` explicitly to select the intended run when several match. If the recommendation artifact is unavailable, the workflow reconstructs the published settings and records the fallback. A disagreement in available evidence causes an error. [Issue #11 experiments](cross-validation-experiments.md)

| Method | Projection | Classifier |
|---|---|---|
| PCA + Bayes | 10 principal components | Full covariance, empirical priors, ridge 0.01 |
| LDA + Bayes | One Fisher LDA feature, ridge 0.01 | Shared covariance, empirical priors, ridge 0.01 |
| PCA + MLP | 20 principal components | 16 tanh units |
| LDA + MLP | One Fisher LDA feature, ridge 0.01 | 8 tanh units |

All methods use 64 × 64 unit-scale grayscale doubles. Both MLPs use training-fitted standardization, Glorot uniform initialization, seed 42 in a local `mt19937ar` stream, learning rate 0.01, momentum 0.9, 300 epochs, batch size 16, and L2 coefficient `1e-4`. The 4,096 input features are the flattened preprocessed pixels. Each pipeline fits every stage on the complete 200-image training set. The fixed MLP epoch budget uses no early stopping.

## Run and reuse

Run from the repository root with base MATLAB R2026a Update 5 and its JVM. Git is used for source revision metadata. Supply flat, separate local PNG directories. Relative paths resolve from the repository root.

```powershell
matlab -wait -batch "run_tests;"
matlab -wait -batch "setup; run_final_training('train');"
matlab -wait -batch "setup; run_independent_testing('results/final-evaluation/run-REPLACE-ME','test');"
```

Replace `run-REPLACE-ME` with the training command's reported run directory. In an active MATLAB session use `training = run_final_training("train");` and `run_independent_testing(training.OutputDirectory, "test");`. The combined entry is `run_final_evaluation("train", "test")`. For custom output use `OutputDirectory` with a local Git-ignored directory. `NestedRun` and `BaselineRun` optionally name the saved Issue #11 runs. An absent test folder produces `unavailable` and leaves official metrics pending. Rerun only the testing command once the course images arrive.

For the Issue #13 demonstration, use `models = loadFinalModels(trainingRunDirectory);` followed by `[labels, scores] = predictPipeline(models(k), images);`. Model order follows the table above. For a single trusted file, use `loadPipelineModel(modelPath)`. Use `loadEvaluationResults` on the attempt's `evaluation-results.json` for recorded metrics, and use `plotIndependentTestRun(attemptDirectory, figureDirectory)` to rebuild figures from saved records. The comparison figure uses the saved fixed-baseline and nested-outer summaries when both are available. Final PCA + MLP has 20 components, and the nested outer fits had 10.

## Bundle and integrity checks

The training run first saves `frozen-configurations.json` with all method options, provenance, source revision, MATLAB version, training fingerprint, seeds, zero threshold, timestamp and configuration SHA-256. It saves the fitted state as four format-v1 MAT files in `models/`, with version, filenames, byte sizes and SHA-256 checksums in `model-manifest.json`. The new run becomes loadable after all four model verifications and metadata writes succeed. Model loading checks dimensions, parameters, settings and training sample counts. MAT artifacts must come from a trusted local source. Hashes detect unexpected changes to the bundle.

On a deterministic subset of up to 16 training images, the verification compares the original and reloaded fitted state, predicted labels, score order including ties, and finite continuous scores. The acceptance tolerance is eight representable steps at each score pair's maximum absolute magnitude, `8*eps(max(abs(a),abs(b)))`. Maximum absolute and relative differences and exact equality are recorded in `persistence-verification.json`. The workflow checks global RNG preservation. Bundles retain no training images or feature matrices.

The loader accepts `man_<person>_<image>.png` and `woman_<person>_<image>.png`. It normalizes zero padding and qualifies subject numbers by class. The training profile requires 200 RGB `uint8` images at 640 × 640 × 3, with 100 images and 50 subject identifiers per class. The aggregate integrity report checks decoding, unique sample IDs, representation, exact duplicate pairs across samples and subjects, conflicting labels and exact duplicates crossing saved outer folds when those folds are available. The near-duplicate diagnostic counts nonexact pairs with RMSE at most 0.02 between 16 × 16 unit-scale grayscale thumbnails. These candidates do not establish subject identity. A conflict or an exact cross-fold duplicate is recorded as an integrity defect affecting Issue #11.

The testing entry validates and reloads all four saved models before resolving or opening the test directory. Canonical paths reject the same directory, and decoded-pixel SHA-256 detects exact overlap even with different filenames or PNG encodings. Exact cross-set overlap or conflicting identical test images block official metrics. The test metadata distinguishes shared numeric and class-qualified subject identifiers from verified content overlap. Numbering alone cannot establish shared real-world identities. The official test requires 50 images per class and uses one common filename-ordered batch. The fitting code is absent from the testing path.

## Metrics, benchmarks and local records

Gaussian Bayes supplies log-posterior odds and MLP supplies raw logits. Female is the course's positive class (1). Scores greater than zero predict 1, and zero predicts 0. Accuracy comes from the confusion matrix, with rows for actual labels and columns for predicted labels, both ordered `[0, 1]`. ROC uses continuous scores and advances all examples at a tied score together. AUC measures ranking, and accuracy measures the fixed decision rule. Independent testing yields one record per method with no fold number.

Fit timings use fresh fits, two warm-ups and five measured repeats, and retain the first measured full fit. Inference timings use the same reloaded model and batch in each repeat. Medians are primary, with all individual timings and summary statistics retained in seconds. Classifier timing uses projected features, and full-pipeline timing includes preprocessing, projection and classifier prediction. Both exclude disk I/O, integrity checks, serialization, metrics, figures, and exports. Batch-derived seconds per image is the batch median divided by image count. Each train/test timing is measured in memory. Parameter totals include fitted projection, normalization and classifier scalars. File sizes in bytes also reflect metadata, redundant storage and compression.

Under the ignored `results/final-evaluation/run-…/` directory, `training-summary.csv` and `training-metadata.json` hold fit measurements. Each `independent-test/attempt-…/` contains its own status metadata and, when complete, the evaluation JSON/CSV, local predictions MAT file and figures in PNG and vector PDF. Incomplete attempts retain an `.incomplete` suffix. Keep the entire output tree local. Real images, detailed integrity records, fingerprints, paths, identifiers, scores, ROC thresholds, predictions and model coefficients are private.

## Verified official training run

The final fit ran from clean revision `984e14d` in MATLAB R2026a Update 5 on Windows with a six-thread limit, matching the Issue #11 timing setup. The completed nested run from revision `91c6d5e` supplied the recommendations. Its decoded training fingerprint matched the official training folder, and all four saved winners matched the final selection evidence.

All four methods used 200 decoded RGB `uint8` images at 640 × 640 × 3. There were 100 images per class, 200 unique sample IDs, and 50 filename-derived subject IDs per class with two images per ID. The audit found zero exact duplicates, zero conflicting labels, zero duplicates crossing the recorded outer folds, and zero nonexact pairs meeting the thumbnail RMSE threshold. The thumbnail diagnostic uses antialiased bilinear resizing and covered 19,900 image pairs. These observations do not establish real-world identity independence.

| Method | Features | Projection | Normalization | Classifier | Combined | Classifier fit (ms) | Pipeline fit (ms) | MAT bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PCA + Bayes | 10 | 45,056 | 0 | 131 | 45,187 | 0.5338 | 695.7095 | 353,830 |
| LDA + Bayes | 1 | 8,192 | 0 | 4 | 8,196 | 0.1665 | 733.0585 | 125,945 |
| PCA + MLP | 20 | 86,016 | 40 | 353 | 86,409 | 239.9194 | 901.7660 | 674,523 |
| LDA + MLP | 1 | 8,192 | 2 | 25 | 8,219 | 150.2264 | 787.4883 | 131,204 |

All four models preserved their fitted state, 16 verification labels and continuous scores exactly, with maximum absolute and relative score differences of zero and unchanged global RNG state. LDA + Bayes had the smallest classifier and the shortest classifier fit. PCA + Bayes had the shortest full-pipeline fit in this run.

The configured `test/` directory was absent. The local attempt recorded `unavailable` after the bundle passed verification, with zero methods evaluated and no test load timestamp. Official accuracy, ROC, AUC, confusion matrices and inference timings are pending. Synthetic tests verified evaluation, integrity rejection, export/reload and figure reconstruction. Issue #11 measured 100% mean accuracy and 1.000 mean AUC in both cross-validation runs. An independent comparison, including any MLP benefit or effect of the reduced PCA dimensions, requires the held-out images. Small, potentially homogeneous course datasets limit generalization claims, and the classifier predicts supplied course labels. Facial appearance cannot establish gender identity.
