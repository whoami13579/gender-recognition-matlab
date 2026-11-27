# PCA feature extraction

[繁體中文](../zh-TW/pca-feature-extraction.md) · [Project home](../../README.md)

Principal component analysis (PCA) reduces an image's feature vector to a smaller set of coordinates that preserve as much training variance as possible. It uses feature values alone. The PCA + Bayes and PCA + MLP pipelines can share this projection interface and keep labels aligned with the feature rows.

## From the lecture's scatter matrix to economy SVD

Chapter 3, §3.8.1, *Principal Component Analysis*, was checked against combined lecture PDF pages 277–290. The lecture derives the sample mean and the least-squares projection onto the leading eigenvectors of the scatter matrix, then extends the argument to multiple directions. See the [lecture reference](lecture-reference.md).

With $N$ training samples in rows and $D$ features in columns, compute the row mean $\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$ and center each row of $X$ by subtracting $\mu$ to form $X_c$. In this orientation, the lecture's scatter matrix is $S = X_c^\top X_c$. The sample covariance is $C = S/(N-1)$. The two matrices have the same directions and their eigenvalues differ by the factor $N-1$.

The implementation computes the economy SVD $X_c = U\Sigma V^\top$. The columns of $V$ are the principal directions, ordered by descending singular value $\sigma_i$. Since $X_c^\top X_c = V\Sigma^\top\Sigma V^\top$ in this factorization, the sample covariance eigenvalues are $\lambda_i = \sigma_i^2/(N-1)$. The variance fraction of component $i$ is $\mathrm{EVR}_i = \lambda_i/\sum_j\lambda_j$ and the cumulative fraction is $\mathrm{CEVR}_k = \sum_{i=1}^k\mathrm{EVR}_i$. The denominator uses the entire economy spectrum, including values below the rank cutoff and those omitted by component selection. Directions omitted by economy SVD have zero eigenvalue in exact arithmetic. The code computes the ratios from $(\sigma_i/\sigma_1)^2$ to keep the denominator in a safe numerical range.

The projection keeps $W = V(:,1{:}K)$ and maps each sample to $Z = (X - \mu)W$, an $N \times K$ matrix for an $N$-row input. Subtraction applies the same fitted mean to every row. Leading directions minimize squared training reconstruction error for a given $K$. SVD may change direction signs, and directions within a repeated-eigenvalue subspace can rotate. Variance and reconstruction within the same subspace are invariant to those changes.

## Rank and component selection

Centering bounds the rank by $\min(N-1,D)$. After SVD, the numerical rank counts singular values strictly greater than `max(N,D) * eps(sigmaMax)`, capped at that bound. `eps(sigmaMax)` is the double precision spacing at the largest singular value. The rank can be smaller with redundant features, duplicate samples, or other linear dependencies. At 4,096 features, the theoretical maxima are 159 directions for 160 training rows and 199 for 200 rows. Economy SVD returns at most $\min(N,D)$ right singular vectors. For wide inputs this keeps the basis compact and avoids forming the $D \times D$ covariance or scatter matrix.

Pass a scalar options struct with at most one of the following settings.

| Setting | Behavior |
|---|---|
| Omitted or `struct()` | Retain `min(40, NumericalRank)`. The 40-component baseline comes from earlier experiments and is an engineering choice. |
| `NumComponents` | Retain the specified positive integer, up to the numerical rank. A request above the rank raises `genderRecognition:PCAComponentsExceedRank`. |
| `ExplainedVarianceThreshold` | Retain the fewest usable components whose cumulative fraction reaches the requested value in $(0,1]$, allowing an absolute tolerance of `8 * eps(1)`. An unreachable threshold raises `genderRecognition:PCAThresholdExceedsRank`. |

Unknown, malformed, or conflicting options raise `genderRecognition:InvalidConfig`. Variance fields and thresholds are fractions. The analysis prints percentages. The component baseline, tolerance, and selection policy are project choices. The variance fraction describes the training features and cannot establish a classification optimum.

## Fit and transform

```matlab
setup;
pcaModel = fitPCA(XTrain, struct('NumComponents', 40));
ZTrain = transformPCA(pcaModel, XTrain);
ZValidation = transformPCA(pcaModel, XValidation);
```

`fitPCA` accepts at least two training rows and one feature column. Both functions accept full, real, finite `single` or `double` matrices. They calculate in double precision and `transformPCA` returns doubles. A `0 × D` input produces a `0 × K` output. The transform checks the fitted coefficient shapes, finiteness, column norms, and dimension metadata. Every row uses the same stored mean and projection, so it can be transformed individually or in a batch to within floating-point tolerance.

Constant training matrices raise `genderRecognition:ZeroPCAVariance`. Centering or projecting values beyond the double precision range, nonfinite SVD output, or a usable covariance eigenvalue that rounds to zero raises `genderRecognition:PCANumericalRange`. The calculation uses mean centering and an ordinary orthonormal projection.

| Fitted field | Meaning |
|---|---|
| `Mean` | $1 \times D$ row of training means. |
| `Components` | $D \times K$ matrix of retained right singular vectors. |
| `Eigenvalues` | Descending sample covariance eigenvalues as a column of length $\min(N,D)$. It includes zero and numerically negligible values. |
| `ExplainedVarianceRatio`, `CumulativeExplainedVariance` | Fraction columns over the same full spectrum. The final cumulative entry is rounded to exactly one. |
| `NumericalRank`, `RankTolerance` | Usable direction count and singular-value cutoff. |
| `NumTrainingSamples`, `NumInputFeatures`, `NumFeatures` | $N$, $D$, and retained $K$. |
| `NumFittedParameters` | $D + DK$ stored coefficients used in projection. |
| `RetainedExplainedVariance` | Cumulative variance fraction at $K$. |
| `Options` | `SelectionMode` is `"default"`, `"fixed"`, or `"explainedVariance"`. `NumComponents` and `ExplainedVarianceThreshold` record the effective request, with the inactive field empty. |

The parameter count includes every stored mean and projection coefficient. Orthonormality constrains the statistically independent degrees of freedom. Count the PCA coefficients separately from fitted classifier parameters. Keep real fitted PCA models local because their coefficients are derived from faces.

### Using one shared training fold

```matlab
[images, y, subjectIDs] = loadDataset(trainingPath);
[folds, ~] = createFolds(y, subjectIDs);

trainIdx = folds(1).TrainIndices;
valIdx = folds(1).ValidationIndices;

preprocessor = fitPreprocessor(images(trainIdx));
XTrain = transformImages(preprocessor, images(trainIdx));
XValidation = transformImages(preprocessor, images(valIdx));

pcaModel = fitPCA(XTrain, struct('NumComponents', 40));
ZTrain = transformPCA(pcaModel, XTrain);
ZValidation = transformPCA(pcaModel, XValidation);
yTrain = y(trainIdx);
yValidation = y(valIdx);
```

Fit both steps afresh using each fold's training indices. Use the frozen state for that fold's validation rows to prevent held-out samples from influencing the projection. The same indices preserve alignment for images, features, labels, and subject IDs. For a smaller training split, choose a count that fits its numerical rank or use threshold mode. See [dataset loading and preprocessing](data-preprocessing.md) and [dataset partitioning](dataset-partitioning.md).

## Local analysis and verified aggregates

```matlab
setup;
run_pca_analysis;  % Full local train/ directory with default settings
stats = run_pca_analysis("path/to/training-data", struct('NumComponents', 40));
foldStats = run_pca_analysis("path/to/training-data", ...
    struct('ExplainedVarianceThreshold', 0.95), struct('ImageSize', [64 64]), 1);
```

The arguments are the training path, PCA options, preprocessing options, optional fold number from the default subject-grouped split, and optional plot flag. Pass `[]` as the fold number for the complete training dataset. Pass `true` as the fifth argument to display the variance fractions and cumulative curve with the retained count and 90%, 95%, and 99% thresholds. Plots default to off. The function prints and returns aggregate statistics and writes no files. The complete-training run is descriptive. Fit separate models for cross-validation.

On Windows with MATLAB R2026a Update 5, the official local training dataset contained 200 RGB images at 640 × 640 × 3, with 100 samples per class. Default preprocessing produced 64 × 64 grayscale `double` features at intensities in $[0,1]$ (observed range 0.00157613–0.994189). Fold 1 had 80 training and 20 validation images per class, with zero overlap in supplied subject IDs. Both fits had a 1 × 4,096 mean, a 4,096 × 40 projection, and 167,936 stored fitted coefficients.

| Metric | Complete training dataset | Fold 1 training | Fold 1 validation |
|---|---:|---:|---:|
| Feature dimensions before → after | 200 × 4,096 → 200 × 40 | 160 × 4,096 → 160 × 40 | 40 × 4,096 → 40 × 40 |
| Numerical rank / theoretical maximum | 199 / 199 | 159 / 159 | Uses fold training fit |
| Explained variance retained by 40 | 93.331369% | 93.764618% | Uses fold training fit |
| Components for 90% / 95% / 99% | 26 / 53 / 128 | 25 / 49 / 109 | Uses fold training fit |
| Matrix storage before → after (bytes) | 6,553,600 → 64,000 | 5,242,880 → 51,200 | 1,310,720 → 12,800 |
| PCA fitting time (s) | 0.065598 | 0.018214 | Uses fold training fit |
| PCA transform time (s) | 0.008630 | 0.004668 | 0.003003 |

Times are batch measurements from one local run, with loading and preprocessing outside the timer. Storage counts feature-matrix contents and excludes the fitted model, input images, and temporary arrays. Actual runtimes vary with hardware and warm-up. The independent test dataset was reserved for final evaluation.

## Automated verification

Run `matlab -batch "run_tests;"` from the repository root, or `run_tests;` in MATLAB. Base MATLAB is sufficient. The tests compare against independent small sample covariances, verify reconstruction and batch-independent projection, cover rank and input boundaries, and fit a synthetic 160 × 4,096 matrix. A temporary PNG dataset verifies subject-grouped fold alignment and that changing held-out images preserves the fitted training state and training projections.
