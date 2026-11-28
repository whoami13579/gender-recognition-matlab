# Gaussian Bayes classifier

[繁體中文](../zh-TW/gaussian-bayes-classifier.md) · [Project home](../../README.md)

## Bayesian decision rule

For label $c$ and a $K$-dimensional feature vector $z$, the prior $P(c)$ is the class probability before observing $z$. The class-conditional likelihood $p(z\mid c)$ is a density, and the posterior $P(c\mid z)=p(z\mid c)P(c)/p(z)$ is the probability after observing the features. With equal misclassification costs, the minimum-error rule chooses the larger posterior. The evidence $p(z)$ is common to both classes and cancels in their comparison.

The Gaussian model uses a class mean $\mu_c$ and positive-definite covariance $\Sigma_c$. The squared Mahalanobis distance $d_c^2=(z-\mu_c)^\top\Sigma_c^{-1}(z-\mu_c)$ measures separation in covariance-scaled units. The multivariate density is $(2\pi)^{-K/2}\det(\Sigma_c)^{-1/2}\exp(-d_c^2/2)$. At $K=1$, the covariance is a scalar variance $\sigma_c^2$ and the density is $\exp(-(z-\mu_c)^2/(2\sigma_c^2))/\sqrt{2\pi\sigma_c^2}$.

The classifier compares the Gaussian log discriminants.

$$
g_c(z)=\log P(c)-\tfrac K2\log(2\pi)-\tfrac12\log\det(\Sigma_c)-\tfrac12 d_c^2
$$

For a full covariance, Cholesky factorization gives $\Sigma_c=L_cL_c^\top$. The code solves for the whitened feature row using `(Z - mu) / L.'`, sums squared entries for $d_c^2$, and obtains $\log\det(\Sigma_c)$ from $2\sum_j\log L_{c,jj}$. The diagonal mode uses elementwise division by standard deviations. The implementation evaluates $g_1-g_0$ by differencing the terms before addition, with the common $K\log(2\pi)/2$ term canceled.

The score is the fitted log-posterior odds $\log[P(1\mid z)/P(0\mid z)]$. Larger values give stronger evidence for female (`1`). A positive score predicts `1`, and a negative or exactly zero score predicts male (`0`). A score depends on its row and the fitted model. Direct Gaussian densities can underflow to zero for narrow distributions. Even a stable logistic conversion can round different log odds to the same posterior probability near 0 or 1, so keep the log odds for ranking. The API returns two outputs.

## Maximum-likelihood fit

For $N_c$ training rows in class $c$ and $N=N_0+N_1$, the default empirical prior is $P(c)=N_c/N$. The `"equal"` option fixes both priors at $0.5$. Each class mean is the training average $\mu_c=\operatorname{mean}(Z_c,1)$, and $R_c$ comprises the centered rows $Z_c-\mu_c$. The class covariance estimate $S_c=R_c^\top R_c/N_c$ uses the Gaussian maximum-likelihood divisor $N_c$. The unbiased sample covariance instead divides by $N_c-1$. The pooled estimate $S_{\mathrm{pooled}}=(R_0^\top R_0+R_1^\top R_1)/N$ uses the pooled maximum-likelihood divisor $N$, where the unbiased pooled estimator uses $N-2$.

| `CovarianceType` | Estimate before ridge | Statistical assumption | Independent covariance parameters |
|---|---|---|---:|
| `"full"` | One $S_c$ per class | Class-specific full Gaussian, generally quadratic discriminant | $K(K+1)$ |
| `"shared"` | One $S_{\mathrm{pooled}}$ | Common full Gaussian covariance, yielding an affine discriminant | $K(K+1)/2$ |
| `"diagonal"` | Diagonal of each $S_c$ | Conditional feature independence, or Gaussian naive Bayes | $2K$ |

The shared covariance classifier operates on the projected features. Fisher LDA is a supervised feature extractor fitted upstream and produces one projection coordinate for this project.

For the fixed two-class case, dense fitting takes $O(NK^2+K^3)$ operations including covariance formation, rank diagnostics, and factorization. A dense triangular solve costs $O(K^2)$ per prediction row. The diagonal mode takes $O(NK)$ to fit and $O(K)$ per row to score.

## Regularization and numerical limits

Repeated or linearly dependent features and a small sample count can yield singular or near-singular estimates. The rank of a class residual matrix is at most $N_c-1$, with a zero class covariance for a singleton. All modes add $\alpha I$ (or $\alpha$ to each retained diagonal variance). The common ridge is $\alpha=\rho s$, where $\rho$ is the positive dimensionless `Regularization` coefficient and $s=\operatorname{tr}(S_{\mathrm{pooled}})/K$.

If all within-class residuals are zero but the means differ, $s=w_0w_1\lVert\mu_1-\mu_0\rVert^2/K$, using the training class fractions $w_c=N_c/N$ even with fixed priors. This is the average total training variance in that case. For identical constant rows across both classes, the within-class and between-class scales are zero, so the convention is $s=1$ in squared input units. The two resulting likelihoods are identical and cancel for any finite feature row, leaving the fitted prior odds.

The fit checks covariance, ridge, and factor representability; prediction checks centering, squared distances, and scores. An unusable positive-definite factorization raises `genderRecognition:BayesCovarianceFactorization`. An overflow or loss of a positive variance, ridge, or entire squared distance to underflow raises `genderRecognition:BayesNumericalRange`. The fit uses the configured ridge directly. The default is a numerical baseline. A ridge proportional to very small within-class LDA variances can remain small, and numerical stability alone gives no guarantee about generalization, calibration, or the best choice of covariance or ridge.

## Interface and fitted state

```matlab
setup;
config = struct('CovarianceType', "full", 'PriorType', "empirical", ...
    'Regularization', 1e-2);  % Defaults
bayesModel = fitBayes(ZTrain, yTrain, config);
[yPred, scores] = predictBayes(bayesModel, ZValidation);
```

Rows are samples and columns are features. Fit accepts full, real, finite single or double $N\times K$ feature matrices with at least one feature and both classes represented. Labels are an aligned double $N\times1$ column of 0s and 1s. Calculations use double precision. Prediction requires the fitted feature dimension and returns two double columns in input row order, including two $0\times1$ columns for a $0\times K$ input. Config accepts the three listed fields and supported, case-sensitive option values. Unknown fields and invalid inputs or fitted scoring state raise `genderRecognition:...` errors.

| Model fields | Contents |
|---|---|
| `ClassLabels`, `ClassCounts`, `Priors` | Label order `[0 1]`, training counts and fitted or fixed priors |
| `ClassMeans` | $2\times K$ training mean matrix |
| `CovarianceType`, `Covariances`, `CovarianceFactors`, `LogDeterminants` | Mode, regularized covariance, factor and log determinant for each stored covariance. Full uses $K\times K\times2$. Shared uses a single $K\times K$ matrix. Diagonal uses $1\times K\times2$ rows of variances or standard deviations. |
| `CovarianceRanks` | Numerical ranks before the ridge, in covariance storage order |
| `RegularizationCoefficient`, `EffectiveRegularization`, `CovarianceScale`, `CovarianceScaleSource` | $\rho$, $\alpha$, $s$ and the source `"withinClass"`, `"betweenClass"` or `"constant"` |
| `NumTrainingSamples`, `NumFeatures`, `NumClassifierParameters`, `Options` | Training sample count, input dimension, independent parameter count, and effective configuration |

The count is $2K$ for the class means plus the covariance entries above plus one for empirical priors. The two probabilities sum to one. Fixed equal priors add zero. Counts describe model structure after choosing the covariance assumption. Symmetric copies, factors, log determinants, the fixed ridge coefficient, derived diagnostics, and metadata add no independent fitted parameters. Projection coefficients are reported separately as `NumFittedParameters` on the feature models.

| Mode | Empirical-prior formula | $K=1$ empirical / equal | $K=40$ empirical / equal |
|---|---|---:|---:|
| Full | $K^2+3K+1$ | 5 / 4 | 1,721 / 1,720 |
| Shared | $2K+K(K+1)/2+1$ | 4 / 3 | 901 / 900 |
| Diagonal | $4K+1$ | 5 / 4 | 161 / 160 |

## Fit on a subject-grouped training split

```matlab
[images, y, subjectIDs] = loadDataset("train");
folds = createFolds(y, subjectIDs);
iTrain = folds(1).TrainIndices;
iValidation = folds(1).ValidationIndices;
prep = fitPreprocessor(images(iTrain));
XTrain = transformImages(prep, images(iTrain));
XValidation = transformImages(prep, images(iValidation));
yTrain = y(iTrain);

pcaModel = fitPCA(XTrain, struct('NumComponents', 40));
pcaBayes = fitBayes(transformPCA(pcaModel, XTrain), yTrain);
[pcaLabels, pcaScores] = predictBayes(pcaBayes, transformPCA(pcaModel, XValidation));

ldaModel = fitLDA(XTrain, yTrain);
ldaBayes = fitBayes(transformLDA(ldaModel, XTrain), yTrain);
[ldaLabels, ldaScores] = predictBayes(ldaBayes, transformLDA(ldaModel, XValidation));
```

Choose the split before fitting. Reuse the fitted preprocessing, projection, and classifier for held-out features. Validation labels are unnecessary for prediction. Keep the covariance and ridge settings independent of held-out data.

## Official training dataset verification

```matlab
setup;
run_bayes_analysis;  % Local train/, default PCA and fold 1
run_bayes_analysis("train", "lda", struct('CovarianceType', "full"));
run_bayes_analysis("train", "pca", struct(), struct('NumComponents', 40), struct(), 1);
```

The optional fourth, fifth, and sixth arguments supply projection settings, preprocessing settings, and the representative fold number. This utility times the classifier fit and one validation batch in seconds, with loading, preprocessing, and projection outside both timers. It returns and prints only aggregate numerical diagnostics.

The local official dataset had 200 RGB images (100 per class) at 640 × 640 × 3. Default preprocessing gave 64 × 64 grayscale images and 200 × 4,096 double features in [0, 1]. Default subject-grouped fold 1 had 160 fitting images (80/80), 40 validation images (20/20), and zero subject-ID overlap. Both runs used class-specific full covariance, priors [0.5, 0.5] and $\rho=0.01$.

| Fold 1 observation | PCA + Bayes | LDA + Bayes |
|---|---:|---:|
| Reduced training / validation shapes | 160 × 40 / 40 × 40 | 160 × 1 / 40 × 1 |
| Class mean / per-class covariance shapes | 2 × 40 / 40 × 40 | 2 × 1 / 1 × 1 |
| Unregularized ranks [male, female] | [40, 40] | [1, 1] |
| Successful Cholesky factorizations | 2/2 | 2/2 |
| Within-class scale $s$ | 2.75740801 | 1.17513355e−6 |
| Effective ridge $\alpha$ | 0.0275740801 | 1.17513355e−8 |
| Classifier parameters / stored fitted projection coefficients | 1,721 / 167,936 | 5 / 8,192 |
| Label / score shapes | 40 × 1 / 40 × 1 | 40 × 1 / 40 × 1 |
| Finite scores / distinct scores | 40/40 / 40 | 40/40 / 40 |
| Score range | [−373.84718, 2281.29189] | [−6839418.22, 11777179.7] |
| Predicted counts [male, female] | [20, 20] | [20, 20] |
| Classifier fit / 40-row prediction (seconds) | 0.029393 / 0.010145 | 0.011338 / 0.003726 |

The times are single local measurements. Individual and batch rankings agreed for both projections. Changing validation pixels in memory changed validation scores and left all fitted training states identical. The run checked numerical behavior in a single fold and left the independent testing dataset untouched.

## Testing and lecture references

Run `matlab -batch "run_tests;"` from the repository root. All 84 tests passed, including 67 existing tests and 17 new Gaussian Bayes tests. The synthetic suite uses independent Gaussian reference calculations, singular and narrow-variance cases, parameter counts, score ranking and tie checks, and temporary PNGs for PCA and LDA integration and data separation. Verification used Windows and base MATLAB R2026a Update 5 (26.1.0.3346908).

The combined lecture PDF was checked locally. Chapter 2 §§2.2–2.6 covers Bayes probabilities, minimum-error decisions, Gaussian densities, and Gaussian discriminants (PDF pages 74–84, 86–115, 117–126, 128–141, 143–179). The lecture compares common spherical, common full, and arbitrary class-dependent covariance. The project implements class-specific diagonal covariance as an additional model assumption. Chapter 3 §3.2 (pages 195–221) covers maximum-likelihood estimates and the covariance bias from fitting the mean. Pages 268–272 discuss fitting cost, distinct symmetric entries, and singular estimates when samples are limited. The ridge scale and fallbacks, Cholesky-based evaluation, and log-odds output convention are project engineering choices. [Lecture reference and system architecture](lecture-reference.md)
