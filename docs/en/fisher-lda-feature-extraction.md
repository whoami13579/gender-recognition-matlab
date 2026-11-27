# Fisher LDA feature extraction

[繁體中文](../zh-TW/fisher-lda-feature-extraction.md) · [Project home](../../README.md)

Fisher linear discriminant analysis uses training labels to find a direction that separates the class means while keeping variation within each class small. PCA learns directions of high overall variance without labels. The two feature extractors run independently on the original preprocessed rows. LDA decomposes **within-class residuals** as part of its numerical solve, whereas PCA decomposes globally centered samples to retain variance.

## Fisher's criterion and the lecture

Chapter 3 §3.8.2, *Fisher Linear Discriminant*, was checked in the local combined lecture PDF on pages 291–312 (slides 103–124). Pages 297–305 define projected means, within-class scatter, between-class scatter, and their quotient. Pages 306–307 give the conventional solution for nonsingular scatter. The ridge and compressed solve below are project choices for singular scatter. [Lecture reference and system architecture](lecture-reference.md)

Let $X_k$ contain the $N_k$ training rows with class $k$, where $k=0$ is male and $k=1$ is female (the positive class). The row means are $\mu_k=\operatorname{mean}(X_k,1)$, the global row mean is $\mu=\operatorname{mean}(X,1)$, and the column mean difference is $\delta=(\mu_1-\mu_0)^\top$. Let $R$ stack the rows of $X_0-\mu_0$ and $X_1-\mu_1$. Summing the residual outer products gives the within-class scatter $S_W=R^\top R$. The rank-one between-class scatter is $S_B=\delta\delta^\top$. The sample-count-weighted definition scales $S_B$ by $N_0N_1/(N_0+N_1)>0$, so it gives the same maximizing direction.

$$
J(w)=\frac{w^\top S_Bw}{w^\top S_Ww}
    =\frac{(w^\top\delta)^2}{\lVert Rw\rVert_2^2}.
$$

For nonsingular $S_W$, the maximizer is proportional to $S_W^{-1}\delta$. Each class's residual rows sum to zero, giving $\operatorname{rank}(S_W)=\operatorname{rank}(R)\le\min(N-2,D)$. With $D=4096$, the bounds are 198 for all 200 training rows and 158 for a 160-row fold, so both scatter matrices are singular. With distinct class means, $S_B$ has one nonzero direction. Binary LDA therefore produces one feature.

## Ridge and low-rank solve

Add a positive ridge $\alpha$ to get a positive definite system. The direction maximizing the regularized Fisher criterion is proportional to the solution of $(S_W+\alpha I)w_{\rm raw}=\delta$.

$$
J_\alpha(w)=\frac{(w^\top\delta)^2}{\lVert Rw\rVert_2^2+\alpha\lVert w\rVert_2^2},
\qquad
\alpha=\rho\,\frac{\operatorname{tr}(S_W)}{D}
      =\rho\,\frac{\lVert R\rVert_F^2}{D}.
$$

Pass a scalar struct with optional `Regularization`, a positive finite real numeric scalar $\rho$. The default `1e-2` is an engineering baseline. $\rho$ is dimensionless and $\alpha$ has the same squared-feature units as scatter. Uniformly multiplying the inputs by $a>0$ multiplies the ridge and scatter by $a^2$, preserving the direction up to rounding. A smaller ridge puts more weight on directions of low within-class scatter, including the null space. A larger ridge moves the direction toward $\delta$.

For exactly zero within-class scatter, the fallback is $\alpha=\rho\lVert\delta\rVert_2^2/D$ and the direction is parallel to $\delta$. Identical class means raise `genderRecognition:IdenticalLDAClassMeans`. The fit requires a representable scatter trace (positive for nonzero residuals) and positive normal double effective and scaled ridges. A nonfinite or unusable result, a nonpositive Fisher score, a relative solver residual above `1e-6`, or an unrepresentable ridge raises `genderRecognition:LDANumericalRange`. These numerical checks make the supported ridge range data dependent. The default was used without tuning on validation or independent test data.

Conceptually, for an economy SVD $R=U\Sigma V^\top$, the scale-invariant solution can be computed as

$$
\alpha w_{\rm raw}
=V\,\operatorname{diag}\!\left(\frac{\alpha}{s_i^2+\alpha}\right)V^\top\delta
 +\bigl(\delta-VV^\top\delta\bigr).
$$

The second term preserves the mean shift orthogonal to the economy right singular vectors. The implementation avoids subtracting nearly equal feature-space vectors to compute it. It scales $R$ to $A=R/c$ with $c=\max(\lVert R\rVert_F,\sqrt{\alpha})$, uses $b=\delta/\lVert\delta\rVert_2$, and sets $\beta=\alpha/c^2$. For wide inputs, a thin QR factorization $[b\;\; A^\top]=QT$ puts both the residual rows and mean shift in a subspace of dimension at most $N+1$. The solution belongs to this span because the ridge acts as a scalar on its orthogonal complement and the right-hand side there is zero. The reduced right-hand side is $T(:,1)$ and the reduced residual matrix is $T(:,2{:}\mathrm{end})^\top$.

Economy SVD of the reduced residual matrix with one zero row appended supplies all right singular directions of the reduced problem, including its null directions. For each scaled singular value $s_i$, the solver weights its right-hand-side coefficient by $\beta/(s_i^2+\beta)$ and maps the result back with $Q$. When $N\ge D$, economy SVD of $A$ supplies all $D$ right directions directly. Both paths compute $u=\beta(A^\top A+\beta I)^{-1}b$ through orthogonal factors and scalar weights. No feature-space scatter, covariance, identity, or inverse is formed. For wide inputs the basis has at most $D(N+1)$ entries and the SVD works on an $(N+1)\times(N+1)$ matrix.

The fit reports $\lVert A^\top(Au)+\beta(u-b)\rVert_2/(\beta\lVert b\rVert_2)$ using only matrix-vector products. Numerical rank counts **unscaled residual singular values** strictly above `max(N,D) * eps(sigmaMax)` and caps the result at $\min(N-2,D)$. Scatter eigenvalues are the squared residual singular values. The rank cutoff is diagnostic and the solve uses the full spectrum of the reduced problem. The direction is normalized to unit Euclidean norm and oriented so that $w^\top\delta>0$.

## Fit, transform, and model fields

```matlab
setup;
ldaModel = fitLDA(XTrain, yTrain, struct('Regularization', 1e-2));
ZTrain = transformLDA(ldaModel, XTrain);
ZValidation = transformLDA(ldaModel, XValidation);
```

Inputs are full, real, finite `single` or `double` matrices with samples in rows. Fit requires an aligned $N\times1$ double column of labels and at least one row of each class. Calculation and output use double precision. Transform applies $Z=(X-\mu)w$ to each row using the frozen global training mean and direction. It takes no labels, preserves row order, and returns an $N\times1$ column, including $0\times1$ for a correctly shaped empty batch. Training centering matches the PCA interface and leaves the Fisher criterion unchanged. The projected coordinate is a feature for use by a classifier.

| Fitted field | Meaning |
|---|---|
| `Mean`, `Components` | $1\times D$ global mean and $D\times1$ unit direction used at transformation. |
| `ClassMeans`, `ClassCounts` | $2\times D$ class means and $1\times2$ counts, in label order `[0 1]`. |
| `WithinClassRank`, `RankTolerance` | Residual numerical rank and singular-value cutoff. |
| `RegularizationCoefficient`, `EffectiveRegularization`, `Options` | Configured $\rho$, computed $\alpha$, and effective configuration struct. |
| `NumTrainingSamples`, `NumInputFeatures`, `NumFeatures` | $N$, $D$, and one output column. |
| `NumFittedParameters` | $2D$ stored mean and direction coefficients used for projection. |
| `MeanDifferenceNorm`, `WithinClassScatterTrace`, `UsedZeroScatterFallback` | Norm of $\delta$, $\lVert R\rVert_F^2$, and fallback flag. |
| `ProjectedClassMeans`, `ProjectedMeanSeparation`, `ProjectedWithinClassScatter` | Centered scalar class means in label order, their female-minus-male difference, and $\lVert Rw\rVert_2^2$. |
| `RegularizedFisherCriterion`, `RelativeResidual` | $J_\alpha(w)$ and the scaled linear-system residual (zero for the analytic fallback). |

At $D=4096$, the projection uses 8,192 stored coefficients. This is the same storage convention as PCA. Count statistical degrees of freedom, serialized model size, and fitted Bayes or MLP parameters separately. Training rows, residuals, and SVD factors are temporary. Keep the model, including its class means and direction, local.

### Use a shared training fold

```matlab
[images, y, subjectIDs] = loadDataset(trainingPath);
[folds, ~] = createFolds(y, subjectIDs);
trainIdx = folds(1).TrainIndices;
valIdx = folds(1).ValidationIndices;

preprocessor = fitPreprocessor(images(trainIdx));
XTrain = transformImages(preprocessor, images(trainIdx));
XValidation = transformImages(preprocessor, images(valIdx));
yTrain = y(trainIdx);

ldaModel = fitLDA(XTrain, yTrain);
ZTrain = transformLDA(ldaModel, XTrain);
ZValidation = transformLDA(ldaModel, XValidation);
```

Use these same indices for image, label, subject, and feature rows. Fit preprocessing and LDA afresh on each fold's training subset to keep validation data out of fitting. LDA + Bayes and LDA + MLP can train on `ZTrain` and `yTrain` with one input feature. The classifiers supply their own continuous positive-class scores. [Dataset loading and preprocessing](data-preprocessing.md) · [Dataset partitioning](dataset-partitioning.md)

## Local analysis and verified aggregates

```matlab
setup;
run_lda_analysis;  % Descriptive fit on the local train/ directory
stats = run_lda_analysis("path/to/training-data", struct('Regularization', 1e-2));
foldStats = run_lda_analysis("path/to/training-data", ...
    struct('Regularization', 1e-2), struct('ImageSize', [64 64]), 1);
```

The fourth argument selects a default subject-grouped fold. Pass `[]` for the complete training dataset. The analysis reads the supplied training folder, prints aggregate statistics, and returns them in memory. The complete-training fit is descriptive. The fold run fits both the preprocessor and LDA on the fold's training rows.

On Windows with MATLAB R2026a Update 5, the official training folder contained 200 RGB images at 640 × 640 × 3. Default preprocessing yielded 64 × 64 grayscale double features in $[0,1]$ (observed training range 0.00157613–0.994189). Fold 1 had zero overlap in supplied subject IDs. Both fits used $\rho=0.01$, a 1 × 4,096 mean, 2 × 4,096 class means, and a 4,096 × 1 direction with 8,192 fitted projection coefficients. The residual shape equals the fitting feature shape, and the maximum output dimension is one.

| Metric | Complete training | Fold 1 training | Fold 1 validation |
|---|---:|---:|---:|
| Male / female samples | 100 / 100 | 80 / 80 | 20 / 20 |
| Features before → after | 200 × 4,096 → 200 × 1 | 160 × 4,096 → 160 × 1 | 40 × 4,096 → 40 × 1 |
| Within-class rank / maximum | 198 / 198 | 158 / 158 | — |
| Effective ridge $\alpha$ | 0.0598684235 | 0.0472629789 | — |
| Projected means [male, female] | −1.48221597, 1.48221597 | −1.76143457, 1.76143457 | −1.76640547, 1.82323874 |
| Projected sample variances [male, female] | 1.68731962e−6, 2.30960494e−6 | 1.10045735e−6, 1.27955996e−6 | 0.155990164, 0.322504941 |
| Female-minus-male mean separation | 2.96443193 | 3.52286913 | 3.58964421 |
| Regularized Fisher criterion | 145.822370 | 261.545739 | — |
| Relative solver residual | 3.73312133e−12 | 4.32189081e−12 | — |
| Feature matrix bytes before → after | 6,553,600 → 1,600 | 5,242,880 → 1,280 | 1,310,720 → 320 |
| LDA fit (s) | 0.097200 | 0.030176 | — |
| LDA transform (s) | 0.006294 | 0.003629 | 0.002458 |

Times are single-run batch measurements excluding loading and preprocessing. Storage covers feature matrices only. Class variances use MATLAB's sample variance, with zero for a singleton. Separation and variance describe each projected batch. Classifier training and evaluation are separate project stages. The independent testing dataset was reserved for final evaluation.

## Automated verification

Run `matlab -batch "run_tests;"` from the repository root or `run_tests;` inside MATLAB. Base MATLAB is sufficient. `TestLDA` uses independent dense solves for small full-rank and singular cases, checks the null-space signal and the regularized objective, and verifies rank and the linear-system relationship for a 160 × 4,096 synthetic matrix. Temporary PNGs exercise subject grouping, row alignment, fold-local fitting, and held-out image changes. The synthetic suite runs without any private assets.
