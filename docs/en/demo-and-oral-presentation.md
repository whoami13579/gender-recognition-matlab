# MATLAB demo and oral presentation

[繁體中文](../zh-TW/demo-and-oral-presentation.md) · [Course requirements, §9](project-requirements.md#9-oral-presentation-requirements) · [Project home](../../README.md)

## Prepare an offline copy

Use MATLAB R2026a with the JVM enabled. Verification used R2026a Update 5 on Windows 11 with base MATLAB and no add-on toolboxes. From the repository root, run `setup`. MATLAB and the saved demo can run offline. Git is used to record the source revision when fitting new models. PowerPoint is only needed to rebuild the presentation on Windows.

Copy the chosen complete directories privately. Keep their internal filenames and contents together. The saved demo only needs the two CV runs and the final bundle. Single-image prediction needs one local RGB or grayscale image. A new fit needs the course training data. An independent evaluation needs the separately issued holdout.

```text
train/                                        # optional after fitting, course PNGs
test/                                         # course holdout, when issued
results/cross-validation/baseline/run-*/       # Issue #11 baseline
results/cross-validation/nested/run-*/         # Issue #11 nested
results/final-evaluation/run-*/                # Issue #12 final bundle and training index
  independent-test/attempt-*/                 # saved status/evaluations, if present
results/demo/                                 # new local plots, runs, and attempts
results/presentation/                         # built slides and offline copy
```

Relative paths are anchored at the repository root. Absolute paths are supported for local use. The option fields are independent.

| Option | Default and purpose |
| --- | --- |
| `TrainingPath` | `train`. Used for explicit training and to validate a training demonstration image's location. |
| `TestingPath` | `test`. Consulted only by independent actions. |
| `BaselineRun`, `NestedRun` | Select a completed directory with `experiment-metadata.json`. The local defaults are the corresponding `results/cross-validation` and demo-output subtrees. |
| `FinalRun` | Select the verified run directory containing `model-manifest.json`. Searches `results/final-evaluation` and the equivalent demo-output subtree when omitted. |
| `IndependentAttempt` | Select a saved attempt directory containing `test-metadata.json`. Independent actions discover a unique completed matching attempt under the bundle or demo output. `figures` uses an explicitly selected attempt for independent figures. |
| `OutputDirectory` | `results/demo`. Place new outputs in an ignored local directory. |

If multiple compatible runs exist, copy the desired candidate from the diagnostic and set its selector explicitly. Select the reported Issue #11 baseline and nested runs at revision `91c6d5e`; their full paths also appear in the saved fit and experiment records. The other local baseline came from `66f82ff`. The official bundle was fitted at `984e14d`. Each loaded result prints its provenance and scope. Fixture runs use `Synthetic=true` with fixture-only artifacts.

## Execute the demo

From a clean MATLAB session in the repository root:

```matlab
setup;
run_demo("quick", BaselineRun="results/cross-validation/baseline/<chosen-run>");
```

The placeholders stand for actual local directories. On this prepared machine the reported baseline directory is `results/cross-validation/baseline/run-tp2b5627d2_ac6f_463a_845f_aad7bd4e2dad`. A unique nested run and unique final run may be discovered automatically. To pin every input for the classroom, fill the following structure with the directories selected in advance. Choose a training PNG locally without putting its name in a tracked file.

```matlab
setup;
demoOptions = struct( ...
    "TrainingPath", "train", "TestingPath", "test", ...
    "BaselineRun", "results/cross-validation/baseline/<chosen-run>", ...
    "NestedRun", "results/cross-validation/nested/<chosen-run>", ...
    "FinalRun", "results/final-evaluation/<chosen-run>", ...
    "OutputDirectory", "results/demo");
[file, folder] = uigetfile("*.png", "Select a local training example", fullfile(pwd, "train"));
if isequal(file, 0), error("Select a local image before the live demo."); end
demoOptions.ImagePath = fullfile(folder, file);
demoOptions.ImageKind = "training";
a = namedargs2cell(demoOptions);
run_demo("quick", a{:}, ShowFigures=true);
run_demo("figures", a{:}, ShowFigures=true);
run_demo("independent-status", a{:});
```

Choose the image before presenting. The picker above is only preparation. In batch, pass a known local `ImagePath` directly and leave `ShowFigures` at its default `false`. Images in `TrainingPath` are labeled as training examples automatically. Set `ImageKind="training"` to require that location. For an image elsewhere, use the default `ImageKind="local"`. `run_demo("menu", a{:})` opens the interactive menu with these inputs. With unique defaults, `setup; run_demo;` opens the menu directly. Options 2, 3 and 7 can perform new computation and are explicit actions.

```powershell
matlab -wait -batch "setup; run_demo('quick',BaselineRun='results/cross-validation/baseline/run-tp2b5627d2_ac6f_463a_845f_aad7bd4e2dad');"
```

| Action | What to show |
| --- | --- |
| `quick` | Saved baseline and nested CV, fitted model settings, parameter counts and fit measurements. Optional local image. No fit, search, or holdout access. |
| `results` | CV-only comparison. Add `FoldDetails=true` for per-fold metrics. |
| `models`, `image` | Verify and inspect the bundle, or apply all four pipelines to one image. `ShowFigures=true` can display the original and fitted preprocessing. |
| `figures` | Rebuild mean ROC, accuracy/AUC and cost figures under a fresh local directory. `FoldROC=true` exports individual-fold curves. `ShowFigures=true` opens the ROC PNGs in MATLAB. Close them when done. |
| `independent-status` | Show a saved completed evaluation or report readiness, missing data, or a selected failed attempt. Never fits or evaluates. |
| `independent` | Reuse a saved completed evaluation. Otherwise evaluate the configured test path using the frozen bundle and existing integrity checks. Inspect the returned status. |
| `baseline`, `train` | Explicitly run fixed baseline CV or fit and verify final models. These actions write new runs. |

The CV tables show accuracy, AUC and sample SD across folds, reduced dimensions, classifier/projection/normalization/combined counts, and classifier/full-pipeline training and inference. The mean ROC figures retain all measured coordinates and the random-ranking diagonal. The four curves overlap on the saved course runs. Fold SD describes fold-to-fold variation. Baseline, nested outer fits and final full-training settings may differ. In particular, final PCA + MLP uses 20 components and the selected nested outer settings used 10.

Timings are in milliseconds. CV shows the mean of the fold medians with a 40-image validation batch for the saved official runs. The final fit tables show medians for 200 training images. Inference uses images in memory. Classifier-only inference starts from projected features. MAT bytes include representation, metadata and compression. A live single-image call measures a different workload from a controlled benchmark.

The image view describes original and processed dimensions, feature counts, all labels and all scores. Male is 0 and female is 1, the positive class. A strictly positive score predicts 1. A zero score predicts 0. Bayes returns log-posterior odds and MLP returns logits, on different scales. The program predicts course labels. Facial appearance cannot establish gender identity. Keep the input, predictions and scores local.

## Independent testing and fallback

For a completed official attempt, use `run_demo("independent-status", a{:}, IndependentAttempt=attempt, ShowFigures=true)`. Its record names the selected frozen bundle, and the display includes accuracy, AUC, ROC, confusion matrices, and classifier and complete in-memory inference for the saved batch. To export all independent plots use `run_demo("figures", a{:}, IndependentAttempt=attempt)`.

If the official holdout is supplied but unevaluated, confirm the intended data and bundle, then run:

```matlab
r = run_demo("independent", a{:});
disp(r.Independent.Status);
```

A new attempt is saved below `OutputDirectory/independent-test`. Check that the status is `complete` before presenting metrics. With the missing official directory, the current display reports the final models ready and all independent metrics pending. Synthetic fixtures have separate labels and do not supply official metrics. A blocked or incomplete attempt needs investigation. Preserve the holdout for final evaluation.

If the model bundle is missing, copy the verified bundle privately or prepare it before the classroom session from the course training data:

```matlab
setup;
r = run_final_training("train", BaselineRun=baselineDir, NestedRun=nestedDir);
run_demo("models", FinalRun=r.OutputDirectory);
```

Use the actual saved directories for `baselineDir` and `nestedDir`. The final trainer can reconstruct the documented frozen settings when recommendation evidence is absent. See the project workflow for the recorded provenance and its integrity implications. [Final training and testing](final-training-and-testing.md)

If figures are slow or MATLAB cannot display windows, use `ShowFigures=false` and open the already exported local PNG/PDF. Regenerate them with `run_demo("figures", a{:}, FoldROC=true)` before class. Use the PDF deck plus saved figures if slide rendering fails. If a chosen artifact fails validation, restore the complete original run rather than combining files from different runs. If `setup` is missing, change to the checked-out root. If ambiguity is reported, set the named selector. Close demo figures before switching back to the slides.

## Materials

- [Editable PowerPoint sources and build instructions](../presentation/README.md)
- [English slide outline](oral-presentation-outline.md)
- [Traditional Chinese speaker notes](../zh-TW/oral-presentation-notes.md)
- [Traditional Chinese Q&A](../zh-TW/oral-presentation-qa.md)
- [Rehearsal checklist and technical log](../zh-TW/demo-rehearsal-checklist.md)
