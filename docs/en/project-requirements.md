# Gender Recognition Grand Challenge Project Requirements

[繁體中文](../zh-TW/project-requirements.md) · [Project home](../../README.md)

English translation of the seven-page handout `PR_2026_專題說明文件_v1.pdf`. The original handout is authoritative for the project requirements.

## 1. Project Background and Motivation

This hands-on project brings together pattern recognition concepts and techniques from the course, including the Bayesian classifier, principal component analysis (PCA), linear discriminant analysis (LDA), the multilayer perceptron (MLP), receiver operating characteristic (ROC) curves, and cross-validation.

Students tackle gender recognition from face images by extracting suitable features, building different classification models, and using systematic experiments and evaluation to compare how feature reduction methods and classifiers affect recognition performance.

Students apply the course theory to a real-world problem and learn to design experiments, analyze results, compare the strengths and weaknesses of different methods, and present their findings in oral and written reports.

## 2. Project Objectives

The main objective is to build a system that recognizes gender from face images and to compare the performance of different combinations of feature reduction methods and classifiers.

Each team must implement all four classification methods below.

1. PCA + Bayes Classifier
2. LDA + Bayes Classifier
3. PCA + MLP Classifier
4. LDA + MLP Classifier

For each of the four methods, teams must do the following.

- Perform five-fold cross-validation.
- Calculate classification accuracy.
- Plot the receiver operating characteristic (ROC) curve.
- Calculate the area under the curve (AUC).
- Analyze feature-vector dimensionality.
- Analyze the model parameter count.
- Analyze training time.
- Analyze inference time.

Use these experiments to analyze how the methods perform on gender recognition and examine how PCA, LDA, the Bayes classifier, and the MLP classifier affect classification results and computational efficiency.

## 3. Image Datasets

The project provides face images divided into training and testing datasets.

| Dataset | Male | Female | Total face images |
|---|---:|---:|---:|
| Training dataset | 100 | 100 | 200 |
| Testing dataset | 50 | 50 | 100 |

### (1) Training dataset

Use the training data to build the classification models and evaluate their performance through five-fold cross-validation.

### (2) Testing dataset

Reserve the testing dataset exclusively for final model testing to assess generalization. Keep it out of model training and cross-validation.

## 4. Programming Language and Development Environment

All teams must use **MATLAB**.

Students may use MATLAB functions or write their own code according to what they learned in class and the needs of their project. They must be able to clearly explain the methods, parameter settings, and experimental workflow they use.

## 5. Software Functional Requirements

Each team must produce an executable gender recognition program with at least the following functionality.

### (1) Training

The program must let users configure the training dataset path. After loading the data, the system must perform classification using each of the following methods.

1. PCA + Bayes Classifier
2. LDA + Bayes Classifier
3. PCA + MLP Classifier
4. LDA + MLP Classifier

For each method, perform five-fold cross-validation and produce the following results.

- Classification accuracy
- ROC curve
- AUC value
- Feature-vector dimensionality
- Model parameter count
- Training time
- Inference time

### (2) Testing

The program must let users configure the testing dataset path. After building the final models with the training data, apply the models to the testing dataset and do the following for each of the four methods.

- Calculate classification accuracy.
- Plot the ROC curve.
- Calculate AUC.
- Measure inference time.

## 6. Experimental and Analysis Requirements

The project requires a working classification system and emphasizes using experiments to analyze differences between methods. The report must analyze and discuss at least the following items.

1. **Classification accuracy**

   Compare the four methods' classification accuracy from cross-validation on the training data and discuss possible reasons for differences.
2. **ROC curves and AUC**

   Plot the four methods' ROC curves and compare their AUC values to analyze how well the classifiers distinguish the male and female classes.
3. **Feature-vector dimensionality**

   Compare the dimensions of the feature vectors produced by PCA and LDA. Discuss how feature dimensionality affects classification performance and computational efficiency.
4. **Model parameter count**

   Compare the parameter counts of the Bayes and MLP classifiers and discuss the relationship between parameter count and classification performance.
5. **Training time**

   Compare the time needed to build the four classification models and analyze how the feature reduction methods and classifiers affect training cost.
6. **Inference time**

   Compare the time the four methods need to perform gender recognition and discuss their computational efficiency.
7. **Overall comparison**

   Consider classification performance, parameter count, and computational efficiency together. Discuss the strengths and limitations of each method and explain which is most suitable overall for the face-image gender recognition problem defined in this project.

## 7. Team Formation

The project is expected to have 15 teams. Team sizes depend on actual course enrollment, with about 5–6 students expected per team.

Team members should work together on program development, experiments, analysis, the oral presentation, and the written report.

## 8. Evaluation Criteria

| Graded component | Weight |
|---|---:|
| Oral presentation and live demonstration | 60% |
| Written report | 30% |
| Test-set AUC | 10% |

Evaluation covers the following aspects.

- Completeness of the project system
- Correctness of the method implementations
- Soundness of the experimental design
- Completeness of the experimental results
- Depth of analysis and discussion
- Understanding of the methods used
- Performance during the live demonstration
- Oral communication skills
- Completeness and professionalism of the written report

## 9. Oral Presentation Requirements

The oral presentation must include at least the following.

1. **Introduction**
   - Project background and objectives
2. **Methods**
   - System flowchart
   - Key parameter settings
3. **Experimental results and analysis**

   Compare the four methods on the following items, and analyze and discuss the results.
   - Classification accuracy
   - ROC curves
   - AUC
   - Feature-vector dimensionality
   - Model parameter count
   - Training time
   - Inference time
4. **Live demonstration**

   Run the gender recognition system live and show its actual output.
5. **Conclusion**

   Summarize the project findings, explain the strengths and weaknesses of the four methods, and present the overall comparison.
6. **References**

   List the papers, books, and other sources consulted during the project.

## 10. Written Report Requirements

The report must follow **National Chung Cheng University's thesis format** and include at least the following.

### Chapter 1. Introduction

- Project background and objectives

### Chapter 2. Methods

Describe the methods used in the project in detail, covering the following items.

- System flowchart
- PCA
- LDA
- Bayes Classifier
- MLP Classifier
- Relevant mathematical formulas
- Experimental parameter settings

### Chapter 3. Experimental Results and Analysis

Describe the experimental environment, datasets, and experimental settings. Compare the four methods on the following items, and analyze and discuss the experimental results in detail.

- Classification accuracy
- ROC curves
- AUC
- Feature-vector dimensionality
- Model parameter count
- Training time
- Inference time

### Chapter 4. Conclusion

Summarize the project's main outcomes and experimental findings and compare the overall strengths and weaknesses of the four classification methods.

### References

List all sources cited in the project, formatted according to National Chung Cheng University's thesis requirements.

## 11. Project Schedule

| Milestone | Date |
|---|---|
| Submit team member list | September 29 |
| Oral presentation | December 15 |
| Submit written report | December 22 |

## 12. Expected Project Deliverables

By the end of the project, each team should deliver the following.

1. An executable MATLAB gender recognition program.
2. Implementations of all four classification methods (PCA + Bayes, LDA + Bayes, PCA + MLP, and LDA + MLP).
3. Five-fold cross-validation on the training data.
4. Comparative analysis of classification accuracy, ROC curves, and AUC.
5. Comparative analysis of feature dimensionality, model parameter count, training time, and inference time.
6. A live system demonstration.
7. An oral presentation.
8. A written project report.

Alongside producing program output, students are expected to understand the methods, design sound experiments, compare approaches, and draw their own observations and conclusions from the experimental results.
