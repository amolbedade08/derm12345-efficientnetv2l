Derm12345 EfficientNetV2-L Skin Lesion Classification

This repository contains the code, experiments, evaluation results, statistical analysis, and explainability outputs for a multiclass skin-lesion image classification study using EfficientNetV2-L on the Derm12345 dataset.

The project also includes:

a balanced Logistic Regression baseline using frozen EfficientNetV2-L features,

3-fold stratified cross-validation,

evaluation on the untouched official Derm12345 test set,

external validation on PAD-UFES-20 using the shared-label subset,

ROC/PR analysis,

statistical comparison between the EfficientNetV2-L model and the baseline,

LayerCAM-based explainability examples.

Important: The datasets and trained model weight files are intentionally not included in this repository because of their size and dataset/licensing constraints. See the dataset section below.

1. Project Overview

The objective is to evaluate EfficientNetV2-L for multiclass skin-lesion classification and compare its performance with a classical machine-learning baseline.

Main dataset

Derm12345

40 classes

Official training set: 9,860 images

Official test set: 2,485 images

Metadata includes patient, image, split, superclass, malignancy, class, and label information.

External validation dataset

PAD-UFES-20

2,298 images

6 diagnostic classes in the released metadata

External evaluation uses the 4 classes shared with the Derm12345 label space:

BCC → bcc

MEL → mel

SCC → scc

SEK → sk

ACK and NEV are excluded from the external evaluation because they do not have a direct shared target class in the current 40-class mapping.

2. Methodology

EfficientNetV2-L

The main deep-learning model is EfficientNetV2-L with ImageNet initialization.

Configuration used in the current experiment:

Input size: 224 × 224 × 3

Number of classes: 40

Batch size: 2

Optimizer: Adam

Learning rate: 1e-4

Dropout: 0.3

L2 regularization: 1e-4

Initial backbone: frozen

Cross-validation: 3-fold stratified

Random seed: 42

Data augmentation includes rotation, width/height shift, zoom, shear, horizontal flip, and brightness adjustment.

Experimental protocol

The official Derm12345 split is preserved.

The 9,860-image official training set is used for 3-fold stratified cross-validation.

The 2,485-image official test set remains untouched until final evaluation.

A final model is trained using all 9,860 official training images.

The final model is evaluated once on the official test set.

External validation is performed on the compatible shared-label subset of PAD-UFES-20.

3. Baseline

A Logistic Regression classifier is trained on frozen EfficientNetV2-L feature representations.

Derm12345 images
       ↓
EfficientNetV2-L feature extraction
       ↓
1280-dimensional feature vectors
       ↓
StandardScaler
       ↓
Balanced Logistic Regression
       ↓
40-class prediction

This baseline is included to determine whether the end-to-end fine-tuned model provides an improvement over a simpler classifier operating on the same pretrained feature representation.

4. Current Results

3-Fold Cross-Validation

Metric

EfficientNetV2-L

Logistic Regression Baseline

Accuracy

0.2305 ± 0.0334

0.5198 ± 0.0073

Macro Precision

0.1952 ± 0.0300

0.5929 ± 0.0365

Macro Recall

0.3859 ± 0.1050

0.5115 ± 0.0099

Macro F1

0.1927 ± 0.0457

0.5322 ± 0.0042

Specificity

0.9800 ± 0.0008

0.9857 ± 0.0002

FPR

0.0200 ± 0.0008

0.0143 ± 0.0002

IoU

0.1133 ± 0.0287

0.4045 ± 0.0063

ROC-AUC

0.8544 ± 0.0545

0.9251 ± 0.0094

PR-AUC

0.3184 ± 0.0959

0.5271 ± 0.0096

At the current configuration, the Logistic Regression baseline performs better than the EfficientNetV2-L model on most classification metrics.

Official Derm12345 Test Set

The final EfficientNetV2-L model was evaluated on the untouched official test set of 2,485 images.

Metric

Score

Accuracy

0.1940

Macro Precision

0.1824

Macro Recall

0.2843

Macro F1

0.1724

Specificity

0.9793

FPR

0.0207

IoU

0.1049

ROC-AUC

0.7779

PR-AUC

0.2404

External PAD-UFES-20 Validation

The external evaluation contains 1,324 images belonging to the four shared classes.

External accuracy is a 4-class shared-label evaluation, not a 40-class evaluation.

Metric

Score

Accuracy

0.4350

Macro Precision

0.3647

Macro Recall

0.3823

Macro F1

0.3520

Specificity

0.7886

FPR

0.2114

IoU

0.2222

ROC-AUC

0.6833

PR-AUC

0.3610

5. Class-Level External Results

PAD-UFES-20 shared-label evaluation:

Class

Precision

Recall

F1

Support

BCC

0.7124

0.4426

0.5460

845

MEL

0.2000

0.1923

0.1961

52

SCC

0.1824

0.4219

0.2547

192

SEK

0.3639

0.4723

0.4111

235

6. Explainability

LayerCAM is used to visualize spatial regions contributing to model predictions.

The current XAI analysis:

selects convolutional layers near the end of EfficientNetV2-L,

produces LayerCAM heatmaps,

includes both correct and incorrect examples,

covers the four shared PAD-UFES-20 classes used for external validation.

LayerCAM visualizations are intended for model explainability, not as proof of clinical causality.

7. Repository Structure

derm12345-efficientnetv2l/
│
├── src/
│   ├── efficientnetv2l.py
│   ├── train_final_all.py
│   ├── statistical_comparison.py
│   └── generate_paper_tables.py
│
├── outputs/
│   └── EfficientNetV2L/
│       ├── models/
│       ├── external_PAD_UFES20/
│       ├── statistics/
│       ├── paper_tables/
│       ├── paper_figures/
│       └── XAI/
│
├── dataset/
│   └── README.md
│
├── .gitignore
├── README.md
└── requirements.txt

Large dataset folders and trained model files are excluded through .gitignore.

8. Dataset Setup

The datasets are not stored in this GitHub repository.

After obtaining the datasets from their official sources, place them in the expected local project directories.

Derm12345

Place the dataset under:

dataset/

The exact image and metadata layout should match the paths expected by the scripts in src/.

PAD-UFES-20

Place the extracted dataset under:

dataset/external/PAD-UFES-20/

Expected files/folders used by the current project include:

dataset/external/PAD-UFES-20/
├── metadata.csv
└── images/
    ├── imgs_part_1/
    ├── imgs_part_2/
    └── imgs_part_3/

Official PAD-UFES-20 source:

https://data.mendeley.com/datasets/zr7vgbcyr2/1

DOI:

10.17632/zr7vgbcyr2.1

9. Environment

The experiments were developed in a WSL2/Linux environment with GPU support.

Example Python environment:

python3 -m venv venv-linux
source venv-linux/bin/activate

Install dependencies:

pip install -r requirements.txt

GPU availability can be checked with:

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

10. Running the Project

Train/evaluate EfficientNetV2-L

python src/efficientnetv2l.py

Train final model on all official training data

python src/train_final_all.py

Statistical comparison

python src/statistical_comparison.py

Generate paper tables

python src/generate_paper_tables.py

Check the generated files under:

outputs/EfficientNetV2L/

11. Reproducibility

Important settings used in the current experiment:

Seed: 42
Image size: 224 × 224
Batch size: 2
Epochs: 20 for cross-validation configuration
Final training epochs: 9
Number of classes: 40
Cross-validation folds: 3
Optimizer: Adam
Learning rate: 1e-4
Dropout: 0.3
L2 regularization: 1e-4

The current experiments should be considered a reproducible research baseline rather than a final optimized clinical model.

12. Statistical Analysis

The repository includes paired fold-level comparisons between EfficientNetV2-L and the Logistic Regression baseline.

The analysis includes:

Shapiro-Wilk normality testing,

paired t-test,

Wilcoxon signed-rank test.

Because only 3 folds are available, these statistical comparisons should be interpreted as exploratory rather than definitive statistical evidence.

13. Important Research Notes

Class imbalance

Derm12345 contains highly imbalanced classes, including several classes with very few images. This can strongly affect macro-averaged metrics and model learning.

Patient overlap

Three patient IDs were found in both the official training and test metadata during dataset inspection. This should be investigated carefully before publication and discussed appropriately in the final research report.

External label mapping

PAD-UFES-20 does not have a one-to-one mapping to all 40 Derm12345 classes. Therefore, external performance is reported only on the four-class shared label intersection.

Model performance

The current EfficientNetV2-L configuration does not outperform the Logistic Regression baseline. Future experiments should investigate class imbalance handling, classifier design, augmentation strength, resolution, fine-tuning strategy, and loss functions before treating the current model as the final optimized result.

14. Outputs Included

The repository contains generated research artifacts such as:

cross-validation results,

official test predictions,

confusion matrices,

ROC curves,

precision-recall curves,

external validation results,

statistical test results,

paper-ready tables,

paper-ready figures,

LayerCAM explainability figures.

Model files such as .keras and raw datasets are excluded from Git tracking.

15. License and Dataset Attribution

The code in this repository should be used according to the license selected for the project.

The datasets remain subject to their respective source licenses, terms of use, and attribution requirements. Users should obtain the datasets directly from their official sources and follow those terms.

16. Status

Current status: Research prototype / experimental study.

The repository reflects the current experimental pipeline and results. The EfficientNetV2-L model is not claimed to be clinically deployable, and the reported metrics should not be interpreted as medical diagnostic performance without further validation.
