import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = "/mnt/c/Users/0M SAI RAM/Desktop/project"

FEATURE_FILE = os.path.join(
    PROJECT_DIR,
    "outputs/EfficientNetV2L/features/train_features.npy"
)

LABEL_FILE = os.path.join(
    PROJECT_DIR,
    "outputs/EfficientNetV2L/features/train_feature_labels.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs/EfficientNetV2L/feature_classifier"
)

N_SPLITS = 3
RANDOM_SEED = 42
FEATURE_SIZE = 1280


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("EFFICIENTNETV2-L FEATURE CLASSIFIER")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(FEATURE_FILE):
    raise FileNotFoundError(
        f"\nFeature file not found:\n{FEATURE_FILE}"
    )

if not os.path.exists(LABEL_FILE):
    raise FileNotFoundError(
        f"\nLabel file not found:\n{LABEL_FILE}"
    )


# ============================================================
# LOAD LABELS
# ============================================================

print("\nLoading labels...")

labels_df = pd.read_csv(LABEL_FILE)

if "class_index" not in labels_df.columns:
    raise ValueError(
        "Column 'class_index' not found in label CSV."
    )

if "label" not in labels_df.columns:
    raise ValueError(
        "Column 'label' not found in label CSV."
    )

y = labels_df["class_index"].to_numpy(
    dtype=np.int64
)

num_samples = len(y)

print(
    "Number of samples:",
    num_samples
)


# ============================================================
# LOAD FEATURE MATRIX
# ============================================================

print("\nLoading feature matrix...")

X = np.memmap(
    FEATURE_FILE,
    dtype="float32",
    mode="r",
    shape=(num_samples, FEATURE_SIZE)
)

print(
    "Feature shape:",
    X.shape
)


# ============================================================
# BASIC VALIDATION
# ============================================================

if X.shape[0] != num_samples:
    raise ValueError(
        f"Feature/label mismatch:\n"
        f"Features = {X.shape[0]}\n"
        f"Labels   = {num_samples}"
    )

if X.shape[1] != FEATURE_SIZE:
    raise ValueError(
        f"Expected {FEATURE_SIZE} features, "
        f"but found {X.shape[1]}"
    )


classes = np.unique(y)

num_classes = len(classes)

print(
    "Number of classes:",
    num_classes
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

class_counts = (
    pd.Series(y)
    .value_counts()
    .sort_index()
)

for class_index, count in class_counts.items():

    print(
        f"Class {int(class_index):2d}: {int(count)}"
    )


# ============================================================
# CHECK WHETHER 3-FOLD CV IS POSSIBLE
# ============================================================

min_class_count = int(
    class_counts.min()
)

print(
    "\nMinimum class count:",
    min_class_count
)

if min_class_count < N_SPLITS:

    raise ValueError(
        f"\nCannot perform {N_SPLITS}-fold stratified CV.\n"
        f"Minimum class count = {min_class_count}"
    )


# ============================================================
# MODEL
# ============================================================

print("\nCreating classifier...")

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_SEED
            )
        )
    ]
)

print(
    "Classifier: StandardScaler + LogisticRegression"
)

print(
    "Class weighting: balanced"
)


# ============================================================
# STRATIFIED K-FOLD
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED
)


# ============================================================
# STORAGE
# ============================================================

fold_results = []

all_true = []
all_pred = []
all_prob = []


# ============================================================
# CROSS VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print(
    f"STARTING {N_SPLITS}-FOLD STRATIFIED CROSS-VALIDATION"
)
print("=" * 70)


for fold, (train_idx, val_idx) in enumerate(
    skf.split(np.zeros(num_samples), y),
    start=1
):

    print("\n")
    print("-" * 70)
    print(
        f"FOLD {fold}/{N_SPLITS}"
    )
    print("-" * 70)

    print(
        f"Train samples: {len(train_idx)}"
    )

    print(
        f"Validation samples: {len(val_idx)}"
    )


    # ========================================================
    # SELECT TRAINING / VALIDATION DATA
    # ========================================================

    X_train = X[train_idx]

    X_val = X[val_idx]

    y_train = y[train_idx]

    y_val = y[val_idx]


    # ========================================================
    # TRAIN
    # ========================================================

    print("\nTraining classifier...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")


    # ========================================================
    # PREDICTION
    # ========================================================

    print(
        "Generating validation predictions..."
    )

    y_pred = model.predict(
        X_val
    )

    y_prob = model.predict_proba(
        X_val
    )


    # ========================================================
    # BASIC METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_val,
        y_pred,
        labels=classes
    )


    # ========================================================
    # CLASS-WISE SPECIFICITY / FPR / IoU / PPV
    # ========================================================

    specificity_values = []

    fpr_values = []

    iou_values = []

    ppv_values = []


    for i in range(num_classes):

        tp = cm[i, i]

        fn = (
            cm[i, :].sum()
            - tp
        )

        fp = (
            cm[:, i].sum()
            - tp
        )

        tn = (
            cm.sum()
            - tp
            - fn
            - fp
        )


        # ----------------------------------------------------
        # Specificity
        # ----------------------------------------------------

        if (tn + fp) > 0:

            specificity = (
                tn / (tn + fp)
            )

        else:

            specificity = 0.0


        # ----------------------------------------------------
        # FPR
        # ----------------------------------------------------

        if (fp + tn) > 0:

            fpr_value = (
                fp / (fp + tn)
            )

        else:

            fpr_value = 0.0


        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        if (
            tp + fp + fn
        ) > 0:

            iou_value = (
                tp
                / (tp + fp + fn)
            )

        else:

            iou_value = 0.0


        # ----------------------------------------------------
        # PPV
        # ----------------------------------------------------

        if (tp + fp) > 0:

            ppv_value = (
                tp
                / (tp + fp)
            )

        else:

            ppv_value = 0.0


        specificity_values.append(
            specificity
        )

        fpr_values.append(
            fpr_value
        )

        iou_values.append(
            iou_value
        )

        ppv_values.append(
            ppv_value
        )


    specificity = float(
        np.mean(
            specificity_values
        )
    )

    fpr = float(
        np.mean(
            fpr_values
        )
    )

    iou = float(
        np.mean(
            iou_values
        )
    )

    ppv = float(
        np.mean(
            ppv_values
        )
    )


    # ========================================================
    # ROC-AUC
    # ========================================================

    try:

        roc_auc = roc_auc_score(
            y_val,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=classes
        )

    except Exception as e:

        print(
            "ROC-AUC warning:",
            e
        )

        roc_auc = np.nan


    # ========================================================
    # PR-AUC
    # ========================================================

    try:

        y_val_onehot = np.zeros(
            (
                len(y_val),
                num_classes
            ),
            dtype=np.float32
        )


        # Convert class labels into positions
        class_position = {
            class_id: pos
            for pos, class_id
            in enumerate(classes)
        }


        for row_index, true_class in enumerate(y_val):

            y_val_onehot[
                row_index,
                class_position[true_class]
            ] = 1.0


        pr_auc = average_precision_score(
            y_val_onehot,
            y_prob,
            average="macro"
        )

    except Exception as e:

        print(
            "PR-AUC warning:",
            e
        )

        pr_auc = np.nan


    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    cm_file = os.path.join(
        OUTPUT_DIR,
        f"confusion_matrix_fold_{fold}.csv"
    )

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    cm_df.to_csv(
        cm_file
    )


    # ========================================================
    # SAVE VALIDATION PREDICTIONS
    # ========================================================

    predictions_df = labels_df.iloc[
        val_idx
    ].copy()

    predictions_df[
        "true_class_index"
    ] = y_val

    predictions_df[
        "predicted_class_index"
    ] = y_pred

    predictions_df[
        "correct"
    ] = (
        y_val == y_pred
    )


    prediction_file = os.path.join(
        OUTPUT_DIR,
        f"predictions_fold_{fold}.csv"
    )

    predictions_df.to_csv(
        prediction_file,
        index=False
    )


    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    fold_result = {

        "fold": fold,

        "accuracy": float(
            accuracy
        ),

        "precision_macro": float(
            precision
        ),

        "recall_macro_sensitivity": float(
            recall
        ),

        "f1_macro": float(
            f1
        ),

        "specificity_macro": float(
            specificity
        ),

        "fpr_macro": float(
            fpr
        ),

        "iou_macro": float(
            iou
        ),

        "ppv_macro": float(
            ppv
        ),

        "roc_auc_macro_ovr": float(
            roc_auc
        )
        if not np.isnan(roc_auc)
        else np.nan,

        "pr_auc_macro": float(
            pr_auc
        )
        if not np.isnan(pr_auc)
        else np.nan,

        "train_samples": int(
            len(train_idx)
        ),

        "validation_samples": int(
            len(val_idx)
        )
    }


    fold_results.append(
        fold_result
    )


    # ========================================================
    # STORE AGGREGATED PREDICTIONS
    # ========================================================

    all_true.extend(
        y_val.tolist()
    )

    all_pred.extend(
        y_pred.tolist()
    )

    all_prob.append(
        y_prob
    )


    # ========================================================
    # PRINT FOLD RESULTS
    # ========================================================

    print("\nFold results:")

    print(
        f"Accuracy       : {accuracy:.4f}"
    )

    print(
        f"Precision      : {precision:.4f}"
    )

    print(
        f"Recall         : {recall:.4f}"
    )

    print(
        f"F1             : {f1:.4f}"
    )

    print(
        f"Specificity    : {specificity:.4f}"
    )

    print(
        f"FPR            : {fpr:.4f}"
    )

    print(
        f"IoU            : {iou:.4f}"
    )

    print(
        f"PPV            : {ppv:.4f}"
    )

    print(
        f"ROC-AUC        : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC         : {pr_auc:.4f}"
    )


# ============================================================
# FOLD RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    fold_results
)


results_file = os.path.join(
    OUTPUT_DIR,
    "fold_results.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# MEAN AND STANDARD DEVIATION
# ============================================================

metric_columns = [

    "accuracy",

    "precision_macro",

    "recall_macro_sensitivity",

    "f1_macro",

    "specificity_macro",

    "fpr_macro",

    "iou_macro",

    "ppv_macro",

    "roc_auc_macro_ovr",

    "pr_auc_macro"
]


summary_rows = []


for metric in metric_columns:

    values = pd.to_numeric(
        results_df[metric],
        errors="coerce"
    )

    summary_rows.append({

        "metric": metric,

        "mean": values.mean(),

        "std": values.std(
            ddof=1
        ),

        "min": values.min(),

        "max": values.max()

    })


summary_df = pd.DataFrame(
    summary_rows
)


summary_file = os.path.join(
    OUTPUT_DIR,
    "cv_summary_mean_sd.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# AGGREGATED CV PREDICTIONS
# ============================================================

all_true = np.asarray(
    all_true,
    dtype=np.int64
)

all_pred = np.asarray(
    all_pred,
    dtype=np.int64
)

all_prob = np.vstack(
    all_prob
)


aggregated_accuracy = accuracy_score(
    all_true,
    all_pred
)

aggregated_precision = precision_score(
    all_true,
    all_pred,
    average="macro",
    zero_division=0
)

aggregated_recall = recall_score(
    all_true,
    all_pred,
    average="macro",
    zero_division=0
)

aggregated_f1 = f1_score(
    all_true,
    all_pred,
    average="macro",
    zero_division=0
)


aggregated_predictions = pd.DataFrame({

    "true_class_index":
        all_true,

    "predicted_class_index":
        all_pred,

    "correct":
        all_true == all_pred

})


aggregated_prediction_file = os.path.join(
    OUTPUT_DIR,
    "aggregated_cv_predictions.csv"
)


aggregated_predictions.to_csv(
    aggregated_prediction_file,
    index=False
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {

    "n_splits":
        N_SPLITS,

    "random_seed":
        RANDOM_SEED,

    "num_samples":
        int(num_samples),

    "num_features":
        int(FEATURE_SIZE),

    "num_classes":
        int(num_classes),

    "classifier":
        "StandardScaler + LogisticRegression",

    "solver":
        "lbfgs",

    "class_weight":
        "balanced",

    "C":
        1.0,

    "max_iter":
        1000

}


config_file = os.path.join(
    OUTPUT_DIR,
    "config.json"
)


with open(
    config_file,
    "w"
) as file:

    json.dump(
        config,
        file,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("3-FOLD CROSS-VALIDATION COMPLETE")
print("=" * 70)


print("\nMean ± SD:")


for _, row in summary_df.iterrows():

    print(

        f"{row['metric']:32s}"

        f"{row['mean']:.4f} ± "

        f"{row['std']:.4f}"

    )


print("\nAggregated CV metrics:")


print(
    f"Accuracy       : "
    f"{aggregated_accuracy:.4f}"
)

print(
    f"Precision      : "
    f"{aggregated_precision:.4f}"
)

print(
    f"Recall         : "
    f"{aggregated_recall:.4f}"
)

print(
    f"F1             : "
    f"{aggregated_f1:.4f}"
)


print("\nOutput directory:")

print(
    OUTPUT_DIR
)


print("\nSaved files:")

print(
    "fold_results.csv"
)

print(
    "cv_summary_mean_sd.csv"
)

print(
    "aggregated_cv_predictions.csv"
)

print(
    "confusion_matrix_fold_1.csv"
)

print(
    "confusion_matrix_fold_2.csv"
)

print(
    "confusion_matrix_fold_3.csv"
)

print(
    "predictions_fold_1.csv"
)

print(
    "predictions_fold_2.csv"
)

print(
    "predictions_fold_3.csv"
)

print(
    "config.json"
)


print("=" * 70)	
