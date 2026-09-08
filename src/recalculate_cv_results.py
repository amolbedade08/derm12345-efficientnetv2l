import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_CSV = os.path.join(
    PROJECT_DIR,
    "outputs",
    "EfficientNetV2L",
    "csv_results",
    "train_loaded.csv",
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "EfficientNetV2L",
)

RESULT_DIR = os.path.join(
    OUTPUT_DIR,
    "csv_results",
)

PRED_DIR = os.path.join(
    OUTPUT_DIR,
    "predictions",
)

MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "models",
)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 2
N_SPLITS = 3
RANDOM_SEED = 42


# ============================================================
# GPU
# ============================================================

print("=" * 70)
print("RECALCULATE 3-FOLD CV RESULTS")
print("=" * 70)

print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)

if not gpus:
    raise RuntimeError(
        "GPU not detected. Stop rather than running inference on CPU."
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(TRAIN_CSV)

classes = sorted(df["label"].unique())

class_to_index = {
    c: i
    for i, c in enumerate(classes)
}

index_to_class = {
    i: c
    for c, i in class_to_index.items()
}

NUM_CLASSES = len(classes)

df["class_index"] = df["label"].map(class_to_index)
df["class_index_str"] = df["class_index"].astype(str)

print("Samples:", len(df))
print("Classes:", NUM_CLASSES)


# ============================================================
# EXACT SAME 3-FOLD SPLIT
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED,
)

splits = list(
    skf.split(
        df,
        df["class_index"],
    )
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob,
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    specificities = []
    fprs = []
    ious = []

    for i in range(NUM_CLASSES):

        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp

        tn = (
            cm.sum()
            - tp
            - fn
            - fp
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        iou = (
            tp / (tp + fp + fn)
            if (tp + fp + fn) > 0
            else 0.0
        )

        specificities.append(
            specificity
        )

        fprs.append(
            fpr
        )

        ious.append(
            iou
        )

    specificity = np.mean(
        specificities
    )

    fpr = np.mean(
        fprs
    )

    iou = np.mean(
        ious
    )

    try:

        roc_auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
        )

    except Exception:

        roc_auc = np.nan

    try:

        y_onehot = tf.keras.utils.to_categorical(
            y_true,
            num_classes=NUM_CLASSES,
        )

        pr_auc = average_precision_score(
            y_onehot,
            y_prob,
            average="macro",
        )

    except Exception:

        pr_auc = np.nan

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "specificity_macro": specificity,
        "fpr_macro": fpr,
        "iou_macro": iou,
        "roc_auc_macro_ovr": roc_auc,
        "pr_auc_macro": pr_auc,
    }


# ============================================================
# MODELS
# ============================================================

model_paths = {
    1: os.path.join(
        MODEL_DIR,
        "efficientnetv2l_fold_1.keras",
    ),

    2: os.path.join(
        MODEL_DIR,
        "efficientnetv2l_fold_2.keras",
    ),

    3: os.path.join(
        MODEL_DIR,
        "efficientnetv2l_fold_3_fixed.keras",
    ),
}


# ============================================================
# VALIDATION INFERENCE
# ============================================================

all_results = []

for fold in [1, 2, 3]:

    print()
    print("=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    _, val_idx = splits[fold - 1]

    fold_val = df.iloc[val_idx].copy()

    print("Validation samples:", len(fold_val))

    generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=fold_val,
        x_col="filepath",
        y_col="class_index_str",
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        class_mode="sparse",
        classes=[str(i) for i in range(NUM_CLASSES)],
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model_path = model_paths[fold]

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    print("Loading:", model_path)

    model = tf.keras.models.load_model(
        model_path
    )

    print(
        "Model output:",
        model.output_shape
    )

    generator.reset()

    probabilities = model.predict(
        generator,
        verbose=1,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    true_labels = (
        fold_val["class_index"]
        .to_numpy()
    )

    metrics = calculate_metrics(
        true_labels,
        predictions,
        probabilities,
    )

    metrics["fold"] = fold

    all_results.append(metrics)

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = fold_val[
        ["image_id", "filepath", "label"]
    ].copy()

    prediction_df["true_index"] = true_labels

    prediction_df["predicted_index"] = predictions

    prediction_df["predicted_label"] = [
        index_to_class[i]
        for i in predictions
    ]

    prediction_df["confidence"] = np.max(
        probabilities,
        axis=1,
    )

    prediction_file = os.path.join(
        PRED_DIR,
        f"recalculated_validation_predictions_fold_{fold}.csv",
    )

    prediction_df.to_csv(
        prediction_file,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
    )

    cm_file = os.path.join(
        RESULT_DIR,
        f"recalculated_confusion_matrix_fold_{fold}.csv",
    )

    pd.DataFrame(
        cm,
        index=classes,
        columns=classes,
    ).to_csv(cm_file)

    print("\nFold metrics:")

    for key, value in metrics.items():

        if key != "fold":

            print(
                f"{key:25s}: {value:.6f}"
            )

    del model
    tf.keras.backend.clear_session()


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df = results_df[
    [
        "fold",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "specificity_macro",
        "fpr_macro",
        "iou_macro",
        "roc_auc_macro_ovr",
        "pr_auc_macro",
    ]
]

fold_results_file = os.path.join(
    RESULT_DIR,
    "corrected_fold_results.csv",
)

results_df.to_csv(
    fold_results_file,
    index=False,
)


# ============================================================
# MEAN / SD
# ============================================================

metric_columns = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "specificity_macro",
    "fpr_macro",
    "iou_macro",
    "roc_auc_macro_ovr",
    "pr_auc_macro",
]

summary_rows = []

for metric in metric_columns:

    values = (
        results_df[metric]
        .dropna()
        .to_numpy()
    )

    summary_rows.append({

        "metric": metric,

        "mean": np.mean(values),

        "sd": np.std(
            values,
            ddof=1,
        ),

        "min": np.min(values),

        "max": np.max(values),

    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "statistics",
    "corrected_3fold_mean_sd.csv",
)

os.makedirs(
    os.path.dirname(summary_file),
    exist_ok=True,
)

summary_df.to_csv(
    summary_file,
    index=False,
)


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("CORRECTED 3-FOLD CV SUMMARY")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print()
print(
    summary_df.to_string(
        index=False
    )
)

print()
print("Fold results:")
print(fold_results_file)

print()
print("Mean / SD:")
print(summary_file)
