import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TEST_CSV = os.path.join(
    PROJECT_DIR,
    "outputs",
    "EfficientNetV2L",
    "csv_results",
    "test_loaded.csv",
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "outputs",
    "EfficientNetV2L",
    "models",
    "efficientnetv2l_final.keras",
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

CM_DIR = os.path.join(
    OUTPUT_DIR,
    "confusion_matrices",
)

PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    "plots",
)

for d in [
    RESULT_DIR,
    PRED_DIR,
    CM_DIR,
    PLOT_DIR,
]:
    os.makedirs(d, exist_ok=True)


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 2


# ============================================================
# GPU
# ============================================================

print("=" * 75)
print("OFFICIAL DERM12345 TEST EVALUATION")
print("=" * 75)

print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

print("GPUs:", gpus)

if not gpus:
    raise RuntimeError(
        "GPU not detected."
    )

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(
            gpu,
            True
        )
    except RuntimeError:
        pass


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(TEST_CSV)

print("\nTest samples:", len(test_df))

if len(test_df) != 2485:
    raise ValueError(
        f"Expected 2485 test images, found {len(test_df)}"
    )


# ============================================================
# LABEL MAPPING
# ============================================================

classes = sorted(
    pd.read_csv(
        os.path.join(
            PROJECT_DIR,
            "outputs",
            "EfficientNetV2L",
            "csv_results",
            "train_loaded.csv",
        )
    )["label"].unique()
)

NUM_CLASSES = len(classes)

class_to_index = {
    c: i
    for i, c in enumerate(classes)
}

index_to_class = {
    i: c
    for c, i in class_to_index.items()
}

print("Number of classes:", NUM_CLASSES)

if NUM_CLASSES != 40:
    raise ValueError(
        f"Expected 40 classes, found {NUM_CLASSES}"
    )


test_df["class_index"] = (
    test_df["label"].map(class_to_index)
)

if test_df["class_index"].isna().any():
    raise ValueError(
        "Test set contains labels not present in training mapping."
    )


# ============================================================
# PATH VALIDATION
# ============================================================

if test_df["filepath"].isna().any():

    raise ValueError(
        "Test set contains missing filepaths."
    )

exists = test_df["filepath"].map(
    os.path.exists
)

if not exists.all():

    print(
        "Missing files:",
        (~exists).sum()
    )

    raise ValueError(
        "Some test images do not exist."
    )

print("All test image paths verified.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading final model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded.")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)
print("Parameters  :", model.count_params())


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    # Keep 0-255 range because the EfficientNetV2-L
    # model was built with include_preprocessing=True.
    return image


# ============================================================
# PREDICTION
# ============================================================

true_labels = (
    test_df["class_index"]
    .to_numpy()
)

probabilities = []

print("\nStarting prediction...")

for start in range(
    0,
    len(test_df),
    BATCH_SIZE
):

    batch = test_df.iloc[
        start:start + BATCH_SIZE
    ]

    images = tf.stack([
        load_image(path)
        for path in batch["filepath"]
    ])

    probs = model.predict(
        images,
        verbose=0
    )

    probabilities.append(
        probs
    )

    processed = min(
        start + BATCH_SIZE,
        len(test_df)
    )

    if (
        processed % 200 == 0
        or processed == len(test_df)
    ):

        print(
            f"Processed {processed}/{len(test_df)}"
        )


probabilities = np.concatenate(
    probabilities,
    axis=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)

confidence = np.max(
    probabilities,
    axis=1
)


# ============================================================
# METRICS
# ============================================================

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=np.arange(NUM_CLASSES)
)

accuracy = accuracy_score(
    true_labels,
    predictions
)

precision = precision_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# SPECIFICITY / FPR / IOU
# ============================================================

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

    fprs.append(fpr)

    ious.append(iou)


specificity = np.mean(
    specificities
)

fpr = np.mean(
    fprs
)

iou = np.mean(
    ious
)


# ============================================================
# ROC-AUC
# ============================================================

try:

    roc_auc = roc_auc_score(
        true_labels,
        probabilities,
        multi_class="ovr",
        average="macro",
    )

except Exception as e:

    print("ROC-AUC warning:", e)

    roc_auc = np.nan


# ============================================================
# PR-AUC
# ============================================================

try:

    y_onehot = tf.keras.utils.to_categorical(
        true_labels,
        num_classes=NUM_CLASSES
    )

    pr_auc = average_precision_score(
        y_onehot,
        probabilities,
        average="macro"
    )

except Exception as e:

    print("PR-AUC warning:", e)

    pr_auc = np.nan


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    true_labels,
    predictions,
    labels=np.arange(NUM_CLASSES),
    target_names=classes,
    zero_division=0,
    output_dict=True,
)

report_df = pd.DataFrame(
    report
).transpose()


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([{

    "dataset": "Derm12345_official_test",

    "n_samples": len(test_df),

    "n_classes": NUM_CLASSES,

    "accuracy": accuracy,

    "precision_macro": precision,

    "recall_macro": recall,

    "f1_macro": f1,

    "specificity_macro": specificity,

    "fpr_macro": fpr,

    "iou_macro": iou,

    "roc_auc_macro_ovr": roc_auc,

    "pr_auc_macro": pr_auc,
}])

metrics_path = os.path.join(
    RESULT_DIR,
    "official_test_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(
    RESULT_DIR,
    "official_test_classification_report.csv"
)

report_df.to_csv(
    report_path
)


# ============================================================
# SAVE CONFUSION MATRIX CSV
# ============================================================

cm_csv_path = os.path.join(
    RESULT_DIR,
    "official_test_confusion_matrix.csv"
)

pd.DataFrame(
    cm,
    index=classes,
    columns=classes,
).to_csv(
    cm_csv_path
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = test_df[
    [
        "image_id",
        "patient_id",
        "filepath",
        "label",
    ]
].copy()

prediction_df["true_index"] = true_labels

prediction_df["predicted_index"] = predictions

prediction_df["predicted_label"] = [
    index_to_class[i]
    for i in predictions
]

prediction_df["confidence"] = confidence

# ------------------------------------------------------------
# SAVE FULL 40-CLASS PROBABILITIES
# ------------------------------------------------------------

for class_index, class_name in index_to_class.items():

    prediction_df[
        f"prob_{class_name}"
    ] = probabilities[:, class_index]

prediction_path = os.path.join(
    PRED_DIR,
    "official_test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)

print(
    "\nOfficial prediction file saved with "
    f"{NUM_CLASSES} probability columns."
)

print(
    "Prediction shape:",
    prediction_df.shape
)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(18, 16)
)

im = ax.imshow(
    cm,
    interpolation="nearest",
)

ax.set_title(
    "Derm12345 Official Test Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "True Class"
)

ax.set_xticks(
    np.arange(NUM_CLASSES)
)

ax.set_yticks(
    np.arange(NUM_CLASSES)
)

ax.set_xticklabels(
    classes,
    rotation=90,
    fontsize=7
)

ax.set_yticklabels(
    classes,
    fontsize=7
)

fig.colorbar(im, ax=ax)

plt.tight_layout()

cm_png_path = os.path.join(
    CM_DIR,
    "official_test_confusion_matrix.png"
)

plt.savefig(
    cm_png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# MULTI-CLASS ROC CURVES
# One-vs-rest
# ============================================================

y_onehot = tf.keras.utils.to_categorical(
    true_labels,
    num_classes=NUM_CLASSES
)

plt.figure(
    figsize=(10, 8)
)

valid_roc = 0

for i in range(NUM_CLASSES):

    positives = y_onehot[:, i]

    if (
        positives.sum() == 0
        or positives.sum() == len(positives)
    ):
        continue

    fpr_i, tpr_i, _ = roc_curve(
        positives,
        probabilities[:, i]
    )

    auc_i = roc_auc_score(
        positives,
        probabilities[:, i]
    )

    plt.plot(
        fpr_i,
        tpr_i,
        linewidth=1,
        label=f"{classes[i]} (AUC={auc_i:.3f})"
    )

    valid_roc += 1


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Derm12345 Official Test ROC Curves"
)

if valid_roc <= 15:
    plt.legend(
        fontsize=7,
        loc="lower right"
    )

plt.tight_layout()

roc_path = os.path.join(
    PLOT_DIR,
    "official_test_roc_curves.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# MULTI-CLASS PRECISION-RECALL CURVES
# ============================================================

plt.figure(
    figsize=(10, 8)
)

valid_pr = 0

for i in range(NUM_CLASSES):

    positives = y_onehot[:, i]

    if (
        positives.sum() == 0
        or positives.sum() == len(positives)
    ):
        continue

    precision_i, recall_i, _ = precision_recall_curve(
        positives,
        probabilities[:, i]
    )

    ap_i = average_precision_score(
        positives,
        probabilities[:, i]
    )

    plt.plot(
        recall_i,
        precision_i,
        linewidth=1,
        label=f"{classes[i]} (AP={ap_i:.3f})"
    )

    valid_pr += 1


plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Derm12345 Official Test Precision-Recall Curves"
)

if valid_pr <= 15:
    plt.legend(
        fontsize=7,
        loc="lower left"
    )

plt.tight_layout()

pr_path = os.path.join(
    PLOT_DIR,
    "official_test_pr_curves.png"
)

plt.savefig(
    pr_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 75)
print("OFFICIAL TEST RESULTS")
print("=" * 75)

print(
    f"Accuracy          : {accuracy:.6f}"
)

print(
    f"Macro Precision   : {precision:.6f}"
)

print(
    f"Macro Recall      : {recall:.6f}"
)

print(
    f"Macro F1          : {f1:.6f}"
)

print(
    f"Specificity       : {specificity:.6f}"
)

print(
    f"FPR               : {fpr:.6f}"
)

print(
    f"IoU               : {iou:.6f}"
)

print(
    f"ROC-AUC           : {roc_auc:.6f}"
)

print(
    f"PR-AUC            : {pr_auc:.6f}"
)

print()
print("Metrics CSV:")
print(metrics_path)

print()
print("Classification report:")
print(report_path)

print()
print("Confusion matrix:")
print(cm_csv_path)

print()
print("Predictions:")
print(prediction_path)

print()
print("Confusion matrix figure:")
print(cm_png_path)

print()
print("ROC figure:")
print(roc_path)

print()
print("PR figure:")
print(pr_path)

print()
print("=" * 75)
print("OFFICIAL TEST EVALUATION COMPLETED")
print("=" * 75)
