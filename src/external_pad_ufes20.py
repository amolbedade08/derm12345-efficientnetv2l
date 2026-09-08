import os
import glob
import json
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
    classification_report,
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "external",
    "PAD-UFES-20",
    "metadata.csv"
)

IMAGE_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "external",
    "PAD-UFES-20",
    "images"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "models",
    "efficientnetv2l_final.keras"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "external_PAD_UFES20"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8

# PAD-UFES -> Derm12345 class names
CLASS_MAPPING = {
    "BCC": "bcc",
    "MEL": "mel",
    "SCC": "scc",
    "SEK": "sk",
}

EXTERNAL_CLASSES = ["BCC", "MEL", "SCC", "SEK"]
MODEL_CLASSES = ["bcc", "mel", "scc", "sk"]

# ============================================================
# GPU INFORMATION
# ============================================================

print("=" * 70)
print("PAD-UFES-20 EXTERNAL VALIDATION")
print("=" * 70)

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
print("GPUs detected:", gpus)

if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading metadata...")

df = pd.read_csv(METADATA_PATH)

required_columns = ["img_id", "diagnostic"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Required column '{col}' not found in metadata.csv"
        )

print("Total metadata rows:", len(df))

# Keep only compatible classes
df = df[df["diagnostic"].isin(EXTERNAL_CLASSES)].copy()

print("\nCompatible class distribution:")
print(df["diagnostic"].value_counts())

# ============================================================
# RESOLVE IMAGE PATHS
# ============================================================

print("\nResolving image paths...")

def resolve_image_path(filename):
    matches = glob.glob(
        os.path.join(IMAGE_ROOT, "*", filename)
    )

    if not matches:
        return None

    return matches[0]


df["image_path"] = df["img_id"].apply(resolve_image_path)

missing = df["image_path"].isna().sum()

print("Compatible images:", len(df))
print("Missing images:", missing)

if missing > 0:
    print("\nMissing image examples:")
    print(df[df["image_path"].isna()][["img_id", "diagnostic"]].head(20))
    raise RuntimeError(
        "Some compatible images could not be found."
    )

# ============================================================
# MAP LABELS
# ============================================================

df["mapped_label"] = df["diagnostic"].map(CLASS_MAPPING)

if df["mapped_label"].isna().any():
    raise RuntimeError("Some labels could not be mapped.")

label_to_index = {
    label: i for i, label in enumerate(MODEL_CLASSES)
}

df["target_index"] = df["mapped_label"].map(label_to_index)

print("\nExternal class mapping:")
for k, v in CLASS_MAPPING.items():
    print(f"  {k} -> {v} -> index {label_to_index[v]}")

# ============================================================
# VERIFY MODEL CLASS MAPPING
# ============================================================

mapping_csv = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "csv_results",
    "final_class_mapping.csv"
)

if os.path.exists(mapping_csv):
    print("\nChecking final model class mapping...")

    mapping_df = pd.read_csv(mapping_csv)

    print(mapping_df.to_string(index=False))

    # Try to identify expected index of each mapped class
    if "label" in mapping_df.columns and "class_index" in mapping_df.columns:

        model_mapping = dict(
            zip(
                mapping_df["label"].astype(str),
                mapping_df["class_index"].astype(int)
            )
        )

        print("\nVerified Derm12345 model indices:")

        for model_class in MODEL_CLASSES:
            if model_class in model_mapping:
                print(
                    f"  {model_class}: {model_mapping[model_class]}"
                )
else:
    print(
        "\nWARNING: final_class_mapping.csv was not found."
    )

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading EfficientNetV2-L model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model input:", model.input_shape)
print("Model output:", model.output_shape)
print("Parameters:", model.count_params())

# ============================================================
# DATASET PIPELINE
# ============================================================

paths = df["image_path"].values
targets = df["target_index"].values

def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.image.decode_png(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(image, tf.float32)

    return image, label


dataset = tf.data.Dataset.from_tensor_slices(
    (paths, targets)
)

dataset = dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset = dataset.batch(
    BATCH_SIZE
).prefetch(tf.data.AUTOTUNE)

# ============================================================
# PREDICTION
# ============================================================

print("\nRunning external inference...")

all_probs = []

for batch_images, _ in dataset:
    probs = model.predict(
        batch_images,
        verbose=0
    )

    all_probs.append(probs)

all_probs = np.concatenate(
    all_probs,
    axis=0
)

print("Raw model probability shape:", all_probs.shape)

# ============================================================
# GET MODEL INDICES FOR THE FOUR TARGET CLASSES
# ============================================================

if os.path.exists(mapping_csv):
    mapping_df = pd.read_csv(mapping_csv)

    if "label" in mapping_df.columns and "class_index" in mapping_df.columns:

        model_mapping = dict(
            zip(
                mapping_df["label"].astype(str),
                mapping_df["class_index"].astype(int)
            )
        )

        selected_indices = [
            model_mapping[c]
            for c in MODEL_CLASSES
        ]

    else:
        raise RuntimeError(
            "final_class_mapping.csv does not contain "
            "'label' and 'class_index'."
        )

else:
    raise FileNotFoundError(
        "final_class_mapping.csv is required."
    )

print("\nSelected model output indices:")
for c, idx in zip(MODEL_CLASSES, selected_indices):
    print(f"  {c}: {idx}")

# ============================================================
# RESTRICT PROBABILITIES TO FOUR TARGET CLASSES
# ============================================================

selected_probs = all_probs[
    :,
    selected_indices
]

# Renormalize probabilities over the four selected classes
prob_sum = selected_probs.sum(axis=1, keepdims=True)

selected_probs = selected_probs / np.maximum(
    prob_sum,
    1e-12
)

pred_indices = np.argmax(
    selected_probs,
    axis=1
)

true_indices = targets

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    true_indices,
    pred_indices
)

precision = precision_score(
    true_indices,
    pred_indices,
    average="macro",
    zero_division=0
)

recall = recall_score(
    true_indices,
    pred_indices,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    true_indices,
    pred_indices,
    average="macro",
    zero_division=0
)

cm = confusion_matrix(
    true_indices,
    pred_indices,
    labels=np.arange(len(MODEL_CLASSES))
)

# ------------------------------------------------------------
# Specificity / FPR / IoU
# ------------------------------------------------------------

specificities = []
fprs = []
ious = []

for i in range(len(MODEL_CLASSES)):

    tp = cm[i, i]

    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp

    tn = cm.sum() - tp - fn - fp

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

    specificities.append(specificity)
    fprs.append(fpr)
    ious.append(iou)

specificity_macro = np.mean(specificities)
fpr_macro = np.mean(fprs)
iou_macro = np.mean(ious)

# ------------------------------------------------------------
# ROC-AUC and PR-AUC
# ------------------------------------------------------------

y_true_onehot = tf.keras.utils.to_categorical(
    true_indices,
    num_classes=len(MODEL_CLASSES)
)

try:
    roc_auc = roc_auc_score(
        y_true_onehot,
        selected_probs,
        average="macro",
        multi_class="ovr"
    )
except Exception as e:
    print("ROC-AUC calculation failed:", e)
    roc_auc = np.nan

try:
    pr_auc = average_precision_score(
        y_true_onehot,
        selected_probs,
        average="macro"
    )
except Exception as e:
    print("PR-AUC calculation failed:", e)
    pr_auc = np.nan

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("EXTERNAL VALIDATION RESULTS")
print("=" * 70)

print(f"Accuracy           : {accuracy:.6f}")
print(f"Macro Precision    : {precision:.6f}")
print(f"Macro Recall       : {recall:.6f}")
print(f"Macro F1           : {f1:.6f}")
print(f"Specificity        : {specificity_macro:.6f}")
print(f"FPR                : {fpr_macro:.6f}")
print(f"IoU                : {iou_macro:.6f}")
print(f"ROC-AUC            : {roc_auc:.6f}")
print(f"PR-AUC             : {pr_auc:.6f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    true_indices,
    pred_indices,
    labels=np.arange(len(MODEL_CLASSES)),
    target_names=MODEL_CLASSES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

# ============================================================
# CONFUSION MATRIX CSV
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=EXTERNAL_CLASSES,
    columns=EXTERNAL_CLASSES
)

# ============================================================
# PREDICTIONS CSV
# ============================================================

prediction_df = df[
    [
        "patient_id",
        "lesion_id",
        "img_id",
        "diagnostic",
        "mapped_label",
        "image_path"
    ]
].copy()

prediction_df["true_class_index"] = true_indices
prediction_df["predicted_class_index"] = pred_indices

prediction_df["predicted_class"] = [
    MODEL_CLASSES[i]
    for i in pred_indices
]

prediction_df["correct"] = (
    prediction_df["true_class_index"]
    == prediction_df["predicted_class_index"]
)

for i, label in enumerate(MODEL_CLASSES):
    prediction_df[
        f"prob_{label}"
    ] = selected_probs[:, i]

# ============================================================
# METRICS CSV
# ============================================================

metrics_df = pd.DataFrame(
    {
        "dataset": ["PAD-UFES-20_external"],
        "n_images": [len(df)],
        "n_classes": [len(MODEL_CLASSES)],
        "accuracy": [accuracy],
        "precision_macro": [precision],
        "recall_macro": [recall],
        "f1_macro": [f1],
        "specificity_macro": [specificity_macro],
        "fpr_macro": [fpr_macro],
        "iou_macro": [iou_macro],
        "roc_auc_macro_ovr": [roc_auc],
        "pr_auc_macro": [pr_auc],
    }
)

# ============================================================
# SAVE CSV FILES
# ============================================================

metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "external_metrics.csv"
    ),
    index=False
)

report_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.csv"
    )
)

cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    )
)

prediction_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "predictions.csv"
    ),
    index=False
)

# ============================================================
# SAVE CLASS DISTRIBUTION
# ============================================================

class_distribution = (
    df["diagnostic"]
    .value_counts()
    .reindex(EXTERNAL_CLASSES)
    .fillna(0)
    .astype(int)
    .rename_axis("class")
    .reset_index(name="count")
)

class_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "class_distribution.csv"
    ),
    index=False
)

# ============================================================
# SAVE CONFIG
# ============================================================

config = {
    "dataset": "PAD-UFES-20",
    "source_metadata": METADATA_PATH,
    "image_root": IMAGE_ROOT,
    "model": MODEL_PATH,
    "total_dataset_images": 2298,
    "evaluated_images": int(len(df)),
    "excluded_classes": ["ACK", "NEV"],
    "class_mapping": CLASS_MAPPING,
    "external_classes": EXTERNAL_CLASSES,
    "model_classes": MODEL_CLASSES,
    "image_size": list(IMAGE_SIZE),
    "batch_size": BATCH_SIZE,
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "config.json"
    ),
    "w"
) as f:
    json.dump(
        config,
        f,
        indent=2
    )

# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(8, 7))

plt.imshow(cm)

plt.title(
    "PAD-UFES-20 External Validation\nConfusion Matrix"
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(
    range(len(EXTERNAL_CLASSES)),
    EXTERNAL_CLASSES,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(EXTERNAL_CLASSES)),
    EXTERNAL_CLASSES
)

for i in range(len(EXTERNAL_CLASSES)):
    for j in range(len(EXTERNAL_CLASSES)):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# PER-CLASS ROC CURVES
# ============================================================

plt.figure(figsize=(8, 7))

for i, label  in enumerate(EXTERNAL_CLASSES):

    y_true_class = (
        true_indices == i
    ).astype(int)

    try:
        from sklearn.metrics import roc_curve

        fpr_values, tpr_values, _ = roc_curve(
            y_true_class,
            selected_probs[:, i]
        )

        auc_value = roc_auc_score(
            y_true_class,
            selected_probs[:, i]
        )

        plt.plot(
            fpr_values,
            tpr_values,
            label=f"{label} (AUC={auc_value:.3f})"
        )

    except Exception:
        pass

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "PAD-UFES-20 External Validation\nROC Curves"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curves.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# PER-CLASS PR CURVES
# ============================================================

plt.figure(figsize=(8, 7))

from sklearn.metrics import precision_recall_curve

for i, label in enumerate(EXTERNAL_CLASSES):

    y_true_class = (
        true_indices == i
    ).astype(int)

    try:
        precision_values, recall_values, _ = (
            precision_recall_curve(
                y_true_class,
                selected_probs[:, i]
            )
        )

        ap_value = average_precision_score(
            y_true_class,
            selected_probs[:, i]
        )

        plt.plot(
            recall_values,
            precision_values,
            label=f"{label} (AP={ap_value:.3f})"
        )

    except Exception:
        pass

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "PAD-UFES-20 External Validation\nPrecision-Recall Curves"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "pr_curves.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFiles saved to:")
print(OUTPUT_DIR)

print("\nExternal validation completed successfully.")
