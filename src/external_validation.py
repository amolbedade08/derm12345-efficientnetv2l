import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "models",
    "final_model.keras"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "external_validation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)

# DERM12345 classes
CLASS_NAMES = [
    "acb", "acd", "ajb", "ajd", "ak",
    "alm", "angk", "anm", "bcc", "bd",
    "bdb", "cb", "ccb", "ccd", "cd",
    "ch", "cjb", "db", "df", "dfsp",
    "ha", "isl", "jb", "jd", "ks",
    "la", "lk", "lm", "lmm", "ls",
    "mcb", "mel", "mpd", "pg", "rd",
    "sa", "scc", "sk", "sl", "srjd"
]

# External dataset classes mapped to DERM12345
EXTERNAL_TO_DERM = {
    "BCC": "bcc",
    "MEL": "mel",
    "SCC": "scc",
    "SEK": "sk"
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("EXTERNAL DATASET VALIDATION")
print("=" * 70)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Final model not found:\n{MODEL_PATH}\n"
        "Train the final EfficientNetV2-L model first."
    )

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# EXTERNAL DATASET PATH
# ============================================================

EXTERNAL_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "external",
    "PAD-UFES-20"
)

CSV_PATH = os.path.join(
    EXTERNAL_ROOT,
    "metadata.csv"
)

IMAGE_DIR = os.path.join(
    EXTERNAL_ROOT,
    "images"
)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Metadata not found:\n{CSV_PATH}"
    )

if not os.path.exists(IMAGE_DIR):
    raise FileNotFoundError(
        f"Image directory not found:\n{IMAGE_DIR}"
    )


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("\nExternal dataset:")
print("Total samples:", len(df))
print("Columns:")
print(df.columns.tolist())


# ============================================================
# ADAPT COLUMN NAMES
# ============================================================

# PAD-UFES-20 commonly contains:
# img_id / diagnosis
# Adjust automatically if necessary.

image_column = None
label_column = None

for col in ["img_id", "image_id", "image", "filename"]:
    if col in df.columns:
        image_column = col
        break

for col in ["diagnosis", "label", "dx"]:
    if col in df.columns:
        label_column = col
        break

if image_column is None:
    raise ValueError(
        "Could not identify image filename column."
    )

if label_column is None:
    raise ValueError(
        "Could not identify diagnosis column."
    )

print("Image column:", image_column)
print("Label column:", label_column)


# ============================================================
# MAP EXTERNAL CLASSES
# ============================================================

df["external_label"] = (
    df[label_column]
    .astype(str)
    .str.upper()
    .str.strip()
)

df["derm_label"] = df["external_label"].map(
    EXTERNAL_TO_DERM
)

# Keep only four directly mappable classes
df = df[df["derm_label"].notna()].copy()

print("\nMapped external samples:", len(df))

print(
    df["derm_label"]
    .value_counts()
)


# ============================================================
# CLASS INDEX
# ============================================================

class_to_index = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}

df["true_index"] = df["derm_label"].map(
    class_to_index
)


# ============================================================
# IMAGE PATH
# ============================================================

def find_image(filename):

    filename = str(filename)

    path = os.path.join(
        IMAGE_DIR,
        filename
    )

    if os.path.exists(path):
        return path

    # Try common extensions
    for ext in [".jpg", ".jpeg", ".png"]:

        path = os.path.join(
            IMAGE_DIR,
            filename + ext
        )

        if os.path.exists(path):
            return path

    return None


df["image_path"] = df[
    image_column
].apply(find_image)

df = df[
    df["image_path"].notna()
].copy()

print(
    "Images successfully found:",
    len(df)
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(path):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image


# ============================================================
# PREDICTION
# ============================================================

images = []
true_labels = []

print("\nLoading external images...")

for _, row in df.iterrows():

    try:

        image = load_image(
            row["image_path"]
        )

        images.append(image.numpy())
        true_labels.append(
            row["true_index"]
        )

    except Exception as e:

        print(
            "Skipping:",
            row["image_path"],
            e
        )


X_external = np.asarray(
    images,
    dtype=np.float32
)

y_external = np.asarray(
    true_labels,
    dtype=np.int32
)

print(
    "External tensor:",
    X_external.shape
)


# ============================================================
# PREDICT
# ============================================================

print("\nRunning predictions...")

probabilities = model.predict(
    X_external,
    batch_size=32,
    verbose=1
)

predictions = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_external,
    predictions
)

precision = precision_score(
    y_external,
    predictions,
    labels=list(EXTERNAL_TO_DERM.values()),
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_external,
    predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_external,
    predictions,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 70)
print("EXTERNAL VALIDATION RESULTS")
print("=" * 70)

print(f"Accuracy        : {accuracy:.4f}")
print(f"Macro Precision : {precision:.4f}")
print(f"Macro Recall    : {recall:.4f}")
print(f"Macro F1        : {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

target_indices = [
    class_to_index[x]
    for x in EXTERNAL_TO_DERM.values()
]

target_names = list(
    EXTERNAL_TO_DERM.keys()
)

report = classification_report(
    y_external,
    predictions,
    labels=target_indices,
    target_names=target_names,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

report_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.csv"
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_external,
    predictions,
    labels=target_indices
)

cm_df = pd.DataFrame(
    cm,
    index=target_names,
    columns=target_names
)

cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

df_result = df.iloc[
    :len(predictions)
].copy()

df_result["predicted_index"] = predictions

df_result["predicted_label"] = [
    CLASS_NAMES[i]
    for i in predictions
]

df_result["correct"] = (
    df_result["true_index"]
    ==
    df_result["predicted_index"]
)

df_result.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "external_predictions.csv"
    ),
    index=False
)

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nEXTERNAL VALIDATION COMPLETE")