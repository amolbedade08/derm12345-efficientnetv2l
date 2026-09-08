import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetV2L
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
    "train_loaded.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "EfficientNetV2L",
    "features"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

IMAGE_SIZE = (224, 224)

# Small test first
SAMPLE_SIZE = 100

BATCH_SIZE = 2


# ============================================================
# GPU SETUP
# ============================================================

print("=" * 70)
print("EFFICIENTNETV2-L FEATURE EXTRACTION TEST")
print("=" * 70)

print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

print("GPU:", gpus)

if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )
        except RuntimeError:
            pass


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(TRAIN_CSV)

print("\nTotal training images:", len(df))

df = df[
    df["filepath"].map(os.path.exists)
].copy()

print("Valid images:", len(df))


# ============================================================
# TAKE SMALL SAMPLE
# ============================================================

sample_df = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=42
).reset_index(drop=True)

print(
    "Images for feature extraction:",
    len(sample_df)
)


# ============================================================
# DATA GENERATOR
# ============================================================

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

generator = datagen.flow_from_dataframe(
    dataframe=sample_df,
    x_col="filepath",
    y_col=None,
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LOAD EFFICIENTNETV2-L
# ============================================================

print("\nLoading EfficientNetV2-L...")

model = EfficientNetV2L(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
    include_preprocessing=False
)

model.trainable = False

print("Backbone loaded.")
print("Backbone frozen.")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("\nExtracting features...")

features = model.predict(
    generator,
    verbose=1
)

print("\nRaw feature shape:", features.shape)


# ============================================================
# GLOBAL AVERAGE POOLING
# ============================================================

features = np.mean(
    features,
    axis=(1, 2)
)

print(
    "Pooled feature shape:",
    features.shape
)


# ============================================================
# SAVE FEATURES
# ============================================================

feature_path = os.path.join(
    OUTPUT_DIR,
    "diagnostic_features.npy"
)

label_path = os.path.join(
    OUTPUT_DIR,
    "diagnostic_labels.csv"
)

np.save(
    feature_path,
    features
)

sample_df[
    ["filepath", "label"]
].to_csv(
    label_path,
    index=False
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE EXTRACTION SUCCESS")
print("=" * 70)

print(
    "Features:",
    feature_path
)

print(
    "Labels:",
    label_path
)

print(
    "Number of samples:",
    features.shape[0]
)

print(
    "Number of features:",
    features.shape[1]
)
