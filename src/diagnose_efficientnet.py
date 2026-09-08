import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
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

IMAGE_SIZE = (224, 224)

# Small dataset ONLY for diagnosis
SAMPLE_SIZE = 320

BATCH_SIZE = 4

EPOCHS = 5

LEARNING_RATE = 1e-3

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# GPU
# ============================================================

print("=" * 70)
print("EFFICIENTNETV2-L DIAGNOSTIC TEST")
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


# Mixed precision
tf.keras.mixed_precision.set_global_policy(
    "mixed_float16"
)

print("Mixed precision: ENABLED")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(TRAIN_CSV)

print("\nTotal training images:", len(df))

# Keep only samples with valid files
df = df[df["filepath"].map(os.path.exists)].copy()

print("Valid images:", len(df))


# ============================================================
# LABEL MAPPING
# ============================================================

classes = sorted(
    df["label"].unique()
)

class_to_index = {
    label: i
    for i, label in enumerate(classes)
}

df["class_index"] = df["label"].map(
    class_to_index
)

df["class_index_str"] = df["class_index"].astype(str)

NUM_CLASSES = len(classes)

print("Number of classes:", NUM_CLASSES)


# ============================================================
# CREATE A BALANCED SMALL SAMPLE
# ============================================================

# Take up to 8 images per class
samples_per_class = max(
    1,
    SAMPLE_SIZE // NUM_CLASSES
)

parts = []

for label in classes:

    class_df = df[
        df["label"] == label
    ]

    n = min(
        len(class_df),
        samples_per_class
    )

    if n > 0:
        parts.append(
            class_df.sample(
                n=n,
                random_state=RANDOM_SEED
            )
        )

sample_df = pd.concat(
    parts
).sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

print("\nDiagnostic samples:", len(sample_df))

print(
    "\nSamples per class:"
)

print(
    sample_df["label"].value_counts().sort_index()
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
    y_col="class_index_str",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    class_mode="sparse",
    classes=[
        str(i)
        for i in range(NUM_CLASSES)
    ],
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=RANDOM_SEED
)


# ============================================================
# MODEL
# ============================================================

print("\nCreating EfficientNetV2-L...")

inputs = layers.Input(
    shape=(224, 224, 3)
)

base_model = EfficientNetV2L(
    weights="imagenet",
    include_top=False,
    include_preprocessing=False,
    input_tensor=inputs
)

# IMPORTANT:
# For this diagnostic test, allow the backbone to learn.
base_model.trainable = True

x = base_model(
    inputs,
    training=True
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    dtype="float32"
)(x)

model = models.Model(
    inputs,
    outputs
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 70)
print("STARTING DIAGNOSTIC TRAINING")
print("=" * 70)

history = model.fit(
    generator,
    epochs=EPOCHS,
    verbose=1
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("DIAGNOSTIC RESULT")
print("=" * 70)

for epoch in range(len(history.history["accuracy"])):

    acc = history.history["accuracy"][epoch]

    loss = history.history["loss"][epoch]

    print(
        f"Epoch {epoch + 1}: "
        f"accuracy={acc:.4f}, "
        f"loss={loss:.4f}"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

final_accuracy = history.history[
    "accuracy"
][-1]

first_accuracy = history.history[
    "accuracy"
][0]

print("\nFirst epoch accuracy:", first_accuracy)
print("Final epoch accuracy:", final_accuracy)

if final_accuracy > first_accuracy:
    print("\nSUCCESS:")
    print(
        "Training accuracy increased."
    )
    print(
        "The image -> label -> model pipeline "
        "is learning."
    )
else:
    print("\nWARNING:")
    print(
        "Training accuracy did not increase."
    )
    print(
        "We need to investigate preprocessing, "
        "labels, or dataset configuration."
    )
