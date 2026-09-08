import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger


# ============================================================
# FINAL MODEL TRAINING ON ALL 9,860 OFFICIAL TRAIN IMAGES
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

MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "models",
)

RESULT_DIR = os.path.join(
    OUTPUT_DIR,
    "csv_results",
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 2

# Median of observed best epochs from the corrected CV runs:
# Fold 1 = 19
# Fold 2 = 9
# Fold 3 = 3
# Median = 9
EPOCHS = 9

RANDOM_SEED = 42

DROPOUT_RATE = 0.3
L2_REG = 1e-4
LEARNING_RATE = 1e-4


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# GPU SETUP
# ============================================================

print("=" * 75)
print("FINAL EFFICIENTNETV2-L TRAINING")
print("=" * 75)

print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

print("GPUs:", gpus)

if not gpus:
    raise RuntimeError(
        "GPU NOT DETECTED. Do NOT train this final model on CPU."
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
# MIXED PRECISION
# ============================================================

tf.keras.mixed_precision.set_global_policy(
    "mixed_float16"
)

print("Mixed precision: ENABLED")


# ============================================================
# LOAD ALL OFFICIAL TRAINING DATA
# ============================================================

print("\nLoading training metadata...")

df = pd.read_csv(TRAIN_CSV)

print("Training samples:", len(df))

if len(df) != 9860:
    raise ValueError(
        f"Expected 9860 training images, found {len(df)}."
    )


# ============================================================
# LABEL ENCODING
# ============================================================

classes = sorted(
    df["label"].unique()
)

class_to_index = {
    cls: i
    for i, cls in enumerate(classes)
}

index_to_class = {
    i: cls
    for cls, i in class_to_index.items()
}

NUM_CLASSES = len(classes)

if NUM_CLASSES != 40:
    raise ValueError(
        f"Expected 40 classes, found {NUM_CLASSES}."
    )

df["class_index"] = (
    df["label"].map(class_to_index)
)

df["class_index_str"] = (
    df["class_index"].astype(str)
)

print("Number of classes:", NUM_CLASSES)


# ============================================================
# VERIFY FILEPATHS
# ============================================================

missing = df["filepath"].isna().sum()

if missing > 0:
    raise ValueError(
        f"{missing} images have missing filepaths."
    )

exists = df["filepath"].map(
    os.path.exists
)

if not exists.all():
    missing_count = (~exists).sum()

    raise ValueError(
        f"{missing_count} image files do not exist."
    )

print("All 9,860 image paths verified.")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=df["class_index"],
)

class_weights = {
    int(i): float(w)
    for i, w in enumerate(weights)
}

print("\nClass weights calculated.")


# ============================================================
# DATA AUGMENTATION
# Same augmentation used during CV
# ============================================================

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.10,
    shear_range=0.05,
    horizontal_flip=True,
    brightness_range=(0.85, 1.15),
)


# ============================================================
# TRAINING GENERATOR
# ============================================================

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    x_col="filepath",
    y_col="class_index_str",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    class_mode="sparse",
    classes=[str(i) for i in range(NUM_CLASSES)],
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=RANDOM_SEED,
)

print(
    "\nGenerator samples:",
    train_generator.samples
)


# ============================================================
# MODEL
# ============================================================

def build_model():

    inputs = layers.Input(
        shape=(224, 224, 3)
    )

    backbone = EfficientNetV2L(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
        include_preprocessing=True,
    )

    # Same transfer-learning configuration
    # used in the corrected Fold-3 run.
    backbone.trainable = False

    x = backbone(
        inputs,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(
        DROPOUT_RATE
    )(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        kernel_regularizer=regularizers.l2(
            L2_REG
        ),
        dtype="float32",
    )(x)

    model = models.Model(
        inputs,
        outputs
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


model = build_model()

print("\nModel parameters:", model.count_params())


# ============================================================
# OUTPUT PATHS
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "efficientnetv2l_final.keras",
)

HISTORY_PATH = os.path.join(
    RESULT_DIR,
    "history_final_all.csv",
)

CLASS_MAPPING_PATH = os.path.join(
    RESULT_DIR,
    "final_class_mapping.csv",
)


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

mapping_df = pd.DataFrame({
    "class_index": list(index_to_class.keys()),
    "label": list(index_to_class.values())
})

mapping_df.to_csv(
    CLASS_MAPPING_PATH,
    index=False
)


# ============================================================
# CSV LOGGER
# ============================================================

csv_logger = CSVLogger(
    HISTORY_PATH,
    append=False
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 75)
print("TRAINING FINAL MODEL ON ALL 9,860 IMAGES")
print("=" * 75)

print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)

print()
print("IMPORTANT:")
print("The official 2,485-image test set is NOT used during training.")


history = model.fit(
    train_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[csv_logger],
    verbose=1,
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print()
print("Saving final model...")

model.save(
    MODEL_PATH
)

print("Final model saved.")


# ============================================================
# TRAINING SUMMARY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

best_epoch = (
    history_df["accuracy"].idxmax() + 1
)

best_train_accuracy = (
    history_df["accuracy"].max()
)

final_train_accuracy = (
    history_df["accuracy"].iloc[-1]
)

final_train_loss = (
    history_df["loss"].iloc[-1]
)


print()
print("=" * 75)
print("FINAL MODEL TRAINING COMPLETED")
print("=" * 75)

print("Training images :", len(df))
print("Classes         :", NUM_CLASSES)

print(
    "Best train accuracy:",
    f"{best_train_accuracy:.4f}"
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Final train accuracy:",
    f"{final_train_accuracy:.4f}"
)

print(
    "Final train loss:",
    f"{final_train_loss:.4f}"
)

print()
print("MODEL:")
print(MODEL_PATH)

print()
print("HISTORY:")
print(HISTORY_PATH)

print()
print("CLASS MAPPING:")
print(CLASS_MAPPING_PATH)

print()
print("=" * 75)
print("READY FOR OFFICIAL TEST EVALUATION")
print("=" * 75)
