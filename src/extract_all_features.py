import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = "/mnt/c/Users/0M SAI RAM/Desktop/project"

TRAIN_CSV = os.path.join(
    PROJECT_DIR,
    "outputs/EfficientNetV2L/csv_results/train_loaded.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs/EfficientNetV2L/features"
)

FEATURE_FILE = os.path.join(
    OUTPUT_DIR,
    "train_features.npy"
)

LABEL_FILE = os.path.join(
    OUTPUT_DIR,
    "train_feature_labels.csv"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 2
FEATURE_SIZE = 1280


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("EFFICIENTNETV2-L FULL FEATURE EXTRACTION")
print("=" * 70)

print("TensorFlow:", tf.__version__)


# ============================================================
# GPU
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

print("GPUs detected:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("GPU memory growth enabled.")

    except Exception as e:
        print("GPU memory growth warning:", e)

else:
    print("WARNING: No GPU detected.")


# ============================================================
# LOAD TRAINING CSV
# ============================================================

if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(
        f"\nTraining CSV not found:\n{TRAIN_CSV}"
    )

df = pd.read_csv(TRAIN_CSV)

print("\nTraining CSV loaded.")
print("Number of rows:", len(df))

required_columns = ["filepath", "label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )


# ============================================================
# CHECK IMAGE FILES
# ============================================================

df["filepath"] = df["filepath"].astype(str)

missing = ~df["filepath"].apply(os.path.exists)

if missing.sum() > 0:

    print(
        f"\nWARNING: {missing.sum()} image files are missing."
    )

    df = df.loc[~missing].reset_index(drop=True)


print("Usable images:", len(df))


# ============================================================
# CLASS MAPPING
# ============================================================

df["label"] = df["label"].astype(str)

classes = sorted(df["label"].unique())

NUM_CLASSES = len(classes)

class_to_index = {
    class_name: index
    for index, class_name in enumerate(classes)
}


df["class_index"] = df["label"].map(
    class_to_index
).astype(int)


# IMPORTANT:
# Keras sparse mode requires string values in y_col.

df["class_index_str"] = df["class_index"].astype(str)


print("Number of classes:", NUM_CLASSES)

print("\nClass mapping:")

for class_name, index in class_to_index.items():
    print(f"{index:2d} -> {class_name}")


# ============================================================
# IMAGE GENERATOR
# ============================================================

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)


generator = datagen.flow_from_dataframe(

    dataframe=df,

    x_col="filepath",

    y_col="class_index_str",

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="sparse",

    classes=[
        str(i)
        for i in range(NUM_CLASSES)
    ],

    shuffle=False,

    validate_filenames=False
)


print("\nGenerator created.")

print("Samples:", generator.samples)

print("Batches:", len(generator))


# ============================================================
# LOAD EFFICIENTNETV2-L
# ============================================================

print("\nLoading EfficientNetV2-L...")


base_model = EfficientNetV2L(

    weights="imagenet",

    include_top=False,

    input_shape=(224, 224, 3),

    include_preprocessing=False
)


base_model.trainable = False


# ============================================================
# GLOBAL AVERAGE POOLING
# ============================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)


feature_model = Model(

    inputs=base_model.input,

    outputs=x
)


print("Feature model ready.")

print(
    "Feature dimension:",
    feature_model.output_shape[-1]
)


# ============================================================
# CREATE MEMORY-MAPPED FEATURE FILE
# ============================================================

num_samples = len(df)


features = np.memmap(

    FEATURE_FILE,

    dtype="float32",

    mode="w+",

    shape=(num_samples, FEATURE_SIZE)
)


print("\nFeature storage created.")

print("Expected shape:")

print(
    (num_samples, FEATURE_SIZE)
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("\n")
print("=" * 70)
print("STARTING FEATURE EXTRACTION")
print("=" * 70)

print(
    f"Total images: {num_samples}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    "\nPlease keep this terminal open."
)

print()


generator.reset()

write_index = 0

total_batches = len(generator)


for batch_number in range(total_batches):

    images, labels = generator[batch_number]


    batch_features = feature_model.predict(

        images,

        verbose=0
    )


    batch_features = np.asarray(

        batch_features,

        dtype=np.float32
    )


    current_batch_size = len(
        batch_features
    )


    end_index = (
        write_index
        + current_batch_size
    )


    features[
        write_index:end_index
    ] = batch_features


    write_index = end_index


    # Print progress every 100 batches
    if (
        (batch_number + 1) % 100 == 0
        or batch_number == 0
        or batch_number == total_batches - 1
    ):

        percentage = (
            write_index
            / num_samples
            * 100
        )


        print(
            f"Processed "
            f"{write_index}/{num_samples} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# FLUSH
# ============================================================

features.flush()


# ============================================================
# SAVE LABELS
# ============================================================

labels_df = df[
    [
        "filepath",
        "label",
        "class_index"
    ]
].copy()


labels_df.to_csv(

    LABEL_FILE,

    index=False
)


# ============================================================
# VERIFY
# ============================================================

print("\nVerifying saved feature file...")


del feature_model

del base_model

del generator

gc.collect()


saved_features = np.load(

    FEATURE_FILE,

    mmap_mode="r"
)


print("\n")
print("=" * 70)
print("FEATURE EXTRACTION SUCCESS")
print("=" * 70)

print("\nFeature file:")

print(FEATURE_FILE)


print("\nLabel file:")

print(LABEL_FILE)


print("\nFeature shape:")

print(saved_features.shape)


print(
    "\nNumber of samples:",
    saved_features.shape[0]
)


print(
    "Number of features:",
    saved_features.shape[1]
)


# ============================================================
# FINAL CHECK
# ============================================================

expected_shape = (

    num_samples,

    FEATURE_SIZE
)


if tuple(saved_features.shape) != expected_shape:

    raise RuntimeError(

        "\nFeature shape verification FAILED.\n"

        f"Expected: {expected_shape}\n"

        f"Got: {saved_features.shape}"
    )


print("\nShape verification: PASSED")

print(
    "All training features were extracted successfully."
)

print("=" * 70)
