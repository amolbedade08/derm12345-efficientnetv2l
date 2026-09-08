import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# CONFIGURATION
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

PRED_DIR = os.path.join(
    OUTPUT_DIR,
    "predictions",
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


IMAGE_SIZE = (224, 224)

# Same setting as the original production run
BATCH_SIZE = 1

EPOCHS = 20

N_SPLITS = 3

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

print("=" * 70)
print("EFFICIENTNETV2-L - FINAL FOLD 3")
print("=" * 70)

print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

print("GPU:", gpus)

if not gpus:
    raise RuntimeError(
        "GPU was not detected. "
        "Do not start training on CPU."
    )

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(
            gpu,
            True,
        )
    except RuntimeError:
        pass


try:
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16"
    )
    print("Mixed precision: ENABLED")
except Exception as e:
    print("Mixed precision error:", e)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(TRAIN_CSV)

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

df["class_index"] = (
    df["label"]
    .map(class_to_index)
)

df["class_index_str"] = (
    df["class_index"]
    .astype(str)
)

print("Training samples:", len(df))
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

# Third split = Fold 3
train_idx, val_idx = splits[2]

fold_train = df.iloc[train_idx].copy()

fold_val = df.iloc[val_idx].copy()

print()
print("Fold 3")
print("Training samples:", len(fold_train))
print("Validation samples:", len(fold_val))


# ============================================================
# DATA AUGMENTATION
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

val_datagen = ImageDataGenerator()


# ============================================================
# DATA GENERATORS
# ============================================================

train_generator = train_datagen.flow_from_dataframe(
    dataframe=fold_train,
    x_col="filepath",
    y_col="class_index_str",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    class_mode="sparse",
    classes=[str(i) for i in range(NUM_CLASSES)],
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=RANDOM_SEED + 3,
)

val_generator = val_datagen.flow_from_dataframe(
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


# ============================================================
# CLASS WEIGHTS
# ============================================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=fold_train["class_index"],
)

class_weights = {
    int(i): float(w)
    for i, w in enumerate(weights)
}

print("Class weights calculated.")


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    inputs = layers.Input(
        shape=(224, 224, 3)
    )

    base_model = EfficientNetV2L(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
        include_preprocessing=True,
    )

    # Same transfer-learning setup
    base_model.trainable = False

    x = base_model(
        inputs,
        training=False,
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
        outputs,
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


# ============================================================
# CHECKPOINT
# ============================================================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "efficientnetv2l_fold_3_final.keras",
)

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        patience=2,
        factor=0.5,
        min_lr=1e-6,
        verbose=1,
    ),

    ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
]


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("TRAINING FOLD 3")
print("=" * 70)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "history_fold_3_final.csv",
    ),
    index=False,
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

val_generator.reset()

probabilities = model.predict(
    val_generator,
    verbose=1,
)

predictions = np.argmax(
    probabilities,
    axis=1,
)

true_labels = fold_val[
    "class_index"
].to_numpy()


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=np.arange(NUM_CLASSES),
)


# ============================================================
# BASIC METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predictions,
)

precision = precision_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0,
)

recall = recall_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0,
)

f1 = f1_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0,
)


# ============================================================
# SPECIFICITY / FPR / IoU
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

except Exception:

    roc_auc = np.nan


# ============================================================
# PR-AUC
# ============================================================

try:

    true_onehot = tf.keras.utils.to_categorical(
        true_labels,
        num_classes=NUM_CLASSES,
    )

    pr_auc = average_precision_score(
        true_onehot,
        probabilities,
        average="macro",
    )

except Exception:

    pr_auc = np.nan


# ============================================================
# SAVE RESULTS
# ============================================================

result_df = pd.DataFrame([{

    "fold": 3,

    "accuracy": accuracy,

    "precision": precision,

    "recall_sensitivity": recall,

    "f1": f1,

    "specificity": specificity,

    "fpr": fpr,

    "iou": iou,

    "ppv": precision,

    "roc_auc": roc_auc,

    "pr_auc": pr_auc,

}])


result_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "fold_3_final_results.csv",
    ),
    index=False,
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes,
)

cm_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix_fold_3_final.csv",
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = fold_val[
    ["filepath", "label"]
].copy()

prediction_df["true_index"] = (
    true_labels
)

prediction_df["predicted_index"] = (
    predictions
)

prediction_df["predicted_label"] = [
    index_to_class[x]
    for x in predictions
]

prediction_df["confidence"] = (
    np.max(
        probabilities,
        axis=1,
    )
)

prediction_df.to_csv(
    os.path.join(
        PRED_DIR,
        "validation_predictions_fold_3_final.csv",
    ),
    index=False,
)


# ============================================================
# PRINT FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("FOLD 3 FINAL COMPLETED")
print("=" * 70)

for column in result_df.columns:

    if column != "fold":

        print(
            f"{column:25s}: "
            f"{result_df[column].iloc[0]:.4f}"
        )

print()
print("Model:")
print(checkpoint_path)

print()
print("Results:")
print(
    os.path.join(
        RESULT_DIR,
        "fold_3_final_results.csv",
    )
)
