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

for d in [MODEL_DIR, RESULT_DIR, PRED_DIR]:
    os.makedirs(d, exist_ok=True)


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 2
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
# GPU
# ============================================================

print("=" * 70)
print("FIXED FOLD 3 TRAINING")
print("=" * 70)

print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

print("GPUs:", gpus)

if not gpus:
    raise RuntimeError(
        "GPU not detected. Stop. Do not train on CPU."
    )

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(
            gpu,
            True,
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
    df["label"].map(class_to_index)
)

df["class_index_str"] = (
    df["class_index"].astype(str)
)

print("Training samples:", len(df))
print("Classes:", NUM_CLASSES)


# ============================================================
# EXACT FOLD 3 SPLIT
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

train_idx, val_idx = splits[2]

fold_train = df.iloc[train_idx].copy()
fold_val = df.iloc[val_idx].copy()

print()
print("Fold 3 training:", len(fold_train))
print("Fold 3 validation:", len(fold_val))


# ============================================================
# DATA GENERATORS
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
# MODEL
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

print("Parameters:", model.count_params())


# ============================================================
# CHECKPOINTS
# ============================================================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "efficientnetv2l_fold_3_fixed.keras",
)

history_path = os.path.join(
    RESULT_DIR,
    "history_fold_3_fixed.csv",
)

result_path = os.path.join(
    RESULT_DIR,
    "fold_3_fixed_results.csv",
)

prediction_path = os.path.join(
    PRED_DIR,
    "validation_predictions_fold_3_fixed.csv",
)


# ============================================================
# CALLBACKS
# ============================================================

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
print("STARTING FIXED FOLD 3")
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
    history_path,
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
# METRICS
# ============================================================

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=np.arange(NUM_CLASSES),
)

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

specificities = []
fprs = []
ious = []

for i in range(NUM_CLASSES):

    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp

    tn = cm.sum() - (
        tp + fn + fp
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


try:
    roc_auc = roc_auc_score(
        true_labels,
        probabilities,
        multi_class="ovr",
        average="macro",
    )
except Exception:
    roc_auc = np.nan


try:
    y_onehot = tf.keras.utils.to_categorical(
        true_labels,
        num_classes=NUM_CLASSES,
    )

    pr_auc = average_precision_score(
        y_onehot,
        probabilities,
        average="macro",
    )
except Exception:
    pr_auc = np.nan


# ============================================================
# SAVE RESULT
# ============================================================

result_df = pd.DataFrame([{
    "fold": 3,
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

result_df.to_csv(
    result_path,
    index=False,
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = os.path.join(
    RESULT_DIR,
    "confusion_matrix_fold_3_fixed.csv",
)

pd.DataFrame(
    cm,
    index=classes,
    columns=classes,
).to_csv(cm_path)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = fold_val[
    ["filepath", "label"]
].copy()

prediction_df["true_index"] = true_labels

prediction_df["predicted_index"] = predictions

prediction_df["predicted_label"] = [
    index_to_class[i]
    for i in predictions
]

prediction_df["confidence"] = (
    np.max(
        probabilities,
        axis=1
    )
)

prediction_df.to_csv(
    prediction_path,
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("FIXED FOLD 3 COMPLETED")
print("=" * 70)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1       :", f1)
print("ROC-AUC  :", roc_auc)
print("PR-AUC   :", pr_auc)

print()
print("Model:", checkpoint_path)
print("History:", history_path)
print("Results:", result_path)
print("Predictions:", prediction_path)
