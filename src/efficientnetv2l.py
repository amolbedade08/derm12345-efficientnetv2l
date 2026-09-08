import os
import random
import json
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
    average_precision_score
)

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_CSV = os.path.join(
    PROJECT_DIR,
    "outputs",
    "EfficientNetV2L",
    "csv_results",
    "train_loaded.csv"
)

TEST_CSV = os.path.join(
    PROJECT_DIR,
    "outputs",
    "EfficientNetV2L",
    "csv_results",
    "test_loaded.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "EfficientNetV2L"
)

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULT_DIR = os.path.join(OUTPUT_DIR, "csv_results")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

for folder in [MODEL_DIR, PLOT_DIR, RESULT_DIR, PRED_DIR]:
    os.makedirs(folder, exist_ok=True)


IMAGE_SIZE = (224, 224)

# Start safely on RTX 3050 4 GB
BATCH_SIZE = 2

EPOCHS = 20

N_SPLITS = 3

RANDOM_SEED = 42

DROPOUT_RATE = 0.3

L2_REG = 1e-4

LEARNING_RATE = 1e-4


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# 3. GPU SETUP
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

print("=" * 70)
print("EFFICIENTNETV2-L PRODUCTION TRAINING")
print("=" * 70)

print("TensorFlow:", tf.__version__)
print("GPU:", gpus)

if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


# Mixed precision
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision: ENABLED")
except Exception as e:
    print("Mixed precision unavailable:", e)


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\nLoading training data...")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Training samples:", len(train_df))
print("Test samples:", len(test_df))


# ============================================================
# 5. LABEL ENCODING
# ============================================================

classes = sorted(train_df["label"].unique())

class_to_index = {
    cls: idx for idx, cls in enumerate(classes)
}

index_to_class = {
    idx: cls for cls, idx in class_to_index.items()
}

NUM_CLASSES = len(classes)

train_df["class_index"] = train_df["label"].map(class_to_index)
test_df["class_index"] = test_df["label"].map(class_to_index)

train_df["class_index_str"] = train_df["class_index"].astype(str)
test_df["class_index_str"] = test_df["class_index"].astype(str)

print("Number of classes:", NUM_CLASSES)


# ============================================================
# 6. CHECK DATA
# ============================================================

if train_df["filepath"].isna().any():
    raise ValueError("Training data contains missing filepaths.")

if test_df["filepath"].isna().any():
    raise ValueError("Test data contains missing filepaths.")

if not train_df["filepath"].map(os.path.exists).all():
    raise ValueError("Some training images do not exist.")

if not test_df["filepath"].map(os.path.exists).all():
    raise ValueError("Some test images do not exist.")


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.10,
    shear_range=0.05,
    horizontal_flip=True,
    brightness_range=(0.85, 1.15)
)

val_datagen = ImageDataGenerator()


# ============================================================
# 8. MODEL CREATION
# ============================================================

def build_model(num_classes):

    inputs = layers.Input(
        shape=(224, 224, 3)
    )

    base_model = EfficientNetV2L(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
        include_preprocessing=True
    )

    # Freeze backbone initially
    base_model.trainable = False

    x = base_model(
        inputs,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(
        DROPOUT_RATE
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizers.l2(L2_REG),
        dtype="float32"
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
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 9. CLASS WEIGHTS
# ============================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["class_index"]
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

print("\nClass weights calculated.")


# ============================================================
# 10. TEST GENERATOR
# ============================================================

test_generator = val_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col="class_index_str",
    classes=[str(i) for i in range(NUM_CLASSES)],
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    class_mode="sparse",
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 11. METRICS FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES)
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    specificity_values = []
    fpr_values = []
    iou_values = []

    for i in range(NUM_CLASSES):

        tp = cm[i, i]

        fn = cm[i, :].sum() - tp

        fp = cm[:, i].sum() - tp

        tn = cm.sum() - (
            tp + fn + fp
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0 else 0
        )

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0 else 0
        )

        iou = (
            tp / (tp + fp + fn)
            if (tp + fp + fn) > 0 else 0
        )

        specificity_values.append(
            specificity
        )

        fpr_values.append(
            fpr
        )

        iou_values.append(
            iou
        )

    specificity = np.mean(
        specificity_values
    )

    fpr = np.mean(
        fpr_values
    )

    iou = np.mean(
        iou_values
    )

    ppv = precision

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro"
        )
    except Exception:
        roc_auc = np.nan

    # PR-AUC
    try:
        y_true_onehot = tf.keras.utils.to_categorical(
            y_true,
            num_classes=NUM_CLASSES
        )

        pr_auc = average_precision_score(
            y_true_onehot,
            y_prob,
            average="macro"
        )
    except Exception:
        pr_auc = np.nan

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "iou": iou,
        "ppv": ppv,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }, cm


# ============================================================
# 12. 3-FOLD STRATIFIED CROSS VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED
)

fold_results = []


for fold, (train_idx, val_idx) in enumerate(
    skf.split(
        train_df,
        train_df["class_index"]
    ),
    start=1
):

    print("\n")
    print("=" * 70)
    print(f"STARTING FOLD {fold}/{N_SPLITS}")
    print("=" * 70)

    fold_train = train_df.iloc[train_idx].copy()

    fold_val = train_df.iloc[val_idx].copy()

    print(
        "Training samples:",
        len(fold_train)
    )

    print(
        "Validation samples:",
        len(fold_val)
    )


    # --------------------------------------------------------
    # Generators
    # --------------------------------------------------------

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=fold_train,
        x_col="filepath",
        y_col="class_index_str",        
        classes=[str(i) for i in range(NUM_CLASSES)],
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        class_mode="sparse",
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED + fold
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=fold_val,
        x_col="filepath",
        y_col="class_index_str",
        classes=[str(i) for i in range(NUM_CLASSES)],
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        class_mode="sparse",
        batch_size=BATCH_SIZE,
        shuffle=False
    )
 

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model(
        NUM_CLASSES
    )


    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    checkpoint_path = os.path.join(
        MODEL_DIR,
        f"efficientnetv2l_fold_{fold}.keras"
    )

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            patience=2,
            factor=0.5,
            min_lr=1e-6,
            verbose=1
        ),

        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )


    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history.history
    )

    history_path = os.path.join(
        RESULT_DIR,
        f"history_fold_{fold}.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )


    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    val_generator.reset()

    val_prob = model.predict(
        val_generator,
        verbose=1
    )

    val_pred = np.argmax(
        val_prob,
        axis=1
    )

    val_true = fold_val["class_index"].values


    metrics, cm = calculate_metrics(
        val_true,
        val_pred,
        val_prob
    )


    metrics["fold"] = fold

    fold_results.append(
        metrics
    )


    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    cm_path = os.path.join(
        RESULT_DIR,
        f"confusion_matrix_fold_{fold}.csv"
    )

    cm_df.to_csv(
        cm_path
    )


    # --------------------------------------------------------
    # Save validation predictions
    # --------------------------------------------------------

    pred_df = fold_val[
        ["filepath", "label"]
    ].copy()

    pred_df["true_index"] = val_true

    pred_df["predicted_index"] = val_pred

    pred_df["predicted_label"] = [
        index_to_class[x]
        for x in val_pred
    ]

    pred_df["confidence"] = np.max(
        val_prob,
        axis=1
    )

    pred_path = os.path.join(
        PRED_DIR,
        f"validation_predictions_fold_{fold}.csv"
    )

    pred_df.to_csv(
        pred_path,
        index=False
    )


    print("\nFold metrics:")

    for key, value in metrics.items():

        if key != "fold":

            print(
                f"{key:25s}: {value:.4f}"
            )


    # Clear model from memory
    del model

    tf.keras.backend.clear_session()


# ============================================================
# 13. CROSS-VALIDATION SUMMARY
# ============================================================

results_df = pd.DataFrame(
    fold_results
)

results_path = os.path.join(
    RESULT_DIR,
    "cross_validation_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


metric_columns = [
    "accuracy",
    "precision",
    "recall_sensitivity",
    "f1",
    "specificity",
    "fpr",
    "iou",
    "ppv",
    "roc_auc",
    "pr_auc"
]


summary_rows = []

for metric in metric_columns:

    summary_rows.append({

        "metric": metric,

        "mean": results_df[metric].mean(),

        "std": results_df[metric].std(),

        "min": results_df[metric].min(),

        "max": results_df[metric].max()
    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_path = os.path.join(
    RESULT_DIR,
    "cross_validation_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 14. PRINT FINAL CV SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)

for _, row in summary_df.iterrows():

    print(
        f"{row['metric']:25s} "
        f"{row['mean']:.4f} ± {row['std']:.4f}"
    )


# ============================================================
# 15. SAVE CLASS MAPPING
# ============================================================

mapping_path = os.path.join(
    RESULT_DIR,
    "class_mapping.json"
)

with open(
    mapping_path,
    "w"
) as f:

    json.dump(
        class_to_index,
        f,
        indent=4
    )


print("\n")
print("=" * 70)
print("CROSS-VALIDATION FINISHED")
print("=" * 70)

print("Results saved to:")
print(RESULT_DIR)

print("\nIMPORTANT:")
print("Official test set has NOT been used for model selection.")