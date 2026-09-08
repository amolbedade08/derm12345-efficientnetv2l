import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = "/mnt/c/Users/0M SAI RAM/Desktop/project"

FEATURE_FILE = os.path.join(
    PROJECT_DIR,
    "outputs/EfficientNetV2L/features/train_features.npy"
)

LABEL_FILE = os.path.join(
    PROJECT_DIR,
    "outputs/EfficientNetV2L/features/train_feature_labels.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs/EfficientNetV2L/mlp_classifier"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_SIZE = 1280
N_SPLITS = 3
EPOCHS = 30
BATCH_SIZE = 32
RANDOM_SEED = 42

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================
# START
# ============================================================

print("=" * 70)
print("EFFICIENTNETV2-L MLP FEATURE CLASSIFIER")
print("=" * 70)

print("TensorFlow:", tf.__version__)

# ============================================================
# LOAD LABELS
# ============================================================

if not os.path.exists(FEATURE_FILE):
    raise FileNotFoundError(FEATURE_FILE)

if not os.path.exists(LABEL_FILE):
    raise FileNotFoundError(LABEL_FILE)

print("\nLoading labels...")

labels_df = pd.read_csv(LABEL_FILE)

y = labels_df["class_index"].to_numpy(
    dtype=np.int64
)

num_samples = len(y)

classes = np.unique(y)
num_classes = len(classes)

print("Samples:", num_samples)
print("Classes:", num_classes)

# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading feature matrix...")

X = np.memmap(
    FEATURE_FILE,
    dtype="float32",
    mode="r",
    shape=(num_samples, FEATURE_SIZE)
)

print("Feature shape:", X.shape)

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_counts = (
    pd.Series(y)
    .value_counts()
    .sort_index()
)

print("\nMinimum class count:", class_counts.min())

# ============================================================
# MODEL
# ============================================================

def build_model():

    model = Sequential([
        Input(shape=(FEATURE_SIZE,)),

        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.40),

        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.30),

        Dense(128, activation="relu"),
        Dropout(0.20),

        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# CLASS WEIGHTS
# ============================================================

total = len(y)

class_weights = {}

for class_id, count in class_counts.items():

    class_weights[int(class_id)] = (
        total
        / (
            num_classes * count
        )
    )

print("\nClass weights created.")

# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):

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

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=classes
    )

    specificity_values = []
    fpr_values = []
    iou_values = []
    ppv_values = []

    for i in range(num_classes):

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

        ppv = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        specificity_values.append(specificity)
        fpr_values.append(fpr)
        iou_values.append(iou)
        ppv_values.append(ppv)

    specificity = np.mean(
        specificity_values
    )

    fpr = np.mean(
        fpr_values
    )

    iou = np.mean(
        iou_values
    )

    ppv = np.mean(
        ppv_values
    )

    # ROC-AUC
    try:

        roc_auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=classes
        )

    except Exception as e:

        print(
            "ROC-AUC warning:",
            e
        )

        roc_auc = np.nan

    # PR-AUC
    try:

        y_onehot = np.zeros(
            (
                len(y_true),
                num_classes
            ),
            dtype=np.float32
        )

        class_to_position = {
            class_id: pos
            for pos, class_id
            in enumerate(classes)
        }

        for row, class_id in enumerate(y_true):

            y_onehot[
                row,
                class_to_position[class_id]
            ] = 1.0

        pr_auc = average_precision_score(
            y_onehot,
            y_prob,
            average="macro"
        )

    except Exception as e:

        print(
            "PR-AUC warning:",
            e
        )

        pr_auc = np.nan

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "specificity_macro": specificity,
        "fpr_macro": fpr,
        "iou_macro": iou,
        "ppv_macro": ppv,
        "roc_auc_macro_ovr": roc_auc,
        "pr_auc_macro": pr_auc
    }


# ============================================================
# 3-FOLD STRATIFIED CV
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED
)

fold_results = []

print("\n")
print("=" * 70)
print("STARTING 3-FOLD MLP TRAINING")
print("=" * 70)

for fold, (train_idx, val_idx) in enumerate(
    skf.split(np.zeros(num_samples), y),
    start=1
):

    print("\n")
    print("-" * 70)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("-" * 70)

    X_train = np.asarray(
        X[train_idx],
        dtype=np.float32
    )

    X_val = np.asarray(
        X[val_idx],
        dtype=np.float32
    )

    y_train = y[train_idx]
    y_val = y[val_idx]

    print(
        "Train samples:",
        len(train_idx)
    )

    print(
        "Validation samples:",
        len(val_idx)
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_model()

    model.summary()

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    checkpoint_file = os.path.join(
        OUTPUT_DIR,
        f"best_model_fold_{fold}.keras"
    )

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_file,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nTraining MLP...")

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1
    )

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history.history
    )

    history_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"history_fold_{fold}.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    y_prob = model.predict(
        X_val,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    y_pred = np.argmax(
        y_prob,
        axis=1
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_val,
        y_pred,
        y_prob
    )

    metrics["fold"] = fold
    metrics["train_samples"] = len(train_idx)
    metrics["validation_samples"] = len(val_idx)

    fold_results.append(
        metrics
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_val,
        y_pred,
        labels=classes
    )

    pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"confusion_matrix_fold_{fold}.csv"
        )
    )

    # --------------------------------------------------------
    # PREDICTION CSV
    # --------------------------------------------------------

    pred_df = labels_df.iloc[
        val_idx
    ].copy()

    pred_df["true_class_index"] = y_val

    pred_df["predicted_class_index"] = y_pred

    pred_df["correct"] = (
        y_val == y_pred
    )

    pred_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"predictions_fold_{fold}.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\nFold results:")

    for key in [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "specificity_macro",
        "fpr_macro",
        "iou_macro",
        "ppv_macro",
        "roc_auc_macro_ovr",
        "pr_auc_macro"
    ]:

        value = metrics[key]

        if np.isnan(value):

            print(
                f"{key:30s}: NaN"
            )

        else:

            print(
                f"{key:30s}: {value:.4f}"
            )

    # Clear model
    del model
    del X_train
    del X_val

    tf.keras.backend.clear_session()


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    fold_results
)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "fold_results.csv"
    ),
    index=False
)


# ============================================================
# MEAN ± SD
# ============================================================

metric_columns = [

    "accuracy",

    "precision_macro",

    "recall_macro",

    "f1_macro",

    "specificity_macro",

    "fpr_macro",

    "iou_macro",

    "ppv_macro",

    "roc_auc_macro_ovr",

    "pr_auc_macro"
]

summary = []

for metric in metric_columns:

    values = pd.to_numeric(
        results_df[metric],
        errors="coerce"
    )

    summary.append({

        "metric": metric,

        "mean": values.mean(),

        "std": values.std(
            ddof=1
        ),

        "min": values.min(),

        "max": values.max()
    })


summary_df = pd.DataFrame(
    summary
)

summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cv_summary_mean_sd.csv"
    ),
    index=False
)


# ============================================================
# FINAL PRINT
# ============================================================

print("\n")
print("=" * 70)
print("MLP 3-FOLD CROSS-VALIDATION COMPLETE")
print("=" * 70)

print("\nMean ± SD:")

for _, row in summary_df.iterrows():

    print(
        f"{row['metric']:32s}"
        f"{row['mean']:.4f} ± "
        f"{row['std']:.4f}"
    )

print("\nOutput directory:")

print(
    OUTPUT_DIR
)

print("=" * 70)
