import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

CSV_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "csv_results"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "paper_figures"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ------------------------------------------------------------
# HISTORY FILES
# ------------------------------------------------------------

history_files = {
    "Fold 1": "history_fold_1.csv",
    "Fold 2": "history_fold_2.csv",
    "Fold 3": "history_fold_3_fixed.csv",
}

histories = {}

for name, filename in history_files.items():

    path = os.path.join(
        CSV_DIR,
        filename
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing history file: {path}"
        )

    df = pd.read_csv(path)

    print(
        f"{name}: {len(df)} epochs"
    )

    print(
        "Columns:",
        df.columns.tolist()
    )

    histories[name] = df


# ============================================================
# ACCURACY CURVE
# ============================================================

plt.figure(figsize=(11, 7))

for name, df in histories.items():

    epoch_col = (
        "epoch"
        if "epoch" in df.columns
        else df.columns[0]
    )

    acc_col = None
    val_acc_col = None

    for col in df.columns:

        lower = col.lower()

        if lower == "accuracy":
            acc_col = col

        if lower in [
            "val_accuracy",
            "val_acc"
        ]:
            val_acc_col = col

    if acc_col is not None:
        plt.plot(
            df[epoch_col] + 1,
            df[acc_col],
            label=f"{name} Train"
        )

    if val_acc_col is not None:
        plt.plot(
            df[epoch_col] + 1,
            df[val_acc_col],
            linestyle="--",
            label=f"{name} Validation"
        )

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "EfficientNetV2-L Training and Validation Accuracy"
)

plt.legend()

plt.tight_layout()

accuracy_path = os.path.join(
    OUTPUT_DIR,
    "Figure3_Training_Validation_Accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(figsize=(11, 7))

for name, df in histories.items():

    epoch_col = (
        "epoch"
        if "epoch" in df.columns
        else df.columns[0]
    )

    loss_col = None
    val_loss_col = None

    for col in df.columns:

        lower = col.lower()

        if lower == "loss":
            loss_col = col

        if lower == "val_loss":
            val_loss_col = col

    if loss_col is not None:
        plt.plot(
            df[epoch_col] + 1,
            df[loss_col],
            label=f"{name} Train"
        )

    if val_loss_col is not None:
        plt.plot(
            df[epoch_col] + 1,
            df[val_loss_col],
            linestyle="--",
            label=f"{name} Validation"
        )

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "EfficientNetV2-L Training and Validation Loss"
)

plt.legend()

plt.tight_layout()

loss_path = os.path.join(
    OUTPUT_DIR,
    "Figure3_Training_Validation_Loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("FIGURE 3 GENERATED")
print("=" * 70)

print(
    "Accuracy:",
    accuracy_path
)

print(
    "Loss:",
    loss_path
)
