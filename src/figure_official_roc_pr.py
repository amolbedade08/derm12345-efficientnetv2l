import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

PRED_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "predictions",
    "official_test_predictions.csv"
)

MAPPING_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "csv_results",
    "final_class_mapping.csv"
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


# ============================================================
# LOAD PREDICTIONS
# ============================================================

pred = pd.read_csv(PRED_PATH)

print("Prediction rows:", len(pred))
print("Columns:")
print(pred.columns.tolist())


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

mapping = pd.read_csv(MAPPING_PATH).sort_values(
    "class_index"
)

class_names = (
    mapping["label"]
    .astype(str)
    .tolist()
)

n_classes = len(class_names)

print("Number of classes:", n_classes)


# ============================================================
# FIND TRUE / PREDICTED COLUMNS
# ============================================================

if "true_index" not in pred.columns:
    raise RuntimeError(
        "true_index not found in official predictions."
    )

true = pred["true_index"].astype(int).values


# ============================================================
# FIND PROBABILITY COLUMNS
# ============================================================

prob_cols = [
    c for c in pred.columns
    if c.startswith("prob_")
]

if len(prob_cols) != n_classes:
    raise RuntimeError(
        f"Expected {n_classes} probability columns, "
        f"found {len(prob_cols)}."
    )

probs = pred[prob_cols].values

print("Probability shape:", probs.shape)


# ============================================================
# ONE-VS-REST BINARIZATION
# ============================================================

y_true = label_binarize(
    true,
    classes=np.arange(n_classes)
)


# ============================================================
# CREATE FIGURE — ROC
# ============================================================

plt.figure(figsize=(10, 8))

auc_values = []

for i, class_name in enumerate(class_names):

    # Skip classes without positive or negative examples.
    if (
        y_true[:, i].sum() == 0
        or
        y_true[:, i].sum() == len(y_true)
    ):
        continue

    fpr, tpr, _ = roc_curve(
        y_true[:, i],
        probs[:, i]
    )

    auc_value = roc_auc_score(
        y_true[:, i],
        probs[:, i]
    )

    auc_values.append(auc_value)

    plt.plot(
        fpr,
        tpr,
        linewidth=1.0,
        label=f"{class_name} ({auc_value:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)

macro_auc = roc_auc_score(
    y_true,
    probs,
    average="macro",
    multi_class="ovr"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    f"Official Derm12345 Test ROC Curves\n"
    f"Macro-AUC = {macro_auc:.3f}"
)

plt.legend(
    fontsize=7,
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

roc_path = os.path.join(
    OUTPUT_DIR,
    "Figure5_Official_Test_ROC.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CREATE FIGURE — PR
# ============================================================

plt.figure(figsize=(10, 8))

ap_values = []

for i, class_name in enumerate(class_names):

    if y_true[:, i].sum() == 0:
        continue

    precision, recall, _ = precision_recall_curve(
        y_true[:, i],
        probs[:, i]
    )

    ap_value = average_precision_score(
        y_true[:, i],
        probs[:, i]
    )

    ap_values.append(ap_value)

    plt.plot(
        recall,
        precision,
        linewidth=1.0,
        label=f"{class_name} ({ap_value:.3f})"
    )

macro_ap = average_precision_score(
    y_true,
    probs,
    average="macro"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    f"Official Derm12345 Test Precision-Recall Curves\n"
    f"Macro-AP = {macro_ap:.3f}"
)

plt.legend(
    fontsize=7,
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

pr_path = os.path.join(
    OUTPUT_DIR,
    "Figure5_Official_Test_PR.png"
)

plt.savefig(
    pr_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "metric": [
            "macro_roc_auc",
            "macro_pr_auc"
        ],
        "value": [
            macro_auc,
            macro_ap
        ]
    }
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Figure5_ROC_PR_summary.csv"
    ),
    index=False
)


print("=" * 70)
print("FIGURE 5 GENERATED")
print("=" * 70)

print("ROC:", roc_path)
print("PR :", pr_path)
print("Macro ROC-AUC:", macro_auc)
print("Macro PR-AUC :", macro_ap)
