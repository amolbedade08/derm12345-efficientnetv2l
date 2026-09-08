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

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

PRED_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "external_PAD_UFES20",
    "predictions.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "paper_figures"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD EXTERNAL PREDICTIONS
# ============================================================

df = pd.read_csv(PRED_PATH)

print("Rows:", len(df))

classes = ["bcc", "mel", "scc", "sk"]

prob_cols = [
    f"prob_{c}"
    for c in classes
]

for col in [
    "true_class_index",
    *prob_cols
]:
    if col not in df.columns:
        raise RuntimeError(
            f"Missing column: {col}"
        )

y_true = df[
    "true_class_index"
].astype(int).values

probs = df[
    prob_cols
].values

print(
    "Probability shape:",
    probs.shape
)

# ============================================================
# ONE-VS-REST ROC
# ============================================================

plt.figure(figsize=(9, 8))

auc_values = []

for i, class_name in enumerate(classes):

    y_binary = (
        y_true == i
    ).astype(int)

    fpr, tpr, _ = roc_curve(
        y_binary,
        probs[:, i]
    )

    auc_value = roc_auc_score(
        y_binary,
        probs[:, i]
    )

    auc_values.append(
        auc_value
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=1.5,
        label=f"{class_name.upper()} (AUC={auc_value:.3f})"
    )

macro_auc = roc_auc_score(
    np.eye(len(classes))[y_true],
    probs,
    average="macro",
    multi_class="ovr"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    f"PAD-UFES-20 External Validation ROC\n"
    f"Macro-AUC = {macro_auc:.3f}"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

roc_path = os.path.join(
    OUTPUT_DIR,
    "Figure7_PAD_UFES20_External_ROC.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# ONE-VS-REST PR
# ============================================================

plt.figure(figsize=(9, 8))

ap_values = []

for i, class_name in enumerate(classes):

    y_binary = (
        y_true == i
    ).astype(int)

    precision, recall, _ = (
        precision_recall_curve(
            y_binary,
            probs[:, i]
        )
    )

    ap_value = average_precision_score(
        y_binary,
        probs[:, i]
    )

    ap_values.append(
        ap_value
    )

    plt.plot(
        recall,
        precision,
        linewidth=1.5,
        label=f"{class_name.upper()} (AP={ap_value:.3f})"
    )

macro_ap = average_precision_score(
    np.eye(len(classes))[y_true],
    probs,
    average="macro"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    f"PAD-UFES-20 External Validation "
    f"Precision-Recall Curves\n"
    f"Macro-AP = {macro_ap:.3f}"
)

plt.legend(
    loc="upper right"
)

plt.tight_layout()

pr_path = os.path.join(
    OUTPUT_DIR,
    "Figure7_PAD_UFES20_External_PR.png"
)

plt.savefig(
    pr_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "class": [c.upper() for c in classes],
        "roc_auc": auc_values,
        "average_precision": ap_values
    }
)

summary.loc[len(summary)] = [
    "MACRO",
    macro_auc,
    macro_ap
]

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Figure7_external_ROC_PR_summary.csv"
    ),
    index=False
)

print("=" * 70)
print("FIGURE 7 GENERATED")
print("=" * 70)

print("ROC:", roc_path)
print("PR :", pr_path)
print("Macro ROC-AUC:", macro_auc)
print("Macro PR-AUC :", macro_ap)
