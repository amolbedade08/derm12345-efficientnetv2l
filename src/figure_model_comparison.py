import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L"
)

TABLE_PATH = os.path.join(
    OUTPUT_ROOT,
    "paper_tables",
    "table2_three_fold_cv_comparison.csv"
)

FIGURE_DIR = os.path.join(
    OUTPUT_ROOT,
    "paper_figures"
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)

# ============================================================
# LOAD TABLE
# ============================================================

df = pd.read_csv(TABLE_PATH)

selected_metrics = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "ROC-AUC",
    "PR-AUC"
]

df = df[
    df["Metric"].isin(selected_metrics)
].copy()

# Preserve intended order
df["order"] = df["Metric"].map(
    {
        metric: i
        for i, metric in enumerate(
            selected_metrics
        )
    }
)

df = df.sort_values(
    "order"
)

# ============================================================
# VALUES
# ============================================================

metrics = df["Metric"].tolist()

efficientnet_mean = (
    df["EfficientNetV2L_mean"].values * 100
)

efficientnet_sd = (
    df["EfficientNetV2L_SD"].values * 100
)

baseline_mean = (
    df["LogisticRegression_mean"].values * 100
)

baseline_sd = (
    df["LogisticRegression_SD"].values * 100
)

x = np.arange(
    len(metrics)
)

width = 0.36

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 7)
)

bars1 = ax.bar(
    x - width / 2,
    efficientnet_mean,
    width,
    yerr=efficientnet_sd,
    capsize=4,
    label="EfficientNetV2-L"
)

bars2 = ax.bar(
    x + width / 2,
    baseline_mean,
    width,
    yerr=baseline_sd,
    capsize=4,
    label="Logistic Regression"
)

ax.set_ylabel(
    "Score (%)"
)

ax.set_xlabel(
    "Performance Metric"
)

ax.set_title(
    "Three-Fold Cross-Validation Performance Comparison"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    metrics,
    rotation=20
)

ax.set_ylim(
    0,
    100
)

ax.legend()

# ============================================================
# VALUE LABELS
# ============================================================

for i in range(len(metrics)):

    ax.text(
        x[i] - width / 2,
        efficientnet_mean[i] + efficientnet_sd[i] + 1,
        f"{efficientnet_mean[i]:.1f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

    ax.text(
        x[i] + width / 2,
        baseline_mean[i] + baseline_sd[i] + 1,
        f"{baseline_mean[i]:.1f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()

# ============================================================
# SAVE FIGURE
# ============================================================

output_path = os.path.join(
    FIGURE_DIR,
    "Figure9_Model_vs_Baseline_Comparison.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# SAVE DATA
# ============================================================

comparison_df = pd.DataFrame(
    {
        "Metric": metrics,

        "EfficientNetV2L_mean_percent":
            efficientnet_mean,

        "EfficientNetV2L_SD_percent":
            efficientnet_sd,

        "LogisticRegression_mean_percent":
            baseline_mean,

        "LogisticRegression_SD_percent":
            baseline_sd
    }
)

comparison_df.to_csv(
    os.path.join(
        FIGURE_DIR,
        "Figure9_Model_Comparison_Data.csv"
    ),
    index=False
)

print("=" * 70)
print("FIGURE 9 UPDATED")
print("=" * 70)

print(
    "Mean ± SD error bars added."
)

print(
    "Output:",
    output_path
)

print(
    "Data:",
    os.path.join(
        FIGURE_DIR,
        "Figure9_Model_Comparison_Data.csv"
    )
)