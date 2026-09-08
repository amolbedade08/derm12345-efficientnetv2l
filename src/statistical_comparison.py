import os
import numpy as np
import pandas as pd

from scipy.stats import (
    shapiro,
    ttest_rel,
    wilcoxon
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

EFF_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "csv_results",
    "corrected_fold_results.csv"
)

BASE_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "feature_classifier",
    "fold_results.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "statistics"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# LOAD RESULTS
# ============================================================

eff = pd.read_csv(EFF_PATH)
base = pd.read_csv(BASE_PATH)

# Sort by fold to guarantee pairing
eff = eff.sort_values("fold").reset_index(drop=True)
base = base.sort_values("fold").reset_index(drop=True)

if not np.array_equal(
    eff["fold"].values,
    base["fold"].values
):
    raise RuntimeError(
        "Fold numbers do not match between models."
    )

# ============================================================
# METRICS
# ============================================================

metric_pairs = {
    "accuracy": (
        "accuracy",
        "accuracy"
    ),
    "precision_macro": (
        "precision_macro",
        "precision_macro"
    ),
    "recall_macro": (
        "recall_macro",
        "recall_macro_sensitivity"
    ),
    "f1_macro": (
        "f1_macro",
        "f1_macro"
    ),
    "specificity_macro": (
        "specificity_macro",
        "specificity_macro"
    ),
    "fpr_macro": (
        "fpr_macro",
        "fpr_macro"
    ),
    "iou_macro": (
        "iou_macro",
        "iou_macro"
    ),
    "roc_auc_macro_ovr": (
        "roc_auc_macro_ovr",
        "roc_auc_macro_ovr"
    ),
    "pr_auc_macro": (
        "pr_auc_macro",
        "pr_auc_macro"
    )
}

# ============================================================
# MAIN STATISTICAL TABLE
# ============================================================

rows = []

for metric_name, (eff_col, base_col) in metric_pairs.items():

    eff_values = eff[eff_col].astype(float).values
    base_values = base[base_col].astype(float).values

    differences = eff_values - base_values

    # --------------------------------------------------------
    # Shapiro-Wilk on paired differences
    # --------------------------------------------------------

    shapiro_stat, shapiro_p = shapiro(
        differences
    )

    # --------------------------------------------------------
    # Paired t-test
    # --------------------------------------------------------

    t_stat, t_p = ttest_rel(
        eff_values,
        base_values
    )

    # --------------------------------------------------------
    # Wilcoxon signed-rank
    # --------------------------------------------------------

    try:

        w_stat, w_p = wilcoxon(
            eff_values,
            base_values,
            alternative="two-sided"
        )

    except ValueError:

        w_stat = np.nan
        w_p = np.nan

    # --------------------------------------------------------
    # Effect direction
    # --------------------------------------------------------

    mean_eff = np.mean(eff_values)
    mean_base = np.mean(base_values)
    mean_diff = np.mean(differences)

    if mean_diff > 0:
        direction = "EfficientNetV2-L higher"
    elif mean_diff < 0:
        direction = "Baseline higher"
    else:
        direction = "Equal"

    # --------------------------------------------------------
    # Statistical significance
    # --------------------------------------------------------

    rows.append(
        {
            "metric": metric_name,
            "n_folds": len(eff_values),

            "efficientnet_mean": mean_eff,
            "baseline_mean": mean_base,
            "mean_difference": mean_diff,

            "efficientnet_std": np.std(
                eff_values,
                ddof=1
            ),
            "baseline_std": np.std(
                base_values,
                ddof=1
            ),

            "shapiro_statistic": shapiro_stat,
            "shapiro_p_value": shapiro_p,

            "paired_t_statistic": t_stat,
            "paired_t_p_value": t_p,

            "wilcoxon_statistic": w_stat,
            "wilcoxon_p_value": w_p,

            "direction": direction,

            "t_test_significant_alpha_0_05": (
                t_p < 0.05
            ),

            "wilcoxon_significant_alpha_0_05": (
                w_p < 0.05
                if not np.isnan(w_p)
                else False
            )
        }
    )

stats_df = pd.DataFrame(rows)

# ============================================================
# SAVE MAIN STATISTICAL RESULTS
# ============================================================

stats_path = os.path.join(
    OUTPUT_DIR,
    "statistical_comparison.csv"
)

stats_df.to_csv(
    stats_path,
    index=False
)

# ============================================================
# SAVE SHAPIRO RESULTS
# ============================================================

shapiro_df = stats_df[
    [
        "metric",
        "n_folds",
        "shapiro_statistic",
        "shapiro_p_value"
    ]
].copy()

shapiro_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shapiro_results.csv"
    ),
    index=False
)

# ============================================================
# SAVE PAIRED T-TEST RESULTS
# ============================================================

ttest_df = stats_df[
    [
        "metric",
        "n_folds",
        "efficientnet_mean",
        "baseline_mean",
        "mean_difference",
        "paired_t_statistic",
        "paired_t_p_value",
        "t_test_significant_alpha_0_05"
    ]
].copy()

ttest_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "paired_ttest.csv"
    ),
    index=False
)

# ============================================================
# SAVE WILCOXON RESULTS
# ============================================================

wilcoxon_df = stats_df[
    [
        "metric",
        "n_folds",
        "mean_difference",
        "wilcoxon_statistic",
        "wilcoxon_p_value",
        "wilcoxon_significant_alpha_0_05"
    ]
].copy()

wilcoxon_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "wilcoxon_results.csv"
    ),
    index=False
)

# ============================================================
# SAVE FOLD-LEVEL PAIRED DATA
# ============================================================

paired_rows = []

for fold in eff["fold"]:

    eff_row = eff[
        eff["fold"] == fold
    ].iloc[0]

    base_row = base[
        base["fold"] == fold
    ].iloc[0]

    paired_rows.append(
        {
            "fold": int(fold),

            "efficientnet_accuracy":
                float(eff_row["accuracy"]),
            "baseline_accuracy":
                float(base_row["accuracy"]),

            "efficientnet_precision":
                float(eff_row["precision_macro"]),
            "baseline_precision":
                float(base_row["precision_macro"]),

            "efficientnet_recall":
                float(eff_row["recall_macro"]),
            "baseline_recall":
                float(base_row[
                    "recall_macro_sensitivity"
                ]),

            "efficientnet_f1":
                float(eff_row["f1_macro"]),
            "baseline_f1":
                float(base_row["f1_macro"]),

            "efficientnet_specificity":
                float(eff_row["specificity_macro"]),
            "baseline_specificity":
                float(base_row["specificity_macro"]),

            "efficientnet_iou":
                float(eff_row["iou_macro"]),
            "baseline_iou":
                float(base_row["iou_macro"]),

            "efficientnet_roc_auc":
                float(eff_row["roc_auc_macro_ovr"]),
            "baseline_roc_auc":
                float(base_row["roc_auc_macro_ovr"]),

            "efficientnet_pr_auc":
                float(eff_row["pr_auc_macro"]),
            "baseline_pr_auc":
                float(base_row["pr_auc_macro"])
        }
    )

paired_df = pd.DataFrame(
    paired_rows
)

paired_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "paired_fold_results.csv"
    ),
    index=False
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 80)
print("STATISTICAL COMPARISON")
print("=" * 80)

for _, row in stats_df.iterrows():

    print(
        f"\nMetric: {row['metric']}"
    )

    print(
        f"  EfficientNetV2-L mean: "
        f"{row['efficientnet_mean']:.6f}"
    )

    print(
        f"  Baseline mean: "
        f"{row['baseline_mean']:.6f}"
    )

    print(
        f"  Mean difference: "
        f"{row['mean_difference']:.6f}"
    )

    print(
        f"  Shapiro p: "
        f"{row['shapiro_p_value']:.6f}"
    )

    print(
        f"  Paired t-test p: "
        f"{row['paired_t_p_value']:.6f}"
    )

    print(
        f"  Wilcoxon p: "
        f"{row['wilcoxon_p_value']:.6f}"
    )

    print(
        f"  Direction: "
        f"{row['direction']}"
    )

print("\nSaved files:")
print(
    " - statistical_comparison.csv"
)
print(
    " - shapiro_results.csv"
)
print(
    " - paired_ttest.csv"
)
print(
    " - wilcoxon_results.csv"
)
print(
    " - paired_fold_results.csv"
)

print(
    "\nIMPORTANT: Statistical inference is exploratory "
    "because only 3 folds are available."
)

