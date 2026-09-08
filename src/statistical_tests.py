import os
import numpy as np
import pandas as pd

from scipy.stats import (
    shapiro,
    wilcoxon,
    ttest_rel
)

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CV_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "cv"
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
# LOAD FOLD RESULTS
# ============================================================

cv_file = os.path.join(
    CV_DIR,
    "fold_results.csv"
)

if not os.path.exists(cv_file):

    raise FileNotFoundError(
        f"CV results not found:\n{cv_file}"
    )

df = pd.read_csv(
    cv_file
)

print("=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

print("\nFold results:")
print(df)


# ============================================================
# NUMERIC METRICS
# ============================================================

metrics = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "specificity_macro",
    "fpr_macro",
    "iou_macro",
    "roc_auc_macro_ovr",
    "pr_auc_macro"
]


# ============================================================
# MEAN ± SD + 95% CI
# ============================================================

results = []

for metric in metrics:

    if metric not in df.columns:
        continue

    values = (
        df[metric]
        .dropna()
        .values
    )

    n = len(values)

    mean = np.mean(values)

    sd = (
        np.std(
            values,
            ddof=1
        )
        if n > 1
        else 0
    )

    # t-based 95% CI
    if n > 1:

        from scipy.stats import t

        critical = t.ppf(
            0.975,
            df=n - 1
        )

        margin = (
            critical *
            sd /
            np.sqrt(n)
        )

    else:

        margin = np.nan

    results.append({

        "metric": metric,

        "mean": mean,

        "sd": sd,

        "lower_95CI": (
            mean - margin
            if not np.isnan(margin)
            else np.nan
        ),

        "upper_95CI": (
            mean + margin
            if not np.isnan(margin)
            else np.nan
        )

    })


summary = pd.DataFrame(
    results
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "mean_sd_95CI.csv"
    ),
    index=False
)

print("\nMean ± SD + 95% CI:")
print(summary)


# ============================================================
# SHAPIRO-WILK
# ============================================================

shapiro_results = []

for metric in metrics:

    if metric not in df.columns:
        continue

    values = (
        df[metric]
        .dropna()
        .values
    )

    if len(values) >= 3:

        statistic, p_value = (
            shapiro(values)
        )

        shapiro_results.append({

            "metric": metric,

            "W": statistic,

            "p_value": p_value

        })


shapiro_df = pd.DataFrame(
    shapiro_results
)

shapiro_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "shapiro_wilk.csv"
    ),
    index=False
)


# ============================================================
# MODEL COMPARISON
# EfficientNetV2-L vs Logistic Regression
# ============================================================

LR_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L",
    "feature_classifier",
    "fold_results.csv"
)

if os.path.exists(LR_FILE):

    lr = pd.read_csv(
        LR_FILE
    )

    print(
        "\nLogistic Regression baseline found."
    )

    comparison = []

    for metric in [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_macro_ovr",
        "pr_auc_macro"
    ]:

        if (
            metric not in df.columns
            or metric not in lr.columns
        ):
            continue

        a = (
            df[metric]
            .dropna()
            .values
        )

        b = (
            lr[metric]
            .dropna()
            .values
        )

        n = min(
            len(a),
            len(b)
        )

        a = a[:n]
        b = b[:n]

        difference = a - b

        # ----------------------------------------------------
        # Wilcoxon
        # ----------------------------------------------------

        try:

            w_stat, w_p = wilcoxon(
                difference
            )

        except Exception:

            w_stat = np.nan
            w_p = np.nan

        # ----------------------------------------------------
        # Paired t-test
        # ----------------------------------------------------

        try:

            t_stat, t_p = ttest_rel(
                a,
                b
            )

        except Exception:

            t_stat = np.nan
            t_p = np.nan

        comparison.append({

            "metric": metric,

            "EfficientNetV2L_mean":
                np.mean(a),

            "LR_mean":
                np.mean(b),

            "mean_difference":
                np.mean(difference),

            "wilcoxon_statistic":
                w_stat,

            "wilcoxon_p":
                w_p,

            "paired_t_statistic":
                t_stat,

            "paired_t_p":
                t_p

        })

    comparison_df = pd.DataFrame(
        comparison
    )

    comparison_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "EfficientNetV2L_vs_LR_statistics.csv"
        ),
        index=False
    )

    print(
        "\nEfficientNetV2-L vs LR:"
    )

    print(
        comparison_df
    )

else:

    print(
        "\nLR baseline file not found."
    )


# ============================================================
# CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nFiles saved in:"
)

print(
    OUTPUT_DIR
)