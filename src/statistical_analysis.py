#!/usr/bin/env python3
"""
Corrected statistical analysis for the EfficientNetV2-L cached-feature baseline.

Uses the EXISTING files produced by:
  src/train_feature_classifier.py

Reads:
  outputs/EfficientNetV2L/feature_classifier/fold_results.csv
  outputs/EfficientNetV2L/feature_classifier/aggregated_cv_predictions.csv

The aggregated prediction file in this project contains exactly:
  true_class_index
  predicted_class_index
  correct

So this script does NOT assume probability columns.

Writes:
  outputs/EfficientNetV2L/statistics/confidence_intervals.csv
  outputs/EfficientNetV2L/statistics/bootstrap_ci.csv
  outputs/EfficientNetV2L/statistics/statistical_summary.csv
  outputs/EfficientNetV2L/statistics/bootstrap_samples.csv
  outputs/EfficientNetV2L/statistics/config.json

Notes:
- Fold-level 95% CI: t interval across the 3 CV folds.
- Bootstrap 95% CI: percentile bootstrap over the 9,860 out-of-fold predictions.
- ROC-AUC and PR-AUC cannot be bootstrapped from the current aggregated file because
  class probabilities were not saved there. Their fold-level mean/SD/CI still come
  from fold_results.csv.
- With only 3 folds, Shapiro-Wilk is extremely underpowered and is reported only
  descriptively.
"""

from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd
from scipy.stats import t, shapiro
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


SEED = 42
N_BOOT = 2000
CI = 0.95
NUM_CLASSES = 40

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "outputs" / "EfficientNetV2L" / "feature_classifier"
OUT_DIR = PROJECT_ROOT / "outputs" / "EfficientNetV2L" / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_FILE = RESULT_DIR / "fold_results.csv"
PRED_FILE = RESULT_DIR / "aggregated_cv_predictions.csv"

# Exact fold-result columns produced by the existing classifier script.
METRIC_MAP = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro_sensitivity": "recall_macro_sensitivity",
    "f1_macro": "f1_macro",
    "specificity_macro": "specificity_macro",
    "fpr_macro": "fpr_macro",
    "iou_macro": "iou_macro",
    "ppv_macro": "ppv_macro",
    "roc_auc_macro_ovr": "roc_auc_macro_ovr",
    "pr_auc_macro": "pr_auc_macro",
}


def find_col(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def mean_t_ci(values, confidence=CI):
    """Mean, sample SD, and t-based CI across CV folds."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan

    mean = float(np.mean(x))

    if n < 2:
        return mean, np.nan, np.nan, np.nan

    sd = float(np.std(x, ddof=1))
    se = sd / np.sqrt(n)
    critical = t.ppf(
        1 - (1 - confidence) / 2,
        df=n - 1,
    )

    lower = mean - critical * se
    upper = mean + critical * se

    return mean, sd, float(lower), float(upper)


def multiclass_specificity_fpr_iou(
    y_true,
    y_pred,
    num_classes=NUM_CLASSES,
):
    """Macro one-vs-rest specificity, FPR and IoU."""
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
    )

    total = cm.sum()
    specificities = []
    fprs = []
    ious = []

    for k in range(num_classes):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp

        spec_den = tn + fp
        fpr_den = fp + tn
        iou_den = tp + fp + fn

        specificity = tn / spec_den if spec_den > 0 else 0.0
        fpr = fp / fpr_den if fpr_den > 0 else 0.0
        iou = tp / iou_den if iou_den > 0 else 0.0

        specificities.append(specificity)
        fprs.append(fpr)
        ious.append(iou)

    return (
        float(np.mean(specificities)),
        float(np.mean(fprs)),
        float(np.mean(ious)),
    )


def compute_prediction_metrics(y_true, y_pred, num_classes=NUM_CLASSES):
    """Metrics that can be calculated from hard OOF predictions."""
    precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    specificity, fpr, iou = multiclass_specificity_fpr_iou(
        y_true, y_pred, num_classes
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro_sensitivity": float(recall),
        "f1_macro": float(f1),
        "specificity_macro": float(specificity),
        "fpr_macro": float(fpr),
        "iou_macro": float(iou),
        # PPV is precision in this one-vs-rest macro setup, matching the
        # existing classifier script.
        "ppv_macro": float(precision),
    }


def bootstrap_hard_prediction_metrics(
    y_true,
    y_pred,
    n_boot=N_BOOT,
    seed=SEED,
):
    """
    Percentile bootstrap over out-of-fold predictions.

    Each replicate samples the 9,860 OOF observations with replacement.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    metric_names = [
        "accuracy",
        "precision_macro",
        "recall_macro_sensitivity",
        "f1_macro",
        "specificity_macro",
        "fpr_macro",
        "iou_macro",
        "ppv_macro",
    ]

    samples = {
        name: np.empty(n_boot, dtype=np.float64)
        for name in metric_names
    }

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        metrics = compute_prediction_metrics(
            y_true[idx],
            y_pred[idx],
            NUM_CLASSES,
        )

        for name in metric_names:
            samples[name][b] = metrics[name]

    return samples


def main():
    print("=" * 70)
    print("EFFICIENTNETV2-L BASELINE STATISTICAL ANALYSIS")
    print("=" * 70)

    if not FOLD_FILE.exists():
        raise FileNotFoundError(f"Missing file: {FOLD_FILE}")

    if not PRED_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PRED_FILE}")

    folds = pd.read_csv(FOLD_FILE)
    preds = pd.read_csv(PRED_FILE)

    print(f"Fold results shape: {folds.shape}")
    print(f"OOF prediction shape: {preds.shape}")
    print(f"OOF columns: {list(preds.columns)}")

    # ---------------------------------------------------------------
    # 1. Fold-level mean, SD and 95% CI
    # ---------------------------------------------------------------
    ci_rows = []
    source_columns = {}

    for metric, column in METRIC_MAP.items():
        if column not in folds.columns:
            raise KeyError(
                f"Expected fold-results column '{column}' not found. "
                f"Available columns: {list(folds.columns)}"
            )

        source_columns[metric] = column
        values = pd.to_numeric(
            folds[column],
            errors="coerce",
        ).dropna().to_numpy()

        mean, sd, lower, upper = mean_t_ci(values)

        ci_rows.append(
            {
                "metric": metric,
                "n_folds": len(values),
                "mean": mean,
                "sd": sd,
                "ci95_lower_t": lower,
                "ci95_upper_t": upper,
            }
        )

    confidence_df = pd.DataFrame(ci_rows)
    confidence_path = OUT_DIR / "confidence_intervals.csv"
    confidence_df.to_csv(confidence_path, index=False)

    # ---------------------------------------------------------------
    # 2. Load exact OOF prediction columns
    # ---------------------------------------------------------------
    true_col = find_col(
        preds,
        ["true_class_index", "y_true", "true_label", "actual", "label"],
    )
    pred_col = find_col(
        preds,
        [
            "predicted_class_index",
            "y_pred",
            "pred_label",
            "prediction",
            "predicted_class",
        ],
    )

    y_true = pd.to_numeric(preds[true_col], errors="coerce").to_numpy()
    y_pred = pd.to_numeric(preds[pred_col], errors="coerce").to_numpy()

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[valid].astype(int)
    y_pred = y_pred[valid].astype(int)

    print(f"Valid OOF samples used for bootstrap: {len(y_true)}")

    # ---------------------------------------------------------------
    # 3. Point metrics from all OOF predictions
    # ---------------------------------------------------------------
    oof_metrics = compute_prediction_metrics(
        y_true,
        y_pred,
        NUM_CLASSES,
    )

    print("\nOOF point estimates:")
    for metric, value in oof_metrics.items():
        print(f"{metric:32s}: {value:.4f}")

    # ---------------------------------------------------------------
    # 4. Bootstrap 95% CI
    # ---------------------------------------------------------------
    print(f"\nRunning {N_BOOT} bootstrap replicates...")
    bootstrap_samples = bootstrap_hard_prediction_metrics(
        y_true,
        y_pred,
        N_BOOT,
        SEED,
    )

    bootstrap_rows = []

    for metric, values in bootstrap_samples.items():
        low = np.percentile(
            values,
            (1 - CI) / 2 * 100,
        )
        high = np.percentile(
            values,
            (1 + CI) / 2 * 100,
        )

        bootstrap_rows.append(
            {
                "metric": metric,
                "n_oof_samples": len(y_true),
                "bootstrap_replicates": len(values),
                "bootstrap_mean": float(np.mean(values)),
                "ci95_lower_percentile": float(low),
                "ci95_upper_percentile": float(high),
            }
        )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_path = OUT_DIR / "bootstrap_ci.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)

    sample_df = pd.DataFrame(bootstrap_samples)
    sample_path = OUT_DIR / "bootstrap_samples.csv"
    sample_df.to_csv(sample_path, index=False)

    # ---------------------------------------------------------------
    # 5. Shapiro-Wilk on each metric across 3 folds
    # ---------------------------------------------------------------
    shapiro_rows = []

    for metric, column in source_columns.items():
        values = pd.to_numeric(
            folds[column],
            errors="coerce",
        ).dropna().to_numpy()

        if len(values) >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p_value = shapiro(values)

            shapiro_rows.append(
                {
                    "metric": metric,
                    "shapiro_W": float(stat),
                    "p_value": float(p_value),
                    "n_folds": len(values),
                }
            )
        else:
            shapiro_rows.append(
                {
                    "metric": metric,
                    "shapiro_W": np.nan,
                    "p_value": np.nan,
                    "n_folds": len(values),
                }
            )

    shapiro_df = pd.DataFrame(shapiro_rows)

    # ---------------------------------------------------------------
    # 6. Combined summary
    # ---------------------------------------------------------------
    summary_df = (
        confidence_df
        .merge(
            bootstrap_df,
            on="metric",
            how="outer",
        )
        .merge(
            shapiro_df,
            on="metric",
            how="left",
        )
    )

    # For ROC-AUC and PR-AUC, bootstrap columns are intentionally blank
    # because probabilities were not saved in aggregated_cv_predictions.csv.
    summary_path = OUT_DIR / "statistical_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    config = {
        "seed": SEED,
        "bootstrap_replicates": N_BOOT,
        "confidence_level": CI,
        "num_classes": NUM_CLASSES,
        "fold_ci_method": "t-distribution across 3 CV folds",
        "bootstrap_method": (
            "percentile bootstrap over 9860 out-of-fold hard predictions"
        ),
        "probabilities_available_in_aggregated_predictions": False,
        "auc_bootstrap_status": (
            "Not computed because the existing aggregated prediction file "
            "contains labels/predictions/correct only, not class probabilities."
        ),
        "shapiro_note": (
            "With only 3 folds, Shapiro-Wilk has extremely low power and "
            "should be interpreted descriptively."
        ),
        "ppv_note": (
            "PPV equals macro precision in the current one-vs-rest setup."
        ),
    }

    with open(
        OUT_DIR / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Saved: {confidence_path}")
    print(f"Saved: {bootstrap_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {sample_path}")
    print(f"Saved: {OUT_DIR / 'config.json'}")
    print("\nROC-AUC and PR-AUC bootstrap CI were skipped because")
    print("the existing OOF prediction CSV has no probability columns.")


if __name__ == "__main__":
    main()
