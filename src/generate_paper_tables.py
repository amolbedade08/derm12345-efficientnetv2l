import os
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "EfficientNetV2L"
)

TABLE_DIR = os.path.join(
    OUTPUT_ROOT,
    "paper_tables"
)

os.makedirs(
    TABLE_DIR,
    exist_ok=True
)


# ============================================================
# HELPER
# ============================================================

def save_table(df, filename):
    path = os.path.join(
        TABLE_DIR,
        filename
    )

    df.to_csv(
        path,
        index=False
    )

    print("Saved:", path)


# ============================================================
# TABLE 1 — DATASET CHARACTERISTICS
# ============================================================

table1 = pd.DataFrame(
    [
        {
            "Dataset": "Derm12345",
            "Role": "Primary dataset",
            "Classes": 40,
            "Train_images": 9860,
            "Test_images": 2485,
            "Total_images": 12345
        },
        {
            "Dataset": "PAD-UFES-20",
            "Role": "External validation",
            "Classes": 6,
            "Train_images": "-",
            "Test_images": "-",
            "Total_images": 2298
        },
        {
            "Dataset": "PAD-UFES-20 shared-label subset",
            "Role": "External evaluation",
            "Classes": 4,
            "Train_images": "-",
            "Test_images": 1324,
            "Total_images": 1324
        }
    ]
)

save_table(
    table1,
    "table1_dataset_characteristics.csv"
)


# ============================================================
# LOAD INTERNAL CV RESULTS
# ============================================================

eff_path = os.path.join(
    OUTPUT_ROOT,
    "csv_results",
    "corrected_fold_results.csv"
)

baseline_path = os.path.join(
    OUTPUT_ROOT,
    "feature_classifier",
    "fold_results.csv"
)

eff = pd.read_csv(
    eff_path
)

baseline = pd.read_csv(
    baseline_path
)

eff = eff.sort_values(
    "fold"
).reset_index(drop=True)

baseline = baseline.sort_values(
    "fold"
).reset_index(drop=True)


# ============================================================
# TABLE 2 — 3-FOLD CV COMPARISON
# ============================================================

metrics = [
    ("Accuracy", "accuracy"),
    ("Macro Precision", "precision_macro"),
    ("Macro Recall", "recall_macro"),
    ("Macro F1", "f1_macro"),
    ("Specificity", "specificity_macro"),
    ("FPR", "fpr_macro"),
    ("IoU", "iou_macro"),
    ("ROC-AUC", "roc_auc_macro_ovr"),
    ("PR-AUC", "pr_auc_macro")
]

rows = []

for metric_name, eff_col in metrics:

    if metric_name == "Macro Recall":
        base_col = "recall_macro_sensitivity"
    else:
        base_col = eff_col

    eff_mean = eff[eff_col].mean()
    eff_sd = eff[eff_col].std(ddof=1)

    base_mean = baseline[base_col].mean()
    base_sd = baseline[base_col].std(ddof=1)

    rows.append(
        {
            "Metric": metric_name,
            "EfficientNetV2-L_mean_SD":
                f"{eff_mean:.4f} ± {eff_sd:.4f}",
            "LogisticRegression_mean_SD":
                f"{base_mean:.4f} ± {base_sd:.4f}",
            "EfficientNetV2L_mean":
                eff_mean,
            "EfficientNetV2L_SD":
                eff_sd,
            "LogisticRegression_mean":
                base_mean,
            "LogisticRegression_SD":
                base_sd
        }
    )

table2 = pd.DataFrame(rows)

save_table(
    table2,
    "table2_three_fold_cv_comparison.csv"
)


# ============================================================
# TABLE 3 — OFFICIAL TEST
# ============================================================

official_path = os.path.join(
    OUTPUT_ROOT,
    "csv_results",
    "official_test_metrics.csv"
)

official = pd.read_csv(
    official_path
)

save_table(
    official,
    "table3_official_test.csv"
)


# ============================================================
# TABLE 4 — EXTERNAL VALIDATION
# ============================================================

external_path = os.path.join(
    OUTPUT_ROOT,
    "external_PAD_UFES20",
    "external_metrics.csv"
)

external = pd.read_csv(
    external_path
)

save_table(
    external,
    "table4_external_validation.csv"
)


# ============================================================
# TABLE 5 — STATISTICAL COMPARISON
# ============================================================

stats_path = os.path.join(
    OUTPUT_ROOT,
    "statistics",
    "statistical_comparison.csv"
)

stats = pd.read_csv(
    stats_path
)

table5 = stats[
    [
        "metric",
        "efficientnet_mean",
        "baseline_mean",
        "mean_difference",
        "shapiro_p_value",
        "paired_t_p_value",
        "wilcoxon_p_value",
        "direction"
    ]
].copy()

table5.columns = [
    "Metric",
    "EfficientNetV2L_mean",
    "Baseline_mean",
    "Mean_difference",
    "Shapiro_p",
    "Paired_t_test_p",
    "Wilcoxon_p",
    "Higher_mean"
]

save_table(
    table5,
    "table5_statistical_comparison.csv"
)


# ============================================================
# TABLE 6 — EXTERNAL PER-CLASS PERFORMANCE
# ============================================================

report_path = os.path.join(
    OUTPUT_ROOT,
    "external_PAD_UFES20",
    "classification_report.csv"
)

report = pd.read_csv(
    report_path
)

# Keep the four actual evaluated classes.
target_classes = [
    "bcc",
    "mel",
    "scc",
    "sk"
]

table6 = report[
    report.iloc[:, 0].isin(target_classes)
].copy()

if len(table6) > 0:

    first_col = table6.columns[0]

    table6.columns = [
        "Class",
        "Precision",
        "Recall",
        "F1",
        "Support"
    ]

save_table(
    table6,
    "table6_external_per_class.csv"
)


# ============================================================
# HUMAN-READABLE SUMMARY TXT
# ============================================================

summary_path = os.path.join(
    TABLE_DIR,
    "paper_table_summary.txt"
)

with open(
    summary_path,
    "w"
) as f:

    f.write(
        "PAPER TABLE SUMMARY\n"
    )

    f.write(
        "===================\n\n"
    )

    f.write(
        "Table 1: Dataset characteristics\n"
    )

    f.write(
        table1.to_string(
            index=False
        )
    )

    f.write(
        "\n\nTable 2: Three-fold CV comparison\n"
    )

    f.write(
        table2[
            [
                "Metric",
                "EfficientNetV2-L_mean_SD",
                "LogisticRegression_mean_SD"
            ]
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n\nTable 3: Official test\n"
    )

    f.write(
        official.to_string(
            index=False
        )
    )

    f.write(
        "\n\nTable 4: PAD-UFES-20 external validation\n"
    )

    f.write(
        external.to_string(
            index=False
        )
    )

    f.write(
        "\n\nTable 5: Statistical comparison\n"
    )

    f.write(
        table5.to_string(
            index=False
        )
    )

    f.write(
        "\n\nTable 6: External per-class performance\n"
    )

    f.write(
        table6.to_string(
            index=False
        )
    )

print("\n" + "=" * 70)
print("PAPER TABLE GENERATION COMPLETED")
print("=" * 70)

print("\nOutput directory:")
print(TABLE_DIR)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(TABLE_DIR)
):
    print(" -", filename)
