import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "archive",
    "derm12345_metadata_train.csv"
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

df = pd.read_csv(METADATA_PATH)

if "label" not in df.columns:
    raise RuntimeError(
        "Column 'label' not found in metadata."
    )

counts = (
    df["label"]
    .value_counts()
    .sort_values(ascending=False)
)

plt.figure(figsize=(14, 8))

plt.bar(
    range(len(counts)),
    counts.values
)

plt.xticks(
    range(len(counts)),
    counts.index,
    rotation=90
)

plt.xlabel("Class")
plt.ylabel("Number of images")
plt.title(
    "Derm12345 Training-Set Class Distribution"
)

plt.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR,
    "Figure2_Derm12345_Class_Distribution.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 70)
print("FIGURE 2 GENERATED")
print("=" * 70)
print("Classes:", len(counts))
print("Images:", counts.sum())
print("Output:")
print(output_path)
