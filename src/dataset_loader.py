from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset" / "archive"

TRAIN_CSV = DATASET_DIR / "derm12345_metadata_train.csv"
TEST_CSV = DATASET_DIR / "derm12345_metadata_test.csv"

TRAIN_PART_1 = DATASET_DIR / "derm12345_train_part_1"
TRAIN_PART_2 = DATASET_DIR / "derm12345_train_part_2"
TEST_DIR = DATASET_DIR / "derm12345_test"


# ============================================================
# CHECK PATHS
# ============================================================

print("=" * 70)
print("DERM12345 DATASET CHECK")
print("=" * 70)

paths_to_check = {
    "Train CSV": TRAIN_CSV,
    "Test CSV": TEST_CSV,
    "Train Part 1": TRAIN_PART_1,
    "Train Part 2": TRAIN_PART_2,
    "Test Images": TEST_DIR,
}

for name, path in paths_to_check.items():

    print(
        f"{name:<20}: "
        f"{'FOUND' if path.exists() else 'NOT FOUND'}"
    )

    if not path.exists():
        print(f"  Path: {path}")


# ============================================================
# STOP IF FILES ARE MISSING
# ============================================================

if not TRAIN_CSV.exists():
    raise FileNotFoundError(
        f"Training CSV not found:\n{TRAIN_CSV}"
    )

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_CSV}"
    )


# ============================================================
# READ CSV FILES
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)

test_df = pd.read_csv(TEST_CSV)


# ============================================================
# SHOW BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("CSV INFORMATION")
print("=" * 70)

print("Training records:", len(train_df))
print("Testing records :", len(test_df))

print("\nColumns:")
print(train_df.columns.tolist())


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "image_id",
    "label"
]

for column in required_columns:

    if column not in train_df.columns:
        raise ValueError(
            f"Missing column in training CSV: {column}"
        )

    if column not in test_df.columns:
        raise ValueError(
            f"Missing column in test CSV: {column}"
        )


# ============================================================
# CREATE IMAGE INDEX
# ============================================================

print("\n" + "=" * 70)
print("INDEXING IMAGE FILES")
print("=" * 70)


def build_image_index(*directories):

    image_index = {}

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    for directory in directories:

        if not directory.exists():
            continue

        print(f"Scanning: {directory}")

        for file_path in directory.rglob("*"):

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in valid_extensions:
                continue

            image_id = file_path.stem

            # Store first occurrence
            if image_id not in image_index:

                image_index[image_id] = str(
                    file_path.resolve()
                )

    return image_index


train_image_index = build_image_index(
    TRAIN_PART_1,
    TRAIN_PART_2
)

test_image_index = build_image_index(
    TEST_DIR
)


print("\nTraining images found:", len(train_image_index))
print("Test images found    :", len(test_image_index))


# ============================================================
# ATTACH IMAGE PATHS
# ============================================================

train_df["filepath"] = train_df["image_id"].map(
    train_image_index
)

test_df["filepath"] = test_df["image_id"].map(
    test_image_index
)


# ============================================================
# CHECK MISSING IMAGES
# ============================================================

missing_train = train_df[
    train_df["filepath"].isna()
]

missing_test = test_df[
    test_df["filepath"].isna()
]


print("\n" + "=" * 70)
print("IMAGE MATCHING")
print("=" * 70)

print(
    "Training images matched:",
    len(train_df) - len(missing_train)
)

print(
    "Training images missing:",
    len(missing_train)
)

print(
    "Test images matched    :",
    len(test_df) - len(missing_test)
)

print(
    "Test images missing    :",
    len(missing_test)
)


# ============================================================
# DISPLAY MISSING IMAGE IDs
# ============================================================

if len(missing_train) > 0:

    print("\nFirst missing training image IDs:")

    print(
        missing_train["image_id"]
        .head(20)
        .tolist()
    )


if len(missing_test) > 0:

    print("\nFirst missing test image IDs:")

    print(
        missing_test["image_id"]
        .head(20)
        .tolist()
    )


# ============================================================
# CLASS INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("CLASS INFORMATION")
print("=" * 70)

train_classes = sorted(
    train_df["label"].dropna().unique()
)

test_classes = sorted(
    test_df["label"].dropna().unique()
)

print("Number of training classes:", len(train_classes))
print("Number of test classes    :", len(test_classes))

print("\nClasses:")

for i, class_name in enumerate(train_classes):

    print(
        f"{i:2d}. {class_name}"
    )


# ============================================================
# CLASS COUNTS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

print(
    train_df["label"]
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 70)
print("TEST CLASS DISTRIBUTION")
print("=" * 70)

print(
    test_df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# DATASET PREVIEW
# ============================================================

print("\n" + "=" * 70)
print("TRAINING DATA PREVIEW")
print("=" * 70)

print(
    train_df[
        [
            "image_id",
            "label",
            "filepath"
        ]
    ].head(10).to_string(index=False)
)


print("\n" + "=" * 70)
print("TEST DATA PREVIEW")
print("=" * 70)

print(
    test_df[
        [
            "image_id",
            "label",
            "filepath"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# SAVE LOADED DATA
# ============================================================

output_dir = PROJECT_DIR / "outputs" / "EfficientNetV2L" / "csv_results"

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

train_output = output_dir / "train_loaded.csv"
test_output = output_dir / "test_loaded.csv"

train_df.to_csv(
    train_output,
    index=False
)

test_df.to_csv(
    test_output,
    index=False
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)

if len(missing_train) == 0 and len(missing_test) == 0:

    print("SUCCESS: ALL CSV RECORDS HAVE MATCHING IMAGES")

else:

    print("WARNING: SOME CSV RECORDS DO NOT HAVE MATCHING IMAGES")

print("=" * 70)