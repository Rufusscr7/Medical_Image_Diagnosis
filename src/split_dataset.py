import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Dataset paths
DATASET_PATH = "dataset/HAM10000"

METADATA_PATH = os.path.join(
    DATASET_PATH,
    "HAM10000_metadata.csv"
)


# Load metadata
df = pd.read_csv(METADATA_PATH)

print("=" * 50)
print("DATASET SPLITTING")
print("=" * 50)

print("\nTotal images:", len(df))


# First split:
# 70% Training
# 30% Temporary

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["dx"],
    random_state=42
)


# Second split:
# 15% Validation
# 15% Testing

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["dx"],
    random_state=42
)


# Create splits folder
SPLIT_PATH = os.path.join(
    DATASET_PATH,
    "splits"
)

os.makedirs(SPLIT_PATH, exist_ok=True)


# Save CSV files

train_df.to_csv(
    os.path.join(SPLIT_PATH, "train.csv"),
    index=False
)

val_df.to_csv(
    os.path.join(SPLIT_PATH, "validation.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(SPLIT_PATH, "test.csv"),
    index=False
)


# Print results

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))


print("\nTraining class distribution:")
print(train_df["dx"].value_counts())


print("\nValidation class distribution:")
print(val_df["dx"].value_counts())


print("\nTesting class distribution:")
print(test_df["dx"].value_counts())


print("\n" + "=" * 50)
print("DATASET SPLITTING COMPLETED")
print("=" * 50)