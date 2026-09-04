import os
import pandas as pd

# Project dataset folder
DATASET_PATH = "dataset/HAM10000"

# Paths
METADATA_PATH = os.path.join(
    DATASET_PATH,
    "HAM10000_metadata.csv"
)

PART1_PATH = os.path.join(
    DATASET_PATH,
    "HAM10000_images_part_1"
)

PART2_PATH = os.path.join(
    DATASET_PATH,
    "HAM10000_images_part_2"
)

print("=" * 50)
print("HAM10000 DATASET CHECK")
print("=" * 50)

# Check folders
print("\nChecking dataset files...")

print("Metadata exists:", os.path.exists(METADATA_PATH))
print("Part 1 exists:", os.path.exists(PART1_PATH))
print("Part 2 exists:", os.path.exists(PART2_PATH))

# Load metadata
df = pd.read_csv(METADATA_PATH)

print("\nDataset loaded successfully!")

print("\nNumber of records:", len(df))

# Count images
part1_images = [
    f for f in os.listdir(PART1_PATH)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

part2_images = [
    f for f in os.listdir(PART2_PATH)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("\nImages in Part 1:", len(part1_images))
print("Images in Part 2:", len(part2_images))
print("Total images:", len(part1_images) + len(part2_images))

print("\nMetadata columns:")
print(df.columns.tolist())

print("\nDisease categories:")
print(df["dx"].value_counts())

print("\nFirst 5 rows:")
print(df.head())

print("\n" + "=" * 50)
print("DATASET CHECK COMPLETED")
print("=" * 50)