import os
import pandas as pd

DATASET_PATH = "dataset/HAM10000"
METADATA_PATH = os.path.join(DATASET_PATH, "HAM10000_metadata.csv")
PART1_PATH = os.path.join(DATASET_PATH, "HAM10000_images_part_1")
PART2_PATH = os.path.join(DATASET_PATH, "HAM10000_images_part_2")


def verify_dataset():
    """Checks directory structure, image counts, and metadata integrity for HAM10000."""
    print("Verifying HAM10000 dataset structure...")

    metadata_ok = os.path.exists(METADATA_PATH)
    part1_ok = os.path.exists(PART1_PATH)
    part2_ok = os.path.exists(PART2_PATH)

    print(f"  Metadata CSV:       {'Found' if metadata_ok else 'Missing'} ({METADATA_PATH})")
    print(f"  Images Part 1 dir:  {'Found' if part1_ok else 'Missing'} ({PART1_PATH})")
    print(f"  Images Part 2 dir:  {'Found' if part2_ok else 'Missing'} ({PART2_PATH})")

    if not metadata_ok:
        print("\nDataset metadata file not found. Please download HAM10000 dataset files.")
        return

    df = pd.read_csv(METADATA_PATH)
    print(f"\nMetadata Summary:")
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {', '.join(df.columns)}")

    # Count image files
    valid_extensions = (".jpg", ".jpeg", ".png")
    part1_images = [f for f in os.listdir(PART1_PATH) if f.lower().endswith(valid_extensions)] if part1_ok else []
    part2_images = [f for f in os.listdir(PART2_PATH) if f.lower().endswith(valid_extensions)] if part2_ok else []
    total_images = len(part1_images) + len(part2_images)

    print(f"\nImage Counts:")
    print(f"  Part 1: {len(part1_images):>5} images")
    print(f"  Part 2: {len(part2_images):>5} images")
    print(f"  Total:  {total_images:>5} images")

    print("\nClass distribution (dx column):")
    for category, count in df["dx"].value_counts().items():
        print(f"  {category:<8} : {count:>5} ({count / len(df) * 100:.1f}%)")


if __name__ == "__main__":
    verify_dataset()