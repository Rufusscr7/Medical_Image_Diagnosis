"""
Day 2: Lesion ID analysis for HAM10000.

HAM10000 contains multiple images for some lesions.
This script shows how many lesions have more than one image
and why that matters for splitting the dataset.
"""

import os
import pandas as pd

DATASET_PATH = "dataset/HAM10000"
METADATA_PATH = os.path.join(DATASET_PATH, "HAM10000_metadata.csv")
PART1_PATH = os.path.join(DATASET_PATH, "HAM10000_images_part_1")
PART2_PATH = os.path.join(DATASET_PATH, "HAM10000_images_part_2")


def analyze_lesions():
    if not os.path.exists(METADATA_PATH):
        print(f"Metadata not found at: {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)

    print("=" * 55)
    print("DATASET INTEGRITY CHECK")
    print("=" * 55)
    print(f"Total records in metadata:  {len(df)}")
    print(f"Unique image IDs:           {df['image_id'].nunique()}")
    print(f"Duplicate image IDs:        {len(df) - df['image_id'].nunique()}")
    print(f"Missing dx labels:          {df['dx'].isnull().sum()}")
    print(f"Missing lesion_ids:         {df['lesion_id'].isnull().sum()}")

    # Check for missing values in all columns
    missing = df.isnull().sum()
    if missing.any():
        print("\nMissing values per column:")
        for col, cnt in missing[missing > 0].items():
            print(f"  {col}: {cnt}")
    else:
        print("No missing values found in any column.")

    # Verify actual image files exist
    valid_ext = (".jpg", ".jpeg", ".png")
    image_lookup = {}
    for folder in [PART1_PATH, PART2_PATH]:
        if os.path.exists(folder):
            for fname in os.listdir(folder):
                if fname.lower().endswith(valid_ext):
                    img_id = os.path.splitext(fname)[0]
                    image_lookup[img_id] = True

    images_found = sum(1 for img_id in df["image_id"] if img_id in image_lookup)
    images_missing = len(df) - images_found

    print(f"\nImage file check:")
    print(f"  Images found on disk:  {images_found}")
    print(f"  Images missing:        {images_missing}")

    # Class distribution
    print("\n" + "=" * 55)
    print("CLASS DISTRIBUTION")
    print("=" * 55)
    print(f"{'Class':<10} {'Count':>6}  {'Percentage':>10}")
    print("-" * 30)
    for cls, count in df["dx"].value_counts().items():
        pct = count / len(df) * 100
        print(f"{cls:<10} {count:>6}  {pct:>9.2f}%")

    # Lesion ID analysis
    print("\n" + "=" * 55)
    print("LESION_ID ANALYSIS")
    print("=" * 55)
    lesion_counts = df.groupby("lesion_id")["image_id"].count()

    total_lesions = lesion_counts.shape[0]
    multi_image_lesions = (lesion_counts > 1).sum()
    max_images = lesion_counts.max()
    single_image_lesions = (lesion_counts == 1).sum()

    print(f"Unique lesions:                      {total_lesions}")
    print(f"Lesions with 1 image only:           {single_image_lesions}")
    print(f"Lesions with multiple images:        {multi_image_lesions}")
    print(f"Max images for a single lesion:      {max_images}")

    print("\nDistribution of images per lesion:")
    count_dist = lesion_counts.value_counts().sort_index()
    for n_imgs, n_lesions in count_dist.items():
        print(f"  {n_imgs} image(s): {n_lesions} lesion(s)")

    print("\nWhy this matters for splitting:")
    print("  If we split randomly on images (not lesions), the same")
    print("  lesion can have one image in train and another in test.")
    print("  The model may have 'seen' the lesion already, which makes")
    print("  test performance look better than it really is.")
    print("  Solution: always split on lesion_id, never on image_id.")


if __name__ == "__main__":
    analyze_lesions()
