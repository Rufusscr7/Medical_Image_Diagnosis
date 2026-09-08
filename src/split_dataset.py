import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset/HAM10000"
METADATA_PATH = os.path.join(DATASET_PATH, "HAM10000_metadata.csv")
SPLIT_PATH = os.path.join(DATASET_PATH, "splits")


def split_data(metadata_path=METADATA_PATH, output_dir=SPLIT_PATH, seed=42):
    """
    Performs a leakage-aware 70/15/15 split on HAM10000 metadata.

    The key difference from a simple train_test_split is that we split
    on lesion_id first, not on individual images. HAM10000 can have
    multiple images for the same skin lesion. If we split on images
    directly, the same lesion could appear in both train and test,
    which would make test accuracy look better than it really is.

    So: group by lesion_id, split the *lesions*, then collect all
    images belonging to each lesion group.
    """
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found at: {metadata_path}")
        print("Please ensure the HAM10000 dataset is downloaded into the dataset/ folder.")
        return

    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} records from {metadata_path}")

    # Get one representative row per lesion (to get the dx label for stratification)
    # We use the most common dx for each lesion_id
    lesion_df = df.groupby("lesion_id")["dx"].agg(lambda x: x.mode()[0]).reset_index()
    lesion_df.columns = ["lesion_id", "dx"]
    print(f"Unique lesions: {len(lesion_df)}")

    # Split lesion IDs: 70% train, 30% temp
    train_lesions, temp_lesions = train_test_split(
        lesion_df, test_size=0.30, stratify=lesion_df["dx"], random_state=seed
    )

    # Split temp: 50% val, 50% test  (i.e. 15% / 15% of total)
    val_lesions, test_lesions = train_test_split(
        temp_lesions, test_size=0.50, stratify=temp_lesions["dx"], random_state=seed
    )

    # Now collect all images for each lesion group
    train_ids = set(train_lesions["lesion_id"])
    val_ids = set(val_lesions["lesion_id"])
    test_ids = set(test_lesions["lesion_id"])

    train_df = df[df["lesion_id"].isin(train_ids)].copy()
    val_df = df[df["lesion_id"].isin(val_ids)].copy()
    test_df = df[df["lesion_id"].isin(test_ids)].copy()

    # Verify no lesion overlap between splits
    train_lesion_set = set(train_df["lesion_id"])
    val_lesion_set = set(val_df["lesion_id"])
    test_lesion_set = set(test_df["lesion_id"])

    overlap_tv = train_lesion_set & val_lesion_set
    overlap_tt = train_lesion_set & test_lesion_set
    overlap_vt = val_lesion_set & test_lesion_set

    print(f"\nLeakage check:")
    print(f"  Train/Val lesion overlap:  {len(overlap_tv)}")
    print(f"  Train/Test lesion overlap: {len(overlap_tt)}")
    print(f"  Val/Test lesion overlap:   {len(overlap_vt)}")

    if overlap_tv or overlap_tt or overlap_vt:
        print("WARNING: Lesion overlap detected! Something went wrong.")
    else:
        print("  OK - no lesion overlap found")

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"\nSplit summary:")
    print(f"  Training samples:   {len(train_df):>5}  ({len(train_lesion_set)} unique lesions)")
    print(f"  Validation samples: {len(val_df):>5}  ({len(val_lesion_set)} unique lesions)")
    print(f"  Testing samples:    {len(test_df):>5}  ({len(test_lesion_set)} unique lesions)")
    print(f"\nSaved split files to '{output_dir}/'.")


if __name__ == "__main__":
    split_data()