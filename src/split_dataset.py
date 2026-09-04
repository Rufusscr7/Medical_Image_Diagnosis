import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset/HAM10000"
METADATA_PATH = os.path.join(DATASET_PATH, "HAM10000_metadata.csv")
SPLIT_PATH = os.path.join(DATASET_PATH, "splits")


def split_data(metadata_path=METADATA_PATH, output_dir=SPLIT_PATH, seed=42):
    """
    Performs stratified 70/15/15 split on HAM10000 metadata
    and saves train.csv, validation.csv, and test.csv.
    """
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found at: {metadata_path}")
        print("Please ensure the HAM10000 dataset is downloaded into the dataset/ folder.")
        return

    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} records from {metadata_path}")

    # 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["dx"], random_state=seed
    )

    # 15% validation, 15% test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["dx"], random_state=seed
    )

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"\nSplit summary:")
    print(f"  Training samples:   {len(train_df):>5} (70%)")
    print(f"  Validation samples: {len(val_df):>5} (15%)")
    print(f"  Testing samples:    {len(test_df):>5} (15%)")
    print(f"\nSaved split files to '{output_dir}/'.")


if __name__ == "__main__":
    split_data()