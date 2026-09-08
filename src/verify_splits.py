"""
Day 2: Leakage check and split verification.

Run this after running split_dataset.py to confirm:
  - Train/val/test sizes and class distributions
  - Zero lesion overlap between splits
"""

import os
import pandas as pd

DATASET_PATH = "dataset/HAM10000"
SPLIT_PATH = os.path.join(DATASET_PATH, "splits")
METADATA_PATH = os.path.join(DATASET_PATH, "HAM10000_metadata.csv")
CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def print_split_info(name, df):
    print(f"\n{name.upper()}")
    print(f"  Images:         {len(df)}")
    print(f"  Unique lesions: {df['lesion_id'].nunique()}")
    print(f"  Class distribution:")
    for cls in CLASS_ORDER:
        count = (df["dx"] == cls).sum()
        pct = count / len(df) * 100 if len(df) > 0 else 0
        print(f"    {cls:<8}: {count:>4}  ({pct:5.1f}%)")


def verify_splits():
    train_path = os.path.join(SPLIT_PATH, "train.csv")
    val_path = os.path.join(SPLIT_PATH, "validation.csv")
    test_path = os.path.join(SPLIT_PATH, "test.csv")

    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            print(f"Missing split file: {path}")
            print("Please run: python src/split_dataset.py")
            return

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print("=" * 55)
    print("SPLIT VERIFICATION")
    print("=" * 55)

    print_split_info("train", train_df)
    print_split_info("validation", val_df)
    print_split_info("test", test_df)

    # Lesion overlap check
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    overlap_tv = train_lesions & val_lesions
    overlap_tt = train_lesions & test_lesions
    overlap_vt = val_lesions & test_lesions

    print("\n" + "=" * 55)
    print("LESION OVERLAP (must all be 0)")
    print("=" * 55)
    print(f"  Train  ∩ Validation: {len(overlap_tv)}")
    print(f"  Train  ∩ Test:       {len(overlap_tt)}")
    print(f"  Val    ∩ Test:       {len(overlap_vt)}")

    if len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0:
        print("\n  PASS - No leakage detected. Splits are clean.")
    else:
        print("\n  FAIL - Lesion overlap found. Re-run split_dataset.py.")


if __name__ == "__main__":
    verify_splits()
