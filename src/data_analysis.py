import os
import pandas as pd
import matplotlib.pyplot as plt

DATASET_PATH = "dataset/HAM10000"
METADATA_PATH = os.path.join(DATASET_PATH, "HAM10000_metadata.csv")
OUTPUT_DIR = "outputs"


def analyze_data(metadata_path=METADATA_PATH, output_dir=OUTPUT_DIR):
    """Inspects class distribution and missing values in HAM10000 metadata, saving a bar plot."""
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found at: {metadata_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(metadata_path)

    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Missing values check
    missing = df.isnull().sum()
    print("\nMissing Values per column:")
    print(missing[missing > 0] if missing.any() else "  No missing values found.")

    # Class distribution
    class_counts = df["dx"].value_counts()
    print("\nClass Distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls:<8}: {count:>5} ({count / len(df) * 100:.1f}%)")

    # Plot class distribution
    plt.figure(figsize=(9, 5))
    bars = plt.bar(class_counts.index, class_counts.values, color="#2b5c8f", edgecolor="black", alpha=0.85)

    # Annotate bars with counts
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 40, f"{int(height)}", ha="center", va="bottom", fontsize=9)

    plt.title("HAM10000 - Disease Category Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Disease Category (dx)", fontsize=11)
    plt.ylabel("Sample Count", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_file = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"\nDistribution plot saved to: {out_file}")


if __name__ == "__main__":
    analyze_data()