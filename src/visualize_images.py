import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

DATASET_PATH = "dataset/HAM10000"
METADATA_PATH = os.path.join(DATASET_PATH, "HAM10000_metadata.csv")
IMAGE_DIRS = [
    os.path.join(DATASET_PATH, "HAM10000_images_part_1"),
    os.path.join(DATASET_PATH, "HAM10000_images_part_2")
]
OUTPUT_DIR = "outputs"


def visualize_samples(metadata_path=METADATA_PATH, image_dirs=IMAGE_DIRS, output_dir=OUTPUT_DIR):
    """Plots representative sample images for each skin lesion class in HAM10000."""
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found at: {metadata_path}")
        return

    df = pd.read_csv(metadata_path)
    classes = sorted(df["dx"].unique())

    # Build image ID -> filepath mapping
    image_paths = {}
    for folder in image_dirs:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_id = os.path.splitext(filename)[0]
                    image_paths[img_id] = os.path.join(folder, filename)

    if not image_paths:
        print("No image files found in part 1 / part 2 directories.")
        return

    # Plot sample images side by side
    fig, axes = plt.subplots(1, len(classes), figsize=(18, 4))
    if len(classes) == 1:
        axes = [axes]

    for i, disease in enumerate(classes):
        sample_rows = df[df["dx"] == disease]
        found_image = False

        for _, row in sample_rows.iterrows():
            img_id = row["image_id"]
            if img_id in image_paths:
                img = Image.open(image_paths[img_id]).convert("RGB")
                axes[i].imshow(img)
                axes[i].set_title(disease, fontsize=12, fontweight="bold")
                axes[i].axis("off")
                found_image = True
                break

        if not found_image:
            axes[i].set_title(f"{disease}\n(missing)", fontsize=10)
            axes[i].axis("off")

    plt.suptitle("HAM10000 - Sample Skin Lesion Images by Class", fontsize=14, y=1.05)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "sample_images.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Sample visualization saved to: {out_file}")


if __name__ == "__main__":
    visualize_samples()