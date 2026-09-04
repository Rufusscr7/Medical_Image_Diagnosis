import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Dataset paths
DATASET_PATH = "dataset/HAM10000"

METADATA_PATH = os.path.join(
    DATASET_PATH,
    "HAM10000_metadata.csv"
)

IMAGE_PART_1 = os.path.join(
    DATASET_PATH,
    "HAM10000_images_part_1"
)

IMAGE_PART_2 = os.path.join(
    DATASET_PATH,
    "HAM10000_images_part_2"
)

# Load metadata
df = pd.read_csv(METADATA_PATH)

# Disease categories
classes = df["dx"].unique()

print("Classes:")
print(classes)

# Create image dictionary
image_paths = {}

for folder in [IMAGE_PART_1, IMAGE_PART_2]:

    for filename in os.listdir(folder):

        if filename.endswith(".jpg"):
            image_id = filename.replace(".jpg", "")

            image_paths[image_id] = os.path.join(
                folder,
                filename
            )

# Create figure
fig, axes = plt.subplots(
    1,
    len(classes),
    figsize=(20, 5)
)

for i, disease in enumerate(classes):

    # Get first image of this disease
    row = df[df["dx"] == disease].iloc[0]

    image_id = row["image_id"]

    image_path = image_paths[image_id]

    # Open image
    image = Image.open(image_path)

    # Display image
    axes[i].imshow(image)

    axes[i].set_title(disease)

    axes[i].axis("off")

plt.tight_layout()

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Save figure
plt.savefig(
    "outputs/sample_images.png"
)

plt.show()

print("\nSample images saved to:")
print("outputs/sample_images.png")