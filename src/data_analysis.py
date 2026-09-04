import os
import pandas as pd
import matplotlib.pyplot as plt

# Dataset paths
DATASET_PATH = "dataset/HAM10000"
METADATA_PATH = os.path.join(
    DATASET_PATH,
    "HAM10000_metadata.csv"
)

# Output folder
OUTPUT_PATH = "outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load metadata
df = pd.read_csv(METADATA_PATH)

print("=" * 50)
print("HAM10000 DATA ANALYSIS")
print("=" * 50)

# Basic information
print("\nTotal images:", len(df))

print("\nMissing values:")
print(df.isnull().sum())

# Class distribution
print("\nClass distribution:")
class_counts = df["dx"].value_counts()
print(class_counts)

# Create class distribution graph
plt.figure(figsize=(10, 6))

class_counts.plot(
    kind="bar"
)

plt.title("HAM10000 Class Distribution")
plt.xlabel("Disease Category")
plt.ylabel("Number of Images")

plt.tight_layout()

# Save graph
output_file = os.path.join(
    OUTPUT_PATH,
    "class_distribution.png"
)

plt.savefig(output_file)

plt.show()

print(f"\nGraph saved to: {output_file}")

print("\n" + "=" * 50)
print("DATA ANALYSIS COMPLETED")
print("=" * 50)