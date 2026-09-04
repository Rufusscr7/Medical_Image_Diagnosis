import os
import sys
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import (
    HAM10000Dataset,
    test_transform,
    CLASS_NAMES
)
from model import create_model

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "dataset/HAM10000"
TEST_CSV = os.path.join(DATASET_PATH, "splits", "test.csv")
MODEL_PATH = "models/best_model.pth"
OUTPUT_DIR = "outputs"
IMAGE_FOLDERS = [
    os.path.join(DATASET_PATH, "HAM10000_images_part_1"),
    os.path.join(DATASET_PATH, "HAM10000_images_part_2")
]


def load_image_lookup(folders):
    """Builds a mapping from image ID to filepath across dataset parts."""
    lookup = {}
    for folder in folders:
        if os.path.exists(folder):
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_id = os.path.splitext(fname)[0]
                    lookup[img_id] = os.path.join(folder, fname)
    return lookup


def plot_confusion_matrix(cm, class_names, output_path):
    """Draws and saves a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Ticks & labels
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="HAM10000 Confusion Matrix (Test Set)",
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate matrix cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix plot saved to: {output_path}")


def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"Model checkpoint not found at '{MODEL_PATH}'. Run train.py first.")
        return

    if not os.path.exists(TEST_CSV):
        print(f"Test dataset split not found at '{TEST_CSV}'.")
        print("Please place the HAM10000 dataset and split files in the expected location.")
        return

    print(f"Evaluating model on held-out TEST set using {DEVICE}...")

    # Load test split
    test_df = pd.read_csv(TEST_CSV)
    image_lookup = load_image_lookup(IMAGE_FOLDERS)
    test_dataset = HAM10000Dataset(test_df, image_lookup, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load model
    model = create_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    total = len(all_targets)
    acc = accuracy_score(all_targets, all_preds) * 100.0 if total > 0 else 0.0

    macro_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    macro_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0

    weighted_prec = precision_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0
    weighted_rec = recall_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0) * 100.0

    print("\n" + "=" * 55)
    print(f"Test Accuracy: {acc:.2f}% ({sum(p == t for p, t in zip(all_preds, all_targets))}/{total})")
    print("=" * 55)

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0))

    print("Summary Metrics:")
    print(f"  Macro Precision:    {macro_prec:.2f}%")
    print(f"  Macro Recall:       {macro_rec:.2f}%")
    print(f"  Macro F1:           {macro_f1:.2f}%")
    print("")
    print(f"  Weighted Precision: {weighted_prec:.2f}%")
    print(f"  Weighted Recall:    {weighted_rec:.2f}%")
    print(f"  Weighted F1:        {weighted_f1:.2f}%")
    print("=" * 55)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))


if __name__ == "__main__":
    evaluate()