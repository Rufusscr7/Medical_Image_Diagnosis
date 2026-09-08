import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import create_model
from preprocessing import CLASS_NAMES, CLASS_TO_IDX, test_transform

DATASET_PATH = os.environ.get("HAM10000_DATASET_PATH", "dataset/HAM10000")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pth")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_FOLDERS = [
    os.path.join(DATASET_PATH, "HAM10000_images_part_1"),
    os.path.join(DATASET_PATH, "HAM10000_images_part_2"),
]


def load_image_lookup():
    lookup = {}
    for folder in IMAGE_FOLDERS:
        for filename in os.listdir(folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                lookup[os.path.splitext(filename)[0]] = os.path.join(folder, filename)
    return lookup


def plot_confusion_matrix(matrix):
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("HAM10000 Baseline Test Confusion Matrix")
    threshold = matrix.max() / 2 if matrix.max() else 1
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_baseline.png"), dpi=200)
    plt.close(fig)


def plot_confidence(predictions):
    correct = predictions[predictions["correct"] == True]["confidence"]
    incorrect = predictions[predictions["correct"] == False]["confidence"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(correct, bins=20, alpha=0.7, label="Correct")
    ax.hist(incorrect, bins=20, alpha=0.7, label="Incorrect")
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Number of test images")
    ax.set_title("Baseline Prediction Confidence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "confidence_distribution.png"), dpi=200)
    plt.close(fig)


def save_sample_grid(predictions, image_lookup):
    selected = pd.concat(
        [
            predictions[predictions["correct"] == True].sort_values("confidence", ascending=False).head(3),
            predictions[predictions["correct"] == False].sort_values("confidence", ascending=False).head(3),
        ]
    ).drop_duplicates("image_id")
    tiles = []
    for _, row in selected.iterrows():
        with Image.open(image_lookup[row["image_id"]]) as image:
            tile = image.convert("RGB").resize((224, 224))
        canvas = Image.new("RGB", (224, 270), "white")
        canvas.paste(tile, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 230), f"True: {row['true_label']}", fill="black")
        draw.text((5, 245), f"Pred: {row['predicted_label']} {row['confidence']:.1%}", fill="black")
        tiles.append(canvas)
    if not tiles:
        return
    grid = Image.new("RGB", (224 * len(tiles), 270), "white")
    for index, tile in enumerate(tiles):
        grid.paste(tile, (index * 224, 0))
    grid.save(os.path.join(OUTPUT_DIR, "baseline_sample_predictions.png"))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_path = os.path.join(DATASET_PATH, "splits", "test.csv")
    test_df = pd.read_csv(test_path)
    image_lookup = load_image_lookup()
    model = create_model(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=True)
    model.to(DEVICE).eval()

    records = []
    with torch.no_grad():
        for start in range(0, len(test_df), BATCH_SIZE):
            batch = test_df.iloc[start:start + BATCH_SIZE]
            images = torch.stack([
                test_transform(Image.open(image_lookup[row.image_id]).convert("RGB"))
                for row in batch.itertuples()
            ]).to(DEVICE)
            probabilities = torch.softmax(model(images), dim=1).cpu().numpy()
            for row, scores in zip(batch.itertuples(), probabilities):
                prediction = int(np.argmax(scores))
                records.append({
                    "image_id": row.image_id,
                    "true_label": row.dx,
                    "predicted_label": CLASS_NAMES[prediction],
                    "confidence": float(scores[prediction]),
                    "correct": row.dx == CLASS_NAMES[prediction],
                })

    predictions = pd.DataFrame(records)
    predictions.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
    plot_confidence(predictions)
    matrix = confusion_matrix(
        predictions["true_label"], predictions["predicted_label"], labels=CLASS_NAMES
    )
    plot_confusion_matrix(matrix)
    save_sample_grid(predictions, image_lookup)

    true = predictions["true_label"].map(CLASS_TO_IDX)
    predicted = predictions["predicted_label"].map(CLASS_TO_IDX)
    report = classification_report(
        true, predicted, labels=range(len(CLASS_NAMES)), target_names=CLASS_NAMES,
        output_dict=True, zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(true, predicted),
        "macro_precision": precision_score(true, predicted, average="macro", zero_division=0),
        "macro_recall": recall_score(true, predicted, average="macro", zero_division=0),
        "macro_f1": f1_score(true, predicted, average="macro", zero_division=0),
        "weighted_precision": precision_score(true, predicted, average="weighted", zero_division=0),
        "weighted_recall": recall_score(true, predicted, average="weighted", zero_division=0),
        "weighted_f1": f1_score(true, predicted, average="weighted", zero_division=0),
    }
    print("METRICS", metrics)
    print("CONFIDENCE", {
        "mean": predictions["confidence"].mean(),
        "min": predictions["confidence"].min(),
        "max": predictions["confidence"].max(),
        "correct_mean": predictions.loc[predictions["correct"], "confidence"].mean(),
        "incorrect_mean": predictions.loc[~predictions["correct"], "confidence"].mean(),
    })
    print("REPORT", report)
    print("MISCLASSIFICATIONS", Counter(
        f"{row.true_label}->{row.predicted_label}"
        for row in predictions.itertuples()
        if not row.correct
    ).most_common(10))
    for class_name in CLASS_NAMES:
        subset = predictions[predictions["true_label"] == class_name]
        errors = subset[subset["correct"] == False]
        common = errors["predicted_label"].mode().iloc[0] if not errors.empty else "none"
        print("CLASS", class_name, {
            "support": len(subset),
            "correct": int(subset["correct"].sum()),
            "incorrect": int((~subset["correct"]).sum()),
            "recall": report[class_name]["recall"],
            "f1": report[class_name]["f1-score"],
            "average_confidence": subset["confidence"].mean(),
            "common_error": common,
        })


if __name__ == "__main__":
    main()
