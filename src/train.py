import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path for direct execution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import (
    HAM10000Dataset,
    train_transform,
    test_transform,
    CLASS_NAMES
)
from model import create_model

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = "dataset/HAM10000"
SPLIT_PATH = os.path.join(DATASET_PATH, "splits")
IMAGE_FOLDERS = [
    os.path.join(DATASET_PATH, "HAM10000_images_part_1"),
    os.path.join(DATASET_PATH, "HAM10000_images_part_2")
]


def load_image_lookup(folders):
    """Indexes image IDs to their absolute filepaths across dataset parts."""
    lookup = {}
    for folder in folders:
        if os.path.exists(folder):
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_id = os.path.splitext(fname)[0]
                    lookup[img_id] = os.path.join(folder, fname)
    return lookup


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Executes a single training epoch and returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Evaluates the model on validation data and returns loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def main():
    print(f"Training on device: {DEVICE}")

    train_csv = os.path.join(SPLIT_PATH, "train.csv")
    val_csv = os.path.join(SPLIT_PATH, "validation.csv")

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        print(f"Split CSVs not found in {SPLIT_PATH}. Please run split_dataset.py first.")
        return

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    image_lookup = load_image_lookup(IMAGE_FOLDERS)
    print(f"Loaded {len(train_df)} train samples, {len(val_df)} validation samples.")
    print(f"Total images indexed: {len(image_lookup)}")

    train_dataset = HAM10000Dataset(train_df, image_lookup, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, image_lookup, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model
    model = create_model(num_classes=len(CLASS_NAMES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs("models", exist_ok=True)
    best_val_acc = 0.0
    best_model_path = "models/best_model.pth"

    print("\nStarting model training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        saved_note = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            saved_note = " -> [Checkpoint Saved]"

        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:5.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:5.2f}%{saved_note}"
        )

    print(f"\nTraining finished. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best weights preserved at: {best_model_path}")


if __name__ == "__main__":
    main()