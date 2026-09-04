import os
import sys
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

# Allow importing files from src
sys.path.append("src")

from preprocessing import (
    HAM10000Dataset,
    train_transform,
    test_transform,
    CLASS_NAMES
)

from model import create_model


# ==============================
# CONFIGURATION
# ==============================

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ==============================
# LOAD DATA
# ==============================

DATASET_PATH = "dataset/HAM10000"

train_df = pd.read_csv(
    os.path.join(DATASET_PATH, "splits", "train.csv")
)

val_df = pd.read_csv(
    os.path.join(DATASET_PATH, "splits", "validation.csv")
)


# ==============================
# CREATE IMAGE PATH DICTIONARY
# ==============================

image_paths = {}

folders = [

    os.path.join(
        DATASET_PATH,
        "HAM10000_images_part_1"
    ),

    os.path.join(
        DATASET_PATH,
        "HAM10000_images_part_2"
    )
]


print("Loading image paths...")

for folder in folders:

    for filename in os.listdir(folder):

        if filename.endswith(".jpg"):

            image_id = filename.replace(".jpg", "")

            image_paths[image_id] = os.path.join(
                folder,
                filename
            )


print("Images found:", len(image_paths))


# ==============================
# CREATE DATASETS
# ==============================

train_dataset = HAM10000Dataset(
    train_df,
    image_paths,
    transform=train_transform
)

val_dataset = HAM10000Dataset(
    val_df,
    image_paths,
    transform=test_transform
)


# ==============================
# CREATE DATALOADERS
# ==============================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


# ==============================
# CREATE MODEL
# ==============================

model = create_model(
    num_classes=len(CLASS_NAMES)
)

model = model.to(DEVICE)


# ==============================
# LOSS + OPTIMIZER
# ==============================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE
)


# ==============================
# TRAINING
# ==============================

best_accuracy = 0


for epoch in range(EPOCHS):

    print("\n" + "=" * 50)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("=" * 50)


    # TRAINING MODE
    model.train()

    running_loss = 0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        # Forward pass
        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )


        # Backward pass
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        running_loss += loss.item()

        _, predicted = torch.max(
            outputs.data,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_accuracy = (
        100 * correct / total
    )


    # ==============================
    # VALIDATION
    # ==============================

    model.eval()

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(images)


            _, predicted = torch.max(
                outputs.data,
                1
            )


            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = (
        100 * correct / total
    )


    print(
        f"Training Loss: "
        f"{running_loss / len(train_loader):.4f}"
    )

    print(
        f"Training Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )


    # ==============================
    # SAVE BEST MODEL
    # ==============================

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        os.makedirs(
            "models",
            exist_ok=True
        )


        torch.save(

            model.state_dict(),

            "models/best_model.pth"

        )


        print(
            "Best model saved!"
        )


print("\n" + "=" * 50)

print("TRAINING COMPLETED")

print(
    f"Best Validation Accuracy: "
    f"{best_accuracy:.2f}%"
)

print("=" * 50)