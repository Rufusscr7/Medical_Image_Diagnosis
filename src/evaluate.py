import os
import sys
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


# ==========================================
# ALLOW IMPORTING FILES FROM SRC
# ==========================================

sys.path.append("src")


from preprocessing import (
    HAM10000Dataset,
    test_transform,
    CLASS_NAMES
)

from model import create_model


# ==========================================
# CONFIGURATION
# ==========================================

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print("Using device:", DEVICE)


# ==========================================
# DATASET PATH
# ==========================================

DATASET_PATH = "dataset/HAM10000"


# ==========================================
# LOAD VALIDATION DATA
# ==========================================

val_df = pd.read_csv(

    os.path.join(
        DATASET_PATH,
        "splits",
        "validation.csv"
    )
)

print("\nValidation images:", len(val_df))


# ==========================================
# CREATE IMAGE PATH DICTIONARY
# ==========================================

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


print("\nLoading image paths...")


for folder in folders:

    for filename in os.listdir(folder):

        if filename.endswith(".jpg"):

            image_id = filename.replace(
                ".jpg",
                ""
            )

            image_paths[image_id] = os.path.join(
                folder,
                filename
            )


print(
    "Images found:",
    len(image_paths)
)


# ==========================================
# CREATE VALIDATION DATASET
# ==========================================

val_dataset = HAM10000Dataset(

    val_df,

    image_paths,

    transform=test_transform

)


# ==========================================
# CREATE DATALOADER
# ==========================================

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


# ==========================================
# CREATE MODEL
# ==========================================

model = create_model(

    num_classes=len(CLASS_NAMES)

)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = "models/best_model.pth"


if not os.path.exists(MODEL_PATH):

    print(
        "\nERROR: Model file not found!"
    )

    print(
        "Expected location:",
        MODEL_PATH
    )

    sys.exit()


model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

)


model = model.to(DEVICE)


model.eval()


print(
    "\nTrained model loaded successfully!"
)


# ==========================================
# EVALUATION
# ==========================================

correct = 0

total = 0


all_predictions = []

all_labels = []


print("\nEvaluating model...")


with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        outputs = model(images)


        _, predicted = torch.max(

            outputs,

            1

        )


        total += labels.size(0)


        correct += (

            predicted == labels

        ).sum().item()


        all_predictions.extend(

            predicted.cpu().numpy()

        )


        all_labels.extend(

            labels.cpu().numpy()

        )


# ==========================================
# CALCULATE ACCURACY
# ==========================================

accuracy = (

    100 * correct / total

)


print("\n" + "=" * 50)

print(
    f"VALIDATION ACCURACY: "
    f"{accuracy:.2f}%"
)

print("=" * 50)


# ==========================================
# CONFUSION MATRIX
# ==========================================

num_classes = len(CLASS_NAMES)


confusion_matrix = [

    [0 for _ in range(num_classes)]

    for _ in range(num_classes)

]


for true_label, predicted_label in zip(

    all_labels,

    all_predictions

):

    confusion_matrix[

        true_label

    ][

        predicted_label

    ] += 1


# ==========================================
# PRINT CONFUSION MATRIX
# ==========================================

print("\nCONFUSION MATRIX\n")


print("Classes:")

for i, class_name in enumerate(CLASS_NAMES):

    print(i, "-", class_name)


print("\nMatrix:")


for row in confusion_matrix:

    print(row)


# ==========================================
# PRECISION, RECALL, F1 SCORE
# ==========================================

print("\n" + "=" * 65)

print("CLASSIFICATION REPORT")

print("=" * 65)


print(

    f"{'Class':<10}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1-Score':<12}"
    f"{'Support':<10}"

)


print("-" * 65)


for i in range(num_classes):


    # True Positives

    true_positive = confusion_matrix[i][i]


    # False Positives

    false_positive = sum(

        confusion_matrix[row][i]

        for row in range(num_classes)

        if row != i

    )


    # False Negatives

    false_negative = sum(

        confusion_matrix[i][column]

        for column in range(num_classes)

        if column != i

    )


    # Support

    support = sum(

        confusion_matrix[i]

    )


    # Precision

    if (

        true_positive +
        false_positive

    ) == 0:

        precision = 0


    else:

        precision = (

            true_positive /

            (
                true_positive +
                false_positive
            )

        )


    # Recall

    if (

        true_positive +
        false_negative

    ) == 0:

        recall = 0


    else:

        recall = (

            true_positive /

            (
                true_positive +
                false_negative
            )

        )


    # F1 Score

    if (

        precision +
        recall

    ) == 0:

        f1_score = 0


    else:

        f1_score = (

            2 *

            (
                precision *
                recall
            )

            /

            (
                precision +
                recall
            )

        )


    print(

        f"{CLASS_NAMES[i]:<10}"

        f"{precision:<12.3f}"

        f"{recall:<12.3f}"

        f"{f1_score:<12.3f}"

        f"{support:<10}"

    )


# ==========================================
# SAVE CONFUSION MATRIX IMAGE
# ==========================================

os.makedirs(

    "outputs",

    exist_ok=True

)


plt.figure(

    figsize=(10, 8)

)


plt.imshow(

    confusion_matrix

)


plt.colorbar()


plt.xticks(

    range(num_classes),

    CLASS_NAMES,

    rotation=45

)


plt.yticks(

    range(num_classes),

    CLASS_NAMES

)


plt.xlabel(

    "Predicted Label"

)


plt.ylabel(

    "True Label"

)


plt.title(

    "HAM10000 Confusion Matrix"

)


# Add numbers inside matrix

for i in range(num_classes):

    for j in range(num_classes):

        plt.text(

            j,

            i,

            str(
                confusion_matrix[i][j]
            ),

            ha="center",

            va="center"

        )


plt.tight_layout()


plt.savefig(

    "outputs/confusion_matrix.png"

)


plt.show()


print("\nConfusion matrix saved!")

print(

    "Location: "
    "outputs/confusion_matrix.png"

)


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n" + "=" * 50)

print(
    "EVALUATION COMPLETED"
)

print("=" * 50)# ==========================================
# SAVE CONFUSION MATRIX IMAGE
# ==========================================

os.makedirs(

    "outputs",

    exist_ok=True

)


plt.figure(

    figsize=(10, 8)

)


plt.imshow(

    confusion_matrix

)


plt.colorbar()


plt.xticks(

    range(num_classes),

    CLASS_NAMES,

    rotation=45

)


plt.yticks(

    range(num_classes),

    CLASS_NAMES

)


plt.xlabel(

    "Predicted Label"

)


plt.ylabel(

    "True Label"

)


plt.title(

    "HAM10000 Confusion Matrix"

)


# Add numbers inside matrix

for i in range(num_classes):

    for j in range(num_classes):

        plt.text(

            j,

            i,

            str(
                confusion_matrix[i][j]
            ),

            ha="center",

            va="center"

        )


plt.tight_layout()


plt.savefig(

    "outputs/confusion_matrix.png",

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print("\nConfusion matrix saved!")

print(

    "Location: "
    "outputs/confusion_matrix.png"

)