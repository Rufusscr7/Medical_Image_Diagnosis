import os
import sys
import torch
from PIL import Image

# ==========================================
# ALLOW IMPORTING FILES FROM SRC
# ==========================================

sys.path.append("src")

from preprocessing import test_transform, CLASS_NAMES
from model import create_model


# ==========================================
# CONFIGURATION
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/best_model.pth"


# ==========================================
# LOAD MODEL
# ==========================================

print("=" * 50)
print("SKIN DISEASE IMAGE PREDICTION")
print("=" * 50)

print("Using device:", DEVICE)


if not os.path.exists(MODEL_PATH):

    print("\nERROR: Model file not found!")

    print("Expected location:", MODEL_PATH)

    sys.exit()


# Create model

model = create_model(
    num_classes=len(CLASS_NAMES)
)


# Load trained weights

model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

)


# Send model to device

model = model.to(DEVICE)


# Evaluation mode

model.eval()


print("\nTrained model loaded successfully!")


# ==========================================
# GET IMAGE PATH
# ==========================================

print("\nEnter the path of the image.")

image_path = input(
    "Image path: "
)


# Remove extra quotes if pasted

image_path = image_path.strip(
    '"'
)


# Check image exists

if not os.path.exists(image_path):

    print("\nERROR: Image not found!")

    print("Given path:", image_path)

    sys.exit()


# ==========================================
# LOAD IMAGE
# ==========================================

try:

    image = Image.open(
        image_path
    ).convert("RGB")

except Exception as e:

    print("\nERROR: Could not open image!")

    print(e)

    sys.exit()


print("\nImage loaded successfully!")


# ==========================================
# PREPROCESS IMAGE
# ==========================================

image_tensor = test_transform(
    image
)


# Add batch dimension

image_tensor = image_tensor.unsqueeze(
    0
)


# Move to device

image_tensor = image_tensor.to(
    DEVICE
)


# ==========================================
# MAKE PREDICTION
# ==========================================

with torch.no_grad():

    outputs = model(
        image_tensor
    )


# Convert outputs to probabilities

probabilities = torch.softmax(
    outputs,
    dim=1
)


# Get highest probability

confidence, predicted_index = torch.max(
    probabilities,
    1
)


# Get predicted class

predicted_class = CLASS_NAMES[
    predicted_index.item()
]


confidence_percentage = (
    confidence.item() * 100
)


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n" + "=" * 50)

print("PREDICTION RESULT")

print("=" * 50)

print(
    "Predicted Disease:",
    predicted_class
)

print(
    f"Confidence: "
    f"{confidence_percentage:.2f}%"
)

print("=" * 50)


# ==========================================
# SHOW ALL CLASS PROBABILITIES
# ==========================================

print("\nALL CLASS PROBABILITIES")

print("-" * 50)


for i, class_name in enumerate(CLASS_NAMES):

    probability = (
        probabilities[0][i].item() * 100
    )

    print(
        f"{class_name:<10}"
        f"{probability:.2f}%"
    )


print("\nPrediction completed!")