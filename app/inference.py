import os
import sys
import torch
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(BASE_DIR)

SRC_DIR = os.path.join(PROJECT_DIR, "src")

sys.path.insert(0, SRC_DIR)


from preprocessing import (
    test_transform,
    CLASS_NAMES,
    DISEASE_NAMES
)

from model import create_model


# =====================================
# IMPORT PROJECT MODULES
# =====================================

from preprocessing import (
    test_transform,
    CLASS_NAMES,
    DISEASE_NAMES
)

from model import create_model


# =====================================
# DEVICE
# =====================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =====================================
# MODEL PATH
# =====================================

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "best_model.pth"
)


# =====================================
# MODEL CACHE
# =====================================

_loaded_model = None


# =====================================
# LOAD TRAINED MODEL
# =====================================

def load_model():

    global _loaded_model

    # Return cached model
    if _loaded_model is not None:
        return _loaded_model


    # Check model file
    if not os.path.exists(MODEL_PATH):

        print(
            f"Model not found: {MODEL_PATH}"
        )

        return None


    # Create ResNet-18 model
    model = create_model(
        num_classes=len(CLASS_NAMES),
        pretrained=False
    )


    # Load trained weights
    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


    model.load_state_dict(
        state_dict
    )


    # Move model to CPU/GPU
    model = model.to(DEVICE)


    # Evaluation mode
    model.eval()


    _loaded_model = model

    return _loaded_model


# =====================================
# CHECK MODEL AVAILABILITY
# =====================================

def is_model_available():

    return load_model() is not None


# =====================================
# PREDICT IMAGE
# =====================================

def predict(image: Image.Image):


    model = load_model()


    if model is None:

        return {
            "error":
            "best_model.pth not found."
        }


    # Convert image to RGB
    image = image.convert("RGB")


    # Apply SAME preprocessing
    # used during evaluation/prediction
    tensor = test_transform(
        image
    ).unsqueeze(0)


    tensor = tensor.to(DEVICE)


    # Run inference
    with torch.no_grad():

        outputs = model(
            tensor
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        ).squeeze(0)


    # Get highest probability
    top_idx = torch.argmax(
        probabilities
    ).item()


    predicted_class = CLASS_NAMES[
        top_idx
    ]


    confidence = (
        probabilities[top_idx].item()
        * 100.0
    )


    # All probabilities
    class_probabilities = {

        CLASS_NAMES[i]:
        probabilities[i].item() * 100.0

        for i in range(
            len(CLASS_NAMES)
        )

    }


    # Full disease name
    disease_name = DISEASE_NAMES.get(
        predicted_class,
        "Unknown"
    )


    return {

        "label":
        predicted_class,

        "disease_name":
        disease_name,

        "confidence":
        confidence,

        "probabilities":
        class_probabilities

    }