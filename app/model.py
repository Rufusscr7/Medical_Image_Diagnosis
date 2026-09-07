"""Placeholder inference module for skin-lesion cancer prediction.

The UI is fully functional, but the actual machine-learning model is not
trained yet. Plug your real model into this module:

1. Train a CNN (e.g. TensorFlow/Keras) on a skin-lesion dataset such as
   HAM10000 / ISIC 2018.
2. Save it (e.g. `model.keras`).
3. Replace `_MODEL_PATH`, `load_model()` and `predict()` below.

The rest of the app (app.py) does not need to change.
"""

import numpy as np
from PIL import Image

_MODEL_PATH = "model.keras"
_loaded_model = None


def is_model_available():
    """Return True once a real trained model file has been placed
    in the project and loaded successfully."""
    return _loaded_model is not None


def load_model():
    """Load the trained model. No-op until a real model file exists."""
    global _loaded_model
    # TODO: uncomment once a trained model file is available.
    # from tensorflow import keras
    # _loaded_model = keras.models.load_model(_MODEL_PATH)
    _loaded_model = None
    return _loaded_model


def predict(image: Image.Image):
    """Predict probability of malignancy for an uploaded skin-lesion image.

    Placeholder logic — returns a deterministic fake score so the UI flow
    can be demonstrated end-to-end. Replace with real model inference:

        x = np.array(image.resize((224, 224))) / 255.0
        x = x.reshape(1, 224, 224, 3)
        prob = float(_loaded_model.predict(x)[0][0])   # 0 = benign, 1 = malignant

    Returns:
        dict with keys: malignant_prob (float), label (str),
                        confidence (float), message (str)
    """
    arr = np.asarray(image.resize((64, 64))).astype(np.float32)

    # Simple colour heuristics give varied-but-sensible demo results:
    # darker/redder pigmented areas raise the mock malignancy score.
    redness = float(arr[..., 0].mean() - arr[..., 1].mean())
    darkness = float(255.0 - arr[..., 0].mean() * 0.3
                     - arr[..., 1].mean() * 0.59
                     - arr[..., 2].mean() * 0.11)

    score = 0.5 + (redness / 120.0) + ((darkness - 128.0) / 500.0)
    prob = float(np.clip(score, 0.02, 0.98))

    label = "Malignant" if prob >= 0.5 else "Benign"
    confidence = abs(prob - 0.5) * 2.0  # 0..1, how decisive the model is

    if label == "Malignant":
        message = "The model flags this lesion as suspicious. Please consult a dermatologist."
    else:
        message = "The model does not detect signs of malignancy in this image."

    return {
        "malignant_prob": prob,
        "label": label,
        "confidence": confidence,
        "message": message,
    }