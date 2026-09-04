import os
import sys
import argparse
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import test_transform, CLASS_NAMES, DISEASE_NAMES
from model import create_model

DEFAULT_MODEL_PATH = "models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path=DEFAULT_MODEL_PATH, device=DEVICE):
    """Loads ResNet-18 model weights from checkpoint file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at '{model_path}'. Train the model first.")

    model = create_model(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def predict(image_path, model, device=DEVICE):
    """Runs inference on a single image and returns top prediction and class probabilities."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at '{image_path}'.")

    image = Image.open(image_path).convert("RGB")
    tensor = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0)

    top_idx = torch.argmax(probs).item()
    top_class = CLASS_NAMES[top_idx]
    top_conf = probs[top_idx].item() * 100.0

    class_probs = {CLASS_NAMES[i]: probs[i].item() * 100.0 for i in range(len(CLASS_NAMES))}
    return top_class, top_conf, class_probs


def main():
    parser = argparse.ArgumentParser(description="Predict skin lesion disease from an image.")
    parser.add_argument("image_path", nargs="?", help="Path to input skin lesion image")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained model weights")
    args = parser.parse_args()

    # If no image path provided via CLI, prompt interactively
    image_path = args.image_path
    if not image_path:
        image_path = input("Enter path to skin lesion image: ").strip().strip('"')

    if not image_path:
        print("No image path provided. Exiting.")
        return

    try:
        model = load_model(args.model, DEVICE)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    try:
        predicted_class, confidence, probabilities = predict(image_path, model, DEVICE)
    except Exception as err:
        print(f"\nError processing image: {err}")
        return

    full_name = DISEASE_NAMES.get(predicted_class, "Unknown")

    print("\n" + "=" * 55)
    print(f"  Prediction : {predicted_class.upper()} ({full_name})")
    print(f"  Confidence : {confidence:.2f}%")
    print("=" * 55)

    print("\nClass Probability Breakdown:")
    for cls_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(prob / 5)
        print(f"  {cls_name:<8}: {prob:6.2f}% | {bar}")

    print("\nNote: This is an AI-based experimental classification result for educational/research purposes only.")
    print("It is not a medical diagnosis and should not be used as a substitute for professional medical advice.")


if __name__ == "__main__":
    main()