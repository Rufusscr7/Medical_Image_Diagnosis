import argparse
import sys
from pathlib import Path

import numpy as np
import torch


try:
    _TORCHVISION_LIBRARY = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_LIBRARY.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    _TORCHVISION_LIBRARY.define("qnms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except RuntimeError:
    _TORCHVISION_LIBRARY = None

from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent))

from model import create_model
from preprocessing import CLASS_NAMES, test_transform


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path=DEFAULT_MODEL_PATH, device=DEVICE):
    model = create_model(num_classes=len(CLASS_NAMES), pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def make_gradcam(image_path, model, target_class=None, device=DEVICE):
    image = Image.open(image_path).convert("RGB")
    input_tensor = test_transform(image).unsqueeze(0).to(device)
    activations = {}
    gradients = {}
    target_layer = model.layer4[-1].conv2

    def save_activation(_module, _inputs, output):
        activations["value"] = output

    def save_gradient(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(save_activation)
    backward_handle = target_layer.register_full_backward_hook(save_gradient)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())
        class_index = predicted_index if target_class is None else CLASS_NAMES.index(target_class)
        logits[0, class_index].backward()

        activation = activations["value"][0]
        gradient = gradients["value"][0]
        weights = gradient.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * activation).sum(dim=0))
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam_image = Image.fromarray((cam.detach().cpu().numpy() * 255).astype(np.uint8))
        cam_image = cam_image.resize(image.size, Image.Resampling.BILINEAR)
    finally:
        forward_handle.remove()
        backward_handle.remove()

    heatmap = Image.new("RGB", image.size)
    heatmap_pixels = np.asarray(cam_image)
    heatmap_array = np.zeros((*heatmap_pixels.shape, 3), dtype=np.uint8)
    heatmap_array[:, :, 0] = heatmap_pixels
    heatmap_array[:, :, 1] = np.maximum(0, 255 - 2 * np.abs(heatmap_pixels.astype(int) - 128)).astype(np.uint8)
    heatmap_array[:, :, 2] = 255 - heatmap_pixels
    heatmap = Image.fromarray(heatmap_array)
    overlay = Image.blend(image, heatmap, alpha=0.45)
    return image, overlay, CLASS_NAMES[predicted_index], float(probabilities[predicted_index].item()), class_index


def save_visualization(image, overlay, output_path, predicted_class, confidence, true_class=None):
    width, height = image.size
    canvas = Image.new("RGB", (width * 2, height + 55), "white")
    canvas.paste(image, (0, 0))
    canvas.paste(overlay, (width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, height + 8), "Original", fill="black")
    draw.text((width + 10, height + 8), "Grad-CAM", fill="black")
    label = f"True: {true_class or 'unknown'} | Predicted: {predicted_class} | Confidence: {confidence:.2%}"
    draw.text((10, height + 30), label, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Create a Grad-CAM explanation for a HAM10000 image.")
    parser.add_argument("--image", required=True, help="Path to an input image")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "gradcam" / "gradcam.png"))
    parser.add_argument("--class", dest="target_class", choices=CLASS_NAMES, help="Class to explain; defaults to prediction")
    parser.add_argument("--true-class", choices=CLASS_NAMES, help="Optional true class to display")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to the frozen model checkpoint")
    args = parser.parse_args()

    model = load_model(Path(args.model), DEVICE)
    image, overlay, predicted_class, confidence, _ = make_gradcam(
        Path(args.image), model, args.target_class, DEVICE
    )
    save_visualization(
        image,
        overlay,
        Path(args.output),
        predicted_class,
        confidence,
        args.true_class,
    )
    print(f"Predicted: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()