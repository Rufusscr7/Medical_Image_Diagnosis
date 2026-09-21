import base64
import io
import sys
import tempfile
from pathlib import Path

import torch
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gradcam import load_model, make_gradcam
from preprocessing import DISEASE_NAMES


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MODEL = load_model(MODEL_PATH, torch.device("cpu"))


def image_data_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(413)
def file_too_large(_error):
    return render_template("index.html", error="The image is too large. Please use a file smaller than 8 MB."), 413


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return render_template("index.html", error="Please choose an image before analyzing.")
    if not allowed_file(uploaded.filename):
        return render_template("index.html", error="Unsupported file type. Please upload a JPG, JPEG, or PNG image.")

    try:
        image_bytes = uploaded.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return render_template("index.html", error="The selected file is not a readable image.")

    try:
        temporary_path = None
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded.filename).suffix.lower(), delete=False) as temporary_file:
            temporary_file.write(image_bytes)
            temporary_file.flush()
            temporary_path = Path(temporary_file.name)
        try:
            original, overlay, predicted_class, confidence, _ = make_gradcam(
                temporary_path, MODEL, device=torch.device("cpu")
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    except Exception:
        return render_template("index.html", error="The image could not be analyzed. Please try another image.")

    result = {
        "predicted_class": predicted_class,
        "class_name": DISEASE_NAMES[predicted_class],
        "confidence": f"{confidence:.2%}",
        "original": image_data_uri(original),
        "overlay": image_data_uri(overlay),
    }
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)