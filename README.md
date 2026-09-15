# Medical Image Diagnosis

This repository contains an end-to-end computer vision workflow for skin-lesion image classification with the HAM10000 dataset. It combines a frozen ResNet-18 checkpoint with Grad-CAM so individual predictions can be inspected rather than treated as opaque scores.

The current interface is a local Flask inference service. It reports the existing model prediction, confidence, readable class name, and a visual explanation. It is a research interface, not a clinical product.

## Run the inference service

From this directory:

```powershell
python -m pip install -r app/requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/` and upload a JPG, JPEG, or PNG image. Uploads are validated, processed temporarily, and removed after the prediction is generated.

## System overview

- `src/model.py`: the shared ResNet-18 model definition
- `src/preprocessing.py`: class mapping and deterministic test preprocessing
- `src/predict.py`: command-line prediction helper
- `src/gradcam.py`: Grad-CAM generation for the final convolutional layer
- `app.py`: Flask interface using the frozen model and existing Grad-CAM code
- `DAY1_REPORT.md` through `DAY5_REPORT.md`: project notes and verification records

The seven class codes are `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, and `vasc`. Their readable names are defined in `src/preprocessing.py`.

The inference path is intentionally small: Flask receives an image, the shared preprocessing pipeline creates the model input, the frozen checkpoint produces class probabilities, and the existing Grad-CAM module generates the explanation overlay. The checkpoint is loaded once at application startup and the model stays in evaluation mode.

## Model provenance

The application uses only `models/best_model.pth`. Its verified SHA256 is:

`407DCFF64528FD40C97F9BA57262C46D418A010105933E4170F5F0A95DD450EA`

No training is performed when the web app starts. Grad-CAM is an interpretability visualization, not evidence of clinical reasoning or medical correctness. This project is intended for research and evaluation, not clinical decision-making.

## Validation

Compile the Python code with:

```powershell
python -m compileall -q src app.py
```

The Day 5 report records the web upload, error-handling, real-image, and model-integrity checks. The validation suite covers normal images, multiple classes, a minority class, missing files, unsupported files, corrupt images, oversized uploads, startup, and live HTTP page delivery.