# Medical Image Diagnosis

Medical Image Diagnosis is a local Flask application for skin-lesion image classification using the HAM10000 dataset. It combines a frozen ResNet-18 checkpoint with Grad-CAM so predictions can be inspected alongside an interpretability visualization.

## Dataset and model

The project uses HAM10000 and its seven classes:

- `akiec`
- `bcc`
- `bkl`
- `df`
- `mel`
- `nv`
- `vasc`

The application uses the existing frozen ResNet-18 checkpoint at `models/best_model.pth`. The model is loaded for inference only; the web application does not train, retrain, fine-tune, or replace it.

## Workflow

`Upload -> Preprocess -> Predict -> Confidence -> Grad-CAM`

Features include image upload, skin-lesion classification, confidence reporting, Grad-CAM visualization, and input validation.

## Installation and running

From the repository directory, install the verified dependencies:

```powershell
python -m pip install -r requirements.txt
```

Ensure `models/best_model.pth` is present, then start the application:

```powershell
python app.py
```

Open `http://127.0.0.1:5000/`, upload a JPG, JPEG, or PNG image, and review the predicted class, confidence, and Grad-CAM output. Uploads are validated, processed temporarily, and removed after prediction.

For a production-style local server, install the same requirements and run:

```powershell
python wsgi.py
```

The WSGI server listens on `http://127.0.0.1:8000/` by default. Set the `PORT` environment variable when the deployment platform provides its own port.

## System overview

- `src/model.py`: the shared ResNet-18 model definition
- `src/preprocessing.py`: class mapping and deterministic test preprocessing
- `src/predict.py`: command-line prediction helper
- `src/gradcam.py`: Grad-CAM generation for the final convolutional layer
- `app.py`: Flask interface using the frozen model and existing Grad-CAM code
- `DAY1_REPORT.md` through `DAY6_REPORT.md`: project notes and verification records

The seven class codes are `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, and `vasc`. Their readable names are defined in `src/preprocessing.py`.

The inference path is intentionally small: Flask receives an image, the shared preprocessing pipeline creates the model input, the frozen checkpoint produces class probabilities, and the existing Grad-CAM module generates the explanation overlay. The checkpoint is loaded once at application startup and the model stays in evaluation mode.

## Model provenance

The application uses only `models/best_model.pth`. Its verified SHA256 is:

`407DCFF64528FD40C97F9BA57262C46D418A010105933E4170F5F0A95DD450EA`

No training is performed when the web app starts. Grad-CAM is an interpretability visualization, not evidence of clinical reasoning or medical correctness.

## Limitations and responsible use

This project is for educational and research use only and is not a clinical diagnostic tool. Model performance varies between classes and is limited by the training data, class imbalance, dataset coverage, and image quality. HAM10000 is a finite research dataset and does not represent every patient or clinical setting. Grad-CAM shows image regions associated with the model output; it does not establish causation, correctness, or clinical evidence. Predictions must not be treated as medical advice or used for clinical decisions.

## Validation

Compile the Python code with:

```powershell
python -m compileall -q src app.py
```

The Day 6 report records the frozen-model end-to-end and robustness checks. The validation suite covers valid JPG/JPEG/PNG uploads, missing files, unsupported files, corrupt images, very small and oversized images, repeated uploads, startup, direct-versus-web consistency, and Grad-CAM delivery. Raw HAM10000 image files are not stored in this repository, so rerunning the historical class-specific image cases requires the local dataset used by the earlier reports.