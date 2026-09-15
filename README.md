# Medical Image Diagnosis

This is a student research project that explores skin-lesion image classification with the HAM10000 dataset. The project uses a frozen ResNet-18 checkpoint and Grad-CAM to make an individual prediction easier to inspect.

The web page is intentionally a demo, not a diagnostic product. It shows what the existing model predicts and which broad image regions contributed to that prediction.

## Run the web demo

From this directory:

```powershell
python -m pip install -r app/requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/` and upload a JPG, JPEG, or PNG image. The app keeps uploads temporary and removes them after the prediction is generated.

## What is included

- `src/model.py`: the shared ResNet-18 model definition
- `src/preprocessing.py`: class mapping and deterministic test preprocessing
- `src/predict.py`: command-line prediction helper
- `src/gradcam.py`: Grad-CAM generation for the final convolutional layer
- `app.py`: Flask interface using the frozen model and existing Grad-CAM code
- `DAY1_REPORT.md` through `DAY5_REPORT.md`: project notes and verification records

The seven class codes are `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, and `vasc`. Their readable names are defined in `src/preprocessing.py`.

## Model provenance

The application uses only `models/best_model.pth`. Its verified SHA256 is:

`407DCFF64528FD40C97F9BA57262C46D418A010105933E4170F5F0A95DD450EA`

No training is performed when the web app starts. Grad-CAM is an interpretability visualization, not evidence of clinical reasoning or medical correctness. This project is for educational and research use only and must not be used for clinical decision-making.

## Checks

Compile the Python code with:

```powershell
python -m compileall -q src app.py
```

The Day 5 report records the web upload, error-handling, real-image, and model-integrity checks.