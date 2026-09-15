# Day 5: Web App Integration

## 1. Objective

Integrate the existing frozen ResNet-18 classifier and Day 4 Grad-CAM implementation into a simple local web interface for prediction and visual explanation.

## 2. Model

- Architecture: ResNet-18
- Outputs: 7 classes
- Checkpoint: `models/best_model.pth`
- The existing checkpoint was loaded once when the Flask application started.
- The model was set to evaluation mode and run on CPU.
- The model was not retrained, fine-tuned, or otherwise modified.

## 3. Web Application

The application uses Flask with HTML and CSS. Users can upload a JPG, JPEG, or PNG image and receive:

- predicted dataset class and readable class name
- prediction confidence
- original image
- Grad-CAM explanation overlay

The upload is validated and processed through a temporary file that is deleted after inference. The existing `src/gradcam.py` implementation is reused directly, including its model, preprocessing, target layer, and compatibility shim.

Start the application from the repository root with:

```text
python app.py
```

Then open `http://127.0.0.1:5000/`.

## 4. Class Mapping

| Index | Code | Class name |
|---:|---|---|
| 0 | akiec | Actinic keratoses and intraepithelial carcinoma |
| 1 | bcc | Basal cell carcinoma |
| 2 | bkl | Benign keratosis-like lesions |
| 3 | df | Dermatofibroma |
| 4 | mel | Melanoma |
| 5 | nv | Melanocytic nevi |
| 6 | vasc | Vascular lesions |

## 5. Testing

All tests passed:

| Test | Result |
|---|---|
| Python compilation | Pass |
| Flask startup and one-time model load | Pass |
| Live homepage request | Pass, HTTP 200 |
| Normal HAM10000 test image | Pass |
| Different-class HAM10000 test image | Pass |
| Minority `vasc` test image | Pass |
| No file selected | Pass, human-readable message |
| Unsupported file type | Pass, human-readable message |
| Corrupt image | Pass, human-readable message |
| Upload larger than 8 MB | Pass, HTTP 413 message |
| Grad-CAM output in web response | Pass |

The tested real images were existing HAM10000 test images: `ISIC_0029977`, `ISIC_0028760`, and minority-class image `ISIC_0031201`.

## 6. Model Integrity

The checkpoint SHA256 after implementation and testing was:

`407DCFF64528FD40C97F9BA57262C46D418A010105933E4170F5F0A95DD450EA`

No model weights, dataset files, or train/validation/test split files were changed. No training script was executed.

## 7. Limitations

This is an educational and research system, not a clinical diagnostic tool. Predictions are not clinical diagnoses and should not be used for clinical decision-making. Existing model performance is limited. Grad-CAM provides an interpretability visualization of contributing image regions, not proof of clinical reasoning or correctness.