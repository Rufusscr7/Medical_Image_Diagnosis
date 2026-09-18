# Day 6: System Validation and Robustness

## Objective

Validate the complete frozen-model web application without training, retraining, fine-tuning, architecture changes, or checkpoint replacement.

## System Workflow

`Upload -> Preprocess -> Predict -> Confidence -> Grad-CAM -> Result`

The active application is the root Flask service in `app.py`. It reuses `src/preprocessing.py` and `src/gradcam.py` with the frozen ResNet-18 checkpoint.

## End-to-End Tests

| Test | Result | Notes |
|---|---|---|
| Application startup and model load | Pass | Cold import and model startup: approximately 14.7 seconds. |
| Valid PNG upload | Pass | Committed `outputs/baseline_sample_predictions.png`; result page and both images returned. |
| Valid JPG/JPEG upload | Pass | A JPEG encoding of the committed image artifact completed successfully. |
| Different-class HAM10000 image | Not rerunnable | Raw HAM10000 images are not present in this repository checkout. |
| Minority-class HAM10000 image | Not rerunnable | Historical Day 5 case `ISIC_0031201` is documented, but its raw file is absent here. |
| High-confidence HAM10000 case | Not rerunnable | Raw HAM10000 images are not present in this repository checkout. |
| Historical Day 3 error case | Not rerunnable | The prediction table records the case, but the source image is absent. |

The executable image test produced a class, confidence, original image, and Grad-CAM result without an application crash. Predictions are not medical diagnoses.

## Robustness Tests

| Input | Result |
|---|---|
| JPG | Pass; HTTP 200 and result returned. |
| JPEG | Pass; HTTP 200 and result returned. |
| PNG | Pass; HTTP 200 and result returned. |
| No file | Pass; readable selection error, no traceback. |
| Unsupported `.txt` file | Pass; readable file-type error, no traceback. |
| Corrupted PNG payload | Pass; readable image error, no traceback. |
| Very large valid image | Pass; HTTP 200 and result returned. |
| Oversized upload above 8 MB | Pass; HTTP 413 and readable size error. |
| Very small 1x1 image | Pass; HTTP 200 and result returned. |
| Repeated uploads | Pass; repeated requests completed without a crash. |

## Direct vs Web Inference

Using the same committed image artifact and frozen checkpoint:

| Path | Class | Confidence |
|---|---|---:|
| Direct inference | `nv` | 72.63163924% |
| Web Grad-CAM inference | `nv` | 72.63163924% |

The predicted class and confidence matched exactly to the recorded precision.

## Grad-CAM

Grad-CAM generation through the web application succeeded. The overlay was readable, had the same dimensions as the source image, and was non-blank. Grad-CAM is an interpretability visualization of model attention, not clinical evidence, a diagnosis, or proof of correctness.

## Performance

Measurements were approximate on the validation environment:

| Operation | Time |
|---|---:|
| Cold application import and model startup | 14.7 s |
| Direct inference | 0.07 s |
| Grad-CAM generation | 0.17 s |

No model optimization was performed.

## Model Integrity

Checkpoint: `models/best_model.pth`

SHA256:

`407DCFF64528FD40C97F9BA57262C46D418A010105933E4170F5F0A95DD450EA`

- Training: NO
- Retraining: NO
- Fine-tuning: NO
- Architecture change: NO
- Checkpoint changed: NO

The dataset split files were not present in this checkout and were not modified.

## Limitations

This system is for educational and research use only. It is not a clinical diagnostic tool, and its outputs must not be used for clinical decisions. Raw HAM10000 test images were unavailable in this checkout, so class-specific historical image tests could not be independently rerun during Day 6.