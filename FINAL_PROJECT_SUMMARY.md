# Final Project Summary

## 1. Problem Statement

Skin-lesion images can be difficult to classify consistently from visual inspection alone. This project packages a computer-vision workflow that produces a seven-class research prediction and a visual explanation for an uploaded image.

## 2. Objective

The system accepts a skin-lesion image, applies the existing preprocessing pipeline, predicts a HAM10000 class, reports confidence, and generates a Grad-CAM visualization through a local web application.

## 3. Dataset

The project uses HAM10000 with seven classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, and `vasc`.

## 4. Model

The classifier is a ResNet-18 with seven output classes using the existing trained checkpoint at `models/best_model.pth`. The model was not retrained, fine-tuned, replaced, or otherwise changed during this project phase.

## 5. Preprocessing

Input images are converted to RGB, resized to `224 x 224`, converted to tensors, and normalized with ImageNet mean and standard deviation values. The same deterministic test transform is used for inference. Uploaded files are checked for an allowed image extension, readable image content, and the 8 MB request limit.

## 6. Evaluation Results

The verified baseline results are:

| Metric | Result |
|---|---:|
| Accuracy | 75.03% |
| Macro Precision | 52.97% |
| Macro Recall | 44.90% |
| Macro F1 | 46.78% |
| Weighted F1 | 72.59% |

## 7. Important Findings

The dataset is imbalanced, with `nv` dominant in the evaluation distribution. Performance differs substantially between classes, and minority classes generally have weaker results than the dominant class. These findings limit how broadly the aggregate accuracy should be interpreted.

## 8. Explainability

Grad-CAM creates an attention heatmap from the final convolutional features of the ResNet-18. It highlights image regions associated with the selected output, helping reviewers inspect model behavior. It is an interpretability visualization, not proof of correctness or a clinical explanation.

## 9. Web Application

The local Flask interface follows:

`Upload -> Prediction -> Confidence -> Grad-CAM`

It displays the predicted class, readable disease name, confidence score, original image, and explanation overlay. Invalid, unsupported, corrupt, missing, and oversized inputs receive readable error messages.

## 10. Validation

Day 6 validation confirmed application startup and model loading, valid PNG/JPG/JPEG handling, prediction and Grad-CAM delivery, direct-versus-web prediction consistency, repeated uploads, and robust handling of invalid inputs. The recorded direct and web inference result matched at `nv` with `72.63163924%` confidence for the committed demonstration artifact.

## 11. Limitations

This is an educational and research system, not a clinical diagnostic tool. Results are limited by HAM10000's class balance, coverage, labeling, image quality, and domain shift. Historical class-specific images were not available in the final checkout for independent reruns. Grad-CAM does not establish medical evidence, and predictions must not be treated as medical advice or used for clinical decisions.

## 12. Future Scope

Possible future work includes broader external validation, improved minority-class analysis, calibrated uncertainty, additional data, and deployment safeguards. These improvements were not implemented during Day 7.