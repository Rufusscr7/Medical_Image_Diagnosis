# Day 4: Grad-CAM Explainability

## Objective

Apply Grad-CAM to the existing frozen ResNet-18 HAM10000 classifier to inspect which image regions contribute to its predictions.

## Model

- Checkpoint: `models/best_model.pth`
- Architecture: ResNet-18
- Outputs: 7
- Class mapping: `akiec, bcc, bkl, df, mel, nv, vasc`
- SHA256: `407DCFF64528FD40C97F9BA57262C46D418A010105933E4170F5F0A95DD450EA`
- Dataset test split: 1,502 images from the existing verified split

## Method

The script in `src/gradcam.py` uses the existing model construction and test preprocessing. It captures activations and gradients from the final convolutional layer, averages the gradients to obtain channel weights, combines the weighted activations, applies ReLU and normalization, and overlays the resized heatmap on the original image. The default target is the predicted class; `--class` can select another class.

No training, retraining or fine-tuning was performed.

## Examples

| Case | True | Predicted | Confidence | Correct |
|---|---|---|---:|---|
| Correct | nv | nv | 100.00% | Yes |
| Incorrect | mel | nv | 99.78% | No |
| Minority class | vasc | vasc | 100.00% | Yes |
| High confidence | nv | nv | 100.00% | Yes |

Visualizations:

- [Correct nv](outputs/gradcam/correct_nv.png)
- [Incorrect mel to nv](outputs/gradcam/incorrect_mel_to_nv.png)
- [Minority vasc](outputs/gradcam/minority_vasc.png)
- [High-confidence nv](outputs/gradcam/high_confidence_nv.png)

## Observations

- The correct `nv` example emphasizes the central lesion and nearby lesion boundary.
- The incorrect `mel` to `nv` example has a strong diffuse activation across the upper background, with less focused emphasis on the lesion. This may help explain the confident error.
- The rare `vasc` example produces a compact heatmap centered on the vascular lesion.
- The high-confidence `nv` example emphasizes the main lesion region, although the activation is broader than the lesion boundary.
- These maps show regions contributing to the model prediction; they do not prove medical correctness or clinical relevance.

## Limitations

Grad-CAM is an interpretability method, not clinical validation. Heatmaps can be coarse, may include background or acquisition artifacts, and should not be treated as medical evidence.

## Conclusion

Grad-CAM provides a visual way to inspect whether the existing classifier is using lesion-focused or diffuse image evidence for individual predictions. It makes confident correct and incorrect predictions easier to compare without changing the trained model.