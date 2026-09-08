# Day 3 - Baseline Error Analysis

## 1. Objective
Day 3 evaluates and analyzes the existing trained Member 1 ResNet-18 model on the verified HAM10000 test split. **No model training, retraining, or fine-tuning was performed.**

## 2. Model
- Architecture: ResNet-18
- Checkpoint: `models/best_model.pth`
- Output classes: 7
- Class order: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`
- Strict checkpoint load: passed
- SHA256: `407DCFF64528FD40C97F9BA57262C46D418A010105933E4170F5F0A95DD450EA`

## 3. Dataset
- Dataset: HAM10000
- Metadata records: 10,015
- Test samples: 1,502
- Split method: lesion-level grouping
- Train/validation lesion overlap: 0
- Train/test lesion overlap: 0
- Validation/test lesion overlap: 0
- Test preprocessing: resize to 224 x 224, tensor conversion, ImageNet normalization, no training augmentation

## 4. Overall Test Metrics
| Metric | Result |
|---|---:|
| Accuracy | 75.0333% |
| Macro Precision | 52.9676% |
| Macro Recall | 44.8971% |
| Macro F1 | 46.7842% |
| Weighted Precision | 72.7538% |
| Weighted Recall | 75.0333% |
| Weighted F1 | 72.5915% |

## 5. Per-Class Metrics
| Class | Precision | Recall | F1 | Support | Correct | Incorrect | Average confidence | Most common error |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| akiec | 0.3714 | 0.5000 | 0.4262 | 52 | 26 | 26 | 0.4741 | bkl |
| bcc | 0.5357 | 0.4225 | 0.4724 | 71 | 30 | 41 | 0.4630 | nv |
| bkl | 0.4620 | 0.5090 | 0.4843 | 167 | 85 | 82 | 0.6337 | nv |
| df | 0.0000 | 0.0000 | 0.0000 | 20 | 0 | 20 | 0.4189 | bkl |
| mel | 0.5763 | 0.2036 | 0.3009 | 167 | 34 | 133 | 0.5661 | nv |
| nv | 0.8393 | 0.9363 | 0.8851 | 1,004 | 940 | 64 | 0.8587 | bkl |
| vasc | 0.9231 | 0.5714 | 0.7059 | 21 | 12 | 9 | 0.7962 | nv |

## 6. Confusion Matrix
The confusion matrix is saved at `outputs/confusion_matrix_baseline.png`.

Rows are true classes and columns are predicted classes in the order `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.

| True / Predicted | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec | 26 | 8 | 11 | 0 | 0 | 7 | 0 |
| bcc | 13 | 30 | 7 | 0 | 1 | 20 | 0 |
| bkl | 10 | 3 | 85 | 0 | 5 | 64 | 0 |
| df | 4 | 4 | 7 | 0 | 1 | 4 | 0 |
| mel | 12 | 1 | 40 | 0 | 34 | 80 | 0 |
| nv | 5 | 9 | 34 | 0 | 15 | 940 | 1 |
| vasc | 0 | 1 | 0 | 0 | 3 | 5 | 12 |

The strongest class is `nv`, with 0.9363 recall and 0.8851 F1. The weakest class is `df`, with zero correct predictions. The model predicted `nv` for 1,120 of the 1,502 test images, indicating a strong majority-class preference. The most common error pair was `mel -> nv` with 80 cases, followed by `bkl -> nv` with 64 cases.

## 7. Error Analysis
Predictions for every test image are saved in `outputs/test_predictions.csv`. The highest-count error pairs were:

| True -> Predicted | Count |
|---|---:|
| mel -> nv | 80 |
| bkl -> nv | 64 |
| mel -> bkl | 40 |
| nv -> bkl | 34 |
| bcc -> nv | 20 |
| nv -> mel | 15 |
| bcc -> akiec | 13 |
| mel -> akiec | 12 |
| akiec -> bkl | 11 |
| bkl -> akiec | 10 |

The model has a mean confidence of 0.7624. Mean confidence was 0.8360 for correct predictions and 0.5412 for incorrect predictions. Incorrect predictions were therefore less confident on average, but confidence alone did not prevent errors. The minimum confidence was 0.2413 and the maximum was 1.0000.

Representative correct and incorrect predictions are saved in `outputs/baseline_sample_predictions.png`.

## 8. Confidence Analysis
The confidence histogram is saved at `outputs/confidence_distribution.png`. The generated CSV contains `image_id`, `true_label`, `predicted_label`, `confidence`, and `correct` for all 1,502 test images.

## 9. Important Findings
- Overall accuracy is 75.0333%, but macro F1 is only 46.7842% because performance varies substantially by class.
- `nv` dominates the test set with 1,004 samples and is classified well, which strongly influences accuracy and weighted metrics.
- `df` has 20 test samples and no correct predictions.
- `mel` recall is 0.2036, with 80 melanoma images classified as `nv` and 40 classified as `bkl`.
- Minority classes have smaller support and more variable recall, so macro metrics expose weaknesses hidden by aggregate accuracy.

## 10. Limitations
- This is a single existing baseline model evaluation; no training or comparison experiment was performed in this task.
- The test set is class-imbalanced, and rare-class metrics have limited support.
- Confidence values are model probabilities and were not calibrated.
- HAM10000 predictions are for educational/research analysis and are not medical diagnoses.
