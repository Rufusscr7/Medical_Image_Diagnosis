# Day 2 Report: Dataset Verification and Baseline Evaluation

## 1. Objective
Verify HAM10000, create lesion-aware train/validation/test splits, and evaluate the original Member 1 baseline on the untouched test split.

## 2. Dataset Verification
- Dataset: `C:\Users\muham\Documents\HAM10000\HAM10000`
- Metadata records: 10,015
- Unique image IDs: 10,015
- Unique lesions: 7,470
- Image files: 10,015
- Missing image files: 0
- Extra image files: 0
- Unreadable images: 0
- Duplicate image IDs: 0
- Required columns: `lesion_id`, `image_id`, `dx`
- Diagnosis labels: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`

## 3. Split Verification
The previous split was invalid because it had lesion overlap: train/validation 485, train/test 512, and validation/test 115.

The split was regenerated with lesion-level grouping, stratification, and random seed 42. Every metadata record was assigned exactly once.

| Split | Images | Lesions |
|---|---:|---:|
| Train | 6,981 | 5,229 |
| Validation | 1,532 | 1,120 |
| Test | 1,502 | 1,121 |

Final lesion overlap:

- Train/Validation: 0
- Train/Test: 0
- Validation/Test: 0

Duplicate image assignments: 0. Missing metadata images: 0. Extra split images: 0.

## 4. Preprocessing
- Resize: 224 x 224
- Normalization: ImageNet mean and standard deviation
- Training: horizontal flip and rotation augmentation
- Validation/test: deterministic resize, tensor conversion, and normalization only
- Class mapping: `0=akiec`, `1=bcc`, `2=bkl`, `3=df`, `4=mel`, `5=nv`, `6=vasc`

## 5. Baseline Checkpoint
- Architecture: ResNet-18
- Output classes: 7
- Checkpoint: `models/best_model.pth`
- Source: recovered original Member 1 checkpoint from `C:\Users\muham\Downloads\best_model.pth`
- Size: 44,798,283 bytes
- SHA256: `407DCFF64528FD40C97F9BA57262C46D418A010105933E4170F5F0A95DD450EA`
- Strict load: passed; missing keys 0; unexpected keys 0

## 6. Baseline Test Results
The baseline was evaluated on the 1,502-image test split.

| Metric | Result |
|---|---:|
| Accuracy | 75.03% |
| Macro Precision | 52.97% |
| Macro Recall | 44.90% |
| Macro F1 | 46.78% |
| Weighted Precision | 72.75% |
| Weighted Recall | 75.03% |
| Weighted F1 | 72.59% |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| akiec | 0.3714 | 0.5000 | 0.4262 | 52 |
| bcc | 0.5357 | 0.4225 | 0.4724 | 71 |
| bkl | 0.4620 | 0.5090 | 0.4843 | 167 |
| df | 0.0000 | 0.0000 | 0.0000 | 20 |
| mel | 0.5763 | 0.2036 | 0.3009 | 167 |
| nv | 0.8393 | 0.9363 | 0.8851 | 1,004 |
| vasc | 0.9231 | 0.5714 | 0.7059 | 21 |

Confusion matrix: `outputs/confusion_matrix_baseline.png`

## 7. Training Class Imbalance
Weights use only the 6,981 training samples and `weight = N / (7 * class_count)`.

| Class | Count | Percentage | Weight |
|---|---:|---:|---:|
| akiec | 222 | 3.1801% | 4.49227799 |
| bcc | 361 | 5.1712% | 2.76256431 |
| bkl | 772 | 11.0586% | 1.29182087 |
| df | 71 | 1.0170% | 14.04627767 |
| mel | 773 | 11.0729% | 1.29014970 |
| nv | 4,683 | 67.0821% | 0.21295873 |
| vasc | 99 | 1.4181% | 10.07359307 |

The majority class is `nv`; the smallest class is `df`. Macro metrics are important because weighted metrics are strongly influenced by `nv`.

## 8. Limitations
- HAM10000 is substantially class-imbalanced.
- The rare classes have limited support in the test split.
- This is an educational/research evaluation and not a medical diagnostic system.
- Class-weighted training has not been run in Day 2.
