# Day 1 Development Report: Project Audit, Validation & Baseline Hardening

**Project:** Medical Image Diagnosis (Skin Lesion Classification)  
**Dataset:** HAM10000 (7 Dermatological Classes)  
**Model:** PyTorch ResNet-18 (Transfer Learning)  
**Role:** Person 2 (Codebase Audit & Baseline Verification)  
**Branch:** `feature/member2-day1`  
**Base Commit:** `cf437c0` / `fdb7b56`  

---

## 1. Day 1 Objective

The primary objective for Day 1 was to conduct a comprehensive, read-only audit and validation of the baseline system developed by Member 1 (Rufus Mathew Saji), verify compatibility of the trained checkpoint, identify technical flaws in the evaluation pipeline, and harden the baseline code before introducing new features (such as Grad-CAM or web interfaces).

---

## 2. Member 1 Baseline Review

Member 1 established an initial end-to-end classification pipeline for skin lesion diagnosis:
* **Dataset Target:** HAM10000 dataset composed of dermatoscopic images across 7 classes:
  - `akiec`: Actinic keratoses and intraepithelial carcinoma
  - `bcc`: Basal cell carcinoma
  - `bkl`: Benign keratosis-like lesions
  - `df`: Dermatofibroma
  - `mel`: Melanoma
  - `nv`: Melanocytic nevi
  - `vasc`: Vascular lesions
* **Model Architecture:** Pretrained ResNet-18 from `torchvision.models` with the final linear classification head replaced (`num_features -> 7`).
* **Input Preprocessing:** Resize to 224×224, 3 RGB channels, ImageNet mean/std normalization. Mild augmentation (horizontal flip, 15° rotation) during training, deterministic transforms for validation/test.
* **Pipeline Scripts:**
  - `src/model.py`: Model architecture definition.
  - `src/preprocessing.py`: PyTorch `Dataset` class and transform pipelines.
  - `src/split_dataset.py`: 70% Train / 15% Validation / 15% Test stratified splitting script.
  - `src/check_dataset.py` & `src/data_analysis.py`: Dataset integrity and class distribution visualization.
  - `src/visualize_images.py`: Sample lesion extraction and visualization.
  - `src/train.py`: 5-epoch training loop using Adam ($lr=0.001$) and CrossEntropyLoss, saving `models/best_model.pth`.
  - `src/evaluate.py`: Performance evaluation and confusion matrix generation.
  - `src/predict.py`: CLI inference for individual skin lesion images.

---

## 3. Model & Checkpoint Verification

* **Trained Weights:** `models/best_model.pth` (file size: 44,798,283 bytes / ~44.8 MB).
* **Format:** PyTorch `collections.OrderedDict` (`state_dict`) containing 122 parameter tensors.
* **Compatibility Check:** The state dict was loaded into `model = create_model(num_classes=7)` with `strict=True`.
* **Result:** **PASS**. Zero missing keys, zero unexpected keys. The saved checkpoint matches the ResNet-18 7-class architecture perfectly.
* **Inference Test:** A test forward pass on a sample image executed cleanly without shape errors, generating normalized probability outputs summing to 1.0.

---

## 4. Class Mapping Verification

The class mapping is defined in `src/preprocessing.py` and consistently imported by all training, evaluation, and prediction modules:
```text
Index 0 = akiec (Actinic keratoses and intraepithelial carcinoma)
Index 1 = bcc   (Basal cell carcinoma)
Index 2 = bkl   (Benign keratosis-like lesions)
Index 3 = df    (Dermatofibroma)
Index 4 = mel   (Melanoma)
Index 5 = nv    (Melanocytic nevi)
Index 6 = vasc  (Vascular lesions)
```
*Confirmed: The mapping is uniform across the entire codebase.*

---

## 5. Problems Identified During Audit

1. **Evaluation Split Mismatch (Critical):**
   `src/evaluate.py` was hardcoded to evaluate `dataset/HAM10000/splits/validation.csv` instead of the held-out `test.csv`. The validation set must be reserved for model checkpoint selection during training; the final reported metrics must strictly come from the unseen test set.
2. **Missing Macro Summary Metrics:**
   `src/evaluate.py` only printed raw accuracy and the classification report table, but omitted high-level summary metrics for Macro F1, Macro Recall, and Macro Precision. Because HAM10000 is heavily imbalanced (`nv` represents over 67% of samples), standard accuracy is misleading, making Macro F1 the most critical diagnostic metric.
3. **Missing `requirements.txt`:**
   The repository lacked a dependency specification file, causing initial environment import errors (`torchvision`, `scikit-learn`).
4. **Potential Data Leakage in Splitting:**
   `src/split_dataset.py` stratifies purely by the diagnosis column (`dx`). However, in HAM10000, multiple images frequently correspond to the same physical lesion (`lesion_id`). Splitting without grouping by `lesion_id` risks having images of the same patient lesion present in both the training and test splits.
5. **Absence of Medical Disclaimer:**
   `src/predict.py` displayed predictions without stating that model confidence scores are experimental outputs and not diagnostic medical opinions.
6. **Dataset Not Present Locally:**
   The HAM10000 image folders and metadata CSV are currently not stored on the local system, preventing retraining or re-splitting on Day 1.

---

## 6. Changes Implemented (Day 1)

1. **Fixed `src/evaluate.py`:**
   - Switched evaluation split target from `validation.csv` to `test.csv`.
   - Added graceful error handling informing the user if the test split is missing.
   - Added explicit summary printouts for:
     - Test Accuracy
     - Macro Precision, Macro Recall, Macro F1
     - Weighted Precision, Weighted Recall, Weighted F1
   - Updated confusion matrix title to indicate test set evaluation.
2. **Created `requirements.txt`:**
   - Added core dependencies: `torch`, `torchvision`, `numpy`, `pandas`, `Pillow`, `matplotlib`, `scikit-learn`.
3. **Added Disclaimer to `src/predict.py`:**
   - Added an educational/research disclaimer statement at the conclusion of predictions.
4. **Preserved Baseline Simplicity:**
   - Kept model architecture, loss function (`nn.CrossEntropyLoss`), and hyperparameters unchanged to maintain a clean baseline comparison point.

---

## 7. Testing & Verification Summary

### Tests Successfully Executed:
- **Python Syntax Compilation:** `python -m py_compile` passed with code 0 on all source files.
- **Model State Dict Loading:** Verified `best_model.pth` loads into `ResNet-18` with `strict=True` without warnings.
- **Single Image Inference:** Verified `predict.py` successfully runs image preprocessing, forward pass, softmax conversion, and output formatting.
- **Missing Dataset Handling:** Verified `evaluate.py` gracefully reports missing test split without crashing with uncaught exceptions.

### Tests Deferred (Dataset Missing Locally):
- **Full Test Set Evaluation:** Cannot be executed until `dataset/HAM10000/` and split CSVs are acquired locally.
- **Dataset Re-Splitting & Inspection:** `split_dataset.py`, `check_dataset.py`, and `data_analysis.py` require the raw `HAM10000_metadata.csv` to execute.
- **Model Retraining:** Retraining was neither required nor possible without the image dataset.

---

## 8. Next Steps (Day 2 Plan)

1. Obtain and place the HAM10000 dataset in `dataset/HAM10000/`.
2. Inspect `lesion_id` distribution and implement lesion-aware (grouped) splitting to eliminate data leakage.
3. Run `evaluate.py` on the held-out test split to establish the true verified baseline metrics (Accuracy, Macro F1).
4. Address severe class imbalance via class-weighted CrossEntropyLoss.
5. Plan Phase 2 features (Grad-CAM visualization and interactive web UI).
