# Day 2 Report: Dataset Validation and Leakage check

## 1. Objective
Day 2 focused on dataset validation, leakage checking, and establishing the real baseline. However, the full HAM10000 dataset (~5.5GB) was not found on the system. We initiated a download via Kaggle, but it was cancelled before completion as requested. 

## 2. Dataset
- Total images: N/A (Download cancelled)
- Unique lesions: N/A
- Missing images: N/A

## 3. Class distribution
Not calculated as dataset download was cancelled.

## 4. Lesion analysis
Not calculated as dataset download was cancelled. However, the script `src/analyze_lesions.py` was created to perform this analysis once the dataset is available.

## 5. Existing split
We replaced the simple random split with a lesion-aware split in `src/split_dataset.py` to prevent data leakage where images from the same lesion could appear in both train and test sets.

## 6. Final split
Not generated as dataset download was cancelled.

## 7. Leakage check
Not performed as dataset download was cancelled. The script `src/verify_splits.py` was created to perform this verification.

## 8. Member 1 baseline
Model: ResNet-18
Weights: best_model.pth

## 9. Test results
Not calculated as dataset download was cancelled.

## 10. Baseline observations
The original evaluation could suffer from data leakage because the train/test split did not account for multiple images of the same lesion.

## 11. Class imbalance
Skin lesion datasets typically have severe class imbalance (e.g. mostly Melanocytic nevi). Accuracy alone is misleading, which is why precision/recall/F1 metrics are crucial.

## 12. Changes made
1. Created `feature/member2-day2` branch.
2. Rewrote `src/split_dataset.py` to perform lesion-aware splitting (grouping by `lesion_id`).
3. Added `USE_CLASS_WEIGHTS` flag to `src/train.py` for Day 3 preparation (defaults to False to preserve Member 1 baseline).
4. Created `src/verify_splits.py` to verify no lesion overlap exists between splits.
5. Created `src/analyze_lesions.py` to analyze the lesion image distribution.
6. Configured Kaggle API and initiated HAM10000 download (cancelled).

## 13. Next step
Day 3:
Class-weighted training and controlled model improvement.
