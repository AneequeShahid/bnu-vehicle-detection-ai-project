# Training Results Summary

## Best Model Metrics (Epoch 24)

| Metric | Value |
|--------|-------|
| Precision | 89.4% |
| Recall | 57.3% |
| mAP50 | 76.2% |
| mAP50-95 | 56.5% |

## Per-Class Results

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| bnu_sticker | 78.8% | 34.2% | 52.8% | 34.1% |
| number_plate | 100% | 80.3% | 99.5% | 78.9% |

## Files in This Folder

| File | Description |
|------|-------------|
| `results.png` | Training results overview plot |
| `training_results.zip` | Full YOLO training output archive |
| `training_results/` | Extracted training artifacts |
| `training_results/confusion_matrix.png` | Confusion matrix from training |
| `training_results/results.csv` | Per-epoch metrics |
| `training_results/BoxPR_curve.png` | Precision-Recall curve |
| `training_results/BoxF1_curve.png` | F1 curve |

## Confusion Matrix

Also available at `../06_Confusion_Matrix/confusion_matrix.png`.
