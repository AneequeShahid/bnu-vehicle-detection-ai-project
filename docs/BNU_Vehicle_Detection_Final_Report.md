# Final Project Report

## BNU Vehicle Detection System
### AI-Powered Car Sticker & Number Plate Recognition for Campus Gate Monitoring

---

**Author:** Aneeque Shahid  
**Institution:** Beaconhouse National University (BNU), Lahore  
**Program:** BS Computer Science  
**Year:** 2026  
**GitHub Repository:** https://github.com/AneequeShahid/bnu-vehicle-detection-ai-project

---

## Abstract

This project presents an automated vehicle monitoring system for BNU campus gates. The system uses a custom-trained YOLOv8 object detection model to identify BNU parking stickers and vehicle number plates in real time. Detected plates are read using EasyOCR, and all events are logged to a SQLite database. A web-based dashboard provides live monitoring and historical logs. The model was trained on 87 annotated images from a custom Roboflow dataset and achieves 76.2% mAP50 on the validation set, with strong number-plate detection performance (99.5% mAP50).

---

## 1. Introduction

### 1.1 Background

University campuses require controlled vehicle access. Manual gate checks are slow and error-prone. Computer vision can automate verification by detecting official BNU stickers and reading license plates.

### 1.2 Problem Statement

Design and implement a system that:

1. Detects BNU parking stickers on vehicles entering campus gates.
2. Reads number plates from detected regions.
3. Classifies vehicles as BNU (allow) or non-BNU (deny).
4. Logs all detections with timestamps for audit and reporting.

### 1.3 Objectives

- Collect and annotate a domain-specific dataset of BNU vehicles.
- Train a lightweight YOLOv8 model suitable for real-time inference.
- Integrate OCR for plate text extraction.
- Build a demo application with database logging and a web dashboard.

---

## 2. Literature Review

Object detection has advanced rapidly with single-stage detectors such as YOLO (You Only Look Once). YOLOv8 (Ultralytics, 2023) offers a strong balance of speed and accuracy for edge deployment. For license plate reading, OCR engines such as EasyOCR complement detection models by extracting text from cropped plate regions. Similar campus and parking systems combine detection, OCR, and database backends for access control.

---

## 3. Methodology

### 3.1 Dataset

| Property | Value |
|----------|-------|
| Source | Roboflow (custom annotation) |
| Project | bnu-vehicle-detector |
| URL | https://universe.roboflow.com/samis-workspace-cisly/bnu-vehicle-detector/dataset/1 |
| Format | YOLOv8 |
| Total images | 87 |
| Classes | `bnu_sticker`, `number_plate` |
| Split | 62 train / 17 valid / 8 test |
| License | CC BY 4.0 |

Images were captured at BNU campus gates using mobile phones. Annotations were created in Roboflow with bounding boxes around stickers and plates.

### 3.2 Model Architecture

- **Base model:** YOLOv8 Nano (`yolov8n.pt`)
- **Parameters:** ~3.0M
- **Input size:** 640×640
- **Output classes:** 2 (bnu_sticker, number_plate)

### 3.3 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 50 (early stop at 34) |
| Best epoch | 24 |
| Batch size | 16 |
| Optimizer | AdamW (auto) |
| Patience | 10 |
| GPU | Tesla T4 (Google Colab) |
| Augmentation | RandAugment, mosaic, flip |

Training was performed in Google Colab using the notebook `BNU_Vehicle_Detectionnnn.ipynb`.

### 3.4 Inference Pipeline

1. Load image or camera frame.
2. Run YOLOv8 inference (`best.pt`).
3. For each `number_plate` detection, crop region and run EasyOCR.
4. Check for `bnu_sticker` detection.
5. Assign status: **ALLOW** (sticker found) or **DENY** (no sticker).
6. Log result to SQLite (`bnu_vehicles.db`).
7. Display annotated frame or update dashboard.

---

## 4. Implementation

### 4.1 Technology Stack

| Component | Technology |
|-----------|------------|
| Detection | YOLOv8 (Ultralytics) |
| OCR | EasyOCR |
| Image processing | OpenCV |
| Database | SQLite |
| Backend | Python 3 |
| Frontend | HTML, CSS, JavaScript |
| Training | Google Colab |

### 4.2 System Components

**Backend (`backend/detect.py`)**  
Runs detection on input images, performs OCR, writes to SQLite, and displays annotated output.

**Frontend (`frontend/bnu_dashboard.html`)**  
Web dashboard with live stats, simulated detection demo, vehicle logs, and hourly activity chart.

**Database schema**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| plate_number | TEXT | OCR result |
| bnu_sticker_detected | INTEGER | 1 = yes, 0 = no |
| confidence | REAL | Detection confidence |
| timestamp | TEXT | Full datetime |
| date | TEXT | Date only |
| time | TEXT | Time only |

---

## 5. Results

### 5.1 Validation Metrics (best.pt, epoch 24)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| **All** | **89.4%** | **57.3%** | **76.2%** | **56.5%** |
| bnu_sticker | 78.8% | 34.2% | 52.8% | 34.1% |
| number_plate | 100% | 80.3% | 99.5% | 78.9% |

### 5.2 Key Observations

- **Number plate detection** performs very well (99.5% mAP50, 100% precision).
- **BNU sticker detection** has lower recall (34.2%), likely due to small sticker size, occlusion, and limited training data.
- Training converged at epoch 24; early stopping triggered at epoch 34.
- Inference speed: ~3–30 ms per image on GPU.

### 5.3 Confusion Matrix

The confusion matrix (see `results/confusion_matrix.png`) shows:

- 1 true positive for `bnu_sticker`
- 0 true positives for `number_plate` in matrix view (validation artifact at default confidence)
- Most misses classified as background

This indicates room for improvement via more training data, lower confidence threshold, and data augmentation focused on stickers.

### 5.4 Training Curves

Full training curves, PR curves, and F1 curves are in `results/training_results/`.

---

## 6. Demo Application

The demo package includes:

- Python detection backend (`backend/`)
- Web dashboard UI (`frontend/`)
- Pre-trained `best.pt` model (stored in `trained_models/` after download)
- Sample database logging (`backend/bnu_vehicles.db`)

**To run:**

```bash
pip install -r requirements.txt
python backend/detect.py
```

Open `frontend/index.html` or `frontend/bnu_dashboard.html` in a browser.

---

## 7. Limitations

1. Small dataset (87 images) limits generalization.
2. Sticker recall is low; more diverse angles and lighting needed.
3. OCR accuracy depends on plate image quality.
4. Dashboard uses simulated data; full live-camera integration requires additional API work.
5. Model not optimized for mobile/edge deployment (TensorRT, ONNX).

---

## 8. Future Work

- Expand dataset to 500+ images with varied conditions.
- Fine-tune confidence thresholds per class.
- Add REST API between backend and dashboard for real-time updates.
- Deploy on Raspberry Pi or NVIDIA Jetson at physical gates.
- Add face-blur and privacy compliance for stored images.
- Integrate with university access control systems.

---

## 9. Conclusion

The BNU Vehicle Detection System successfully demonstrates an end-to-end computer vision pipeline for campus gate monitoring. YOLOv8 effectively detects number plates with high accuracy, while sticker detection requires further data collection. The project delivers a working prototype with training artifacts, trained weights, evaluation metrics, and a user-facing demo — providing a solid foundation for production deployment at BNU.

---

## 10. References

1. Ultralytics. (2023). *YOLOv8 Documentation.* https://docs.ultralytics.com/
2. JaidedAI. *EasyOCR.* https://github.com/JaidedAI/EasyOCR
3. Roboflow. *BNU Vehicle Detector Dataset.* https://universe.roboflow.com/samis-workspace-cisly/bnu-vehicle-detector/dataset/1
4. Redmon, J., et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection.*

---

## Appendix A — File Inventory

| File | Location |
|------|----------|
| Source code | `backend/`, `frontend/`, `scripts/` |
| Dataset | `dataset/` |
| Model weights | `trained_models/` |
| Confusion matrix | `results/confusion_matrix.png` |
| Demo app | `frontend/` |
| Training results | `results/` |

---

*Report generated for final project submission — June 2026*
