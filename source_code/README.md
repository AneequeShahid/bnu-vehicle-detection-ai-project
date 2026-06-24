# 🚗 BNU Vehicle Detection System — Source Code & Documentation

This directory contains the source code, database files, and dashboard code for the **BNU Vehicle Detection and Gate Security Monitoring System**.

---

## 📁 Directory Structure
```
source-code/
├── backend/
│   ├── detect.py               # Core python vehicle classification & OCR pipeline
│   └── bnu_vehicles.db         # SQLite database storing vehicle access logs
├── frontend/
│   └── bnu_dashboard.html      # Supervisor monitoring dashboard UI
├── BNU_Vehicle_Detectionnnn.ipynb # Google Colab Jupyter Notebook used for training
└── README.md                   # This file (instructions & download links)
```

---

## 🔗 Download Links for Dataset & Models

The dataset and model weights are packaged in this compiled folder. If you need to download them directly from the primary sources, use the following links:

### 📦 1. Project Dataset
* **Local Package Path:** `bnu-vehicle-detection-project/dataset/bnu-vehicle-detector.v1i.yolov8.zip` (YOLOv8 layout format)
* **Direct Repository Link:** [bnu-vehicle-detector.v1i.yolov8.zip](https://github.com/samin2799-hash/bnu-vehicle-detection/raw/main/bnu-vehicle-detector.v1i.yolov8.zip)
* **Dataset Composition:** 87 total annotated images (Train: 62, Val: 17, Test: 8) with annotations for classes `bnu_sticker` and `number_plate`.

### 🧠 2. Trained Model Weights
* **Local Package Path:** `bnu-vehicle-detection-project/models/best.pt` (PyTorch model weights format)
* **Direct Repository Link:** [best.pt (6.2 MB)](https://github.com/samin2799-hash/bnu-vehicle-detection/raw/main/best.pt)
* **Model Type:** YOLOv8 Nano (`yolov8n.pt` base). Custom-trained for 34 epochs, precision score 95.4%, plate detection mAP50 99.5%, overall mAP50 76.2%.

---

## 🚀 How to Run the Project

### ⚙️ Prerequisites & Dependencies
To run the Python detection backend, install the required libraries. Python 3.8+ is recommended:

```bash
pip install ultralytics easyocr opencv-python sqlite3 datetime
```

*Note: EasyOCR will automatically download the English model language weight file (~15MB) upon its first execution.*

---

### 💻 1. Running the Detection Pipeline
The detection script reads an input image or camera stream, runs YOLOv8 model prediction to find BNU stickers and license plates, applies EasyOCR to read license plate characters, draws labeled boxes, and logs entries into the SQLite database.

1. Ensure the model weights are located at `../models/best.pt` (configured in the code).
2. Run the detection script:
   ```bash
   cd backend
   python detect.py
   ```
3. By default, it will load the test image specified in the code, run detection, display a window with green (sticker) and orange (plate) boxes, overlay the access decision status ("ALLOW ENTRY" or "DENY ENTRY"), and write the record to `bnu_vehicles.db`.

---

### 📊 2. Launching the Supervisor Dashboard
The gate monitoring dashboard reads and displays statistics and simulated logs.
1. Navigate to the `frontend/` directory.
2. Double-click and open `bnu_dashboard.html` in any modern web browser.
3. Use the **Simulate Detection** button to trigger simulated gate vehicle entries, showing live bounding boxes, status changes, updated counters, chart metrics, and scrolling real-time log tables.