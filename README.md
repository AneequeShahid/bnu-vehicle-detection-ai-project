# 🚗 BNU Vehicle Detection & Logging System

BNU Gate Monitoring is an automated vehicle inspection system built with computer vision. It detects BNU registration stickers and license plates, recognizes plate text with OCR, logs access events in SQLite, and provides a web dashboard for gate supervisors.

- Stack: YOLOv8 Nano, EasyOCR, OpenCV, SQLite, HTML/CSS/JS
- Model: custom-trained `best.pt`
- Dataset: 2,697 images (174 original images augmented with rotations, flips, brightness, noise, and blur)
- License: student project submission for academic evaluation

## Project structure

- `Final_Project_Report.md` — full technical report
- `AI_Presentation.pdf` — project presentation
- `source_code/` — backend server, detector pipeline, and frontend dashboard
- `trained_models/` — trained YOLOv8 weights
- `dataset/` — dataset export from Roboflow
- `results/` — validation plots and confusion matrix
- `ui_demo/` — standalone offline interactive demo

## Run the system

### 1. Start the API backend server:
```shell
cd source_code/backend
python server.py
```
This launches a FastAPI server on `http://127.0.0.1:8000` to handle image detection requests and query logs from the SQLite database.

### 2. Launch the frontend dashboard:
- Simply open the [bnu_dashboard.html](file:///C:/Users/Aneeque/Downloads/BNU_Vehicle_Detection_Submission/BNU_Vehicle_Detection_Submission/source_code/frontend/bnu_dashboard.html) dashboard in any web browser.
- The dashboard automatically connects to the local API server. You can upload vehicle images, view real-time prediction overlays, see logged entry records, and browse summary statistics.

### 3. Run standalone command line detection:
```shell
cd source_code/backend
python detect.py --image path/to/image.jpg
```
Use `--no-window` for headless execution.

## Dependencies

```shell
pip install ultralytics opencv-python easyocr sqlite3 fastapi uvicorn python-multipart
```

## Submission package

This repository is a consolidated submission package. Original source: `samin2799-hash/bnu-vehicle-detection`.
© BNU Lahore, BS Computer Science, Session 2026
