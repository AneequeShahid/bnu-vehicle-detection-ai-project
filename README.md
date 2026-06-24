# 🚗 BNU Vehicle Detection & Logging System

BNU Gate Monitoring is an automated vehicle inspection system built with computer vision. It detects BNU registration stickers and license plates, recognizes plate text with OCR, logs access events in SQLite, and provides a web dashboard for gate supervisors.

- Stack: YOLOv8 Nano, EasyOCR, OpenCV, SQLite, HTML/CSS/JS
- Model: custom-trained `best.pt`
- Dataset: 1,044 images (87 original images augmented with rotations, flips, and brightness adjustments)
- License: student project submission for academic evaluation

## Project structure

- `Final_Project_Report.md` — full technical report
- `AI_Presentation.pdf` — project presentation
- `source_code/` — backend detector and frontend dashboard
- `trained_models/` — trained YOLOv8 weights
- `dataset/` — dataset export from Roboflow
- `results/` — validation plots and confusion matrix
- `ui_demo/` — standalone interactive demo

## Run the backend

```shell
cd source_code/backend
python detect.py --image path/to/image.jpg
```

Drop a vehicle image into `source_code/backend/` or pass any image path. A debug OpenCV window opens, the annotated frame is saved as `last_detection.jpg`, and a JSON-style result dict is printed to the terminal. Use `--no-window` for headless inference.

## Dependencies

```shell
pip install ultralytics opencv-python easyocr sqlite3
```

## Submission package

This repository is a consolidated submission package. Original source: `samin2799-hash/bnu-vehicle-detection`.
© BNU Lahore, BS Computer Science, Session 2026
