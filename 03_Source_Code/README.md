# BNU Vehicle Detection — Source Code

AI-powered gate monitoring system that detects BNU car stickers and number plates using YOLOv8 and EasyOCR.

## Repository

- **GitHub:** https://github.com/AneequeShahid/bnu-vehicle-detection-ai-project

## Dataset

| Item | Link |
|------|------|
| Roboflow Universe (public dataset) | https://universe.roboflow.com/samis-workspace-cisly/bnu-vehicle-detector/dataset/1 |
| Local copy (YOLOv8 export) | `../04_Dataset/bnu-vehicle-detector.v1i.yolov8.zip` |
| Extracted dataset | `../04_Dataset/extracted/` |

**Classes:** `bnu_sticker`, `number_plate`  
**Size:** 87 images (62 train / 17 valid / 8 test after split)  
**License:** CC BY 4.0

## Trained Models

| Model | Description | Link |
|-------|-------------|------|
| `best.pt` (YOLOv8n) | Best weights from epoch 24 (early stopping) | [GitHub Releases](https://github.com/AneequeShahid/bnu-vehicle-detection-ai-project/releases) |
| Local copy | Included in submission package | `../05_Trained_Models/best.pt` |
| Demo copy | Used by detection script | `../07_Demo_Application/model/best.pt` |

**Training summary:** YOLOv8 Nano, 640×640, batch 16, 34 epochs (patience 10), Google Colab Tesla T4.

## Project Structure

```
03_Source_Code/
├── backend/
│   └── detect.py          # Detection + SQLite logging
├── frontend/
│   └── bnu_dashboard.html # Web monitoring dashboard
├── BNU_Vehicle_Detectionnnn.ipynb  # Colab training notebook
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Copy `best.pt` into a `model/` folder next to `backend/`:

```
model/best.pt
backend/detect.py
```

## Run Detection

```bash
cd backend
python detect.py
```

Open the dashboard in a browser:

```
frontend/bnu_dashboard.html
```

## Model Performance (validation set)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| All | 89.4% | 57.3% | 76.2% | 56.5% |
| bnu_sticker | 78.8% | 34.2% | 52.8% | 34.1% |
| number_plate | 100% | 80.3% | 99.5% | 78.9% |

## Author

Aneeque Shahid — BNU Lahore, BS Computer Science, 2026
