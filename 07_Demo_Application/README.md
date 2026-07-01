# BNU Vehicle Detection — Demo Application

## Contents

- `backend/detect.py` — Python detection + SQLite logging
- `frontend/bnu_dashboard.html` — Web monitoring dashboard
- `model/best.pt` — Trained YOLOv8 weights
- `bnu_vehicles.db` — Sample detection database

## Quick Start

```bash
pip install ultralytics easyocr opencv-python
cd backend
python detect.py
```

Open `frontend/bnu_dashboard.html` in your browser for the dashboard demo.

## Links

- **Dataset:** https://universe.roboflow.com/samis-workspace-cisly/bnu-vehicle-detector/dataset/1
- **Model on GitHub:** https://github.com/AneequeShahid/bnu-vehicle-detection-ai-project/releases
- **Source repo:** https://github.com/AneequeShahid/bnu-vehicle-detection-ai-project
