# BNU Vehicle Detector Dataset

## Download Links

| Source | URL |
|--------|-----|
| **Roboflow Universe** | https://universe.roboflow.com/samis-workspace-cisly/bnu-vehicle-detector/dataset/1 |
| **Local ZIP (YOLOv8)** | `bnu-vehicle-detector.v1i.yolov8.zip` |
| **Extracted files** | `extracted/` |

## Dataset Info

| Property | Value |
|----------|-------|
| Workspace | samis-workspace-cisly |
| Project | bnu-vehicle-detector |
| Version | 1 |
| License | CC BY 4.0 |
| Format | YOLOv8 |
| Classes | `bnu_sticker`, `number_plate` |
| Total images | 87 |

## Class Descriptions

- **bnu_sticker** — Official BNU parking sticker affixed to vehicle windshield
- **number_plate** — Vehicle license plate (Pakistani format)

## Folder Structure (after extraction)

```
extracted/
├── data.yaml
├── train/images/
├── train/labels/
├── valid/images/
├── valid/labels/
├── test/images/
└── test/labels/
```

## Usage

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='extracted/data.yaml', epochs=50, imgsz=640)
```
