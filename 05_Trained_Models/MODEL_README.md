# Trained Models

## best.pt (YOLOv8 Nano)

| Property | Value |
|----------|-------|
| File | `best.pt` |
| Size | ~6.2 MB |
| Architecture | YOLOv8n |
| Classes | bnu_sticker, number_plate |
| Best epoch | 24 (early stop at 34) |
| Input size | 640×640 |

## Download Links

| Source | URL |
|--------|-----|
| **GitHub Releases** | https://github.com/AneequeShahid/bnu-vehicle-detection-ai-project/releases |
| **Local copy** | `best.pt` (this folder) |
| **Demo copy** | `../07_Demo_Application/model/best.pt` |

## Validation Performance

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| All | 89.4% | 57.3% | 76.2% | 56.5% |
| bnu_sticker | 78.8% | 34.2% | 52.8% | 34.1% |
| number_plate | 100% | 80.3% | 99.5% | 78.9% |

## Usage

```python
from ultralytics import YOLO
model = YOLO('best.pt')
results = model('image.jpg', conf=0.1)
```
