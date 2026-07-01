# Trained Models

Model weights are not committed to this repository due to file size.

## Download

Download `best.pt` from the [GitHub Releases](../../releases) page.

Place it in this directory before running the backend server.

## Model Info
- Architecture: YOLOv8n (nano)
- Classes: `vehicle_sticker`, `number_plate`
- mAP50: 99.5% (number plates), 52.8% (stickers)
- Training epochs: 50
- Dataset size: Custom — BNU campus vehicles (annotated via Roboflow)
