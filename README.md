# 🚗 BNU Vehicle Detection & Logging System — Final Submission

This folder contains the complete submission package for the **BNU Vehicle Detection & Logging System** project, consolidated and optimized from the source repository and local presentation materials.

---

## 📋 Submission Checklist & File Mapping

Here is the mapping of the required project items to the actual files in this submission folder:

| Required Deliverable | Description | File / Folder Link |
| :--- | :--- | :--- |
| **Final Project Report** | Comprehensive technical documentation detailing architecture, dataset, training, and metrics | [Final_Project_Report.md](file:///C:/Users/Aneeque/Downloads/BNU_Vehicle_Detection_Submission/BNU_Vehicle_Detection_Submission/Final_Project_Report.md) |
| **Presentation Slides** | High-quality project slides detailing system design and results | [AI_Presentation.pdf](file:///C:/Users/Aneeque/Downloads/BNU_Vehicle_Detection_Submission/BNU_Vehicle_Detection_Submission/AI_Presentation.pdf) |
| **Source Code & Project Files** | Consolidating the backend model detection and frontend dashboard templates | [source_code/](file:///C:/Users/Aneeque/Downloads/BNU_Vehicle_Detection_Submission/BNU_Vehicle_Detection_Submission/source_code) |
| **Dataset(s) used in the project** | Annotated BNU vehicles dataset in YOLOv8 layout zip | [dataset/bnu-vehicle-detector.v1i.yolov8.zip](file:///C:/Users/Aneeque/Downloads/BNU_Vehicle_Detection_Submission/BNU_Vehicle_Detection_Submission/dataset/bnu-vehicle-detector.v1i.yolov8.zip) |
| **Trained Models** | custom-trained YOLOv8 weights (`best.pt` file) | [trained_models/best.pt](file:///C:/Users/Aneeque/Downloads/BNU_Vehicle_Detection_Submission/BNU_Vehicle_Detection_Submission/trained_models/best.pt) |
| **Confusion Matrix** | Performance confusion matrix plot generated during validation runs | [results/confusion_matrix.png](file:///C:/Users/Aneeque/Downloads/BNU_Vehicle_Detection_Submission/BNU_Vehicle_Detection_Submission/results/confusion_matrix.png) |
| **User Interface / Demo App** | High-fidelity interactive web simulator & original gate supervisor dashboard | [ui_demo/](file:///C:/Users/Aneeque/Downloads/BNU_Vehicle_Detection_Submission/BNU_Vehicle_Detection_Submission/ui_demo) |
| **Supporting Materials / Results** | Validation loss curves, training stats, and complete zipped run logs | [results/](file:///C:/Users/Aneeque/Downloads/BNU_Vehicle_Detection_Submission/BNU_Vehicle_Detection_Submission/results) |

---

## 🔗 Primary Resource Links

For reference, the primary online resource links are included below:
* **Original GitHub Repository:** [samin2799-hash/bnu-vehicle-detection](https://github.com/samin2799-hash/bnu-vehicle-detection)
* **Trained Model Weights (best.pt):** [Download best.pt (6.2 MB)](https://github.com/samin2799-hash/bnu-vehicle-detection/raw/main/best.pt)
* **Annotated Roboflow Dataset Zip:** [Download bnu-vehicle-detector.v1i.yolov8.zip](https://github.com/samin2799-hash/bnu-vehicle-detection/raw/main/bnu-vehicle-detector.v1i.yolov8.zip)

---

## 🛠️ Summary of Missing Deliverables Generated

The original GitHub repository lacked several core materials required for the final project submission. These have been generated and integrated:
1. **Final Project Report (`Final_Project_Report.md`):** Generated from scratch. Details project specifications, deep learning hyperparameters, validation graphs, SQLite DB schemas, and gate decision logic.
2. **Interactive Demo Web App (`ui_demo/index.html`):** Developed a premium web application simulating the gate camera feed. Users can select vehicles, watch a scan line animation, view dynamic YOLO bounding boxes, read license plates (EasyOCR simulation), write entries to an SQLite table, and view statistical counters in real-time.
3. **Slides Organization (`AI_Presentation.pdf`):** Relocated your local PDF presentation to the root folder for streamlined presentation grading.

---

## 🚀 Execution & Quick Start Instructions

### 📊 1. Run the Interactive Web Demo Simulator
- Locate and open [ui_demo/index.html](file:///C:/Users/Aneeque/Downloads/BNU_Vehicle_Detection_Submission/BNU_Vehicle_Detection_Submission/ui_demo/index.html) in any modern web browser.
- Select a vehicle from the right panel to trigger the gate entrance simulator.
- View real-time statistics, OCR plate readings, access decisions, and scrollable log lists.

### 💻 2. Run the Python Backend Pipeline
Ensure Python 3.8+ is installed with the required libraries:
```bash
pip install ultralytics easyocr opencv-python sqlite3 datetime
```
Run the local gate verification script:
```bash
cd source_code/backend
python detect.py
```
This script runs the customized YOLOv8 model and EasyOCR text extraction, displays the annotated bounding boxes, and records access logs to the `bnu_vehicles.db` SQLite database.
