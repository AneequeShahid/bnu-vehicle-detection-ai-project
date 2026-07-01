# BNU Vehicle Detection — Final Project Submission

**Author:** Aneeque Shahid
**Institution:** Beaconhouse National University (BNU), Lahore
**Program:** BS Computer Science, 2026

---

## Submission Checklist

| # | Deliverable | Location | Status |
|---|-------------|----------|--------|
| 1 | Final project report | `01_Final_Project_Report/BNU_Vehicle_Detection_Final_Report.md` | Included |
| 2 | Presentation slides | `02_Presentation_Slides/AI_Presentation.pdf` | Included |
| 3 | Source code & project files | `03_Source_Code/` | Included |
| 4 | Dataset | `04_Dataset/` | Included |
| 5 | Trained models | `05_Trained_Models/best.pt` | Included |
| 6 | Confusion matrix | `06_Confusion_Matrix/` | Included |
| 7 | Demo application | `07_Demo_Application/` | Included |
| 8 | Documentation & results | `08_Documentation_and_Results/` | Included |

---

## Folder Structure

```
BNU_Vehicle_Detection_Submission/
├── 01_Final_Project_Report/
│   └── BNU_Vehicle_Detection_Final_Report.md
├── 02_Presentation_Slides/
│   ├── AI_Presentation.pdf          ← official presentation
│   └── BNU_Vehicle_Detection_Presentation.html  (backup)
├── 03_Source_Code/
│   ├── backend/detect.py
│   ├── frontend/bnu_dashboard.html
│   ├── model/best.pt
│   ├── BNU_Vehicle_Detectionnnn.ipynb
│   ├── requirements.txt
│   └── README.md
├── 04_Dataset/
│   ├── bnu-vehicle-detector.v1i.yolov8.zip
│   ├── extracted/
│   └── DATASET_README.md
├── 05_Trained_Models/
│   ├── best.pt
│   └── MODEL_README.md
├── 06_Confusion_Matrix/
│   ├── confusion_matrix.png
│   └── README.md
├── 07_Demo_Application/
│   ├── backend/, frontend/, model/, bnu_vehicles.db
│   └── README.md
├── 08_Documentation_and_Results/
│   ├── results.png, training_results.zip
│   ├── training_results/ (curves, CSV, plots)
│   └── RESULTS_SUMMARY.md
└── README.md  (this file)
```

---

## How to Run the Demo

```bash
pip install -r 03_Source_Code/requirements.txt
cd 07_Demo_Application/backend
python detect.py
```

Open `07_Demo_Application/frontend/bnu_dashboard.html` in a browser.

---

## Model Performance Summary

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| All | 89.4% | 57.3% | 76.2% |
| bnu_sticker | 78.8% | 34.2% | 52.8% |
| number_plate | 100% | 80.3% | 99.5% |

---

*Package compiled for final project submission — June 2026*
