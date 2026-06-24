import cv2
import os
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import easyocr

# ============ CONFIG ============
# Use absolute path based on this script's location so it works when run directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SCRIPT_DIR, '..', '..', 'trained_models', 'best.pt')
DB_PATH = os.path.join(_SCRIPT_DIR, 'bnu_vehicles.db')
CONFIDENCE = 0.5
TEST_IMAGE = os.path.join(_SCRIPT_DIR, 'test_image.jpg')

# ============ INIT ============
print(f"[INFO] Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)
print("[INFO] Model and OCR ready.")

# ============ DATABASE ============
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT,
            bnu_sticker_detected INTEGER,
            confidence REAL,
            timestamp TEXT,
            date TEXT,
            time TEXT
        )
    ''')
    conn.commit()
    return conn, cursor

def log_vehicle(conn, cursor, plate, bnu, conf):
    now = datetime.now()
    cursor.execute('''
        INSERT INTO vehicle_logs
        (plate_number, bnu_sticker_detected, confidence, timestamp, date, time)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (plate, 1 if bnu else 0, conf,
          now.strftime('%Y-%m-%d %H:%M:%S'),
          now.strftime('%Y-%m-%d'),
          now.strftime('%H:%M:%S')))
    conn.commit()

# ============ DETECTION ============
def detect(image_path):
    conn, cursor = init_db()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    results = model(image, conf=CONFIDENCE)[0]

    plate_text = 'NOT DETECTED'
    bnu_sticker = False
    plate_conf = 0.0
    max_conf = 0.0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)

        if conf > max_conf:
            max_conf = conf

        if label == 'number_plate':
            # Crop plate region for OCR
            cropped = image[y1:y2, x1:x2]
            if cropped.size > 0:
                try:
                    ocr_result = reader.readtext(cropped, detail=0)
                except Exception:
                    ocr_result = []
                if ocr_result:
                    plate_text = ' '.join(ocr_result).upper()
                    plate_conf = conf

        elif label == 'bnu_sticker':
            if conf >= CONFIDENCE:
                bnu_sticker = True

        color = (0, 255, 0) if label == 'bnu_sticker' else (255, 165, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{label} {conf:.2f}',
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Use highest detected confidence, fallback to plate_conf, then 0
    final_conf = max_conf if max_conf > 0 else plate_conf

    # Status overlay
    status = 'BNU VEHICLE - ALLOW' if bnu_sticker else 'NOT BNU - DENY'
    color = (0, 255, 0) if bnu_sticker else (0, 0, 255)
    cv2.putText(image, status, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(image, timestamp, (10, image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    log_vehicle(conn, cursor, plate_text, bnu_sticker, final_conf)

    print(f"Image      : {image_path}")
    print(f"Plate      : {plate_text}")
    print(f"BNU        : {'YES' if bnu_sticker else 'NO'}")
    print(f"Confidence : {final_conf:.2f}")
    print(f"Time       : {timestamp}")

    cv2.imshow('BNU Detection', image)
    print("[INFO] OpenCV window opened. Press any key on the image window to close it, or it will close automatically in 10 seconds...")
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    conn.close()

# ============ MAIN ============
if __name__ == '__main__':
    init_db()
    # Test image
    detect(TEST_IMAGE)
