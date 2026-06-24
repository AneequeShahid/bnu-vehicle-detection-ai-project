import cv2
import os
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import easyocr
import re

# ============ CONFIG ============
# Use absolute path based on this script's location so it works when run directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SCRIPT_DIR, '..', '..', 'trained_models', 'best.pt')
DEFAULT_DB_PATH = os.path.join(_SCRIPT_DIR, 'bnu_vehicles.db')
CONFIDENCE = 0.5
TEST_IMAGE = os.path.join(_SCRIPT_DIR, 'test_image.jpg')
DEFAULT_TIMEOUT_MS = 10000

# ============ INIT ============
print(f"[INFO] Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)
print("[INFO] Model and OCR ready.")

# ============ DATABASE ============
def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
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
    except sqlite3.Error as e:
        print(f"[ERROR] Database initialization failed: {e}")

def log_vehicle(plate, bnu, conf):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
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
    except sqlite3.Error as e:
        print(f"[ERROR] Failed to log vehicle entry: {e}")

def clean_plate_text(text):
    if not text:
        return 'NOT DETECTED'
    # Remove special characters, punctuation, and extra whitespace
    cleaned = re.sub(r'[^A-Z0-9\s-]', '', text.upper())
    # Strip leading/trailing whitespaces and collapse multiple spaces into a single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned:
        return 'NOT DETECTED'
    return cleaned

# ============ DETECTION ============
def detect(image_path, confidence=CONFIDENCE, timeout=10000):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    results = model(image, conf=confidence)[0]

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
                    raw_text = ' '.join(ocr_result)
                    plate_text = clean_plate_text(raw_text)
                    plate_conf = conf

        elif label == 'bnu_sticker':
            if conf >= CONFIDENCE:
                bnu_sticker = True

        color = (0, 255, 0) if label == 'bnu_sticker' else (255, 165, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Bounding box label tag background block
        txt = f'{label} {conf:.2f}'
        (txt_w, txt_h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Handle label position near image edges
        label_y = max(y1, txt_h + 10)
        cv2.rectangle(image, (x1, label_y - txt_h - 10), (x1 + txt_w + 10, label_y), color, -1)
        cv2.putText(image, txt, (x1 + 5, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if label == 'bnu_sticker' else (255, 255, 255), 1, cv2.LINE_AA)

    # Use highest detected confidence, fallback to plate_conf, then 0
    final_conf = max_conf if max_conf > 0 else plate_conf

    # Status overlay
    status = 'BNU VEHICLE - ALLOW' if bnu_sticker else 'NOT BNU - DENY'
    color = (0, 255, 0) if bnu_sticker else (0, 0, 255)
    
    # Draw background box for text readability
    (tw, th), baseline = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(image, (10, 10), (10 + tw + 20, 10 + th + 20), (0, 0, 0), -1)
    cv2.putText(image, status, (20, 15 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(image, timestamp, (10, image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    log_vehicle(plate_text, bnu_sticker, final_conf)

    print(f"Image      : {image_path}")
    print(f"Plate      : {plate_text}")
    print(f"BNU        : {'YES' if bnu_sticker else 'NO'}")
    print(f"Confidence : {final_conf:.2f}")
    print(f"Time       : {timestamp}")

    cv2.namedWindow('BNU Detection', cv2.WINDOW_NORMAL)
    cv2.imshow('BNU Detection', image)
    print(f"[INFO] OpenCV window opened. Press any key on the image window to close it, or it will close automatically in {timeout/1000:.0f} seconds...")
    cv2.waitKey(timeout)
    cv2.destroyAllWindows()

# ============ MAIN ============
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="BNU Vehicle Detection & Logging Pipeline")
    parser.add_argument('--image', type=str, default=TEST_IMAGE, help='Path to input image')
    parser.add_argument('--conf', type=float, default=CONFIDENCE, help='Confidence threshold')
    parser.add_argument('--timeout', type=int, default=10000, help='OpenCV window timeout in ms')
    args = parser.parse_args()

    init_db()
    detect(args.image, confidence=args.conf, timeout=args.timeout)
