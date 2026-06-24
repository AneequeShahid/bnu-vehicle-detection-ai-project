import cv2
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import easyocr

# ============ CONFIG ============
MODEL_PATH = '../trained_models/best.pt'
DB_PATH = 'bnu_vehicles.db'
CONFIDENCE = 0.1

# ============ INIT ============
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)

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
    conn.close()

def log_vehicle(plate, bnu, conf):
    conn = sqlite3.connect(DB_PATH)
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
    conn.close()

# ============ DETECTION ============
def detect(image_path):
    image = cv2.imread(image_path)
    results = model(image, conf=CONFIDENCE)[0]

    plate_text = 'NOT DETECTED'
    bnu_sticker = False
    max_conf = 0.0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        if conf > max_conf:
            max_conf = conf

        if label == 'number_plate':
            cropped = image[y1:y2, x1:x2]
            ocr_result = reader.readtext(cropped)
            if ocr_result:
                plate_text = ' '.join([r[1] for r in ocr_result]).upper()

        elif label == 'bnu_sticker':
            bnu_sticker = True

        color = (0, 255, 0) if label == 'bnu_sticker' else (255, 165, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{label} {conf:.2f}',
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Status overlay
    status = 'BNU VEHICLE - ALLOW' if bnu_sticker else 'NOT BNU - DENY'
    color = (0, 255, 0) if bnu_sticker else (0, 0, 255)
    cv2.putText(image, status, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(image, timestamp, (10, image.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    log_vehicle(plate_text, bnu_sticker, max_conf)

    print(f'Plate: {plate_text}')
    print(f'BNU: {"YES" if bnu_sticker else "NO"}')
    print(f'Confidence: {max_conf:.2f}')
    print(f'Time: {timestamp}')

    cv2.imshow('BNU Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ============ MAIN ============
if __name__ == '__main__':
    init_db()
    # Test image
    detect('test_image.jpg')
