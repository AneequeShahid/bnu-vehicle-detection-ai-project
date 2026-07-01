import os

import cv2
from datetime import datetime
from ultralytics import YOLO
import easyocr
import re
import sqlite3

# ============ CONFIG ============
# Use absolute path based on this script's location so it works when run directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SCRIPT_DIR, "..", "trained_models", "best.pt")
DEFAULT_DB_PATH = os.path.join(_SCRIPT_DIR, "bnu_vehicles.db")
CONFIDENCE = 0.5
TEST_IMAGE = os.path.join(_SCRIPT_DIR, "test_image.jpg")
DEFAULT_TIMEOUT_MS = 10000


def _load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found: {path}")
    return YOLO(path)


model = _load_model(MODEL_PATH)
reader = easyocr.Reader(["en"], gpu=False)


# ============ DATABASE ============
def init_db():
    try:
        with sqlite3.connect(DEFAULT_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicle_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT,
                    bnu_sticker_detected INTEGER,
                    confidence REAL,
                    timestamp TEXT,
                    date TEXT,
                    time TEXT
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON vehicle_logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bnu ON vehicle_logs(bnu_sticker_detected)")
            conn.commit()
    except sqlite3.Error as e:
        print(f"[ERROR] Database initialization failed: {e}")


def log_vehicle(plate, bnu, conf):
    try:
        with sqlite3.connect(DEFAULT_DB_PATH) as conn:
            cursor = conn.cursor()
            now = datetime.now()
            cursor.execute(
                """
                INSERT INTO vehicle_logs
                (plate_number, bnu_sticker_detected, confidence, timestamp, date, time)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    plate,
                    1 if bnu else 0,
                    conf,
                    now.strftime("%Y-%m-%d %H:%M:%S"),
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                ),
            )
            conn.commit()
    except sqlite3.Error as e:
        print(f"[ERROR] Failed to log vehicle entry: {e}")


def clean_plate_text(text):
    if not text:
        return "NOT DETECTED"
    cleaned = re.sub(r"[^A-Z0-9\s-]", "", text.upper())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "NOT DETECTED"
    return cleaned


# ============ DETECTION ============
def detect(image_path, confidence=CONFIDENCE, timeout_ms=DEFAULT_TIMEOUT_MS, show_window=True):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    results = model(image, conf=float(confidence), verbose=False)[0]

    plate_text = "NOT DETECTED"
    bnu_sticker = False
    plate_conf = 0.0
    sticker_conf = 0.0
    max_conf = 0.0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)

        if conf > max_conf:
            max_conf = conf

        if label == "number_plate":
            cropped = image[y1:y2, x1:x2]
            if cropped.size > 0:
                try:
                    ocr_result = reader.readtext(cropped, detail=0)
                except Exception:
                    ocr_result = []
                if ocr_result:
                    raw_text = " ".join(ocr_result)
                    plate_text = clean_plate_text(raw_text)
                    plate_conf = conf

        elif label == "bnu_sticker":
            sticker_conf = conf
            if conf >= confidence:
                bnu_sticker = True

        color = (0, 255, 0) if label == "bnu_sticker" else (255, 165, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    final_conf = max_conf if max_conf > 0 else plate_conf

    status = "BNU VEHICLE - ALLOW" if bnu_sticker else "NOT BNU - DENY"
    color = (0, 255, 0) if bnu_sticker else (0, 0, 255)
    cv2.putText(
        image,
        status,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        image,
        timestamp,
        (10, image.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    log_vehicle(plate_text, bnu_sticker, final_conf)

    print(f"Image      : {image_path}")
    print(f"Plate      : {plate_text}")
    print(f"BNU        : {'YES' if bnu_sticker else 'NO'}")
    print(f"Confidence : {final_conf:.2f}")
    print(f"Time       : {timestamp}")

    if show_window:
        cv2.imshow("BNU Detection", image)
        print(
            f"[INFO] Press any key in the image window to close, or it will close automatically in {int(timeout_ms/1000)}s..."
        )
        cv2.waitKey(int(timeout_ms))
        cv2.destroyAllWindows()

    try:
        cv2.imwrite("last_detection.jpg", image)
        print("[INFO] Saved annotated frame to last_detection.jpg")
    except Exception:
        pass

    return {
        "plate_number": plate_text,
        "bnu_sticker_detected": bnu_sticker,
        "confidence": final_conf,
        "timestamp": timestamp,
        "sticker_confidence": sticker_conf,
        "plate_confidence": plate_conf,
    }


# ============ MAIN ============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BNU Vehicle Detection & Logging Pipeline")
    parser.add_argument("--image", type=str, default=TEST_IMAGE, help="Path to input image")
    parser.add_argument("--conf", type=float, default=CONFIDENCE, help="Confidence threshold")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help="OpenCV window timeout in ms",
    )
    parser.add_argument("--no-window", action="store_true", help="Run without opening OpenCV window")
    args = parser.parse_args()

    init_db()
    result = detect(
        args.image,
        confidence=args.conf,
        timeout_ms=args.timeout,
        show_window=not args.no_window,
    )
    print(f"[RESULT] {result}")
