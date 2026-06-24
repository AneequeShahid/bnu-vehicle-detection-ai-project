import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sqlite3
from datetime import datetime

# Import database config from detect.py
from detect import DEFAULT_DB_PATH

# Initialize FastAPI App
app = FastAPI(
    title="BNU Vehicle Detection API",
    description="Backend API for BNU Gate Vehicle Detection & Logging System",
    version="1.0.0"
)

# Enable CORS for frontend web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "system": "BNU Vehicle Detection & Logging System",
        "version": "1.0.0"
    }

@app.get("/api/stats")
def get_stats():
    try:
        with sqlite3.connect(DEFAULT_DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Total vehicles
            cursor.execute("SELECT COUNT(*) FROM vehicle_logs")
            total = cursor.fetchone()[0]
            
            # BNU vehicles
            cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE bnu_sticker_detected = 1")
            bnu = cursor.fetchone()[0]
            
            # Non-BNU vehicles
            cursor.execute("SELECT COUNT(*) FROM vehicle_logs WHERE bnu_sticker_detected = 0")
            non_bnu = cursor.fetchone()[0]
            
            # Last detection time
            cursor.execute("SELECT timestamp FROM vehicle_logs ORDER BY id DESC LIMIT 1")
            last_row = cursor.fetchone()
            last_time = last_row[0] if last_row else "--:--"
            
            # Format last time for display (e.g. HH:MM)
            if last_time != "--:--":
                try:
                    dt = datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S")
                    last_time = dt.strftime("%H:%M")
                except Exception:
                    pass
            
            return {
                "totalCount": total,
                "bnuCount": bnu,
                "nonBnuCount": non_bnu,
                "precision": "95.4%",  # Static model accuracy metric
                "lastTime": last_time
            }
    except sqlite3.Error as e:
        return {"error": f"Database error: {e}"}

@app.get("/api/logs")
def get_logs(filter_type: str = "all"):
    try:
        with sqlite3.connect(DEFAULT_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT plate_number, bnu_sticker_detected, confidence, timestamp, date, time FROM vehicle_logs"
            params = []
            
            if filter_type == "bnu":
                query += " WHERE bnu_sticker_detected = 1"
            elif filter_type == "non":
                query += " WHERE bnu_sticker_detected = 0"
                
            query += " ORDER BY id DESC LIMIT 50"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            logs = []
            for row in rows:
                logs.append({
                    "plate": row["plate_number"],
                    "bnu": bool(row["bnu_sticker_detected"]),
                    "conf": row["confidence"],
                    "time": row["time"],
                    "date": row["date"],
                    "timestamp": row["timestamp"]
                })
            return logs
    except sqlite3.Error as e:
        return {"error": f"Database error: {e}"}

if __name__ == "__main__":

    # Start the server on port 8000
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
