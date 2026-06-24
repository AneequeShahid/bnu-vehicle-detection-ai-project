import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI App
app = FastAPI(
    title="BNU Vehicle Detection API",
    description="Backend API for BNU Gate Vehicle Detection & Logging System",
    version="1.0.0"
)

# Enable CORS for frontend web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits local file access and local server requests
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

if __name__ == "__main__":
    # Start the server on port 8000
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
