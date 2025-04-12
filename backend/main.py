import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.box_detector import Detector
import base64

app = FastAPI()
detector = Detector()

# CORS settings để frontend kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    # Convert image to numpy array
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process frame (chỉ lấy metadata, không vẽ khung)
    person_count, face_count, person_boxes, face_boxes = detector.process_frame(frame)
    
    # Trả về tọa độ và metadata
    return {
        "persons": person_count,
        "faces": face_count,
        "person_boxes": [
            {"coords": coords, "confidence": conf}
            for (coords, conf) in person_boxes
        ],
        "face_boxes": [
            {"coords": coords, "confidence": conf}
            for (coords, conf) in face_boxes
        ]
    }
