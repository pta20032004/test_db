from ultralytics import YOLO
from .config import PERSON_MODEL_PATH, FACE_MODEL_PATH

def load_models():
    """Tải mô hình YOLO cho nhận diện người và khuôn mặt / Load YOLO models"""
    person_model = YOLO(PERSON_MODEL_PATH)   # Mô hình nhận diện người / Person model
    face_model = YOLO(FACE_MODEL_PATH)       # Mô hình nhận diện khuôn mặt / Face model
    return person_model, face_model