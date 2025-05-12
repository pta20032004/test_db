"""
Package quản lý và nhận diện khuôn mặt sử dụng ChromaDB, hỗ trợ GPU
"""

# Export các class chính để sử dụng dễ dàng
from app.face_db3.chroma_manager import ChromaFaceDB
from app.face_db3.embedding_service import FaceEmbeddingService
from app.face_db3.face_recognizer import FaceRecognizer
from app.face_db3.simple_test import register_folder, test_camera

__all__ = [
    'ChromaFaceDB', 
    'FaceEmbeddingService',
    'FaceRecognizer',
    'register_folder',
    'test_camera'
]