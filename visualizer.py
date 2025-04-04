import cv2
from config import FACE_COLOR, TEXT_SCALE, TEXT_THICKNESS

def draw_faces(annotated_frame, face_boxes):
    """Vẽ khung và nhãn khuôn mặt / Draw face boxes"""
    for (coords, conf) in face_boxes:
        x1, y1, x2, y2 = coords
        # Vẽ khung / Draw rectangle
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), FACE_COLOR, 2)
        # Tạo nhãn / Create label
        label = f"face {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, 2)
        # Vẽ nền nhãn / Draw label background
        cv2.rectangle(annotated_frame, (x1, y1 - th -5), (x1 + tw, y1), FACE_COLOR, -1)
        # Vẽ chữ / Draw text
        cv2.putText(annotated_frame, label, (x1, y1 -5), 
                    cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (255,255,255), TEXT_THICKNESS)
    return annotated_frame

def draw_ui(annotated_frame, person_count, face_count, fps):
    """Hiển thị UI thống kê / Draw UI elements"""
    y_offset = 30
    # Hiển thị số người / Persons count
    cv2.putText(annotated_frame, f"Persons: {person_count}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), TEXT_THICKNESS)
    # Hiển thị số khuôn mặt / Faces count
    cv2.putText(annotated_frame, f"Faces: {face_count}", (10, y_offset*2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    # Hiển thị FPS / FPS counter
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", 
                (annotated_frame.shape[1] - 200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0,255,255), TEXT_THICKNESS)
    return annotated_frame