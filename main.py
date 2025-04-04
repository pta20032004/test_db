import time
import cv2
from capture import VideoCapture
from detector import Detector
from visualizer import draw_faces, draw_ui

def main():
    cap = VideoCapture(0)  # Mở camera / Open camera
    detector = Detector()  # Khởi tạo detector / Initialize detector
    prev_time = 0
    
    while cap.is_opened():
        ret, frame = cap.read_frame()
        if not ret:
            break
        
        # Xử lý khung hình / Process frame
        annotated_frame, persons, faces, face_boxes = detector.process_frame(frame)
        annotated_frame = draw_faces(annotated_frame, face_boxes)
        
        # Tính FPS / Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        # Thêm UI / Add UI elements
        annotated_frame = draw_ui(annotated_frame, persons, faces, fps)
        
        # Hiển thị / Display
        cv2.imshow("Face Tracking", annotated_frame)
        if cv2.waitKey(1) == ord('q'):  # Thoát bằng phím Q / Quit with Q
            break
    
    cap.release()  # Giải phóng camera / Release camera
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()