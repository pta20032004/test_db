import numpy as np
from models import load_models

class Detector:
    """Xử lý nhận diện người và khuôn mặt / Detection handler"""
    def __init__(self):
        self.person_model, self.face_model = load_models()  # Tải mô hình / Load models
    
    def process_frame(self, frame):
        """Xử lý khung hình và trả về kết quả / Process frame"""
        # Nhận diện người / Detect persons
        person_results = self.person_model(frame, classes=[0], conf=0.5, half=False, verbose=False)
        
        person_count = 0
        face_count = 0
        annotated_frame = frame.copy()
        face_boxes = []  # Danh sách khung khuôn mặt / Face boxes list
        
        if person_results:
            for person in person_results:
                # Lấy tọa độ khung người / Get person boxes
                boxes = person.boxes.xyxy.cpu().numpy()
                person_count += len(boxes)
                annotated_frame = person.plot()  # Vẽ khung lên ảnh / Draw boxes
                
                # Chuẩn bị vùng quan tâm (ROI) / Prepare ROIs
                valid_rois, valid_indices = self._prepare_rois(frame, boxes)
                # Nhận diện khuôn mặt trong ROI / Detect faces in ROIs
                face_data = self._detect_faces(valid_rois, valid_indices, boxes)
                face_count += face_data["count"]
                face_boxes.extend(face_data["boxes"])
        
        return annotated_frame, person_count, face_count, face_boxes
    
    def _prepare_rois(self, frame, boxes):
        """Chuẩn bị các vùng ảnh hợp lệ từ khung người / Extract valid ROIs"""
        valid_rois = []
        valid_indices = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if (x2 > x1) and (y2 > y1) and (x1 >= 0) and (y1 >= 0):
                valid_rois.append(frame[y1:y2, x1:x2])  # Cắt vùng ảnh / Crop region
                valid_indices.append(i)
        return valid_rois, valid_indices
    
    def _detect_faces(self, rois, indices, boxes):
        """Nhận diện khuôn mặt và tính toán tọa độ toàn cục / Detect faces"""
        face_boxes = []
        face_count = 0
        if rois:
            face_results = self.face_model(rois, stream=True, conf=0.5, imgsz=160, half=False, verbose=False)
            for idx, face in enumerate(face_results):
                if len(face.boxes) > 0:
                    face_count += len(face.boxes)
                    orig_idx = indices[idx]
                    px1, py1 = int(boxes[orig_idx][0]), int(boxes[orig_idx][1])
                    # Chuyển tọa độ cục bộ sang toàn cục / Convert to global coordinates
                    for box in face.boxes:
                        fx1, fy1, fx2, fy2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        global_coords = (px1 + fx1, py1 + fy1, px1 + fx2, py1 + fy2)
                        face_boxes.append((global_coords, conf))
        return {"count": face_count, "boxes": face_boxes}