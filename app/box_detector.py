import numpy as np
from .models import load_models

class Detector:
    """Xử lý nhận diện người và khuôn mặt / Detection handler"""
    def __init__(self):
        """
        Khởi tạo Detector bằng cách tải các mô hình YOLO.
        Initializes the Detector by loading the YOLO models.

        Đây là hàm khởi tạo của lớp Detector. Nó tải hai mô hình YOLO: một cho việc nhận diện người và một cho việc nhận diện khuôn mặt.
        This is the constructor of the Detector class. It loads two YOLO models: one for person detection and one for face detection.
        """
        self.person_model, self.face_model = load_models()  # Tải mô hình / Load models
    
    def process_frame(self, frame):
        """
        Xử lý khung hình và trả về kết quả nhận diện.
        Processes a frame and returns the detection results.

        Args:
            frame (np.ndarray): Khung hình đầu vào (ảnh). Input frame (image).

        Returns:
            tuple: Một tuple chứa:
                   - annotated_frame (np.ndarray): Khung hình đã được vẽ các khung nhận diện. Annotated frame with bounding boxes.
                   - person_count (int): Số lượng người được nhận diện. Number of detected persons.
                   - face_count (int): Số lượng khuôn mặt được nhận diện. Number of detected faces.
                   - face_boxes (list): Danh sách các khung khuôn mặt, mỗi khung là một tuple (coords, conf). List of face bounding boxes, each box is a tuple (coords, conf).
            
        Đây là hàm xử lý chính của lớp Detector. Nó nhận một khung hình (ảnh) làm đầu vào, thực hiện nhận diện người và khuôn mặt, và trả về kết quả.
        This is the main processing function of the Detector class. It takes a frame (image) as input, performs person and face detection, and returns the results.
        
        Thuật toán:
        1. Nhận diện người bằng mô hình person_model.
        2. Với mỗi người được nhận diện, vẽ khung lên ảnh.
        3. Chuẩn bị các vùng quan tâm (ROI) từ khung người.
        4. Nhận diện khuôn mặt trong các ROI bằng mô hình face_model.
        5. Chuyển đổi tọa độ khuôn mặt từ cục bộ (trong ROI) sang toàn cục (trong khung hình).
        6. Trả về số lượng người, số lượng khuôn mặt và danh sách các khung khuôn mặt.
        Algorithm:
        1. Detect persons using the person_model.
        2. For each detected person, draw a bounding box on the image.
        3. Prepare regions of interest (ROIs) from the person bounding boxes.
        4. Detect faces within the ROIs using the face_model.
        5. Convert face coordinates from local (within the ROI) to global (within the frame).
        6. Return the number of persons, the number of faces, and the list of face bounding boxes.
        """
        # Nhận diện người / Detect persons
        person_results = self.person_model(frame, classes=[0], conf=0.3, imgsz=640, half=True, verbose=False)
        
        person_count = 0
        face_count = 0
        annotated_frame = frame.copy()
        face_boxes = []  # Danh sách khung khuôn mặt / Face boxes list
        person_boxes = []
        
        if person_results:
            for person in person_results:
                # Lấy tọa độ khung người / Get person boxes
                boxes = person.boxes.xyxy.cpu().numpy() #boxes is a numpy array with shape (n,4)
                confs = person.boxes.conf.cpu().numpy() #confidence for person box
                person_count += len(boxes)

                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    person_boxes.append(((x1, y1, x2, y2), float(conf)))
                
                # Chuẩn bị vùng quan tâm (ROI) / Prepare ROIs
                valid_rois, valid_indices = self._prepare_rois(frame, boxes)
                # Nhận diện khuôn mặt trong ROI / Detect faces in ROIs
                face_data = self._detect_faces(valid_rois, valid_indices, boxes)
                face_count += face_data["count"]
                face_boxes.extend(face_data["boxes"])
        
        return person_count, face_count, person_boxes, face_boxes
    
    def _prepare_rois(self, frame, boxes):
        """
        Chuẩn bị các vùng ảnh hợp lệ từ khung người.
        Extracts valid image regions from person bounding boxes.

        Args:
            frame (np.ndarray): Khung hình đầu vào (ảnh). Input frame (image).
            boxes (np.ndarray): Mảng các khung người, mỗi khung là [x1, y1, x2, y2]. Array of person bounding boxes, each box is [x1, y1, x2, y2].

        Returns:
            tuple: Một tuple chứa:
                   - valid_rois (list): Danh sách các vùng ảnh hợp lệ (ROI). List of valid image regions (ROIs).
                   - valid_indices (list): Danh sách các chỉ số tương ứng với các khung người hợp lệ. List of indices corresponding to the valid person boxes.
        
        Đây là hàm trích xuất các vùng quan tâm (ROI) từ các khung người. Nó kiểm tra tính hợp lệ của các khung và cắt các vùng ảnh tương ứng.
        This function extracts regions of interest (ROIs) from the person bounding boxes. It checks the validity of the boxes and crops the corresponding image regions.
        
        Thuật toán:
        1. Duyệt qua từng khung người.
        2. Kiểm tra xem khung có hợp lệ không (x2 > x1, y2 > y1, x1 >= 0, y1 >= 0).
        3. Nếu hợp lệ, cắt vùng ảnh tương ứng từ khung hình và thêm vào danh sách valid_rois.
        4. Thêm chỉ số của khung vào danh sách valid_indices.
        5. Trả về danh sách các ROI và danh sách các chỉ số.
        Algorithm:
        1. Iterate through each person bounding box.
        2. Check if the box is valid (x2 > x1, y2 > y1, x1 >= 0, y1 >= 0).
        3. If valid, crop the corresponding image region from the frame and add it to the valid_rois list.
        4. Add the index of the box to the valid_indices list.
        5. Return the list of ROIs and the list of indices.
        """
        valid_rois = []
        valid_indices = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if (x2 > x1) and (y2 > y1) and (x1 >= 0) and (y1 >= 0):
                valid_rois.append(frame[y1:y2, x1:x2])  # Cắt vùng ảnh / Crop region
                valid_indices.append(i)
        return valid_rois, valid_indices
    
    def _detect_faces(self, rois, indices, boxes):
        """
        Nhận diện khuôn mặt và tính toán tọa độ toàn cục.
        Detects faces and calculates global coordinates.

        Args:
            rois (list): Danh sách các vùng ảnh (ROI). List of image regions (ROIs).
            indices (list): Danh sách các chỉ số tương ứng với các khung người. List of indices corresponding to the person boxes.
            boxes (np.ndarray): Mảng các khung người, mỗi khung là [x1, y1, x2, y2]. Array of person bounding boxes, each box is [x1, y1, x2, y2].

        Returns:
            dict: Một dictionary chứa:
                  - count (int): Số lượng khuôn mặt được nhận diện. Number of detected faces.
                  - boxes (list): Danh sách các khung khuôn mặt, mỗi khung là một tuple (coords, conf). List of face bounding boxes, each box is a tuple (coords, conf).
        
        Đây là hàm nhận diện khuôn mặt trong các ROI và chuyển đổi tọa độ khuôn mặt từ cục bộ (trong ROI) sang toàn cục (trong khung hình).
        This function detects faces within the ROIs and converts the face coordinates from local (within the ROI) to global (within the frame).
        
        Thuật toán:
        1. Nếu có ROI, thực hiện nhận diện khuôn mặt bằng mô hình face_model.
        2. Duyệt qua từng kết quả nhận diện khuôn mặt.
        3. Nếu có khuôn mặt được nhận diện, tăng số lượng khuôn mặt.
        4. Lấy chỉ số gốc của khung người chứa ROI.
        5. Lấy tọa độ của khung người đó.
        6. Duyệt qua từng khung khuôn mặt trong kết quả.
        7. Chuyển đổi tọa độ khuôn mặt từ cục bộ sang toàn cục bằng cách cộng tọa độ của khung người.
        8. Thêm tọa độ toàn cục và độ tin cậy vào danh sách face_boxes.
        9. Trả về số lượng khuôn mặt và danh sách các khung khuôn mặt.
        Algorithm:
        1. If there are ROIs, perform face detection using the face_model.
        2. Iterate through each face detection result.
        3. If faces are detected, increment the face count.
        4. Get the original index of the person box containing the ROI.
        5. Get the coordinates of that person box.
        6. Iterate through each face box in the result.
        7. Convert the face coordinates from local to global by adding the coordinates of the person box.
        8. Add the global coordinates and confidence to the face_boxes list.
        9. Return the face count and the list of face bounding boxes.
        """
        face_boxes = []
        face_count = 0
        if rois:
            face_results = self.face_model(rois, stream=True, conf=0.3, imgsz=320, half=True, verbose=False)
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