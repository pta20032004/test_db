import os
import cv2
import numpy as np
import time
import logging
from pathlib import Path
import sys

# Thêm thư mục gốc của dự án vào Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import các module từ dự án
from app.box_detector import Detector  # Detector từ dự án gốc
from app.face_db3.embedding_service import FaceEmbeddingService  # Service tạo embedding
from app.face_db3.chroma_manager import ChromaFaceDB  # Quản lý vector database

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger('BatchSimpleTest')

# Đường dẫn mặc định
DB_PATH = "./face_db/chroma_data"
COLLECTION_NAME = "face_embeddings"
DEBUG_DIR = "./debug_images"

class SimpleTest:
    """
    Demo nhận diện khuôn mặt thời gian thực, sử dụng các module đã tích hợp
    với xử lý batch cho nhiều khuôn mặt
    """
    def __init__(self, db_path=DB_PATH, debug=False):
        """
        Khởi tạo các thành phần cần thiết
        
        Args:
            db_path (str): Đường dẫn đến ChromaDB
            debug (bool): Bật chế độ debug
        """
        self.debug = debug
        
        if self.debug:
            # Tạo thư mục debug nếu chưa có
            Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)
            logger.info(f"Bật chế độ debug, ảnh sẽ được lưu vào {DEBUG_DIR}")
        
        # Khởi tạo detector từ dự án gốc
        logger.info("Khởi tạo detector từ dự án gốc...")
        self.detector = Detector()
        
        # Khởi tạo dịch vụ embedding 
        logger.info("Khởi tạo embedding service...")
        self.embedding_service = FaceEmbeddingService(detector_type="yolo", debug=debug)
        
        # Khởi tạo ChromaDB để lưu trữ và tìm kiếm khuôn mặt
        logger.info(f"Kết nối đến ChromaDB tại {db_path}...")
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.face_db = ChromaFaceDB(db_path=db_path, collection_name=COLLECTION_NAME)
        
        # Thông tin cơ bản về database
        db_info = self.face_db.get_database_info()
        logger.info(f"Database info: {db_info}")
    
    def register_folder(self, folder_path):
        """
        Đăng ký tất cả ảnh trong thư mục vào database
        
        Args:
            folder_path (str): Đường dẫn đến thư mục chứa ảnh
        """
        # Kiểm tra thư mục tồn tại
        if not os.path.exists(folder_path):
            logger.error(f"Thư mục không tồn tại: {folder_path}")
            return
        
        # Lấy danh sách ảnh
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(Path(folder_path).glob(f"*{ext}")))
            image_files.extend(list(Path(folder_path).glob(f"*{ext.upper()}")))
        
        logger.info(f"Tìm thấy {len(image_files)} ảnh")
        
        # Đăng ký từng ảnh
        success_count = 0
        fail_count = 0
        start_time = time.time()
        
        for i, img_path in enumerate(image_files):
            logger.info(f"Xử lý {i+1}/{len(image_files)}: {img_path.name}")
            
            # Sử dụng tên file làm tên người
            name = img_path.stem
            
            try:
                # Tạo embedding từ ảnh - sử dụng FaceEmbeddingService
                embedding = self.embedding_service.get_embedding_from_file(str(img_path))
                
                if embedding is None:
                    logger.warning(f"Không thể tạo embedding từ ảnh: {img_path}")
                    fail_count += 1
                    continue
                
                # Tạo ID từ tên file
                user_id = f"user_{str(img_path.stem).lower()}"
                
                # Tạo metadata
                metadata = {
                    "name": name,
                    "file_path": str(img_path),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                }
                
                # Thêm vào ChromaDB - sử dụng ChromaFaceDB
                face_id = self.face_db.add_face(embedding, metadata, user_id)
                
                if face_id:
                    success_count += 1
                    logger.info(f"Đã thêm: {name} - {face_id}")
                else:
                    fail_count += 1
                    logger.error(f"Lỗi khi thêm {name} vào database")
                
            except Exception as e:
                logger.error(f"Lỗi khi xử lý ảnh {img_path}: {e}")
                fail_count += 1
                continue
        
        # Tính thời gian
        total_time = time.time() - start_time
        logger.info(f"\nĐã đăng ký {success_count}/{len(image_files)} khuôn mặt ({fail_count} thất bại)")
        logger.info(f"Thời gian: {total_time:.2f}s ({success_count/total_time:.1f} khuôn mặt/giây)")
    
    def search_faces_batch(self, embeddings, threshold=0.7):
        """
        Tìm kiếm nhiều khuôn mặt cùng lúc trong database
        
        Args:
            embeddings (list): Danh sách các vector embedding
            threshold (float): Ngưỡng tương đồng (0-1)
            
        Returns:
            list: Danh sách kết quả tìm kiếm cho từng embedding
        """
        if not embeddings:
            return []
        
        # Chuẩn bị danh sách embedding
        embedding_list = [e["embedding"] for e in embeddings if "embedding" in e]
        
        if not embedding_list:
            return [[] for _ in range(len(embeddings))]
        
        try:
            # Chuyển đổi thành list nếu cần
            query_embeddings = [e.tolist() if isinstance(e, np.ndarray) else e for e in embedding_list]
            
            # Thực hiện tìm kiếm
            results = self.face_db.collection.query(
                query_embeddings=query_embeddings,
                n_results=1
            )
            
            # Xử lý kết quả
            matches_list = []
            
            for i in range(len(query_embeddings)):
                matches = []
                
                if i < len(results["ids"]) and len(results["ids"][i]) > 0:
                    for j in range(len(results["ids"][i])):
                        # Tính độ tương đồng (1 - khoảng cách)
                        similarity = 1 - results["distances"][i][j]
                        
                        # Chỉ lấy kết quả vượt ngưỡng
                        if similarity >= threshold:
                            match = {
                                "id": results["ids"][i][j],
                                "metadata": results["metadatas"][i][j],
                                "similarity": similarity
                            }
                            matches.append(match)
                
                matches_list.append(matches)
            
            return matches_list
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm batch: {e}")
            return [[] for _ in range(len(embeddings))]
    
    def test_webcam(self, camera_id=0, threshold=0.6, max_faces=20, scale_factor=0.75):
        """
        Test nhận diện với webcam sử dụng xử lý batch
        
        Args:
            camera_id (int): ID của camera
            threshold (float): Ngưỡng tương đồng (0-1)
            max_faces (int): Số lượng khuôn mặt tối đa xử lý trong một frame
            scale_factor (float): Hệ số giảm kích thước để tăng tốc (0-1)
        """
        logger.info(f"Khởi động camera {camera_id}...")
        
        # Mở camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Không thể mở camera {camera_id}")
            return
        
        logger.info("Nhấn ESC để thoát")
        
        # Số lượng khuôn mặt trong database
        db_info = self.face_db.get_database_info()
        face_count = db_info.get("count", 0)
        
        if face_count == 0:
            logger.warning("CẢNH BÁO: Database trống, không thể nhận diện. Hãy đăng ký khuôn mặt trước.")
        else:
            logger.info(f"Đã tải {face_count} khuôn mặt từ database")
        
        # Thống kê hiệu năng
        frame_count = 0
        fps_list = []
        start_fps_time = time.time()
        
        logger.info(f"Sử dụng ngưỡng tương đồng: {threshold}, tối đa {max_faces} khuôn mặt/frame")
        logger.info(f"Hệ số giảm kích thước: {scale_factor}")
        
        while True:
            # Đọc frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Lỗi khi đọc từ camera!")
                break
            
            # Bắt đầu tính thời gian
            start_time = time.time()
            
            # Giảm kích thước frame nếu cần
            if scale_factor < 1.0:
                h, w = frame.shape[:2]
                new_w, new_h = int(w * scale_factor), int(h * scale_factor) 
                processing_frame = cv2.resize(frame, (new_w, new_h))
                logger.debug(f"Giảm kích thước frame từ {w}x{h} xuống {new_w}x{new_h}")
            else:
                processing_frame = frame
            
            # Phát hiện khuôn mặt trong frame - dùng detector từ dự án gốc
            _, detected_face_count, _, face_boxes = self.detector.process_frame(processing_frame)
            
            # Giới hạn số lượng khuôn mặt xử lý nếu cần
            if len(face_boxes) > max_faces:
                logger.info(f"Giới hạn xử lý {max_faces}/{len(face_boxes)} khuôn mặt")
                # Sắp xếp theo confidence giảm dần và lấy max_faces khuôn mặt
                face_boxes = sorted(face_boxes, key=lambda x: x[1], reverse=True)[:max_faces]
            
            # Tạo embeddings cho tất cả khuôn mặt cùng lúc
            embeddings = []
            
            if face_boxes:
                # Sử dụng get_embeddings_from_image để lấy embeddings cho tất cả khuôn mặt
                embeddings = self.embedding_service.get_embeddings_from_image(processing_frame, max_faces=len(face_boxes))
                logger.debug(f"Đã tạo {len(embeddings)} embeddings")
            
            # Tìm kiếm tất cả khuôn mặt cùng lúc trong database
            matches_list = self.search_faces_batch(embeddings, threshold)
            
            # Vẽ kết quả
            for i, face_box in enumerate(face_boxes):
                coords, conf, emotion = face_box
                
                # Điều chỉnh tọa độ nếu đã resize frame
                if scale_factor < 1.0:
                    # Scale tọa độ trở lại kích thước gốc
                    x1, y1, x2, y2 = map(int, coords)
                    scaled_coords = (
                        int(x1 / scale_factor),
                        int(y1 / scale_factor),
                        int(x2 / scale_factor),
                        int(y2 / scale_factor)
                    )
                    x1, y1, x2, y2 = scaled_coords
                else:
                    x1, y1, x2, y2 = map(int, coords)
                
                # Kiểm tra nếu có embedding tương ứng
                if i < len(embeddings) and i < len(matches_list):
                    matches = matches_list[i]
                    
                    if matches:
                        # Tìm thấy khuôn mặt - vẽ khung xanh
                        match = matches[0]
                        name = match["metadata"].get("name", "Unknown")
                        similarity = match["similarity"]
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{name} ({similarity:.2f})"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # Không tìm thấy khuôn mặt - vẽ khung đỏ
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # Không có embedding - vẽ khung đỏ
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Hiển thị cảm xúc nếu có
                if emotion:
                    cv2.putText(frame, emotion, (x1, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # Tính thời gian xử lý
            process_time = time.time() - start_time
            
            # Tính FPS
            frame_count += 1
            if frame_count % 10 == 0:  # Cập nhật mỗi 10 frame
                current_time = time.time()
                elapsed = current_time - start_fps_time
                fps = frame_count / elapsed
                fps_list.append(fps)
                
                # Reset bộ đếm
                if len(fps_list) > 10:
                    fps_list.pop(0)
                    
                frame_count = 0
                start_fps_time = current_time
            
            # Hiển thị FPS và thông tin
            avg_fps = sum(fps_list) / max(len(fps_list), 1)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Process: {process_time*1000:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {detected_face_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Hiển thị frame
            cv2.imshow("Face Recognition", frame)
            
            # Thoát nếu nhấn ESC
            if cv2.waitKey(1) == 27:
                break
        
        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()

def register_folder(folder_path, db_path=DB_PATH, debug=False):
    """Hàm wrapper để đăng ký khuôn mặt từ thư mục"""
    tester = SimpleTest(db_path, debug)
    tester.register_folder(folder_path)

def test_camera(camera_id=0, db_path=DB_PATH, threshold=0.6, max_faces=20, scale_factor=0.75, debug=False):
    """Hàm wrapper để test nhận diện với camera"""
    tester = SimpleTest(db_path, debug)
    tester.test_webcam(camera_id, threshold, max_faces, scale_factor)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo nhận diện khuôn mặt tích hợp với xử lý batch")
    parser.add_argument("--mode", type=str, required=True, choices=["register", "webcam"], 
                        help="Chế độ: register (đăng ký ảnh) hoặc webcam (test camera)")
    parser.add_argument("--folder", type=str, help="Thư mục chứa ảnh cần đăng ký")
    parser.add_argument("--camera", type=int, default=0, help="ID của camera (mặc định: 0)")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Đường dẫn đến database")
    parser.add_argument("--threshold", type=float, default=0.6, 
                      help="Ngưỡng tương đồng (0-1), mặc định 0.6")
    parser.add_argument("--max-faces", type=int, default=20,
                      help="Số lượng khuôn mặt tối đa xử lý trong một frame (mặc định: 20)")
    parser.add_argument("--scale", type=float, default=0.75,
                      help="Hệ số giảm kích thước frame để tăng tốc (0-1, mặc định: 0.75)")
    parser.add_argument("--debug", action="store_true", help="Bật chế độ debug (lưu ảnh trung gian)")
    
    args = parser.parse_args()
    
    if args.mode == "register":
        if not args.folder:
            logger.error("Thiếu đường dẫn thư mục ảnh (--folder)")
        else:
            register_folder(args.folder, args.db, args.debug)
    elif args.mode == "webcam":
        test_camera(args.camera, args.db, args.threshold, args.max_faces, args.scale, args.debug)