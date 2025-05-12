import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import time
import logging
from pathlib import Path

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger('FaceEmbedding')

class FaceEmbeddingService:
    """
    Dịch vụ tạo embedding khuôn mặt, hỗ trợ cả InsightFace và YOLO
    """
    def __init__(self, detector_type="insightface", det_size=(640, 640), debug=False):
        """
        Khởi tạo dịch vụ với detector được chọn
        
        Args:
            detector_type (str): Loại detector ('insightface' hoặc 'yolo')
            det_size (tuple): Kích thước ảnh phát hiện khuôn mặt
            debug (bool): Bật chế độ debug
        """
        # Lưu loại detector và kích thước
        self.detector_type = detector_type
        self.det_size = det_size
        self.debug = debug
        
        # Tạo thư mục debug nếu cần
        if self.debug:
            self.debug_dir = "./debug_embedding"
            Path(self.debug_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Bật chế độ debug, ảnh sẽ được lưu vào {self.debug_dir}")
        
        # Khởi tạo InsightFace (luôn cần cho phần tạo embedding)
        self.providers = self._get_best_provider()
        self.using_gpu = 'CPU' not in self.providers[0]
        
        # Log thông tin
        logger.info(f"Khởi tạo với detector: {detector_type}, GPU: {self.using_gpu}")
        
        # Cấu hình và khởi tạo detector
        self._setup_detector()
    
    def _get_best_provider(self):
        """Xác định provider tốt nhất cho ONNX Runtime"""
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX Runtime providers: {available_providers}")
            
            # Ưu tiên theo thứ tự: CUDA (NVIDIA) > ROCm (AMD) > DirectML (Windows) > CPU
            if 'CUDAExecutionProvider' in available_providers:
                logger.info("Sử dụng NVIDIA GPU (CUDA)")
                return ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif 'ROCMExecutionProvider' in available_providers:
                logger.info("Sử dụng AMD GPU (ROCm)")
                return ['ROCMExecutionProvider', 'CPUExecutionProvider']
            elif 'DmlExecutionProvider' in available_providers:
                logger.info("Sử dụng DirectML GPU (Windows)")
                return ['DmlExecutionProvider', 'CPUExecutionProvider']
            else:
                logger.info("Không phát hiện GPU khả dụng, sử dụng CPU")
                return ['CPUExecutionProvider']
        except ImportError:
            logger.info("Không tìm thấy onnxruntime, sử dụng CPU")
            return ['CPUExecutionProvider']
    
    def _setup_detector(self):
        """Cấu hình detector dựa trên loại đã chọn"""
        try:
            start_time = time.time()
            
            # Khởi tạo InsightFace (luôn cần cho embedding)
            logger.info(f"Khởi tạo InsightFace với providers: {self.providers}")
            self.face_analyzer = FaceAnalysis(providers=self.providers)
            self.face_analyzer.prepare(ctx_id=0, det_size=self.det_size)
            
            # Nếu sử dụng YOLO từ dự án gốc
            if self.detector_type == "yolo":
                logger.info("Khởi tạo YOLO detector từ dự án gốc")
                try:
                    from app.box_detector import Detector
                    self.yolo_detector = Detector()
                    logger.info("Đã khởi tạo YOLO detector thành công")
                except Exception as e:
                    logger.error(f"Lỗi khi khởi tạo YOLO detector: {e}")
                    logger.info("Chuyển sang sử dụng InsightFace detector")
                    self.detector_type = "insightface"
            
            # Warm up
            dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
            if self.detector_type == "yolo":
                _ = self.yolo_detector.process_frame(dummy_img)
            _ = self.face_analyzer.get(dummy_img)
            
            logger.info(f"Khởi tạo hoàn tất trong {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình khởi tạo: {e}")
            # Fallback to InsightFace if error occurs
            self.detector_type = "insightface"
            self.face_analyzer = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.face_analyzer.prepare(ctx_id=0, det_size=self.det_size)
            self.using_gpu = False
    
    def _extract_face_with_margin(self, img, bbox, margin_percent=0.2, min_size=112):
        """
        Trích xuất khuôn mặt với margin và đảm bảo kích thước tối thiểu
        
        Args:
            img: Ảnh gốc
            bbox: Bounding box [x1, y1, x2, y2]
            margin_percent: Phần trăm margin so với kích thước bbox
            min_size: Kích thước tối thiểu của ảnh khuôn mặt
            
        Returns:
            Ảnh khuôn mặt đã cắt và tiền xử lý
        """
        # Lấy kích thước ảnh
        height, width = img.shape[:2]
        
        # Lấy tọa độ bbox
        x1, y1, x2, y2 = map(int, bbox)
        
        # Tính margin pixel theo phần trăm kích thước bbox
        w, h = x2 - x1, y2 - y1
        margin_x = int(w * margin_percent)
        margin_y = int(h * margin_percent)
        
        # Mở rộng bbox với margin, đảm bảo không vượt quá kích thước ảnh
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(width, x2 + margin_x)
        y2 = min(height, y2 + margin_y)
        
        # Cắt khuôn mặt
        face_img = img[y1:y2, x1:x2]
        
        # Nếu kích thước quá nhỏ, resize lên
        min_dim = min(face_img.shape[0], face_img.shape[1])
        if min_dim < min_size:
            scale = min_size / min_dim
            new_width = int(face_img.shape[1] * scale)
            new_height = int(face_img.shape[0] * scale)
            face_img = cv2.resize(face_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            logger.debug(f"Đã resize ảnh từ {min_dim}px lên {min_size}px")
        
        # Tiền xử lý ảnh
        # 1. Cân bằng histogram để cải thiện độ tương phản
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced_face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_face
    
    def get_embedding_from_file(self, image_path):
        """
        Tạo embedding từ file ảnh
        
        Args:
            image_path (str): Đường dẫn đến file ảnh
            
        Returns:
            np.ndarray: Vector embedding hoặc None nếu không phát hiện khuôn mặt
        """
        # Kiểm tra file tồn tại
        if not os.path.exists(image_path):
            logger.error(f"Ảnh không tồn tại: {image_path}")
            return None
        
        try:
            # Đọc ảnh
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Không thể đọc ảnh: {image_path}")
                return None
            
            # Debug: Lưu ảnh gốc
            if self.debug:
                debug_orig_path = os.path.join(self.debug_dir, f"{Path(image_path).stem}_original.jpg")
                cv2.imwrite(debug_orig_path, img)
                
            return self.get_embedding_from_image(img)
        except Exception as e:
            logger.error(f"Lỗi khi tạo embedding từ ảnh {image_path}: {e}")
            return None
    
    def get_embedding_from_image(self, img):
        """
        Tạo embedding từ ảnh đã đọc
        
        Args:
            img (np.ndarray): Ảnh dạng numpy array (BGR)
            
        Returns:
            np.ndarray: Vector embedding hoặc None nếu không phát hiện khuôn mặt
        """
        try:
            # Phát hiện khuôn mặt với detector đã chọn
            if self.detector_type == "yolo":
                # Sử dụng YOLO detector từ dự án gốc
                _, face_count, _, face_boxes = self.yolo_detector.process_frame(img)
                
                if face_count == 0:
                    logger.warning("Không phát hiện khuôn mặt trong ảnh với YOLO")
                    return None
                
                # Lấy khuôn mặt đầu tiên (có confidence cao nhất)
                best_face = max(face_boxes, key=lambda x: x[1])
                coords, conf, _ = best_face
                
                logger.debug(f"YOLO đã phát hiện khuôn mặt với confidence {conf:.3f}, bbox: {coords}")
                
                # Cắt khuôn mặt với margin và tiền xử lý
                face_img = self._extract_face_with_margin(img, coords)
                
                # Debug: Lưu ảnh khuôn mặt đã cắt
                if self.debug:
                    timestamp = int(time.time() * 1000)
                    debug_face_path = os.path.join(self.debug_dir, f"face_yolo_{timestamp}.jpg")
                    cv2.imwrite(debug_face_path, face_img)
                
                # Tạo embedding bằng InsightFace
                faces = self.face_analyzer.get(face_img)
                if len(faces) == 0:
                    logger.warning("InsightFace không tìm thấy khuôn mặt trong vùng đã cắt")
                    
                    # Thử lại với ảnh gốc
                    logger.info("Thử lại với ảnh gốc...")
                    faces = self.face_analyzer.get(img)
                    if len(faces) == 0:
                        logger.error("InsightFace không tìm thấy khuôn mặt trong ảnh gốc")
                        return None
                    
                    logger.info("Đã tìm thấy khuôn mặt trong ảnh gốc")
                
                # Debug: Vẽ keypoints của InsightFace nếu có
                if self.debug and hasattr(faces[0], 'kps'):
                    debug_img = img.copy()
                    kps = faces[0].kps
                    for kp in kps:
                        x, y = kp
                        cv2.circle(debug_img, (int(x), int(y)), 2, (0, 255, 0), -1)
                    timestamp = int(time.time() * 1000)
                    debug_kps_path = os.path.join(self.debug_dir, f"keypoints_{timestamp}.jpg")
                    cv2.imwrite(debug_kps_path, debug_img)
                
                return faces[0].embedding
            else:
                # Sử dụng InsightFace (mặc định)
                faces = self.face_analyzer.get(img)
                if len(faces) == 0:
                    logger.warning("InsightFace không tìm thấy khuôn mặt trong ảnh")
                    return None
                
                # Lấy khuôn mặt có confidence cao nhất
                best_face = max(faces, key=lambda x: x.det_score)
                
                # Debug: Vẽ bounding box và keypoints
                if self.debug:
                    debug_img = img.copy()
                    bbox = best_face.bbox.astype(int)
                    cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    
                    if hasattr(best_face, 'kps'):
                        kps = best_face.kps
                        for kp in kps:
                            x, y = kp
                            cv2.circle(debug_img, (int(x), int(y)), 2, (0, 255, 0), -1)
                    
                    timestamp = int(time.time() * 1000)
                    debug_path = os.path.join(self.debug_dir, f"insightface_{timestamp}.jpg")
                    cv2.imwrite(debug_path, debug_img)
                
                return best_face.embedding
        except Exception as e:
            logger.error(f"Lỗi khi tạo embedding: {e}")
            return None
    
    def get_embeddings_from_image(self, img, max_faces=5):
        """
        Tạo embedding cho nhiều khuôn mặt từ một ảnh
        
        Args:
            img (np.ndarray): Ảnh dạng numpy array (BGR)
            max_faces (int): Số lượng khuôn mặt tối đa lấy embedding
            
        Returns:
            list: Danh sách các embedding vectors và tọa độ khuôn mặt
        """
        try:
            embeddings = []
            
            # Phát hiện khuôn mặt với detector đã chọn
            if self.detector_type == "yolo":
                # Sử dụng YOLO detector từ dự án gốc
                _, face_count, _, face_boxes = self.yolo_detector.process_frame(img)
                
                if face_count == 0:
                    logger.warning("Không phát hiện khuôn mặt nào trong ảnh với YOLO")
                    return []
                
                # Sắp xếp theo độ tin cậy giảm dần và giới hạn số lượng
                face_boxes = sorted(face_boxes, key=lambda x: x[1], reverse=True)[:max_faces]
                
                # Xử lý từng khuôn mặt
                for face_box in face_boxes:
                    coords, conf, emotion = face_box
                    
                    # Cắt khuôn mặt với margin và tiền xử lý
                    face_img = self._extract_face_with_margin(img, coords)
                    
                    # Debug: Lưu ảnh khuôn mặt
                    if self.debug:
                        timestamp = int(time.time() * 1000)
                        i = len(embeddings)
                        debug_path = os.path.join(self.debug_dir, f"face_{i}_{timestamp}.jpg")
                        cv2.imwrite(debug_path, face_img)
                    
                    # Tạo embedding bằng InsightFace
                    faces = self.face_analyzer.get(face_img)
                    if len(faces) > 0:
                        embedding = faces[0].embedding
                        embeddings.append({
                            "embedding": embedding,
                            "bbox": coords,
                            "score": float(conf),
                            "emotion": emotion  # Giữ lại cảm xúc từ YOLO nếu có
                        })
                    else:
                        logger.warning(f"InsightFace không tìm thấy khuôn mặt trong vùng đã cắt {coords}")
            else:
                # Sử dụng InsightFace (mặc định)
                faces = self.face_analyzer.get(img)
                
                # Debug: Vẽ tất cả khuôn mặt
                if self.debug and faces:
                    debug_img = img.copy()
                    for i, face in enumerate(faces):
                        bbox = face.bbox.astype(int)
                        cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.putText(debug_img, f"{i}: {face.det_score:.2f}", (bbox[0], bbox[1]-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    timestamp = int(time.time() * 1000)
                    debug_path = os.path.join(self.debug_dir, f"all_faces_{timestamp}.jpg")
                    cv2.imwrite(debug_path, debug_img)
                
                # Sắp xếp theo độ tin cậy và giới hạn số lượng
                faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:max_faces]
                
                # Tạo danh sách embedding và bounding box
                for face in faces:
                    embedding = face.embedding
                    bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
                    embeddings.append({
                        "embedding": embedding,
                        "bbox": bbox,
                        "score": float(face.det_score)
                    })
            
            return embeddings
        except Exception as e:
            logger.error(f"Lỗi khi tạo embeddings: {e}")
            return []
    
    def get_hardware_info(self):
        """
        Lấy thông tin phần cứng đang sử dụng
        
        Returns:
            dict: Thông tin về phần cứng và cấu hình
        """
        info = {
            "detector_type": self.detector_type,
            "providers": self.providers,
            "using_gpu": self.using_gpu,
            "det_size": self.det_size
        }
        
        # Thêm thông tin chi tiết về GPU nếu đang sử dụng
        if self.using_gpu:
            # NVIDIA GPU
            if 'CUDA' in self.providers[0]:
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    gpu_info = []
                    
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        name = pynvml.nvmlDeviceGetName(handle)
                        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        
                        gpu_info.append({
                            "name": name,
                            "memory_total": memory.total / (1024**2),  # MB
                            "memory_used": memory.used / (1024**2),    # MB
                            "memory_free": memory.free / (1024**2),    # MB
                            "utilization": utilization.gpu             # %
                        })
                    
                    info["nvidia_gpus"] = gpu_info
                except:
                    info["nvidia_gpus"] = "Không thể đọc thông tin chi tiết (cần cài pynvml)"
            
            # AMD GPU
            elif 'ROCm' in self.providers[0]:
                try:
                    import subprocess
                    result = subprocess.run(['rocm-smi', '--showuse', '--json'], capture_output=True, text=True)
                    if result.returncode == 0:
                        info["amd_gpus"] = "Xem rocm-smi để biết chi tiết"
                    else:
                        info["amd_gpus"] = "Không thể đọc thông tin chi tiết"
                except:
                    info["amd_gpus"] = "Không thể đọc thông tin chi tiết"
        
        return info