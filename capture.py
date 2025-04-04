import cv2

class VideoCapture:
    """Wrapper for OpenCV VideoCapture with cleaner interface / Lớp bao bọc VideoCapture của OpenCV"""
    
    def __init__(self, src=0):
        # Initialize video capture from source (0=webcam) / Khởi tạo video từ nguồn (0=webcam)
        self.cap = cv2.VideoCapture(src)
    
    def read_frame(self):
        """Read next frame from capture / Đọc khung hình tiếp từ video
        Returns:
            (ret, frame): OpenCV-style tuple / Tuple kiểu OpenCV
        """
        return self.cap.read()  # (ret, frame) or (None if failed / hoặc None nếu lỗi)
    
    def release(self):
        """Release video resources / Giải phóng tài nguyên video"""
        self.cap.release()
    
    def is_opened(self):
        """Check if capture is active / Kiểm tra video có đang mở không
        Returns:
            bool: True if capture is open / True nếu video đang mở
        """
        return self.cap.isOpened()  # Useful for loop conditions / Dùng cho vòng lặp