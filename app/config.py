from os import putenv

# Cấu hình môi trường cho AMD GPU / Environment setup for AMD GPUs
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

# Đường dẫn mô hình / Model paths
PERSON_MODEL_PATH = './models/yolo11n.pt'
FACE_MODEL_PATH = './models/best_face_model_float16.tflite'
EMOTION_MODEL_PATH = './models/emotion_model.tflite'

# Thiết lập hiển thị / Visualization settings
PERSON_COLOR = (0, 0, 255)   # Màu đỏ / Red color
FACE_COLOR = (0, 255, 0)     # Màu xanh lá / Green color
EMOTION_COLOR = (255, 165, 0)  # Màu cam / Orange color for emotion
TEXT_SCALE = 0.6             # Tỷ lệ chữ / Text scale
TEXT_THICKNESS = 2           # Độ dày chữ / Text thickness

# Danh sách cảm xúc / Emotion labels
EMOTION_LABELS = ['Giận dữ', 'Ghê tởm', 'Sợ hãi', 'Vui vẻ', 'Buồn bã', 'Ngạc nhiên', 'Bình thường']