# Face and Person Detection System with YOLO

A real-time detection system using YOLO models to identify persons and faces in video streams, with FPS counter and visualization overlay.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Modules Documentation](#modules-documentation)
  - [config.py](#configpy)
  - [models.py](#modelspy)
  - [detector.py](#detectorpy)
  - [visualizer.py](#visualizerpy)
  - [video.py](#videopy)
  - [main.py](#mainpy)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features
- Real-time person detection using YOLO(nano)
- Face detection within person bounding boxes
- FPS counter overlay
- Dynamic count display for:
  - Detected persons
  - Identified faces
- Confidence scores for face detections
- Keyboard control (Q to quit)

## Project Structure
```bash
yolo-human-face-detection/
├── config.py        # Configuration and constants
├── models.py        # Model loading functions
├── detector.py      # Detection logic
├── visualizer.py    # Visualization utilities
├── video.py         # Video capture handling
├── main.py          # Main application loop
├── requirements.txt # Dependency list
├──README.md
```

## Installation
1. Clone repository:
```bash
git clone https://github.com/hai4h/yolo-human-face-detection.git
cd yolo-human-face-detection
```

2. Environment setup and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install ultralytics
```

3. Download YOLO models:
- Place `yolo11n.pt` (or other object detection model trained with class 'person') and `yolo8n-face-640-50epochs.pt` (or other face detection model) in project root
- Or modify `config.py` with your custom model paths

## Usage
```bash
python main.py
```
- Press `Q` to quit
- Detection results shown in real-time:
  ![Sample Output](screenshot.png)

## Configuration
Modify `config.py` for customization:
```python
# Hardware Configuration (for AMD GPUs with ROCm only, remove if unnecessary)
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")  

# Model Paths
PERSON_MODEL_PATH = 'yolo11n.pt'      # Path to person detection model
FACE_MODEL_PATH = 'yolo8n-face-640-50epochs.pt'  # Path to face detection model

# Visualization Parameters
PERSON_COLOR = (0, 0, 255)    # Red bounding boxes for persons
FACE_COLOR = (0, 255, 0)      # Green bounding boxes for faces
TEXT_SCALE = 0.6              # Text size for labels
TEXT_THICKNESS = 2            # Text stroke width
```

## Modules Documentation

### `config.py`
Contains all environment settings and constants:
- Hardware configuration for AMD GPUs (Optional)
- Model file paths
- Visualization parameters (colors, text properties)

### `models.py`
**Functions:**
- `load_models()`: Initializes YOLO models for person/face detection

**Returns:**
- Tuple of (person_model, face_model) YOLO instances

### `detector.py`
**Classes:**
- `Detector`: Main detection handler

**Methods:**
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `process_frame` | `frame: np.ndarray` | `(annotated_frame, person_count, face_count, face_boxes)` | Processes frame through detection pipeline |
| `_prepare_rois` | `frame`, `boxes` | `(valid_rois, valid_indices)` | Extracts valid regions-of-interest from person boxes |
| `_detect_faces` | `rois`, `indices`, `boxes` | `{"count": int, "boxes": list}` | Performs face detection within ROIs |

### `visualizer.py`
**Functions:**
- `draw_faces()`: Draws face boxes and confidence labels
- `draw_ui()`: Renders HUD with counts and FPS

**Parameters:**
- All functions accept frame numpy arrays and detection data
- Returns modified frames with visual elements

### `video.py`
**Classes:**
- `VideoCapture`: Wrapper for OpenCV video handling

**Methods:**
| Method | Description |
|--------|-------------|
| `read_frame()` | Returns (ret, frame) tuple |
| `release()` | Releases video resources |
| `is_opened()` | Returns capture status |

### `main.py**
Main execution script:
1. Initializes video capture and detector
2. Main loop:
   - Frame processing
   - Visualization
   - FPS calculation
   - Display management

## Dependencies
- Python 3.10+ (3.12+ recommended)
- Required packages:
  ```python
  ultralytics  # YOLO implementation
  opencv-python # Video processing
  numpy        # Array operations
  ```

Install with:
```bash
pip install ultralytics opencv-python numpy
```

## Troubleshooting
**1. Model Loading Issues**
- Verify model files exist at specified paths
- Check Ultralytics version compatibility
- Ensure PyTorch is installed (comes with ultralytics)

**2. Webcam Access Problems**
- Confirm camera is not being used by other applications
- Try different video source index (e.g., `VideoCapture(1)`)

**3. Low FPS**
- Reduce input resolution in `face_model` parameters
- Set `half=True` in model calls for FP16 inference (if supported)
- Disable face detection in `process_frame`

**4. AMD GPU Errors**
- Remove HSA_OVERRIDE_GFX_VERSION in `config.py` if not using AMD or without ROCm supported hardware
- Update ROCm drivers if using AMD GPUs

## License
[MIT License](LICENSE) - Free for academic and commercial use with attribution

---