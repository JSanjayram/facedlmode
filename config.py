# Configuration file for Face Mask Detection App

# Model Configuration
MODEL_PATH = 'accurate_mask_detector.h5'
INPUT_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.5

# Detection Configuration
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 4

# Camera Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 30

# Colors (BGR format)
MASK_COLOR = (0, 255, 0)      # Green
NO_MASK_COLOR = (0, 0, 255)   # Red
TEXT_COLOR = (255, 255, 255)  # White

# Data paths
WITH_MASK_DIR = 'data/train/with_mask'
WITHOUT_MASK_DIR = 'data/train/without_mask'

# Streamlit Configuration
PAGE_TITLE = "Face Mask Detection"
PAGE_ICON = "🎭"