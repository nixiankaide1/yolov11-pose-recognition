import os
import torch

# COCO 17-keypoint standard names
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO_KEYPOINT_NAMES_CN = {
    "nose": "鼻子", "left_eye": "左眼", "right_eye": "右眼",
    "left_ear": "左耳", "right_ear": "右耳",
    "left_shoulder": "左肩", "right_shoulder": "右肩",
    "left_elbow": "左肘", "right_elbow": "右肘",
    "left_wrist": "左腕", "right_wrist": "右腕",
    "left_hip": "左髋", "right_hip": "右髋",
    "left_knee": "左膝", "right_knee": "右膝",
    "left_ankle": "左踝", "right_ankle": "右踝",
}

DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"

POSE_MODELS = [
    "yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt",
    "yolo11l-pose.pt", "yolo11x-pose.pt",
]

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "pose")
EXPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "exports")

CONFIDENCE_THRESHOLD = 0.3

# ── Temporal smoothing (Butterworth low-pass filter) ──
SMOOTHING_DEFAULT_ENABLED = True
SMOOTHING_DEFAULT_CUTOFF = 10.0  # Hz
SMOOTHING_ORDER = 4
SMOOTHING_MIN_FRAMES = 16  # > filtfilt padlen (3 * order = 15 for order=4)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)
