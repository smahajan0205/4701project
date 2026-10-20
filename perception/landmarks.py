import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "hand_landmarker.task"
)

#load pretrained mediapipe model
def create_hand_tracker():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )

    return vision.HandLandmarker.create_from_options(options)

#take webcame frame and return 21 points with x y z
def extract_landmarks(tracker, frame_rgb, timestamp_ms):
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(frame_rgb)
    )

    result = tracker.detect_for_video(
        image,
        timestamp_ms
    )

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]

    return np.array([
        [point.x, point.y, point.z]
        for point in landmarks
    ], dtype=np.float32)

#shifts hand so wrist is at origin
def normalize_landmarks(landmarks):
    #wrist is landmark 0
    wrist = landmarks[0]

    #Move wrist to origin
    centered = landmarks - wrist

    #scale by distance from wrist to middle-finger MCP
    scale = np.linalg.norm(
        centered[9, :2]
    )

    scale = max(float(scale), 1e-6)

    return centered / scale
