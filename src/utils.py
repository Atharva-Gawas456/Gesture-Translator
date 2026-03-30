"""
utils.py — Shared helper functions for the Gesture-Translator pipeline.

Author : Atharva Gawas
Project: Real-Time Sign Language to Speech Translator

Updated to use the MediaPipe Tasks API (v0.10.30+), which replaces the
legacy mp.solutions interface removed in recent releases.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# ---------------------------------------------------------------------------
# Default model path — callers can override via function arguments.
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
)

# Hand connections for drawing (21 landmarks, same topology as legacy API)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


class HandsDetector:
    """
    Wrapper around MediaPipe Tasks HandLandmarker that provides a
    context-manager interface compatible with the rest of the codebase.

    Uses VIDEO running mode for synchronous, frame-by-frame detection.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._landmarker.close()
        return False

    def process(self, rgb_frame: np.ndarray):
        """
        Process an RGB frame and return a result object with a
        .hand_landmarks attribute (list of landmark lists).

        The returned object mimics the legacy API's result structure so
        that callers can check `result.hand_landmarks` the same way they
        checked `result.multi_hand_landmarks`.
        """
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb_frame
        )
        self._frame_timestamp_ms += 33  # ~30 fps
        result = self._landmarker.detect_for_video(
            mp_image, self._frame_timestamp_ms
        )
        return _HandResult(result)


class _HandResult:
    """Thin adapter so callers can use the same attribute names as before."""

    def __init__(self, task_result):
        self._result = task_result
        # Expose as multi_hand_landmarks for backward compatibility.
        if task_result.hand_landmarks:
            self.multi_hand_landmarks = [
                _LandmarkListAdapter(lm_list)
                for lm_list in task_result.hand_landmarks
            ]
        else:
            self.multi_hand_landmarks = None


class _LandmarkListAdapter:
    """
    Wraps a list of NormalizedLandmark so that .landmark works
    the same way the legacy mp.solutions Hands result did.
    """

    def __init__(self, landmark_list):
        self.landmark = landmark_list


def build_hands_detector(
    static_image_mode: bool = False,
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.7,
    min_tracking_confidence: float = 0.6,
    model_path: str = _DEFAULT_MODEL_PATH,
) -> "HandsDetector":
    """
    Return a configured HandsDetector context manager.

    Drop-in replacement for the legacy mp.solutions.hands.Hands() call.
    """
    return HandsDetector(
        model_path=model_path,
        num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def extract_landmark_array(hand_landmarks) -> np.ndarray:
    """
    Flatten 21 MediaPipe hand landmarks into a (63,) float32 NumPy array.

    Each landmark contributes three values: x, y (normalised to [0,1] by the
    image dimensions) and z (depth relative to the wrist; negative values are
    closer to the camera).  The full vector is therefore 21 × 3 = 63 elements.
    """
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def draw_styled_landmarks(frame: np.ndarray, hand_landmarks) -> None:
    """
    Render hand skeleton on *frame* in-place using OpenCV drawing.

    Replaces the legacy mp.solutions.drawing_utils which is no longer
    available in mediapipe >= 0.10.30.
    """
    h, w, _ = frame.shape
    landmarks = hand_landmarks.landmark

    # Draw connections (lines).
    for start_idx, end_idx in _HAND_CONNECTIONS:
        pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw landmarks (circles).
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)


def overlay_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int] = (10, 40),
    font_scale: float = 1.0,
    colour: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """
    Render *text* onto *frame* at *position* in-place.
    """
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and any missing parents) and return it as a Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_gesture_labels(label_file: str | Path) -> list[str]:
    """
    Read gesture class names from a plain-text file (one label per line).
    """
    p = Path(label_file)
    if not p.exists():
        raise FileNotFoundError(f"Label file not found: {p}")
    labels = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return labels
