"""
main.py — Real-time gesture recognition and speech synthesis.

Author : Atharva Gawas
Project: Real-Time Sign Language to Speech Translator

Usage:
    python src/main.py --model models/gesture_model.keras \
                       --labels data/labels.txt

The inference pipeline works as a sliding window over the live webcam feed:
  1. Each frame produces a 63-dimensional landmark vector.
  2. The last SEQ_LEN (30) vectors form a window fed to the LSTM.
  3. If the predicted class exceeds CONFIDENCE_THRESHOLD and has not been
     spoken in the last DEBOUNCE_FRAMES frames, pyttsx3 speaks the label.

Speech synthesis is done offline via pyttsx3, which requires no internet
connection — important for accessibility contexts where connectivity may be
limited.  Google TTS can be substituted by swapping _speak() if needed.
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# Make src importable from the project root.
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    build_hands_detector,
    draw_styled_landmarks,
    ensure_dir,
    extract_landmark_array,
    load_gesture_labels,
    overlay_text,
)

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False

import argparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEQ_LEN: int = 30
FEATURE_DIM: int = 63

# Confidence threshold for accepting a prediction.
# 0.70 chosen empirically: values below this produced too many spurious
# activations when transitioning between gestures; values above 0.85 caused
# the system to miss valid gestures in marginally lit environments.
CONFIDENCE_THRESHOLD: float = 0.70

# Minimum number of frames that must pass before the same gesture can be
# spoken again.  At ~30fps this equals roughly 2 seconds — enough to avoid
# repeated announcements for a held gesture without feeling unresponsive.
DEBOUNCE_FRAMES: int = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-time sign language recognition."
    )
    parser.add_argument(
        "--model",
        default="models/gesture_model.keras",
        help="Path to the saved TensorFlow model directory.",
    )
    parser.add_argument(
        "--labels",
        default="data/labels.txt",
        help="Path to the gesture label file.",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="OpenCV camera index (default: 0)."
    )
    parser.add_argument(
        "--no-speech", action="store_true", help="Disable speech synthesis output."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------

def _build_tts_engine():
    """
    Initialise pyttsx3 TTS engine with a moderate speech rate.

    Rate 150 wpm: the default (200 wpm) was too fast for observers unfamiliar
    with synthesised speech; 150 is close to average conversational pace.
    """
    if not _TTS_AVAILABLE:
        return None
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 0.95)
    return engine


def _speak(engine, text: str) -> None:
    """Non-blocking speech: runAndWait() is called in a brief interval."""
    if engine is None:
        return
    engine.say(text)
    engine.runAndWait()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model_path: str,
    label_file: str,
    camera_index: int,
    speech_enabled: bool,
) -> None:
    """
    Main inference loop: read webcam → extract landmarks → classify → speak.
    """
    if not _TF_AVAILABLE:
        print("[ERROR] TensorFlow is not installed. Run: pip install tensorflow")
        sys.exit(1)

    # Load model and labels.
    model = tf.keras.models.load_model(model_path)
    gesture_labels = load_gesture_labels(label_file)
    num_classes = len(gesture_labels)

    print(f"[INFO] Loaded model from: {model_path}")
    print(f"[INFO] Gesture classes: {gesture_labels}")

    tts_engine = _build_tts_engine() if speech_enabled else None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Sliding window: deque automatically drops oldest frames beyond SEQ_LEN.
    frame_buffer: deque = deque(maxlen=SEQ_LEN)

    last_predicted_label: str = ""
    frames_since_last_prediction: int = DEBOUNCE_FRAMES  # start ready to speak
    last_confidence: float = 0.0

    with build_hands_detector() as hands:
        print("[INFO] Inference running — press Q to quit.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Dropped frame.")
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            result = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            if result.multi_hand_landmarks:
                landmarks = extract_landmark_array(result.multi_hand_landmarks[0])
                draw_styled_landmarks(frame, result.multi_hand_landmarks[0])
            else:
                # Zero vector for frames without a detected hand — the model
                # has seen these during training (see collect_data.py) and
                # generally outputs a low-confidence prediction for them.
                landmarks = np.zeros(FEATURE_DIM, dtype=np.float32)

            frame_buffer.append(landmarks)

            # Only run inference once we have a full window.
            if len(frame_buffer) == SEQ_LEN:
                window = np.array(frame_buffer)[np.newaxis, ...]  # (1, 30, 63)
                predictions = model.predict(window, verbose=0)[0]
                predicted_idx = int(np.argmax(predictions))
                confidence = float(predictions[predicted_idx])
                predicted_label = gesture_labels[predicted_idx]

                last_confidence = confidence

                # Only announce if confident AND debounce window has passed.
                if (
                    confidence >= CONFIDENCE_THRESHOLD
                    and frames_since_last_prediction >= DEBOUNCE_FRAMES
                ):
                    print(
                        f"  -> Detected: '{predicted_label}' "
                        f"(confidence: {confidence:.2f})"
                    )
                    last_predicted_label = predicted_label
                    frames_since_last_prediction = 0
                    _speak(tts_engine, predicted_label)

            frames_since_last_prediction += 1

            # --- HUD overlay ------------------------------------------------
            overlay_text(
                frame,
                f"Gesture: {last_predicted_label}",
                position=(10, 35),
                colour=(0, 255, 0),
            )
            overlay_text(
                frame,
                f"Conf: {last_confidence:.2f}  |  Threshold: {CONFIDENCE_THRESHOLD}",
                position=(10, 70),
                colour=(200, 200, 200),
                font_scale=0.6,
            )
            buffer_fill = int((len(frame_buffer) / SEQ_LEN) * 100)
            overlay_text(
                frame,
                f"Buffer: {buffer_fill}%",
                position=(10, 100),
                colour=(100, 200, 255),
                font_scale=0.6,
            )

            cv2.imshow("Sign Language Translator", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")


def main() -> None:
    args = parse_args()
    run_inference(
        model_path=args.model,
        label_file=args.labels,
        camera_index=args.camera,
        speech_enabled=not args.no_speech,
    )


if __name__ == "__main__":
    main()
