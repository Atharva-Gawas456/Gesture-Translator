"""
collect_data.py — Webcam-based gesture landmark collection script.

Author : Atharva Gawas
Project: Real-Time Sign Language to Speech Translator

Usage:
    python src/collect_data.py --gesture hello --samples 200 --output data/

Each run captures <samples> frames of the specified gesture and saves them as a
single NumPy .npy file whose name encodes the gesture label.  The LSTM training
script later groups these files into sequences.

Design decisions:
  - Sequences are built in a fixed-length sliding window (SEQ_LEN frames).
    Padding is avoided; incomplete windows are discarded.
  - The script waits for the user to press SPACE before recording so that
    hand position can be stabilised — this reduces leading-frame noise that
    otherwise inflates within-class variance.
  - A 30-frame cooldown between windows prevents the same pose from spanning
    two consecutive training examples.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make src importable whether the script is run from the project root or src/.
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    build_hands_detector,
    draw_styled_landmarks,
    ensure_dir,
    extract_landmark_array,
    overlay_text,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEQ_LEN: int = 30          # frames per training sequence — 30 @ ~30fps ≈ 1 second
LANDMARK_DIM: int = 63     # 21 landmarks × 3 coordinates (x, y, z)
COOLDOWN_FRAMES: int = 10  # pause between collected sequences to avoid overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect hand-gesture landmark sequences via webcam."
    )
    parser.add_argument(
        "--gesture", required=True, help="Label for the gesture being recorded."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of sequences to collect (default: 200).",
    )
    parser.add_argument(
        "--output", default="data/", help="Directory to store .npy files."
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="OpenCV camera index (default: 0)."
    )
    return parser.parse_args()


def collect_sequences(
    gesture_label: str,
    num_samples: int,
    output_dir: Path,
    camera_index: int,
) -> None:
    """
    Capture *num_samples* landmark sequences for *gesture_label* and save them.

    Each sequence is a (SEQ_LEN, LANDMARK_DIM) array.  All sequences for one
    gesture are stacked into a (num_samples, SEQ_LEN, LANDMARK_DIM) array and
    written to <output_dir>/<gesture_label>.npy.
    """
    output_dir = ensure_dir(output_dir)
    output_path = output_dir / f"{gesture_label}.npy"

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}.")

    # Increase buffer size so frames aren't stale on slower machines.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    collected_sequences: list[np.ndarray] = []

    with build_hands_detector() as hands:
        print(f"\n[INFO] Recording gesture: '{gesture_label}'")
        print("[INFO] Press SPACE to start each sequence, Q to quit early.\n")

        while len(collected_sequences) < num_samples:
            # --- Wait phase: show live feed until SPACE is pressed ----------
            waiting_for_space = True
            while waiting_for_space:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Dropped frame during wait phase.")
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)

                if result.multi_hand_landmarks:
                    draw_styled_landmarks(frame, result.multi_hand_landmarks[0])

                overlay_text(
                    frame,
                    f"Gesture: {gesture_label}  "
                    f"[{len(collected_sequences)}/{num_samples}]",
                    position=(10, 35),
                    colour=(255, 255, 0),
                )
                overlay_text(
                    frame,
                    "SPACE = record  |  Q = quit",
                    position=(10, 70),
                    colour=(200, 200, 200),
                    font_scale=0.6,
                )

                cv2.imshow("Gesture Collector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    waiting_for_space = False
                elif key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    _save_sequences(collected_sequences, output_path, gesture_label)
                    return

            # --- Record phase: collect SEQ_LEN frames -----------------------
            window_frames: list[np.ndarray] = []
            frame_count = 0

            while frame_count < SEQ_LEN:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Dropped frame during record phase.")
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                result = hands.process(rgb_frame)
                rgb_frame.flags.writeable = True

                if result.multi_hand_landmarks:
                    landmarks = extract_landmark_array(
                        result.multi_hand_landmarks[0]
                    )
                    draw_styled_landmarks(frame, result.multi_hand_landmarks[0])
                else:
                    # Zero-pad missing frames rather than skipping the window.
                    # This models "hand briefly off-screen" in training data.
                    landmarks = np.zeros(LANDMARK_DIM, dtype=np.float32)

                window_frames.append(landmarks)
                frame_count += 1

                progress = int((frame_count / SEQ_LEN) * 100)
                overlay_text(
                    frame,
                    f"RECORDING...  {progress}%",
                    position=(10, 35),
                    colour=(0, 0, 255),
                )
                cv2.imshow("Gesture Collector", frame)
                cv2.waitKey(1)

            sequence = np.stack(window_frames)  # (SEQ_LEN, LANDMARK_DIM)
            collected_sequences.append(sequence)
            print(
                f"  Collected {len(collected_sequences)}/{num_samples} sequences."
            )

            # Brief cooldown to avoid consecutive sequences being too similar.
            time.sleep(COOLDOWN_FRAMES / 30.0)

    cap.release()
    cv2.destroyAllWindows()
    _save_sequences(collected_sequences, output_path, gesture_label)


def _save_sequences(
    sequences: list[np.ndarray], output_path: Path, gesture_label: str
) -> None:
    """Stack collected sequences and write to disk."""
    if not sequences:
        print(f"[WARN] No sequences collected for '{gesture_label}'. Nothing saved.")
        return

    data = np.stack(sequences)  # (N, SEQ_LEN, LANDMARK_DIM)
    np.save(output_path, data)
    print(f"\n[INFO] Saved {data.shape[0]} sequences -> {output_path}")
    print(f"       Array shape: {data.shape}")


def main() -> None:
    args = parse_args()
    collect_sequences(
        gesture_label=args.gesture,
        num_samples=args.samples,
        output_dir=Path(args.output),
        camera_index=args.camera,
    )


if __name__ == "__main__":
    main()
