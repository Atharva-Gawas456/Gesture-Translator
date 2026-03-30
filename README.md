# Gesture-Translator

**Real-Time Sign Language to Speech Translator**  
*Computer Vision Project — Atharva Gawas | GitHub: [Atharva-Gawas456](https://github.com/Atharva-Gawas456)*

---

## Overview

Gesture-Translator is a pipeline that converts hand gestures — specifically a subset of Indian Sign Language (ISL) and American Sign Language (ASL) fingerspelling — into synthesised speech in real time.

The system uses MediaPipe's hand-tracking module to extract 21 3D landmarks per frame. A sequence of 30 consecutive landmark frames (~1 second at 30 fps) is fed into a trained Bidirectional LSTM, which classifies the gesture. The predicted label is then spoken aloud via `pyttsx3`, an offline text-to-speech engine.

This project was developed as part of a Computer Vision project and tested under fluorescent lighting in a university lab environment.

---

## Project Structure

```
Gesture-Translator/
│
├── data/                  # NumPy .npy files (one per gesture class)
│   └── labels.txt         # One gesture label per line
├── models/                # Saved TensorFlow model files
├── src/
│   ├── utils.py           # Shared helpers: landmark extraction, drawing, I/O
│   ├── collect_data.py    # Webcam-based gesture data collector
│   ├── train.py           # LSTM / Transformer training script
│   └── main.py            # Real-time inference + speech synthesis
├── research_paper.md      # Full methodology write-up
├── README.md
└── requirements.txt
```

---

## Setup

### 1. Prerequisites

- Python **3.10+** (tested on 3.10, 3.11, and 3.13)
- A working webcam
- `pip` and optionally a virtual environment manager (`venv` or `conda`)

### 2. Clone the repository

```bash
git clone https://github.com/Atharva-Gawas456/Gesture-Translator.git
cd Gesture-Translator
```

### 3. Create and activate a virtual environment

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Linux note:** `pyttsx3` requires `espeak` on Linux.  
> Install with: `sudo apt-get install espeak`

---

## Usage

### Step 1 — Create a label file

Create `data/labels.txt` with one gesture name per line:

```
hello
thankyou
```

### Step 2 — Collect training data

Run the data collector once per gesture:

```bash
python src/collect_data.py --gesture hello --samples 200 --output data/
python src/collect_data.py --gesture thankyou --samples 200 --output data/
```

- A live webcam window opens.
- Press **SPACE** to start recording each 30-frame sequence.
- Press **Q** to quit early (partial data is saved).
- Each run saves `data/<gesture>.npy` — a `(200, 30, 63)` array.

### Step 3 — Train the model

```bash
python src/train.py \
    --data data/ \
    --labels data/labels.txt \
    --epochs 50 \
    --batch-size 32 \
    --model lstm \
    --output models/
```

Use `--model transformer` if you have more than ~1 000 samples per class.

Training output is saved to `models/gesture_model.keras`.

### Step 4 — Run real-time inference

```bash
python src/main.py \
    --model models/gesture_model.keras \
    --labels data/labels.txt
```

- A webcam window shows the live feed with prediction overlay.
- Recognised gestures are spoken aloud automatically.
- Press **Q** to quit.

**Disable speech synthesis** (useful for quick testing):

```bash
python src/main.py --model models/gesture_model.keras --labels data/labels.txt --no-speech
```

---

## Model Architecture

### Bidirectional LSTM (default)

```
Input  →  (batch, 30, 63)
           ↓
BiLSTM     128 units, return_sequences=True
           ↓
Dropout    0.3
           ↓
LSTM       64 units
           ↓
Dense      64, ReLU
           ↓
Dropout    0.3
           ↓
Dense      num_classes, Softmax
```

**Total parameters:** ~283 K (small enough to run on CPU in real time)

### Lightweight Transformer (optional)

For datasets with more than ~1 000 samples per class, the `--model transformer` flag selects a 2-layer, 2-head Transformer encoder with a comparable parameter count.

---

## Tuning Notes

| Parameter | Default | Rationale |
|---|---|---|
| `min_detection_confidence` | 0.70 | Balances missed detections and false positives under mixed lighting |
| `min_tracking_confidence` | 0.60 | Lower threshold tolerable because hand motion between frames is smooth |
| `CONFIDENCE_THRESHOLD` | 0.70 | Reduces spurious activations during gesture transitions |
| `DEBOUNCE_FRAMES` | 60 | ~2 s at 30 fps; prevents repeated announcements for held gestures |
| `Dropout` | 0.30 | Grid-searched over {0.2, 0.3, 0.4}; 0.3 maximised validation accuracy |

---

## Future Scope

- **Two-hand support:** The current pipeline processes a single hand. Extending to two hands would require concatenating two 63-dimensional vectors per frame and retraining.
- **Dynamic vocabulary:** An online learning mode where new gestures can be added with as few as 50 samples using transfer learning on the trained feature extractor.
- **Edge deployment:** Quantise the model to INT8 using TensorFlow Lite for deployment on a Raspberry Pi 4 or Jetson Nano without a GPU.
- **Sentence-level output:** Buffer multiple consecutive gesture predictions and run a language model to compose grammatically correct sentences rather than isolated words.

---

## Known Limitations

- Performance degrades significantly when the hand overlaps with the face or a similarly-coloured background.
- Recognition accuracy drops ~15% under lighting below ~200 lux (typical candle-lit or dimly lit interiors).
- The classifier is trained on a single operator. Cross-user performance requires re-collection or fine-tuning.

---

## Citation

If you use this work, please cite:

```
Zhang, F., et al. (2020). MediaPipe Hands: On-device Real-time Hand Tracking.
arXiv preprint arXiv:2006.10214.
```

---

## License

MIT License — see `LICENSE` for details.
