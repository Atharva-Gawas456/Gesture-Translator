# A Novel Approach to Real-Time Hand Gesture Recognition for Assistive Communication

**Author:** Atharva Gawas  
**Affiliation:** School of Engineering, VIT Bhopal University  
**Contact:** atharva.23bai10578@vitbhopal.ac.in  
**Date:** March 2026

---

## Abstract

Static image-based hand gesture recognition has been studied extensively, but practical deployment in assistive communication contexts demands temporal sequence analysis — the meaning of many signs depends not on a single frame but on the trajectory of the hand over time. This paper presents a system that uses MediaPipe's real-time hand-tracking module to extract 21 three-dimensional landmark coordinates per frame, forming a 63-dimensional feature vector. Sequences of 30 consecutive frames are classified by a Bidirectional Long Short-Term Memory (BiLSTM) network, enabling gesture recognition at approximately 28 frames per second on commodity hardware without a dedicated GPU. The system targets two high-utility Indian Sign Language (ISL) gestures — HELLO and THANK YOU — and achieves a validation accuracy of 100% on an internally collected dataset of 400 sequences. A lightweight Transformer-based alternative is also described for use with larger datasets. Key engineering constraints — including real-world lighting variance, partial occlusion, and the absence of GPU compute during inference — are addressed directly in the methodology and limitations sections.

---

## 1. Introduction

Communication barriers faced by the Deaf and hard-of-hearing community remain a significant accessibility gap. Trained sign language interpreters are scarce and expensive; automated translation has therefore been the subject of considerable research since at least the mid-1990s. Early systems relied on wired data gloves that measured finger flexion angles mechanically. Vision-based approaches subsequently became dominant, but were constrained by the computational cost of processing raw pixel data and the dependence on controlled laboratory backgrounds.

Two concurrent developments have made real-time, camera-only gesture recognition tractable: the availability of efficient landmark-detection models that run on CPU in near real time, and the maturation of recurrent neural network architectures capable of capturing temporal dynamics in short sequences.

The system described here occupies a specific point in this design space. It is not intended to replace full sign language interpretation — a problem that involves facial grammar, body posture, and regional vocabulary variants well beyond the scope of a single model. Instead, it targets word-level gesture recognition using a compact vocabulary of high-utility phrases, with the explicit design constraint that the entire pipeline must run on a laptop without discrete GPU hardware.

### 1.1 Scope and Vocabulary

The dataset covers two gesture classes selected for practical utility in a university environment: HELLO and THANK YOU. These were chosen because they represent the most common transactional exchanges and because their hand shapes are sufficiently distinct to serve as a proof-of-concept for the pipeline. The architecture is designed to scale to additional gesture classes with minimal code changes — only the label file and training data need to be extended.

---

## 2. Related Work

Hand gesture recognition research divides broadly into two categories: glove-based sensing and vision-based sensing. The former provides high-resolution articulation data at the cost of physical instrumentation and user encumbrance. We focus exclusively on the latter.

Mitra and Acharya (2007) provided a comprehensive survey of gesture recognition approaches, noting that Hidden Markov Models were at that time the dominant temporal classifier for gesture sequences. The introduction of deep learning produced a shift toward convolutional-recurrent architectures that operate directly on image frames (Koller et al., 2018). These approaches achieve strong accuracy on benchmark datasets such as RWTH-PHOENIX but require substantial GPU resources at inference time.

A different line of work focuses on landmark-based representations. Rather than processing raw pixels, these methods first reduce the image to a sparse set of keypoints (joints, landmarks), then classify the keypoint trajectory. Cao et al. (2017) demonstrated that skeletal pose estimation could be performed in real time using OpenPose. MediaPipe Hands (Zhang et al., 2020) extended this approach to hands specifically, providing a production-quality model that runs at over 30 fps on mobile hardware.

Our work builds on the landmark-based approach. By accepting the representational lossy compression from raw pixels to 21 keypoints, we gain both computational efficiency and robustness to minor appearance variation (skin tone, background) at the cost of losing texture information that might help disambiguate gestures with identical hand shapes at a single instant.

---

## 3. Methodology

### 3.1 Data Collection

Data was collected using a standard 1080p USB webcam at 30 fps under two lighting conditions: fluorescent overhead lighting (approximately 400 lux) and mixed natural/artificial lighting from a window-adjacent desk (100–250 lux). This intentional variance was included to prevent the model from learning a lighting-specific representation.

For each gesture class, 200 sequences were collected. Each sequence consists of 30 consecutive frames, corresponding to approximately one second of video. A custom collection script (`src/collect_data.py`) displays the live webcam feed, waits for the user to signal readiness (spacebar press), then records 30 frames. A 10-frame cooldown is applied between sequences to prevent consecutive samples from being near-duplicates.

**Dataset summary:**

| Gesture | Sequences | Lighting conditions |
|---|---|---|
| HELLO | 200 | Fluorescent, mixed |
| THANK YOU | 200 | Fluorescent, mixed |
| **Total** | **400** | |

Data was collected by a single operator. Cross-user generalisation is addressed in the Limitations section.

### 3.2 Landmark Extraction

MediaPipe Hands (Zhang et al., 2020) is used for landmark detection. The model detects the presence of a hand in each frame and returns 21 landmark positions. Each landmark is represented in three coordinates: x and y, normalised to [0, 1] by the image dimensions, and z, a depth estimate relative to the wrist landmark (negative values indicate proximity to the camera).

The 21 × 3 = 63 values are concatenated into a single floating-point vector per frame. This representation is deliberately simple. Alternatives considered included:

- **Velocity features:** First-order differences between consecutive landmark positions. These were ultimately not included because the LSTM hidden state can implicitly learn velocity from consecutive inputs without requiring the feature engineering step.
- **Bone lengths:** Euclidean distances between connected landmark pairs. These are invariant to hand translation within the frame but provide no temporal information. They were found to be useful for static gesture classification but offered no measurable improvement in the temporal setting.
- **Wrist-relative normalisation:** Subtracting the wrist landmark from all others to produce translation-invariant coordinates. This was evaluated but discarded because the absolute position of the hand within the frame carries some discriminative information for a small number of gestures.

For frames in which MediaPipe fails to detect a hand (a detection confidence below the threshold), the landmark vector is set to zeros. This decision models the "hand briefly off-screen" case that occurs when a signer repositions between gestures. The model is exposed to these zero vectors during training and learns to associate them with low-confidence outputs rather than misclassifying them.

**Detection thresholds:**  
`min_detection_confidence = 0.70` — tuned by observing the false-positive rate at different values during collection under the lower-light (100 lux) condition. Values below 0.65 caused MediaPipe to activate on forearm regions.  
`min_tracking_confidence = 0.60` — a looser threshold is appropriate for tracking because the model already has a prior from the previous frame.

### 3.3 Model Architecture

#### 3.3.1 Bidirectional LSTM

The primary architecture is a Bidirectional LSTM (BiLSTM). The bidirectional wrapper processes each input sequence in both forward and reverse temporal order, allowing the model to incorporate future context when making predictions at each time step. This is appropriate for offline evaluation but also works in the online setting because the prediction is emitted at the end of the sequence window rather than at each frame.

```
Input          →  (batch, 30, 63)
BiLSTM         →  128 units, return_sequences=True
Dropout        →  p = 0.30
LSTM           →  64 units
Dense          →  64, ReLU
Dropout        →  p = 0.30
Dense          →  num_classes, Softmax
```

Total trainable parameters: approximately 283,000.

The dropout rate of 0.30 was selected via grid search over {0.20, 0.30, 0.40}. Values of 0.20 produced mild overfitting (train/val accuracy diverged after epoch 25); 0.40 caused underfitting.

#### 3.3.2 Lightweight Transformer (Alternative)

For completeness and for users with larger datasets, a Transformer encoder variant is provided. It uses sinusoidal positional encoding, two encoder blocks each with two attention heads (key dimension 32), and a projection layer that maps the 63-dimensional input to a 64-dimensional embedding. The parameter count is kept comparable to the LSTM model (~120 K) to allow a fair comparison.

The Transformer is not recommended for datasets below approximately 1,000 samples per class because the attention mechanism requires more data to learn meaningful query–key relationships than the inductive bias of the LSTM, which explicitly models sequential dependencies.

### 3.4 Training Protocol

The dataset was split 80/20 into training and validation subsets using a stratified random split (seed 42) to preserve class proportions. This yields 320 training samples and 80 validation samples. No test set is held out explicitly; the validation set serves as the reported performance measure, consistent with the project scale.

- **Optimiser:** Adam, initial learning rate 1×10⁻³
- **Loss:** Sparse categorical cross-entropy
- **Batch size:** 32
- **Early stopping:** Patience 10 epochs, monitoring `val_loss`, restoring best weights
- **Learning rate schedule:** ReduceLROnPlateau, factor 0.5, patience 5 epochs

Training was conducted on a laptop CPU. No GPU was used. A full 50-epoch run completed in approximately 2 minutes for the 2-class, 400-sample dataset.

### 3.5 Inference Pipeline

The real-time inference loop operates as follows:

1. Each webcam frame is passed through MediaPipe to produce a 63-dimensional landmark vector.
2. Vectors are appended to a fixed-length deque of capacity 30. Until the deque is full, no prediction is made.
3. Once full, the deque contents form a (1, 30, 63) batch that is forwarded through the trained model.
4. The class with the highest Softmax probability is selected as the prediction.
5. If this probability exceeds `CONFIDENCE_THRESHOLD = 0.70` and at least `DEBOUNCE_FRAMES = 60` frames (approximately 2 seconds at 30 fps) have elapsed since the last announcement, the gesture label is passed to the text-to-speech engine.

The deque slides by one frame per camera frame, so predictions are made at the camera frame rate after the initial 30-frame fill period. In practice, consecutive predictions for a sustained gesture are highly correlated; the debounce mechanism prevents repetitive announcements.

---

## 4. Results

### 4.1 Validation Accuracy

The model was evaluated on the 80-sample validation set (40 per class) using an 80/20 stratified split.

| Gesture | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| HELLO | 1.0000 | 1.0000 | 1.0000 | 40 |
| THANK YOU | 1.0000 | 1.0000 | 1.0000 | 40 |
| **Macro avg** | **1.0000** | **1.0000** | **1.0000** | **80** |

Overall validation accuracy: **100.0%**

### 4.2 Confusion Matrix

|  | Predicted HELLO | Predicted THANK YOU |
|---|---|---|
| **Actual HELLO** | 40 | 0 |
| **Actual THANK YOU** | 0 | 40 |

The model achieves perfect classification on the validation set. This is expected given that the two gestures — an open-palm wave (HELLO) and a hand-to-chin motion (THANK YOU) — are kinematically distinct and unlikely to be confused. As additional gesture classes are added (particularly pairs with similar hand shapes such as HELLO and I LOVE YOU), validation accuracy is expected to decrease, and the confusion matrix will become more informative.

### 4.3 Inference Latency

| Component | Mean latency (ms) |
|---|---|
| MediaPipe landmark extraction | 12.4 |
| LSTM forward pass | 3.1 |
| pyttsx3 speech synthesis | 180–320 (non-blocking after first call) |
| **End-to-end (frame rate)** | **~28 fps** |

Speech synthesis is the dominant latency contributor but does not block the inference loop (it runs in the background via `runAndWait()` on the TTS engine). The effective display frame rate is approximately 28 fps, which is sufficient for smooth visual feedback.

---

## 5. Limitations

### 5.1 Lighting Sensitivity

MediaPipe's hand detection model was trained predominantly on well-lit, indoor images. In conditions below approximately 150 lux — which occurs in dimly lit rooms or when a strong backlight (e.g., a window behind the signer) creates high contrast — the landmark detection confidence falls below the 0.70 threshold frequently. In such conditions the system falls back to zero vectors, and consecutive missed detections prevent any gesture from being recognised. Practical mitigation strategies include ensuring that a light source faces the signer, or lowering `min_detection_confidence` to 0.60 with the understanding that false positives will increase.

### 5.2 Occlusion

When the signing hand overlaps with the face, the other hand, or a background object of similar skin tone, MediaPipe can lose tracking mid-gesture. The current pipeline handles brief occlusions (1–3 frames) gracefully via zero-padding but fails on occlusions exceeding approximately 5 frames, as the LSTM then processes a sequence that is partly uninformative. Extending the sequence length or implementing a dedicated occlusion-handling mechanism (e.g., optical flow to interpolate missing frames) would improve robustness.

### 5.3 Single-Operator Training

All training sequences were captured by a single person. The model therefore encodes the specific hand size, skin tone, and motor patterns of that individual. When evaluated on a different operator, validation accuracy dropped to approximately 74% in informal testing. Cross-user generalisation requires either a much larger and more diverse training corpus or fine-tuning protocols that allow rapid per-user adaptation with a small number of additional samples.

### 5.4 Vocabulary Constraints

The system currently recognises two gesture classes (HELLO and THANK YOU). While the architecture supports an arbitrary number of classes, extending the vocabulary requires collecting additional training data, retraining the model, and verifying that new classes are discriminable from existing ones. The `data/labels.txt` file and per-gesture `.npy` data files make vocabulary extension straightforward. No mechanism currently exists for online or incremental learning.

### 5.5 Single-Hand Processing

The pipeline processes one hand at a time (configured via `max_num_hands=1` in MediaPipe). Signs that require two hands in coordination — a significant portion of ISL and ASL — are therefore outside the scope of this implementation. Extending to two hands doubles the feature dimension to 126 and requires coordinating the tracking IDs of both hands across frames.

---

## 6. Conclusion

This paper described a real-time sign language recognition system that combines MediaPipe hand tracking with a Bidirectional LSTM classifier operating on sequences of 63-dimensional landmark vectors. The system achieves 100% validation accuracy on a two-class dataset (HELLO and THANK YOU, 400 sequences total) while running at approximately 28 fps on a CPU-only laptop. A lightweight Transformer variant was also described for future use with larger datasets.

The perfect accuracy on two kinematically distinct gestures serves as a successful proof-of-concept for the pipeline. The most important next step is extending the vocabulary to include additional gesture classes that are more likely to produce inter-class confusion, which will provide a more rigorous evaluation of the model's discriminative capacity.

The most pressing open problem in this project is cross-user generalisation. Addressing this through data diversity is the most straightforward path forward, though transfer learning and domain adaptation techniques could offer more sample-efficient alternatives.

This project was developed and tested within the VIT Bhopal computer lab under standard fluorescent lighting (approximately 400 lux at desk level). The practical experience of seeing the system fail under the relatively dim lighting of an adjacent corridor reinforced the importance of the lighting limitations documented above and motivates the edge-case robustness work planned for the next iteration.

---

## References

1. Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C.-L., & Grundmann, M. (2020). *MediaPipe Hands: On-device Real-time Hand Tracking*. arXiv preprint arXiv:2006.10214.

2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

3. Mitra, S., & Acharya, T. (2007). Gesture recognition: A survey. *IEEE Transactions on Systems, Man, and Cybernetics, Part C*, 37(3), 311–324.

4. Koller, O., Camgoz, N. C., Ney, H., & Bowden, R. (2018). Weakly supervised learning with multi-stream CNN-LSTM-HMMs to discover sequential parallelism in sign language videos. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(9), 2306–2320.

5. Cao, Z., Simon, T., Wei, S.-E., & Sheikh, Y. (2017). Realtime multi-person 2D pose estimation using part affinity fields. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 7291–7299.

6. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

7. Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M. (2020). *BlazePose: On-device Real-time Body Pose Tracking*. arXiv preprint arXiv:2006.10204.
