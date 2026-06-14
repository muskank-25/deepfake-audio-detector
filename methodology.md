# Deepfake Audio Detection — Methodology & Technical Description

## 1. Preprocessing

### 1.1 Audio Loading
- All audio files are loaded at a fixed **sample rate of 16,000 Hz (16kHz)** in mono channel using `librosa`.
- 16kHz is the standard rate for speech processing — it captures the full intelligible frequency range of human voice (up to 8kHz by Nyquist).

### 1.2 Fixed-Length Clipping
- Each audio clip is standardised to exactly **4 seconds (64,000 samples)**.
- Clips shorter than 4s are **zero-padded** at the end.
- Clips longer than 4s are **truncated** from the beginning.
- This ensures all inputs to the model have identical dimensions.

### 1.3 Amplitude Normalization
- The log-mel spectrogram is normalized **per clip** to the range [0, 1] using min-max scaling:
  ```
  normalized = (X - X.min) / (X.max - X.min + ε)
  ```
- Prevents the model from relying on absolute volume level, which varies across recordings.

---

## 2. Feature Extraction

### 2.1 Log-Mel Spectrogram (Primary — used by CNN)

The log-mel spectrogram is a 2D time-frequency representation of audio that closely mimics how the human auditory system processes sound.

**Pipeline:**
```
Raw Audio (64,000 samples)
    ↓ Short-Time Fourier Transform (STFT)
Power Spectrogram (n_fft=1024, hop=256)
    ↓ Mel Filterbank (128 triangular filters)
Mel Spectrogram (128 × 251 bins)
    ↓ Power-to-dB conversion
Log-Mel Spectrogram (128 × 251)
    ↓ Min-max normalization
Normalized Feature Map ∈ [0, 1]
```

**Parameters:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| Sample Rate | 16,000 Hz | Standard for speech |
| FFT window size (n_fft) | 1024 | ~64ms window — captures phoneme-level detail |
| Hop length | 256 | ~16ms step — 75% overlap for temporal resolution |
| Mel bands (n_mels) | 128 | Fine-grained frequency resolution |
| Duration | 4 seconds | Captures full utterance patterns |
| Output shape | 1 × 128 × 251 | (channels, freq_bins, time_frames) |

**Why log-mel for deepfake detection?**
- Real human speech has irregular, natural harmonic patterns that differ from TTS-generated audio.
- Deepfake audio often shows overly periodic or smooth spectral patterns.
- Log scale compresses the dynamic range, making these differences visually and computationally easier to detect.

### 2.2 MFCC + Delta Statistics (Secondary — used by Random Forest baseline)

Mel-Frequency Cepstral Coefficients (MFCCs) are compact spectral fingerprints widely used in speech recognition.

**Pipeline:**
```
Log-Mel Spectrogram
    ↓ Discrete Cosine Transform (DCT)
40 MFCC coefficients per frame
    ↓ Temporal delta (1st derivative)
Delta MFCCs (captures rate of change)
    ↓ Mean + Std across all frames
160-dimensional feature vector
```

**Feature vector composition:**

| Component | Dimensions | Captures |
|-----------|-----------|---------|
| MFCC mean | 40 | Average spectral shape |
| MFCC std | 40 | Spectral variability |
| Delta MFCC mean | 40 | Average rate of spectral change |
| Delta MFCC std | 40 | Variability of spectral change |
| **Total** | **160** | Compact audio fingerprint |

---

## 3. Model Architecture

### 3.1 Primary Model — AudioCNN

A compact 4-block 2D Convolutional Neural Network that treats the log-mel spectrogram as an image.

```
Input: (batch, 1, 128, 251)   ← single-channel spectrogram image
│
├── Block 1: Conv2D(1→16,  3×3) + BatchNorm + ReLU + MaxPool(2×2)  → (16, 64, 125)
├── Block 2: Conv2D(16→32, 3×3) + BatchNorm + ReLU + MaxPool(2×2)  → (32, 32, 62)
├── Block 3: Conv2D(32→64, 3×3) + BatchNorm + ReLU + MaxPool(2×2)  → (64, 16, 31)
├── Block 4: Conv2D(64→128,3×3) + BatchNorm + ReLU + GlobalAvgPool → (128, 1, 1)
│
├── Flatten → (128,)
├── Dropout(0.3)
├── FC(128 → 64) + ReLU
├── Dropout(0.3)
└── FC(64 → 2) → Softmax

Output: [P(Genuine), P(Deepfake)]
```

**Design choices:**

| Choice | Rationale |
|--------|-----------|
| 2D convolutions | Learns both frequency and time patterns jointly |
| BatchNorm after each conv | Stabilises training, reduces sensitivity to learning rate |
| Progressive filter doubling (16→32→64→128) | Captures low-level patterns first, then complex combinations |
| Global Average Pooling | Reduces overfitting vs flatten; makes model input-size flexible |
| Dropout (0.3) | Regularization to prevent overfitting |
| 2 FC layers | Non-linear decision boundary in learned feature space |
| Softmax output | Outputs calibrated probabilities for both classes |

**Total parameters:** ~186,000 (lightweight, deployable)

### 3.2 Baseline Model — Random Forest

A classical ML baseline for comparison and interpretability.

- **Input:** 160-dim MFCC feature vector (standardized with `StandardScaler`)
- **Model:** `RandomForestClassifier(n_estimators=300, random_state=42)`
- **Purpose:** Fast sanity check; confirms the task is learnable before investing in deep learning

---

## 4. Training Configuration

| Setting | Value |
|---------|-------|
| Loss function | Cross-Entropy Loss |
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Batch size | 16 |
| Epochs | 25 |
| Best model selection | Highest validation accuracy |
| Device | CPU / CUDA (auto-detected) |
| Random seed | 42 |

---

## 5. Evaluation Metrics

| Metric | Formula | Target | Meaning |
|--------|---------|--------|---------|
| Overall Accuracy | (TP+TN)/(TP+TN+FP+FN) | ≥ 80% | % of all samples classified correctly |
| F1 Score | 2×(P×R)/(P+R) | ≥ 80% | Harmonic mean of precision & recall |
| EER | FAR = FRR point | ≤ 12% | Lower = more balanced and reliable detector |
| Per-class Accuracy | TP/(TP+FN) per class | ≥ 75% | Neither class is systematically misclassified |
| Confusion Matrix | — | Required | Shows distribution of TP, TN, FP, FN |

