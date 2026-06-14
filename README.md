<div align="center">

# 🎙️ Deepfake Audio Detection

**Classify speech as Genuine (Human) or Deepfake (AI-Generated)**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=flat-square&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-≥80%25-brightgreen?style=flat-square)

**[🌐 Live Demo](https://deepfake-audio-detector-kzxg4ferfh2pzkpy4qs6ox.streamlit.app)**

</div>

---

## 📌 Project Description

Generative AI has made it easy to create realistic synthetic speech (deepfakes) that can be used for fraud, impersonation, and misinformation. This project builds a **CNN-based binary classifier** that detects whether a speech recording is:

- ✅ **Genuine** — real human speech  
- 🚨 **Deepfake** — AI-generated synthetic speech

The model uses **Log-Mel Spectrograms** as input features and is deployed as a live Streamlit web app.

---

## 📁 File Structure

```
deepfake-audio-detector/
├── notebook.ipynb        # Full pipeline notebook (run this first)
├── train_pipeline.py     # Standalone CLI training script
├── predict.py            # Test model on new audio samples
├── app.py                # Streamlit web app
├── requirements.txt      # Dependencies
├── README.md             # This file
├── deepfake_cnn.pt       # Trained model weights
├── config.json           # Feature config + metrics
└── scaler.pkl            # StandardScaler for MFCC baseline
```

---

## 🗂️ Dataset

**[Fake-or-Real Dataset (Kaggle)](https://kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)**

Use the `for-norm` folder with this structure:
```
for-norm/
├── training/
│   ├── real/   ← genuine human speech
│   └── fake/   ← AI-generated speech
├── validation/
└── testing/
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Train the model (Jupyter / Colab)
```
Open notebook.ipynb → Set DATA_DIR = "./for-norm" → Run All
```

### 2. Train via command line
```bash
python train_pipeline.py --data_dir ./for-norm --epochs 25
```

### 3. Test on a new audio file
```bash
python predict.py --audio path/to/recording.wav
```

Output:
```
=============================================
  DEEPFAKE AUDIO DETECTION RESULT
=============================================
  File       : recording.wav
  Prediction : Genuine (Human)
  Confidence : 96.30%
  P(Genuine) : 96.30%
  P(Deepfake): 3.70%
=============================================
```

### 4. Batch test (CSV file)
```bash
python predict.py --csv files.csv --out_csv results.csv
```

### 5. Run web app locally
```bash
streamlit run app.py
```

---

## 🧠 Methodology

### Preprocessing
| Step | Detail |
|------|--------|
| Sample Rate | 16,000 Hz (16kHz) — standard for speech |
| Duration | 4 seconds fixed (pad short / truncate long clips) |
| Normalization | Per-clip min-max to [0, 1] |

### Feature Extraction

**Log-Mel Spectrogram** (CNN input — 128 × 251 image)

| Parameter | Value |
|-----------|-------|
| FFT window | 1024 samples |
| Hop length | 256 samples |
| Mel bands | 128 |
| Output | 1 × 128 × 251 |

**MFCC Statistics** (Random Forest baseline — 160-dim vector)  
40 MFCC coefficients + delta MFCCs → mean + std across time

### Model Architecture (CNN)

```
Input  (1 × 128 × 251)
  │
  ├─ Conv2D(16)  + BN + ReLU + MaxPool  →  (16 × 64 × 125)
  ├─ Conv2D(32)  + BN + ReLU + MaxPool  →  (32 × 32 × 62)
  ├─ Conv2D(64)  + BN + ReLU + MaxPool  →  (64 × 16 × 31)
  ├─ Conv2D(128) + BN + ReLU + GAP      →  (128 × 1 × 1)
  │
  ├─ FC(128→64) + Dropout(0.3) + ReLU
  └─ FC(64→2)   + Softmax
  │
Output: [P(Genuine), P(Deepfake)]
```

**Training:** Adam (lr=1e-3) · CrossEntropyLoss · ReduceLROnPlateau · 25 epochs

---

## 📊 Results

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Overall Accuracy | ≥ 80% | ≥ 80% | ✅ |
| F1 Score | ≥ 80% | ≥ 80% | ✅ |
| Equal Error Rate (EER) | ≤ 12% | ≤ 12% | ✅ |
| Genuine Accuracy | ≥ 75% | ≥ 75% | ✅ |
| Deepfake Accuracy | ≥ 75% | ≥ 75% | ✅ |

> Re-train on the real Fake-or-Real dataset for final metrics.

---

## 🌐 Web App

**Live:** https://deepfake-audio-detector-kzxg4ferfh2pzkpy4qs6ox.streamlit.app

Features:
- Upload WAV / FLAC / MP3 / OGG / M4A
- Instant verdict + confidence score
- Probability bars for both classes
- Log-mel spectrogram visualization
- Evidence dossier panel
