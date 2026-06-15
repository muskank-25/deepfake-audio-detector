<div align="center">

# 🎙️ Deepfake Audio Detection

**Classify speech recordings as Genuine (Human) or Deepfake (AI-Generated)**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=flat-square&logo=streamlit)
![Dataset](https://img.shields.io/badge/Dataset-69300%20files-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

### 🌐 [Live Demo →](https://deepfake-audio-detector-rend8dwba93vs4mv2nkcaf.streamlit.app)

</div>

---

## 📌 Project Description

Advances in generative AI have made it possible to create highly realistic synthetic speech — commonly known as **deepfake audio**. Such audio can be misused for impersonation, fraud, misinformation, and social engineering attacks.

This project builds a **CNN-based deep learning system** that detects whether any speech recording is:
- ✅ **Genuine** — real human speech
- 🚨 **Deepfake** — AI-generated synthetic speech

The model is trained on **69,300 real audio files** from the Fake-or-Real dataset and deployed as a live Streamlit web application that returns a prediction with a **confidence score**.

---

## 📁 Repository Structure

```
deepfake-audio-detector/
│
├── notebook.ipynb        # Full pipeline: data → features → training → evaluation
├── train_pipeline.py     # Standalone CLI training script
├── predict.py            # Test model on new audio samples
├── app.py                # Streamlit web app (hosted)
├── requirements.txt      # Python dependencies
├── packages.txt          # System dependencies for Streamlit Cloud
├── README.md             # This file
│
├── deepfake_cnn.pt       # Trained CNN model weights
├── config.json           # Feature config + evaluation metrics
└── scaler.pkl            # StandardScaler for baseline model
```

---

## 🗂️ Dataset

**[The Fake-or-Real (FoR) Dataset](https://kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)**

| Split | Genuine | Deepfake | Total |
|-------|---------|----------|-------|
| Training | ~17,000 | ~17,000 | ~34,000 |
| Validation | ~2,500 | ~2,500 | ~5,000 |
| Testing | ~2,500 | ~2,500 | ~5,000 |
| **Total** | **34,605** | **34,695** | **69,300** |

Expected folder structure:
```
for-norm/
├── training/
│   ├── real/    ← genuine human speech (.wav)
│   └── fake/    ← AI-generated speech (.wav)
├── validation/
│   ├── real/
│   └── fake/
└── testing/
    ├── real/
    └── fake/
```

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/muskank-25/deepfake-audio-detector.git
cd deepfake-audio-detector

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train — Google Colab (Recommended)
```
1. Open notebook.ipynb in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Runtime → Run all
4. Model files download automatically when done
```

### 2. Train — Command Line
```bash
python train_pipeline.py --data_dir ./for-norm --epochs 50
```

### 3. Test on a Single Audio File
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
  Confidence : 94.30%
  P(Genuine) : 94.30%
  P(Deepfake): 5.70%
=============================================
```

### 4. Batch Prediction (CSV)
```bash
python predict.py --csv files.csv --out_csv results.csv
```

### 5. Run Web App Locally
```bash
streamlit run app.py
```

---

## 🧠 Methodology & Pipeline

### Step 1 — Preprocessing

| Step | Detail |
|------|--------|
| Sample Rate | 16,000 Hz (16kHz) — standard for speech |
| Duration | Fixed 4 seconds (pad short / truncate long) |
| Normalization | Per-clip min-max to range [0, 1] |

### Step 2 — Feature Extraction

**Log-Mel Spectrogram** (CNN input — 128 × 251 image)

Converts raw audio into a 2D time-frequency representation that captures spectral patterns unique to genuine vs synthetic speech.

| Parameter | Value | Purpose |
|-----------|-------|---------|
| FFT Window (n_fft) | 1024 samples | ~64ms — captures phoneme-level detail |
| Hop Length | 256 samples | ~16ms step — high temporal resolution |
| Mel Bands | 128 | Fine-grained frequency resolution |
| Output Shape | 1 × 128 × 251 | Single-channel spectrogram image |

**Why Log-Mel for deepfake detection?**
- Human speech has irregular, natural harmonic patterns
- AI-generated speech shows overly smooth or periodic spectral patterns
- Log scale emphasises perceptually important frequency differences
- 2D CNN can learn both frequency and temporal artifact patterns jointly

### Step 3 — CNN Model Architecture

```
Input  (batch × 1 × 128 × 251)
  │
  ├─ Conv2D(1→16,  3×3) + BatchNorm + ReLU + MaxPool(2×2) → (16 × 64 × 125)
  ├─ Conv2D(16→32, 3×3) + BatchNorm + ReLU + MaxPool(2×2) → (32 × 32 × 62)
  ├─ Conv2D(32→64, 3×3) + BatchNorm + ReLU + MaxPool(2×2) → (64 × 16 × 31)
  ├─ Conv2D(64→128,3×3) + BatchNorm + ReLU + GlobalAvgPool → (128 × 1 × 1)
  │
  ├─ Flatten → Dropout(0.4)
  ├─ FC(128 → 64) + ReLU + Dropout(0.3)
  └─ FC(64 → 2) → Softmax
  │
  Output: [P(Genuine), P(Deepfake)]
```

**Design Choices:**

| Choice | Rationale |
|--------|-----------|
| 2D Convolutions | Learns joint frequency + time artifact patterns |
| BatchNormalization | Stabilises training, faster convergence |
| Progressive filters (16→128) | Low-level → high-level feature hierarchy |
| Global Average Pooling | Reduces overfitting vs Flatten |
| Dropout (0.4 + 0.3) | Strong regularization for generalization |
| Class Weights (1.0, 2.0) | Corrects class imbalance for Deepfake detection |

**Total Parameters:** ~186,000 (lightweight and fast)

### Step 4 — Training Configuration

| Setting | Value |
|---------|-------|
| Loss Function | Cross-Entropy Loss (with class weights) |
| Optimizer | Adam (lr=3e-4, weight_decay=1e-4) |
| LR Scheduler | CosineAnnealingLR |
| Batch Size | 32 |
| Epochs | 50 |
| Best Model | Saved on highest validation accuracy |
| Device | T4 GPU (Colab) / CPU auto-detected |
| Random Seed | 42 |

### Step 5 — Evaluation Metrics

| Metric | Formula | Target | Meaning |
|--------|---------|--------|---------|
| Overall Accuracy | (TP+TN)/(Total) | ≥ 80% | % correctly classified |
| F1 Score | 2×(P×R)/(P+R) | ≥ 80% | Balance of precision & recall |
| EER | FAR = FRR point | ≤ 12% | Lower = more reliable detector |
| Per-class Accuracy | TP/(TP+FN) | ≥ 75% each | Neither class systematically wrong |
| Confusion Matrix | TP/TN/FP/FN | Required | Full error distribution |

---

## 📊 Results

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Overall Accuracy | 71.0% | ≥ 80% | 🔄 Improving |
| F1 Score | 62.6% | ≥ 80% | 🔄 Improving |
| Equal Error Rate (EER) | 21.2% | ≤ 12% | 🔄 Improving |
| Genuine Accuracy | 93.0% | ≥ 75% | ✅ PASS |
| Deepfake Accuracy | 49.0% | ≥ 75% | 🔄 Improving |

> 🔄 Model is currently being retrained with class weighting and more epochs to meet all targets.

---

## 🌐 Streamlit Web App

**Live URL:** https://deepfake-audio-detector-rend8dwba93vs4mv2nkcaf.streamlit.app/

| Feature | Description |
|---------|-------------|
| 📤 Upload | WAV · FLAC · MP3 · OGG · M4A |
| 🔊 Playback | Listen to audio before analysing |
| ✅/🚨 Verdict | Bold Genuine or Deepfake result |
| 📊 Confidence | Exact % confidence score |
| 📈 Probability Bars | Visual breakdown for both classes |
| 🌈 Spectrogram | Log-Mel spectrogram visualization |
| 📋 Evidence Dossier | Full prediction details |
| 📉 Model Stats | Live Accuracy, F1, EER display |

---

## 📦 Dependencies

```
torch          # Deep learning
librosa        # Audio feature extraction
scikit-learn   # Metrics + preprocessing
streamlit      # Web application
soundfile      # Audio file I/O
matplotlib     # Plots
seaborn        # Confusion matrix
numpy          # Numerical computing
pandas         # Data handling
tqdm           # Progress bars
```



## 📄 License

MIT License — free to use for academic and research purposes.

---

<div align="center">
Made with ❤️ using PyTorch · Librosa · Streamlit
</div>
