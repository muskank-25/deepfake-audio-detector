"""
app.py — Deepfake Audio Detection Streamlit App

Run locally:
    streamlit run app.py

Expects trained artifacts in ./artifacts/:
    - deepfake_cnn.pt
    - config.json
    - (optional) scaler.pkl
"""

import os
import json
import tempfile

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import streamlit as st


# ----------------------- Model -----------------------
class AudioCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


ARTIFACTS_DIR = "./artifacts"


@st.cache_resource
def load_model_and_config():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.join(ARTIFACTS_DIR, "config.json")
    model_path = os.path.join(ARTIFACTS_DIR, "deepfake_cnn.pt")

    if not (os.path.exists(config_path) and os.path.exists(model_path)):
        return None, None, device

    with open(config_path) as f:
        config = json.load(f)

    model = AudioCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, config, device


def load_audio(filepath, sr, duration):
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    target_len = sr * duration
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def extract_logmel(y, sr, n_fft, hop_length, n_mels):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-8)
    return logmel.astype(np.float32)


def predict_file(filepath, model, config, device):
    y = load_audio(filepath, config["SAMPLE_RATE"], config["DURATION"])
    logmel = extract_logmel(y, config["SAMPLE_RATE"], config["N_FFT"], config["HOP_LENGTH"], config["N_MELS"])
    x = torch.tensor(logmel).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        pred_idx = int(probs.argmax())

    label = "Genuine" if pred_idx == 0 else "Deepfake"
    return {
        "label": label,
        "is_deepfake": pred_idx == 1,
        "confidence": float(probs[pred_idx]),
        "prob_genuine": float(probs[0]),
        "prob_deepfake": float(probs[1]),
    }, y, logmel


# ----------------------- UI -----------------------
st.set_page_config(page_title="Deepfake Audio Detection", page_icon="🎙️", layout="centered")
st.title("🎙️ Deepfake Audio Detection")
st.caption("Upload an audio file to classify it as **Genuine (Human)** or **Deepfake (AI-Generated)**.")

model, config, device = load_model_and_config()

if model is None:
    st.error(
        "Model artifacts not found in `./artifacts/`. "
        "Run `train_pipeline.py` (or the notebook) first to generate "
        "`deepfake_cnn.pt` and `config.json`."
    )
    st.stop()

if "metrics" in config:
    with st.expander("Model performance (test set)"):
        m = config["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
        c2.metric("F1 Score", f"{m['f1']*100:.1f}%")
        c3.metric("EER", f"{m['eer']*100:.1f}%")

tab_single, tab_batch = st.tabs(["Single File", "Batch (CSV)"])

# ---------- Single file ----------
with tab_single:
    uploaded = st.file_uploader("Upload audio file", type=["wav", "flac", "mp3"])

    if uploaded is not None:
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.audio(uploaded)

        with st.spinner("Analyzing audio..."):
            result, y, logmel = predict_file(tmp_path, model, config, device)

        if result["is_deepfake"]:
            st.error(f"### 🚨 Deepfake (AI-Generated)\nConfidence: **{result['confidence']*100:.1f}%**")
        else:
            st.success(f"### ✅ Genuine (Human)\nConfidence: **{result['confidence']*100:.1f}%**")

        c1, c2 = st.columns(2)
        c1.metric("P(Genuine)", f"{result['prob_genuine']*100:.1f}%")
        c2.metric("P(Deepfake)", f"{result['prob_deepfake']*100:.1f}%")

        with st.expander("Log-mel spectrogram"):
            fig, ax = plt.subplots(figsize=(6, 3))
            img = librosa.display.specshow(
                logmel, x_axis="time", y_axis="mel",
                sr=config["SAMPLE_RATE"], hop_length=config["HOP_LENGTH"], ax=ax
            )
            fig.colorbar(img, ax=ax, format="%+2.0f")
            st.pyplot(fig)

        os.remove(tmp_path)

# ---------- Batch CSV ----------
with tab_batch:
    st.write("Upload a CSV with a column named **`filepath`** pointing to local audio files, "
             "or upload multiple audio files directly.")

    csv_file = st.file_uploader("Upload CSV with 'filepath' column", type=["csv"], key="csv")
    multi_audio = st.file_uploader(
        "...or upload multiple audio files", type=["wav", "flac", "mp3"],
        accept_multiple_files=True, key="multi"
    )

    rows = []

    if csv_file is not None:
        df_in = pd.read_csv(csv_file)
        if "filepath" not in df_in.columns:
            st.error("CSV must contain a 'filepath' column.")
        else:
            for fp in df_in["filepath"]:
                if os.path.exists(fp):
                    result, _, _ = predict_file(fp, model, config, device)
                    rows.append({"file": fp, **result})
                else:
                    rows.append({"file": fp, "label": "ERROR: file not found"})

    if multi_audio:
        for f in multi_audio:
            suffix = os.path.splitext(f.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            result, _, _ = predict_file(tmp_path, model, config, device)
            rows.append({"file": f.name, **result})
            os.remove(tmp_path)

    if rows:
        results_df = pd.DataFrame(rows)
        st.dataframe(results_df, use_container_width=True)

        n_fake = (results_df.get("is_deepfake") == True).sum() if "is_deepfake" in results_df else 0
        n_real = (results_df.get("is_deepfake") == False).sum() if "is_deepfake" in results_df else 0
        c1, c2 = st.columns(2)
        c1.metric("Genuine", n_real)
        c2.metric("Deepfake", n_fake)

        csv_out = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions as CSV", csv_out, "predictions.csv", "text/csv")

st.divider()
st.caption("Model: CNN on log-mel spectrograms | Dataset: Fake-or-Real (FoR), ASVspoof 2019 (optional)")
