import os, json, tempfile
import numpy as np
import streamlit as st

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️")
st.title("🎙️ Deepfake Audio Detector")
st.write("Upload a speech recording to detect if it is **Genuine (Human)** or **Deepfake (AI-Generated)**.")

@st.cache_resource
def load_everything():
    import torch
    import torch.nn as nn
    import librosa

    class AudioCNN(nn.Module):
        def __init__(self, n_classes=2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1,16,kernel_size=3,padding=1),
                nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(16,32,kernel_size=3,padding=1),
                nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(32,64,kernel_size=3,padding=1),
                nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(64,128,kernel_size=3,padding=1),
                nn.BatchNorm2d(128),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),nn.Dropout(0.3),
                nn.Linear(128,64),nn.ReLU(),
                nn.Dropout(0.3),nn.Linear(64,n_classes)
            )
        def forward(self,x):
            x = self.features(x)
            return self.classifier(x)

    # Find config and model files
    cfg_path, mdl_path = None, None
    for p in ["./config.json","./artifacts/config.json"]:
        if os.path.exists(p): cfg_path = p; break
    for p in ["./deepfake_cnn.pt","./artifacts/deepfake_cnn.pt"]:
        if os.path.exists(p): mdl_path = p; break

    if not cfg_path or not mdl_path:
        return None, None, None, None

    with open(cfg_path) as f: cfg = json.load(f)
    device = torch.device("cpu")
    model = AudioCNN()
    model.load_state_dict(torch.load(mdl_path, map_location=device))
    model.eval()
    return model, cfg, device, librosa

model, config, device, librosa = load_everything()

if model is None:
    st.error("⚠️ Model files not found. Make sure `deepfake_cnn.pt` and `config.json` are in the repo root.")
    st.stop()

# Show model metrics
if "metrics" in config:
    m = config["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", f"{m['accuracy']*100:.1f}%")
    c2.metric("F1 Score",      f"{m['f1']*100:.1f}%")
    c3.metric("EER",           f"{m['eer']*100:.1f}%")

st.divider()

# Upload audio
uploaded = st.file_uploader("Upload Audio File", type=["wav","flac","mp3","ogg","m4a"])
st.caption("Supported: WAV · FLAC · MP3 · OGG · M4A")

if uploaded:
    st.audio(uploaded)

    suffix = os.path.splitext(uploaded.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("🔍 Analysing audio..."):
        import torch
        SR  = config["SAMPLE_RATE"]
        DUR = config["DURATION"]

        y, _ = librosa.load(tmp_path, sr=SR, mono=True)
        n    = SR * DUR
        y    = np.pad(y, (0, max(0, n - len(y))))[:n]

        mel = librosa.feature.melspectrogram(
            y=y, sr=SR,
            n_fft=config["N_FFT"],
            hop_length=config["HOP_LENGTH"],
            n_mels=config["N_MELS"])
        lm = librosa.power_to_db(mel, ref=np.max)
        lm = ((lm - lm.min()) / (lm.max() - lm.min() + 1e-8)).astype(np.float32)

        x = torch.tensor(lm).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].numpy()
        pred = int(probs.argmax())

    os.remove(tmp_path)

    # Verdict
    st.divider()
    if pred == 1:
        st.error(f"## 🚨 DEEPFAKE (AI-Generated)")
        st.error(f"### Confidence: {probs[1]*100:.1f}%")
    else:
        st.success(f"## ✅ GENUINE (Human)")
        st.success(f"### Confidence: {probs[0]*100:.1f}%")

    # Probabilities
    st.divider()
    st.subheader("📊 Probability Breakdown")
    c1, c2 = st.columns(2)
    c1.metric("✅ Genuine",  f"{probs[0]*100:.1f}%")
    c2.metric("🚨 Deepfake", f"{probs[1]*100:.1f}%")
    st.progress(float(probs[0]), text=f"Genuine:  {probs[0]*100:.1f}%")
    st.progress(float(probs[1]), text=f"Deepfake: {probs[1]*100:.1f}%")

    # Spectrogram
    st.divider()
    st.subheader("🌈 Log-Mel Spectrogram")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(
        lm, x_axis="time", y_axis="mel",
        sr=SR, hop_length=config["HOP_LENGTH"],
        ax=ax, cmap="magma")
    label = "Genuine (Human)" if pred == 0 else "Deepfake (AI-Generated)"
    ax.set_title(label)
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.caption("CNN on Log-Mel Spectrograms · 16kHz · Fake-or-Real Dataset")
