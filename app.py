"""
Deepfake Audio Detection — Streamlit Web App
Run: streamlit run app.py
"""

import os, json, tempfile, pickle
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-size: 2.6rem; font-weight: 900; text-align: center;
    background: linear-gradient(135deg, #6C63FF 0%, #FF6584 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    text-align: center; color: #888; font-size: 1rem; margin-bottom: 2rem;
}
.card {
    background: #1e1e2e; border-radius: 16px; padding: 1.6rem 2rem;
    margin: 1rem 0; border: 1px solid #2e2e4e;
}
.result-genuine {
    background: linear-gradient(135deg, #0f2027, #1a3a2a);
    border: 2px solid #00e676; border-radius: 20px;
    padding: 2rem; text-align: center; margin: 1.5rem 0;
}
.result-deepfake {
    background: linear-gradient(135deg, #2d0000, #3a0a0a);
    border: 2px solid #ff1744; border-radius: 20px;
    padding: 2rem; text-align: center; margin: 1.5rem 0;
}
.result-icon { font-size: 4rem; margin-bottom: 0.4rem; }
.result-label { font-size: 2rem; font-weight: 900; }
.result-label-genuine { color: #00e676; }
.result-label-deepfake { color: #ff1744; }
.result-conf { font-size: 1rem; color: #bbb; margin-top: 0.4rem; }

.prob-bar-wrap { margin: 1.2rem 0; }
.prob-label { font-size: 0.85rem; color: #aaa; margin-bottom: 4px; }
.prob-track {
    background: #2a2a3e; border-radius: 99px; height: 22px;
    overflow: hidden; position: relative;
}
.prob-fill-genuine {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #00b09b, #00e676);
    transition: width 0.8s ease;
}
.prob-fill-deepfake {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #c0392b, #ff1744);
    transition: width 0.8s ease;
}
.prob-pct {
    position: absolute; right: 10px; top: 50%;
    transform: translateY(-50%); font-size: 0.78rem;
    font-weight: 700; color: white;
}

.metric-row { display: flex; gap: 1rem; margin: 0.8rem 0; }
.metric-box {
    flex: 1; background: #12121e; border-radius: 12px;
    padding: 1rem; text-align: center; border: 1px solid #2e2e4e;
}
.metric-val { font-size: 1.6rem; font-weight: 800; color: #6C63FF; }
.metric-name { font-size: 0.75rem; color: #888; margin-top: 2px; }

.step-pill {
    display: inline-block; background: #6C63FF22; color: #6C63FF;
    border-radius: 99px; padding: 2px 12px; font-size: 0.75rem;
    font-weight: 700; margin-bottom: 6px; border: 1px solid #6C63FF44;
}
.uploader-hint { text-align: center; color: #666; font-size: 0.85rem; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
        )
        self.clf = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,2)
        )
    def forward(self, x): return self.clf(self.features(x))

ARTIFACTS = "."

@st.cache_resource
def load_model():
    cfg_path = os.path.join(ARTIFACTS, "config.json")
    mdl_path = os.path.join(ARTIFACTS, "deepfake_cnn.pt")
    if not os.path.exists(cfg_path) or not os.path.exists(mdl_path):
        return None, None
    with open(cfg_path) as f:
        cfg = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = AudioCNN().to(device)
    mdl.load_state_dict(torch.load(mdl_path, map_location=device))
    mdl.eval()
    return mdl, cfg

# ─────────────────────────────────────────────
# AUDIO UTILS
# ─────────────────────────────────────────────
def load_audio(path, sr, dur):
    y, _ = librosa.load(path, sr=sr, mono=True)
    n = sr * dur
    return np.pad(y, (0, max(0, n-len(y))))[:n]

def get_logmel(y, cfg):
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg["SAMPLE_RATE"], n_fft=cfg["N_FFT"],
        hop_length=cfg["HOP_LENGTH"], n_mels=cfg["N_MELS"]
    )
    lm = librosa.power_to_db(mel, ref=np.max)
    return ((lm - lm.min()) / (lm.max() - lm.min() + 1e-8)).astype(np.float32)

def run_inference(path, mdl, cfg):
    device = next(mdl.parameters()).device
    y = load_audio(path, cfg["SAMPLE_RATE"], cfg["DURATION"])
    lm = get_logmel(y, cfg)
    x = torch.tensor(lm).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(mdl(x), dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return {
        "pred": pred,
        "label": "Genuine" if pred == 0 else "Deepfake",
        "is_deepfake": pred == 1,
        "confidence": float(probs[pred]),
        "prob_genuine": float(probs[0]),
        "prob_deepfake": float(probs[1]),
        "audio": y,
        "logmel": lm,
    }

def make_spectrogram_fig(lm, cfg, label):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#12121e")
    ax.set_facecolor("#12121e")
    color = "#00e676" if label == "Genuine" else "#ff1744"
    img = librosa.display.specshow(
        lm, x_axis="time", y_axis="mel",
        sr=cfg["SAMPLE_RATE"], hop_length=cfg["HOP_LENGTH"], ax=ax,
        cmap="magma"
    )
    cb = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    ax.set_title(f"Log-Mel Spectrogram — {label}", color=color, fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor("#2e2e4e")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">🎙️ Deepfake Audio Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload any speech recording — get an instant Genuine / Deepfake verdict with confidence score</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model, config = load_model()

if model is None:
    st.error("⚠️ Model artifacts not found in `./artifacts/`. Run `train_pipeline.py` first to generate `deepfake_cnn.pt` and `config.json`.")
    st.stop()

# Model stats banner
m = config.get("metrics", {})
if m:
    st.markdown("""
    <div class="metric-row">
      <div class="metric-box"><div class="metric-val">{:.0f}%</div><div class="metric-name">Test Accuracy</div></div>
      <div class="metric-box"><div class="metric-val">{:.0f}%</div><div class="metric-name">F1 Score</div></div>
      <div class="metric-box"><div class="metric-val">{:.1f}%</div><div class="metric-name">Equal Error Rate</div></div>
      <div class="metric-box"><div class="metric-val">CNN</div><div class="metric-name">Architecture</div></div>
    </div>
    """.format(
        m.get("accuracy",0)*100, m.get("f1",0)*100, m.get("eer",0)*100
    ), unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────
st.markdown('<span class="step-pill">STEP 1</span>', unsafe_allow_html=True)
st.markdown("#### Upload Audio File")

uploaded = st.file_uploader(
    label="",
    type=["wav", "flac", "mp3", "ogg", "m4a"],
    help="Supports WAV, FLAC, MP3, OGG, M4A"
)
st.markdown('<div class="uploader-hint">Supported formats: WAV · FLAC · MP3 · OGG · M4A &nbsp;|&nbsp; Max recommended: 60 seconds</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
if uploaded:
    st.markdown("---")
    st.markdown('<span class="step-pill">STEP 2</span>', unsafe_allow_html=True)
    st.markdown("#### Playback")
    st.audio(uploaded)

    st.markdown("---")
    st.markdown('<span class="step-pill">STEP 3</span>', unsafe_allow_html=True)
    st.markdown("#### Analysis")

    suffix = os.path.splitext(uploaded.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Analyzing audio..."):
        try:
            res = run_inference(tmp_path, model, config)
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            os.remove(tmp_path)
            st.stop()

    os.remove(tmp_path)

    # ── Verdict card ──
    if res["is_deepfake"]:
        st.markdown(f"""
        <div class="result-deepfake">
            <div class="result-icon">🚨</div>
            <div class="result-label result-label-deepfake">DEEPFAKE</div>
            <div class="result-conf">AI-Generated Speech Detected</div>
            <div style="font-size:2.2rem;font-weight:900;color:#ff1744;margin-top:0.6rem;">
                {res['confidence']*100:.1f}% confident
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-genuine">
            <div class="result-icon">✅</div>
            <div class="result-label result-label-genuine">GENUINE</div>
            <div class="result-conf">Human Speech Detected</div>
            <div style="font-size:2.2rem;font-weight:900;color:#00e676;margin-top:0.6rem;">
                {res['confidence']*100:.1f}% confident
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Probability bars ──
    st.markdown("#### Probability Breakdown")
    pg = res["prob_genuine"] * 100
    pd_ = res["prob_deepfake"] * 100

    st.markdown(f"""
    <div class="prob-bar-wrap">
      <div class="prob-label">✅ Genuine (Human)</div>
      <div class="prob-track">
        <div class="prob-fill-genuine" style="width:{pg:.1f}%"></div>
        <span class="prob-pct">{pg:.1f}%</span>
      </div>
    </div>
    <div class="prob-bar-wrap">
      <div class="prob-label">🚨 Deepfake (AI-Generated)</div>
      <div class="prob-track">
        <div class="prob-fill-deepfake" style="width:{pd_:.1f}%"></div>
        <span class="prob-pct">{pd_:.1f}%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Spectrogram ──
    st.markdown("#### Log-Mel Spectrogram")
    fig = make_spectrogram_fig(res["logmel"], config, res["label"])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Evidence summary ──
    st.markdown("#### Evidence Dossier")
    col1, col2, col3 = st.columns(3)
    col1.metric("Verdict", res["label"])
    col2.metric("Confidence", f"{res['confidence']*100:.1f}%")
    col3.metric("Genuine Prob", f"{res['prob_genuine']*100:.1f}%")
    col1.metric("Deepfake Prob", f"{res['prob_deepfake']*100:.1f}%")
    col2.metric("Sample Rate", f"{config['SAMPLE_RATE']//1000}kHz")
    col3.metric("Clip Length", f"{config['DURATION']}s")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#444;font-size:0.8rem;padding:0.5rem 0">
    Model: 4-block CNN on Log-Mel Spectrograms &nbsp;|&nbsp;
    Dataset: Fake-or-Real (FoR) &nbsp;|&nbsp;
    Features: 128 Mel bands @ 16kHz
</div>
""", unsafe_allow_html=True)
