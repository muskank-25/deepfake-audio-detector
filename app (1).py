import os, json, tempfile
import numpy as np
import streamlit as st

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="centered")

# ─── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
* { font-family: 'Inter', sans-serif; }
.hero { text-align:center; padding:1.5rem 0 0.5rem; }
.hero h1 { font-size:2.4rem; font-weight:900; margin:0;
  background:linear-gradient(135deg,#6C63FF,#FF6584);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { color:#888; font-size:1rem; margin-top:0.4rem; }
.verdict-genuine {
  background:linear-gradient(135deg,#0d2b1f,#0f3d28);
  border:2px solid #00e676; border-radius:18px;
  padding:2rem; text-align:center; margin:1.2rem 0; }
.verdict-deepfake {
  background:linear-gradient(135deg,#2b0d0d,#3d1010);
  border:2px solid #ff1744; border-radius:18px;
  padding:2rem; text-align:center; margin:1.2rem 0; }
.v-icon  { font-size:3.5rem; }
.v-label { font-size:2.2rem; font-weight:900; margin:0.3rem 0; }
.v-gen   { color:#00e676; }
.v-fake  { color:#ff1744; }
.v-conf  { font-size:1rem; color:#ccc; margin-top:0.3rem; }
.v-score { font-weight:900; margin-top:0.5rem; }
.bar-wrap  { margin:0.8rem 0; }
.bar-label { font-size:0.85rem; color:#aaa; margin-bottom:3px; }
.bar-track { background:#2a2a3e; border-radius:99px; height:22px;
  position:relative; overflow:hidden; }
.bar-gen  { height:100%; background:linear-gradient(90deg,#00b09b,#00e676);
  border-radius:99px; }
.bar-fake { height:100%; background:linear-gradient(90deg,#c0392b,#ff1744);
  border-radius:99px; }
.bar-pct  { position:absolute; right:10px; top:50%;
  transform:translateY(-50%); font-size:0.78rem;
  font-weight:700; color:white; }
.stat-row { display:flex; gap:0.8rem; margin:0.8rem 0; flex-wrap:wrap; }
.stat-box { flex:1; min-width:110px; background:#12121e;
  border:1px solid #2e2e4e; border-radius:12px;
  padding:0.8rem; text-align:center; }
.stat-val  { font-size:1.5rem; font-weight:800; color:#6C63FF; }
.stat-name { font-size:0.72rem; color:#888; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# ─── Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_everything():
    import torch
    import torch.nn as nn
    import librosa

    class AudioCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.4),
                nn.Linear(128,64), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(64,2)
            )
        def forward(self,x): return self.classifier(self.features(x))

    # Find config and model — check both root and artifacts/ folder
    cfg_path, mdl_path = None, None
    for p in ["./config.json","./artifacts/config.json"]:
        if os.path.exists(p): cfg_path = p; break
    for p in ["./deepfake_cnn.pt","./artifacts/deepfake_cnn.pt"]:
        if os.path.exists(p): mdl_path = p; break

    if not cfg_path or not mdl_path:
        return None, None, None, None

    with open(cfg_path) as f: cfg = json.load(f)
    device = torch.device("cpu")
    model  = AudioCNN()
    model.load_state_dict(torch.load(mdl_path, map_location=device))
    model.eval()
    return model, cfg, device, librosa

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎙️ Deepfake Audio Detector</h1>
  <p>Upload any speech recording — get an instant <b>Genuine / Deepfake</b> verdict</p>
</div>
""", unsafe_allow_html=True)

model, config, device, librosa = load_everything()

if model is None:
    st.error("⚠️ Model not found. Make sure `deepfake_cnn.pt` and `config.json` are in the repo.")
    st.stop()

# Model stats bar
if "metrics" in config:
    m = config["metrics"]
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-box">
        <div class="stat-val">{m['accuracy']*100:.1f}%</div>
        <div class="stat-name">Test Accuracy</div>
      </div>
      <div class="stat-box">
        <div class="stat-val">{m['f1']*100:.1f}%</div>
        <div class="stat-name">F1 Score</div>
      </div>
      <div class="stat-box">
        <div class="stat-val">{m['eer']*100:.1f}%</div>
        <div class="stat-name">Equal Error Rate</div>
      </div>
      <div class="stat-box">
        <div class="stat-val">{m['per_class_accuracy']['Genuine']*100:.0f}%</div>
        <div class="stat-name">Genuine Acc</div>
      </div>
      <div class="stat-box">
        <div class="stat-val">{m['per_class_accuracy']['Deepfake']*100:.0f}%</div>
        <div class="stat-name">Deepfake Acc</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─── Upload ────────────────────────────────────────────────────────────────
st.markdown("#### 📤 Upload Audio File")
uploaded = st.file_uploader(
    "", type=["wav","flac","mp3","ogg","m4a"],
    help="WAV · FLAC · MP3 · OGG · M4A")
st.caption("Supported formats: WAV · FLAC · MP3 · OGG · M4A — first 4 seconds analysed")

if uploaded:
    st.markdown("---")
    st.markdown("#### 🔊 Playback")
    st.audio(uploaded)

    suffix = os.path.splitext(uploaded.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.markdown("---")
    with st.spinner("🔍 Analysing audio..."):
        try:
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
            lm  = librosa.power_to_db(mel, ref=np.max)
            lm  = ((lm - lm.min()) / (lm.max() - lm.min() + 1e-8)).astype(np.float32)

            x = torch.tensor(lm).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].numpy()
            pred = int(probs.argmax())

        except Exception as e:
            st.error(f"Error processing audio: {e}")
            os.remove(tmp_path)
            st.stop()

    os.remove(tmp_path)

    # ── Verdict ──
    if pred == 1:
        st.markdown(f"""
        <div class="verdict-deepfake">
          <div class="v-icon">🚨</div>
          <div class="v-label v-fake">DEEPFAKE</div>
          <div class="v-conf">AI-Generated Speech Detected</div>
          <div class="v-score v-fake" style="font-size:2rem">
            {probs[1]*100:.1f}% confident
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-genuine">
          <div class="v-icon">✅</div>
          <div class="v-label v-gen">GENUINE</div>
          <div class="v-conf">Human Speech Detected</div>
          <div class="v-score v-gen" style="font-size:2rem">
            {probs[0]*100:.1f}% confident
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Probability bars ──
    pg  = probs[0] * 100
    pd_ = probs[1] * 100
    st.markdown("#### 📊 Probability Breakdown")
    st.markdown(f"""
    <div class="bar-wrap">
      <div class="bar-label">✅ Genuine (Human)</div>
      <div class="bar-track">
        <div class="bar-gen" style="width:{pg:.1f}%"></div>
        <span class="bar-pct">{pg:.1f}%</span>
      </div>
    </div>
    <div class="bar-wrap">
      <div class="bar-label">🚨 Deepfake (AI-Generated)</div>
      <div class="bar-track">
        <div class="bar-fake" style="width:{pd_:.1f}%"></div>
        <span class="bar-pct">{pd_:.1f}%</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Spectrogram ──
    st.markdown("#### 🌈 Log-Mel Spectrogram")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    col = "#00e676" if pred == 0 else "#ff1744"
    img = librosa.display.specshow(
        lm, x_axis="time", y_axis="mel",
        sr=SR, hop_length=config["HOP_LENGTH"],
        ax=ax, cmap="magma")
    cb = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    label = "Genuine (Human)" if pred == 0 else "Deepfake (AI-Generated)"
    ax.set_title(f"{label}", color=col, fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for sp in ax.spines.values(): sp.set_edgecolor("#2e2e4e")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Evidence Dossier ──
    st.markdown("#### 📋 Evidence Dossier")
    c1, c2, c3 = st.columns(3)
    c1.metric("Verdict",       label)
    c2.metric("Confidence",    f"{probs[pred]*100:.1f}%")
    c3.metric("P(Genuine)",    f"{probs[0]*100:.1f}%")
    c1.metric("P(Deepfake)",   f"{probs[1]*100:.1f}%")
    c2.metric("Sample Rate",   f"{config['SAMPLE_RATE']//1000}kHz")
    c3.metric("Clip Analysed", f"{config['DURATION']}s")

st.divider()
st.caption("Model: 4-block CNN on Log-Mel Spectrograms · Dataset: Fake-or-Real (FoR) · 16kHz")
