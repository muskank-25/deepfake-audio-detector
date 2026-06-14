"""
Deepfake Audio Detection — Streamlit Web App
Run: streamlit run app.py
Artifacts (deepfake_cnn.pt, config.json, scaler.pkl) must be in same folder as app.py
"""
import os, json, tempfile
import numpy as np
import librosa, librosa.display
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
import streamlit as st

# ─── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
* { font-family: 'Inter', sans-serif; }
.hero { text-align:center; padding: 1rem 0 0.5rem; }
.hero h1 { font-size:2.4rem; font-weight:900; margin:0;
  background:linear-gradient(135deg,#6C63FF,#FF6584);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p  { color:#888; font-size:1rem; margin-top:0.3rem; }
.verdict-genuine  { background:linear-gradient(135deg,#0d2b1f,#0f3d28);
  border:2px solid #00e676; border-radius:18px; padding:2rem; text-align:center; margin:1.2rem 0; }
.verdict-deepfake { background:linear-gradient(135deg,#2b0d0d,#3d1010);
  border:2px solid #ff1744; border-radius:18px; padding:2rem; text-align:center; margin:1.2rem 0; }
.v-icon  { font-size:3.5rem; }
.v-label { font-size:2rem; font-weight:900; margin:0.3rem 0; }
.v-conf  { font-size:1rem; color:#ccc; }
.v-gen  { color:#00e676; }
.v-fake { color:#ff1744; }
.bar-wrap { margin:0.8rem 0; }
.bar-label { font-size:0.85rem; color:#aaa; margin-bottom:3px; }
.bar-track { background:#2a2a3e; border-radius:99px; height:20px; position:relative; overflow:hidden; }
.bar-gen  { height:100%; background:linear-gradient(90deg,#00b09b,#00e676); border-radius:99px; }
.bar-fake { height:100%; background:linear-gradient(90deg,#c0392b,#ff1744); border-radius:99px; }
.bar-pct  { position:absolute; right:8px; top:50%; transform:translateY(-50%);
  font-size:0.75rem; font-weight:700; color:white; }
.stat-row { display:flex; gap:0.8rem; margin:0.8rem 0; flex-wrap:wrap; }
.stat-box { flex:1; min-width:120px; background:#12121e; border:1px solid #2e2e4e;
  border-radius:12px; padding:0.8rem; text-align:center; }
.stat-val  { font-size:1.5rem; font-weight:800; color:#6C63FF; }
.stat-name { font-size:0.72rem; color:#888; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# ─── Model ─────────────────────────────────────────────────────────────────
class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),
        )
        self.clf = nn.Sequential(
            nn.Flatten(),nn.Dropout(0.3),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,2))
    def forward(self,x): return self.clf(self.features(x))

@st.cache_resource
def load_model():
    for cfg_p in ["./config.json","./artifacts/config.json"]:
        for mdl_p in ["./deepfake_cnn.pt","./artifacts/deepfake_cnn.pt"]:
            if os.path.exists(cfg_p) and os.path.exists(mdl_p):
                with open(cfg_p) as f: cfg=json.load(f)
                dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                m=AudioCNN().to(dev)
                m.load_state_dict(torch.load(mdl_p,map_location=dev))
                m.eval()
                return m,cfg,dev
    return None,None,None

def predict(path, model, cfg, dev):
    y,_=librosa.load(path,sr=cfg["SAMPLE_RATE"],mono=True)
    n=cfg["SAMPLE_RATE"]*cfg["DURATION"]
    y=np.pad(y,(0,max(0,n-len(y))))[:n]
    mel=librosa.feature.melspectrogram(y=y,sr=cfg["SAMPLE_RATE"],
        n_fft=cfg["N_FFT"],hop_length=cfg["HOP_LENGTH"],n_mels=cfg["N_MELS"])
    lm=librosa.power_to_db(mel,ref=np.max)
    lm=((lm-lm.min())/(lm.max()-lm.min()+1e-8)).astype(np.float32)
    x=torch.tensor(lm).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        probs=torch.softmax(model(x),dim=1)[0].cpu().numpy()
    pred=int(probs.argmax())
    return {"label":"Genuine" if pred==0 else "Deepfake",
            "is_deepfake":pred==1,"confidence":float(probs[pred]),
            "prob_genuine":float(probs[0]),"prob_deepfake":float(probs[1]),"lm":lm,"y":y}

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎙️ Deepfake Audio Detector</h1>
  <p>Upload a speech recording — get an instant <b>Genuine / Deepfake</b> verdict with confidence score</p>
</div>
""", unsafe_allow_html=True)

model, config, device = load_model()

if model is None:
    st.error("⚠️ Model not found. Make sure `deepfake_cnn.pt` and `config.json` are in the same folder as `app.py`.")
    st.stop()

# Model stats
if "metrics" in config:
    m=config["metrics"]
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-box"><div class="stat-val">{m['accuracy']*100:.0f}%</div><div class="stat-name">Test Accuracy</div></div>
      <div class="stat-box"><div class="stat-val">{m['f1']*100:.0f}%</div><div class="stat-name">F1 Score</div></div>
      <div class="stat-box"><div class="stat-val">{m['eer']*100:.1f}%</div><div class="stat-name">Equal Error Rate</div></div>
      <div class="stat-box"><div class="stat-val">CNN</div><div class="stat-name">Architecture</div></div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ─── Upload ────────────────────────────────────────────────────────────────
st.markdown("#### 📤 Upload Audio File")
uploaded = st.file_uploader("", type=["wav","flac","mp3","ogg","m4a"],
                             help="WAV · FLAC · MP3 · OGG · M4A")
st.caption("Supported: WAV · FLAC · MP3 · OGG · M4A — any length (first 4s analysed)")

if uploaded:
    st.markdown("---")
    st.markdown("#### 🔊 Playback")
    st.audio(uploaded)

    suffix = os.path.splitext(uploaded.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read()); tmp_path=tmp.name

    st.markdown("---")
    with st.spinner("🔍 Analysing audio..."):
        try:
            res = predict(tmp_path, model, config, device)
        except Exception as e:
            st.error(f"Error: {e}"); os.remove(tmp_path); st.stop()
    os.remove(tmp_path)

    # ── Verdict ──
    if res["is_deepfake"]:
        st.markdown(f"""
        <div class="verdict-deepfake">
          <div class="v-icon">🚨</div>
          <div class="v-label v-fake">DEEPFAKE</div>
          <div class="v-conf">AI-Generated Speech Detected</div>
          <div style="font-size:2rem;font-weight:900;color:#ff1744;margin-top:0.5rem">
            {res['confidence']*100:.1f}% confident</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-genuine">
          <div class="v-icon">✅</div>
          <div class="v-label v-gen">GENUINE</div>
          <div class="v-conf">Human Speech Detected</div>
          <div style="font-size:2rem;font-weight:900;color:#00e676;margin-top:0.5rem">
            {res['confidence']*100:.1f}% confident</div>
        </div>""", unsafe_allow_html=True)

    # ── Probability bars ──
    pg=res["prob_genuine"]*100; pd_=res["prob_deepfake"]*100
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
    fig,ax=plt.subplots(figsize=(7,2.8))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
    col="#00e676" if not res["is_deepfake"] else "#ff1744"
    img=librosa.display.specshow(res["lm"],x_axis="time",y_axis="mel",
        sr=config["SAMPLE_RATE"],hop_length=config["HOP_LENGTH"],ax=ax,cmap="magma")
    cb=fig.colorbar(img,ax=ax,format="%+2.0f dB")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(),color="white",fontsize=8)
    ax.set_title(f"Log-Mel Spectrogram — {res['label']}",color=col,fontsize=11,fontweight="bold",pad=6)
    ax.tick_params(colors="white",labelsize=8)
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    for sp in ax.spines.values(): sp.set_edgecolor("#2e2e4e")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── Evidence dossier ──
    st.markdown("#### 📋 Evidence Dossier")
    c1,c2,c3=st.columns(3)
    c1.metric("Verdict",      res["label"])
    c2.metric("Confidence",   f"{res['confidence']*100:.1f}%")
    c3.metric("P(Genuine)",   f"{res['prob_genuine']*100:.1f}%")
    c1.metric("P(Deepfake)",  f"{res['prob_deepfake']*100:.1f}%")
    c2.metric("Sample Rate",  f"{config['SAMPLE_RATE']//1000}kHz")
    c3.metric("Clip Analysed",f"{config['DURATION']}s")

st.markdown("---")
st.caption("Model: 4-block CNN · Features: 128 Log-Mel bands @ 16kHz · Dataset: Fake-or-Real (FoR)")
