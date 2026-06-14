"""
predict.py — Test the Deepfake Audio Detection model on new audio samples.

Usage:
    # Single file
    python predict.py --audio path/to/file.wav

    # Batch CSV (must have a 'filepath' column)
    python predict.py --csv files.csv --out_csv predictions.csv
"""

import os, json, pickle, argparse
import numpy as np
import librosa
import torch
import torch.nn as nn


# ─── Model (must match training) ───────────────────────────────────────────
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
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, n_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))


# ─── Load artifacts ─────────────────────────────────────────────────────────
def load_artifacts(artifacts_dir="."):
    cfg_path = os.path.join(artifacts_dir, "config.json")
    mdl_path = os.path.join(artifacts_dir, "deepfake_cnn.pt")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found in {artifacts_dir}")
    if not os.path.exists(mdl_path):
        raise FileNotFoundError(f"deepfake_cnn.pt not found in {artifacts_dir}")

    with open(cfg_path) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN().to(device)
    model.load_state_dict(torch.load(mdl_path, map_location=device))
    model.eval()
    print(f"Model loaded from {mdl_path} | Device: {device}")
    return model, config, device


# ─── Audio utils ────────────────────────────────────────────────────────────
def load_audio(path, sr, duration):
    y, _ = librosa.load(path, sr=sr, mono=True)
    n = sr * duration
    return np.pad(y, (0, max(0, n - len(y))))[:n]

def extract_logmel(y, sr, n_fft, hop_length, n_mels):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    lm = librosa.power_to_db(mel, ref=np.max)
    return ((lm - lm.min()) / (lm.max() - lm.min() + 1e-8)).astype(np.float32)


# ─── Predict single file ────────────────────────────────────────────────────
def predict(filepath, model, config, device):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    y   = load_audio(filepath, config["SAMPLE_RATE"], config["DURATION"])
    lm  = extract_logmel(y, config["SAMPLE_RATE"], config["N_FFT"],
                         config["HOP_LENGTH"], config["N_MELS"])
    x   = torch.tensor(lm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        probs    = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())

    label = "Genuine (Human)" if pred_idx == 0 else "Deepfake (AI-Generated)"
    return {
        "filepath":      filepath,
        "prediction":    label,
        "is_deepfake":   bool(pred_idx == 1),
        "confidence":    f"{float(probs[pred_idx])*100:.2f}%",
        "prob_genuine":  f"{float(probs[0])*100:.2f}%",
        "prob_deepfake": f"{float(probs[1])*100:.2f}%",
    }


# ─── Batch predict from CSV ─────────────────────────────────────────────────
def predict_batch(csv_path, model, config, device, out_csv):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "filepath" not in df.columns:
        raise ValueError("CSV must have a 'filepath' column")
    results = []
    for fp in df["filepath"]:
        try:
            results.append(predict(fp, model, config, device))
        except Exception as e:
            results.append({"filepath": fp, "prediction": f"ERROR: {e}"})
    out = pd.DataFrame(results)
    out.to_csv(out_csv, index=False)
    print(f"\nSaved {len(out)} predictions to {out_csv}")
    print(out.to_string())
    return out


# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Audio Detector — Inference Script")
    parser.add_argument("--audio",      type=str, help="Path to a single audio file (.wav/.flac/.mp3)")
    parser.add_argument("--csv",        type=str, help="CSV file with a 'filepath' column for batch prediction")
    parser.add_argument("--out_csv",    type=str, default="predictions.csv")
    parser.add_argument("--artifacts",  type=str, default=".", help="Folder containing deepfake_cnn.pt and config.json")
    args = parser.parse_args()

    model, config, device = load_artifacts(args.artifacts)

    if args.audio:
        result = predict(args.audio, model, config, device)
        print("\n" + "="*45)
        print("  DEEPFAKE AUDIO DETECTION RESULT")
        print("="*45)
        print(f"  File       : {result['filepath']}")
        print(f"  Prediction : {result['prediction']}")
        print(f"  Confidence : {result['confidence']}")
        print(f"  P(Genuine) : {result['prob_genuine']}")
        print(f"  P(Deepfake): {result['prob_deepfake']}")
        print("="*45)

    elif args.csv:
        predict_batch(args.csv, model, config, device, args.out_csv)

    else:
        print("Usage:")
        print("  python predict.py --audio path/to/file.wav")
        print("  python predict.py --csv files.csv --out_csv predictions.csv")
