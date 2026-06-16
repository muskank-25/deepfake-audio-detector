"""
train_pipeline.py — Standalone training script for Deepfake Audio Detection.

Usage:
    python train_pipeline.py --data_dir ./for-norm --epochs 25 --out_dir ./artifacts
"""

import os, glob, json, pickle, argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report, roc_curve)

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# ─── Config ────────────────────────────────────────────────────────────────
SR=16000; DUR=4; N_MELS=128; N_MFCC=40; N_FFT=1024; HOP=256
LABEL_MAP = {"real": 0, "fake": 1}

# ─── Data ──────────────────────────────────────────────────────────────────
def collect(data_dir):
    records = []
    for split in ["training","validation","testing"]:
        for label, lid in LABEL_MAP.items():
            for ext in ("*.wav","*.flac","*.mp3"):
                for fp in glob.glob(f"{data_dir}/{split}/{label}/{ext}"):
                    records.append({"fp":fp,"label":lid,"lname":label,"split":split})
    return pd.DataFrame(records)

def load_audio(fp):
    y,_ = librosa.load(fp, sr=SR, mono=True)
    n = SR*DUR
    return np.pad(y,(0,max(0,n-len(y))))[:n]

def logmel(y):
    m = librosa.feature.melspectrogram(y=y,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS)
    d = librosa.power_to_db(m, ref=np.max)
    return ((d-d.min())/(d.max()-d.min()+1e-8)).astype(np.float32)

def mfcc_feats(y):
    m  = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    dm = librosa.feature.delta(m)
    return np.concatenate([m.mean(1),m.std(1),dm.mean(1),dm.std(1)]).astype(np.float32)

def build(frame, cache):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["logmels"], d["mfccs"], d["labels"]
    mels,mfccs,labels = [],[],[]
    for _,r in tqdm(frame.iterrows(), total=len(frame)):
        y = load_audio(r.fp)
        mels.append(logmel(y)); mfccs.append(mfcc_feats(y)); labels.append(r.label)
    mels,mfccs,labels = np.stack(mels),np.stack(mfccs),np.array(labels)
    np.savez_compressed(cache, logmels=mels, mfccs=mfccs, labels=labels)
    return mels, mfccs, labels

# ─── Dataset ───────────────────────────────────────────────────────────────
class DS(Dataset):
    def __init__(self,m,l): self.m=m; self.l=l
    def __len__(self): return len(self.l)
    def __getitem__(self,i):
        return torch.tensor(self.m[i]).unsqueeze(0), torch.tensor(self.l[i],dtype=torch.long)

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

def compute_eer(y_true, y_scores):
    fpr,tpr,_ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1-tpr
    idx = np.nanargmin(np.abs(fpr-fnr))
    return (fpr[idx]+fnr[idx])/2

# ─── Main ──────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = collect(args.data_dir)
    if len(df)==0:
        raise FileNotFoundError(f"No audio files found in {args.data_dir}")
    print(f"Total files: {len(df)}\n{df['lname'].value_counts()}")

    if {"training","testing"}.issubset(set(df["split"].unique())):
        tr = df[df.split=="training"].reset_index(drop=True)
        te = df[df.split=="testing"].reset_index(drop=True)
        va = df[df.split=="validation"].reset_index(drop=True) if "validation" in df.split.unique() \
             else train_test_split(tr, test_size=0.15, stratify=tr.label, random_state=SEED)[1]
    else:
        tr,tmp = train_test_split(df,test_size=0.30,stratify=df.label,random_state=SEED)
        va,te  = train_test_split(tmp,test_size=0.5,stratify=tmp.label,random_state=SEED)

    os.makedirs(args.cache_dir, exist_ok=True)
    Xm_tr,Xf_tr,y_tr = build(tr, f"{args.cache_dir}/train.npz")
    Xm_v, Xf_v, y_v  = build(va, f"{args.cache_dir}/val.npz")
    Xm_te,Xf_te,y_te = build(te, f"{args.cache_dir}/test.npz")

    sc = StandardScaler()
    sc.fit(Xf_tr)

    trl = DataLoader(DS(Xm_tr,y_tr), args.batch_size, shuffle=True)
    vl  = DataLoader(DS(Xm_v, y_v),  args.batch_size)
    tel = DataLoader(DS(Xm_te,y_te), args.batch_size)

    model   = AudioCNN().to(device)
    crit    = nn.CrossEntropyLoss()
    opt     = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    os.makedirs(args.out_dir, exist_ok=True)
    best, best_path = 0.0, f"{args.out_dir}/deepfake_cnn.pt"

    for epoch in range(1, args.epochs+1):
        model.train(); tl=0
        for x,y in trl:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(); loss=crit(model(x),y); loss.backward(); opt.step()
            tl+=loss.item()*x.size(0)
        tl/=len(y_tr)

        model.eval(); vp,vl_true=[],[]
        with torch.no_grad():
            for x,y in vl:
                vp.extend(model(x.to(device)).argmax(1).cpu().numpy())
                vl_true.extend(y.numpy())
        va_acc=accuracy_score(vl_true,vp); sched.step(va_acc)
        if va_acc>best: best=va_acc; torch.save(model.state_dict(),best_path)
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={tl:.4f} | val_acc={va_acc:.4f}")

    # Test evaluation
    model.load_state_dict(torch.load(best_path,map_location=device)); model.eval()
    tp,tl_true,tprobs=[],[],[]
    with torch.no_grad():
        for x,y in tel:
            out=model(x.to(device))
            tp.extend(out.argmax(1).cpu().numpy())
            tl_true.extend(y.numpy())
            tprobs.extend(torch.softmax(out,1)[:,1].cpu().numpy())

    tp,tl_true,tprobs=np.array(tp),np.array(tl_true),np.array(tprobs)
    acc=accuracy_score(tl_true,tp); f1=f1_score(tl_true,tp)
    eer=compute_eer(tl_true,tprobs)
    cm=confusion_matrix(tl_true,tp); pca=cm.diagonal()/cm.sum(1)

    print(f"\n{'='*40}")
    print(f"TEST RESULTS")
    print(f"{'='*40}")
    print(f"Accuracy : {acc*100:.2f}%  (target ≥80%)")
    print(f"F1 Score : {f1*100:.2f}%   (target ≥80%)")
    print(f"EER      : {eer*100:.2f}%  (target ≤12%)")
    print(f"Genuine  : {pca[0]*100:.2f}%  Deepfake: {pca[1]*100:.2f}%  (target ≥75% each)")
    print(f"\n{classification_report(tl_true,tp,target_names=['Genuine','Deepfake'])}")
    print("Confusion Matrix:\n", cm)

    with open(f"{args.out_dir}/scaler.pkl","wb") as f: pickle.dump(sc,f)
    cfg={"SAMPLE_RATE":SR,"DURATION":DUR,"N_MELS":N_MELS,"N_MFCC":N_MFCC,
         "N_FFT":N_FFT,"HOP_LENGTH":HOP,"LABEL_MAP":LABEL_MAP,
         "INV_LABEL_MAP":{"0":"real","1":"fake"},
         "metrics":{"accuracy":float(acc),"f1":float(f1),"eer":float(eer),
                    "per_class_accuracy":{"Genuine":float(pca[0]),"Deepfake":float(pca[1])},
                    "confusion_matrix":cm.tolist()}}
    with open(f"{args.out_dir}/config.json","w") as f: json.dump(cfg,f,indent=2)
    print(f"\nArtifacts saved to {args.out_dir}/")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data_dir",  type=str, default="./for-norm")
    p.add_argument("--cache_dir", type=str, default="./cache")
    p.add_argument("--out_dir",   type=str, default="./artifacts")
    p.add_argument("--epochs",    type=int, default=25)
    p.add_argument("--batch_size",type=int, default=16)
    p.add_argument("--lr",        type=float,default=1e-3)
    main(p.parse_args())
