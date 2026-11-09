#!/usr/bin/env python3
"""19단계(체인 36) ②: identity-shortcut 검증 — PURE 학습 분할(6 피험자 × ~6 태스크)에서
PhysFormer 인코더 특징(transformer3 출력 풀링)으로 피험자 identity를 선형 probe로 복호.

가설: identity-HR이 무상관인 합성 클립을 포함해 학습한 모델일수록 identity 복호율이 낮다
(= identity shortcut 의존 감소). 통제: 같은 특징으로 chunk HR 회귀(Ridge R²) — HR 정보는 유지돼야 함.

분할: probe 학습 = 태스크 01~04, 평가 = 태스크 05~06 (video 단위 held-out, chance=1/6).
사용: python probe_identity.py [--models NAME=CKPT ...]  (기본 4모델)
"""
import argparse, glob, json, os, re
import numpy as np
import cv2
import torch

ROOT = "/SPECPHYS_ROOT"
PF = f"{ROOT}/_workspace/data/physformer"
TB = f"{ROOT}/repos/rPPG-Toolbox"
import sys
sys.path.insert(0, TB)
from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp

CHUNK, SIZE, FS = 160, 128, 30

DEFAULT_MODELS = [
    ("UBFC_base", f"{TB}/final_model_release/UBFC-rPPG_PhysFormer_DiffNormalized.pth"),
    ("CoTReal_L", f"{PF}/runs/finetune_CoTReal_L/PreTrainedModels/SYNTH_CoTReal_L_physformer_Epoch9.pth"),
    ("CoTAF_L", f"{PF}/runs/finetune_CoTAF_L/PreTrainedModels/SYNTH_CoTAF_L_physformer_Epoch9.pth"),
    ("CoTAF_TW_L", f"{PF}/runs/finetune_CoTAF_TW_L/PreTrainedModels/SYNTH_CoTAF_TW_L_physformer_Epoch9.pth"),
]

def load_model(ckpt, device):
    m = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(CHUNK, SIZE, SIZE), patches=(4, 4, 4), dim=96, ff_dim=144,
        num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    m.load_state_dict(sd); m.eval()
    return m

@torch.no_grad()
def encoder_feat(m, x, gra_sharp=2.0):
    """x: (B,3,T,H,W) → transformer3 출력 토큰 mean+std 풀링 (B, 2*dim)."""
    h = m.Stem2(m.Stem1(m.Stem0(x)))
    h = m.patch_embedding(h).flatten(2).transpose(1, 2)
    h, _ = m.transformer1(h, gra_sharp)
    h, _ = m.transformer2(h, gra_sharp)
    h, _ = m.transformer3(h, gra_sharp)
    return torch.cat([h.mean(1), h.std(1)], dim=1)

def face_box(frame):
    cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    det = cas.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)
    if len(det) == 0:
        h, w = frame.shape[:2]; return 0, 0, w, h
    x, y, w, h = max(det, key=lambda d: d[2] * d[3])
    cx, cy = x + w / 2, y + h / 2; s = max(w, h) * 1.5
    H, W = frame.shape[:2]
    x0 = int(max(0, cx - s / 2)); y0 = int(max(0, cy - s / 2))
    return x0, y0, int(min(W - x0, s)), int(min(H - y0, s))

def video_chunks(path):
    cap = cv2.VideoCapture(path)
    frames = []
    box = None
    while True:
        ok, f = cap.read()
        if not ok: break
        if box is None: box = face_box(f)
        x0, y0, w, h = box
        frames.append(cv2.resize(f[y0:y0 + h, x0:x0 + w], (SIZE, SIZE), interpolation=cv2.INTER_AREA))
    cap.release()
    a = np.asarray(frames, np.float32)[..., ::-1]  # BGR→RGB
    d = (a[1:] - a[:-1]) / (a[1:] + a[:-1] + 1e-7)
    d = d / (d.std() + 1e-7)
    d = np.concatenate([d, np.zeros_like(d[:1])], 0)
    ch = []
    for st in range(0, len(d) - CHUNK + 1, CHUNK):
        ch.append(np.transpose(d[st:st + CHUNK], (3, 0, 1, 2)))  # (3,T,H,W)
    return np.asarray(ch, np.float32)

def chunk_hrs(gt_path, n_chunks):
    import importlib.util
    spec = importlib.util.spec_from_file_location("agg2", f"{ROOT}/scripts/agg_model2.py")
    agg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(agg2)
    sig = np.array(open(gt_path).read().split("\n")[0].split(), dtype=float)
    hrs = []
    for k in range(n_chunks):
        seg = sig[k * CHUNK:(k + 1) * CHUNK]
        hrs.append(agg2.toolbox_hr(np.diff(seg, prepend=seg[:1])) if len(seg) >= 100 else np.nan)
    return hrs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="NAME=CKPT ...")
    a = ap.parse_args()
    models = [tuple(s.split("=", 1)) for s in a.models] if a.models else DEFAULT_MODELS
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vids = sorted(glob.glob(f"{PF}/PURE_TRAIN/subject9*/vid.avi"))
    meta = []  # (subject, task, chunks, hrs)
    for v in vids:
        m = re.search(r"subject9(\d{2})(\d{2})", v)
        subj, task = int(m.group(1)), int(m.group(2))
        ch = video_chunks(v)
        hrs = chunk_hrs(os.path.join(os.path.dirname(v), "ground_truth.txt"), len(ch))
        meta.append((subj, task, ch, hrs))
        print(f"prep subj{subj:02d} task{task:02d} chunks={len(ch)}", flush=True)

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    out = {}
    for name, ckpt in models:
        if not os.path.exists(ckpt):
            print(f"SKIP {name} (no ckpt)"); continue
        model = load_model(ckpt, device)
        X, y, task_arr, hr_arr = [], [], [], []
        for subj, task, ch, hrs in meta:
            for i in range(0, len(ch), 8):
                b = torch.from_numpy(ch[i:i + 8]).to(device)
                X.append(encoder_feat(model, b).cpu().numpy())
            y += [subj] * len(ch); task_arr += [task] * len(ch); hr_arr += hrs
        X = np.concatenate(X); y = np.array(y); task_arr = np.array(task_arr); hr_arr = np.array(hr_arr)
        tr = task_arr <= 4; te = task_arr >= 5
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[tr])
        acc = float((clf.predict(Xte) == y[te]).mean())
        # HR 통제: 같은 특징으로 chunk HR 회귀
        ok = ~np.isnan(hr_arr)
        rg = Ridge(alpha=10.0).fit(sc.transform(X[tr & ok]), hr_arr[tr & ok])
        pred = rg.predict(sc.transform(X[te & ok]))
        hr_mae = float(np.mean(np.abs(pred - hr_arr[te & ok])))
        ss_res = float(np.sum((pred - hr_arr[te & ok]) ** 2)); mu = hr_arr[te & ok].mean()
        r2 = 1 - ss_res / float(np.sum((hr_arr[te & ok] - mu) ** 2))
        out[name] = dict(id_acc=acc, n_test=int(te.sum()), hr_mae=hr_mae, hr_r2=r2)
        print(f"PROBE {name}: identity acc={acc:.3f} (chance 0.167, n={int(te.sum())}) | "
              f"HR probe MAE={hr_mae:.2f} R2={r2:.3f}", flush=True)
        del model; torch.cuda.empty_cache()
    json.dump(out, open(f"{ROOT}/_workspace/data/results/probe_identity.json", "w"), indent=1)
    print("PROBE_DONE")

if __name__ == "__main__":
    main()
