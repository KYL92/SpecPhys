#!/usr/bin/env python3
"""MMPD 전체(33명 661클립)에서 POS/CHROM HR 추정 (mat 직접 읽기, eval_classical_ubfc.py와 동일 프로토콜:
YCrCb 피부마스크 평균 → POS/CHROM). HR은 **정본 toolbox 프로토콜**(detrend λ=100 → butter1 0.6–3.3Hz → periodogram)로
GT/예측 모두 계산 — MMPD GT PPG는 2차 harmonic이 강해 단순 0.7–4.0Hz FFT가 2f를 잡는 경우가 있음(subject1).
출력: _workspace/data/results/pos_pervideo_MMPD.json (속성 포함, per-video)."""
import glob, json, os, sys
import numpy as np, cv2, scipy.io as sio
if not hasattr(np, "mat"): np.mat = np.asmatrix
sys.path.insert(0, "/SPECPHYS_ROOT/repos/rPPG-Toolbox")
from unsupervised_methods.methods.POS_WANG import POS_WANG
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
from evaluation.post_process import _calculate_fft_hr
sys.path.insert(0, '/SPECPHYS_ROOT/scripts')
from agg_model2 import fast_detrend
import scipy.signal as ss
def tb_hr(x, fs=30):
    x = fast_detrend(np.asarray(x, float) - np.mean(x), 100)
    b, a = ss.butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
    return float(_calculate_fft_hr(ss.filtfilt(b, a, x), fs=fs, low_pass=0.6, high_pass=3.3))
SRC = "/SPECPHYS_RAW/MMPD"; OUT = "/SPECPHYS_ROOT/_workspace/data/results/pos_pervideo_MMPD.json"
FPS = 30

def series_from_rgb(frames):  # frames uint8 RGB (T,H,W,3)
    acc = np.zeros(frames.shape[1:3], np.float32); sub = frames[::40]
    for f in sub:
        y = cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)
        acc += ((y[:, :, 1] >= 133) & (y[:, :, 1] <= 173) & (y[:, :, 2] >= 77) & (y[:, :, 2] <= 127)).astype(np.float32)
    m = (acc / len(sub)) > 0.5
    if m.sum() < 500:
        h, w = frames.shape[1:3]; m = np.zeros((h, w), bool); m[h//4:3*h//4, w//4:3*w//4] = True
    return [f.astype(np.float32)[m].mean(0).reshape(1, 1, 3) for f in frames]

rows = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {(r["subject"], r["trial"]) for r in rows}
mats = sorted(glob.glob(f"{SRC}/subject*/p*_*.mat"), key=lambda p: (int(p.split("subject")[1].split("/")[0]), int(p.split("_")[-1][:-4])))
for mp in mats:
    subj = int(mp.split("subject")[1].split("/")[0]); trial = int(mp.split("_")[-1][:-4])
    if (subj, trial) in done: continue
    m = sio.loadmat(mp)
    v = (np.round(np.asarray(m["video"], np.float32) * 255)).astype(np.uint8)
    gtw = np.asarray(m["GT_ppg"], float).reshape(-1)[:len(v)]
    gt = tb_hr(gtw)
    ser = series_from_rgb(v)
    r = dict(subject=subj, trial=trial, light=str(m["light"][0]), motion=str(m["motion"][0]),
             exercise=str(m["exercise"][0]), skin=int(m["skin_color"][0][0]), gt=gt)
    for name, fn in (("POS", POS_WANG), ("CHROM", CHROME_DEHAAN)):
        bvp = np.asarray(fn(ser, FPS)).squeeze()
        r[name] = tb_hr(bvp)
    rows.append(r); json.dump(rows, open(OUT, "w"), indent=0)
    print(f"subject{subj} p{subj}_{trial} gt={gt:.1f} POS={r['POS']:.1f} CHROM={r['CHROM']:.1f}", flush=True)
ho = [r for r in rows if r["subject"] >= 7]
for k in ("POS", "CHROM"):
    e = np.array([abs(r[k]-r["gt"]) for r in rows]); eh = np.array([abs(r[k]-r["gt"]) for r in ho])
    print(f"{k}: all n={len(e)} MAE={e.mean():.2f} | held-out(7-33) n={len(eh)} MAE={eh.mean():.2f}")
