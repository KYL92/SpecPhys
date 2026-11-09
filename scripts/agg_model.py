#!/usr/bin/env python3
"""모델별 9축 집계 — 영상 단위 고해상 FFT HR MAE + 배가/반감 카운트.
사용: python agg_model.py MODEL [MODEL...]  (runs/ev_{MODEL}_{AXIS}/*/saved_test_outputs)"""
import glob, pickle, re, sys
import numpy as np, torch
from scipy.signal import butter, filtfilt

fs = 30
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
AXES = ["MCD_B", "MCD_A", "PURE", "REALTEST", "RTNI", "RTNC", "C3", "C6", "E3"]


def video_hr(chunks):
    sig = np.concatenate([chunks[k].detach().cpu().float().numpy().ravel() for k in sorted(chunks)])
    sig = np.cumsum(sig - sig.mean())
    b, a = butter(3, [0.7 / (fs / 2), 4.0 / (fs / 2)], btype="band")
    f = filtfilt(b, a, sig)
    fr = np.fft.rfftfreq(len(f) * 8, 1 / fs)
    pw = np.abs(np.fft.rfft(f - f.mean(), n=len(f) * 8)) ** 2
    band = (fr >= 0.7) & (fr <= 4.0)
    return float(fr[band][pw[band].argmax()] * 60)


def vidnum(v):
    m = re.search(r"(\d+)$", str(v))
    return int(m.group(1)) if m else -1


def agg(model):
    print(f"--- {model}")
    for ax in AXES:
        pk = glob.glob(f"{PF}/runs/ev_{model}_{ax}/*/saved_test_outputs/*.pickle")
        if not pk:
            continue
        d = pickle.load(open(pk[0], "rb"))
        rows = []
        for vid in d["predictions"]:
            hp = video_hr(d["predictions"][vid]); hl = video_hr(d["labels"][vid])
            rows.append((vidnum(vid), abs(hp - hl), hp / hl))
        errs = np.array([r[1] for r in rows])
        dbl = sum(1 for r in rows if abs(r[2] - 2) < 0.15)
        hlv = sum(1 for r in rows if abs(r[2] - 0.5) < 0.08)
        line = f"  {ax:9s} n={len(errs):3d} MAE={errs.mean():6.2f} med={np.median(errs):5.2f} 배가={dbl} 반감={hlv}"
        if ax == "PURE":
            ho = np.array([r[1] for r in rows if r[0] >= 700])
            dh = sum(1 for r in rows if r[0] >= 700 and abs(r[2] - 2) < 0.15)
            if len(ho):
                line += f" | held-out(07-10) n={len(ho)} MAE={ho.mean():.2f} 배가={dh}"
        print(line)


if __name__ == "__main__":
    for m in sys.argv[1:]:
        agg(m)
