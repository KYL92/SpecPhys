#!/usr/bin/env python3
"""UBFC 레이아웃 데이터셋에서 POS/CHROM HR을 추정하고
ground_truth.txt 1행 파형의 FFT HR 대비 MAE를 계산한다 (실데이터 범용)."""
import argparse, glob, os, sys
import numpy as np
import cv2

if not hasattr(np, "mat"):
    np.mat = np.asmatrix
sys.path.insert(0, "/SPECPHYS_ROOT/repos/rPPG-Toolbox")
from unsupervised_methods.methods.POS_WANG import POS_WANG          # noqa
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN  # noqa
from evaluation.post_process import _calculate_fft_hr  # noqa


def skin_mean_series(avi):
    cap = cv2.VideoCapture(avi)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    acc = np.zeros(frames[0].shape[:2], np.float32)
    for f in frames[::40]:
        y = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
        acc += ((y[:, :, 1] >= 133) & (y[:, :, 1] <= 173) &
                (y[:, :, 2] >= 77) & (y[:, :, 2] <= 127)).astype(np.float32)
    m = (acc / len(frames[::40])) > 0.5
    if m.sum() < 500:
        h, w = frames[0].shape[:2]
        m = np.zeros((h, w), bool); m[h//4:3*h//4, w//4:3*w//4] = True
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32)[m].mean(0).reshape(1, 1, 3)
            for f in frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fps", type=float, default=30)
    a = ap.parse_args()
    errs = {"POS": [], "CHROM": []}
    for d in sorted(glob.glob(os.path.join(a.root, "subject*"))):
        gtw = np.array(open(os.path.join(d, "ground_truth.txt")).readline().split(), float)
        gt_hr = float(_calculate_fft_hr(gtw - gtw.mean(), fs=a.fps, low_pass=0.7, high_pass=4.0))
        series = skin_mean_series(os.path.join(d, "vid.avi"))
        for name, fn in (("POS", POS_WANG), ("CHROM", CHROME_DEHAAN)):
            bvp = np.asarray(fn(series, a.fps)).squeeze()
            hr = float(_calculate_fft_hr(bvp, fs=a.fps, low_pass=0.7, high_pass=4.0))
            errs[name].append(abs(hr - gt_hr))
    print(f"== {a.root} (n={len(errs['POS'])}) ==")
    for m in ("POS", "CHROM"):
        e = np.array(errs[m])
        print(f"{m}: MAE={e.mean():.2f} BPM (median {np.median(e):.2f}, max {e.max():.1f})")


if __name__ == "__main__":
    main()
