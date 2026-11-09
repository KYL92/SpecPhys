#!/usr/bin/env python3
"""eval_classical.py — 조건별 테스트 데이터셋(UBFC 포맷)에서 POS/CHROM으로 HR을 추정하고
달성 GT BPM(meta.json) 대비 MAE를 계산한다. 신경망과의 비교 기준선(무학습 하한선)."""
import argparse, glob, json, os, sys
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
    for f in frames[::50]:
        y = cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb)
        acc += ((y[:, :, 1] >= 133) & (y[:, :, 1] <= 173) &
                (y[:, :, 2] >= 77) & (y[:, :, 2] <= 127)).astype(np.float32)
    m = (acc / len(frames[::50])) > 0.5
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32)[m].mean(0).reshape(1, 1, 3)
            for f in frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="TEST_Cx 데이터셋 루트")
    ap.add_argument("--fps", type=float, default=30)
    a = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(a.root, "subject*"))):
        gt = json.load(open(os.path.join(d, "meta.json")))["achieved_bpm"]
        series = skin_mean_series(os.path.join(d, "vid.avi"))
        for name, fn in (("POS", POS_WANG), ("CHROM", CHROME_DEHAAN)):
            bvp = np.asarray(fn(series, a.fps)).squeeze()
            hr = float(_calculate_fft_hr(bvp, fs=a.fps, low_pass=0.7, high_pass=4.0))
            rows.append((os.path.basename(d), name, gt, hr, abs(hr - gt)))
    print(f"\n== {a.root} ==")
    for m in ("POS", "CHROM"):
        errs = [r[4] for r in rows if r[1] == m]
        print(f"{m}: MAE={np.mean(errs):.2f} BPM  (개별: " +
              ", ".join(f"{r[0]}:{r[3]:.0f}/gt{r[2]:.0f}" for r in rows if r[1] == m) + ")")
    out = os.path.join(a.root, "classical_results.json")
    json.dump([dict(zip(["subject", "method", "gt", "hr", "abs_err"], r)) for r in rows],
              open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
