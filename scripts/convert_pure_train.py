#!/usr/bin/env python3
"""PURE 학습 분할(피험자 01~06) → UBFC-rPPG 포맷 변환.
- 평가 held-out은 피험자 07~10 (subject-disjoint; CoTA의 PURE 성능은 이 부분집합으로 보고)
- 프레임: pngs → FFV1 무손실 vid.avi (원본 해상도 유지)
- GT: json /FullPackage waveform(~60Hz, ns 타임스탬프)을 /Image 타임스탬프에 선형보간
- subject id: 9SSTT (SS=피험자, TT=태스크) 예: 90101
"""
import glob, json, os, subprocess, sys
import numpy as np
import cv2

SRC = "/SPECPHYS_ROOT/_workspace/data/physformer/PURE_RAW"
OUT = "/SPECPHYS_ROOT/_workspace/data/physformer/PURE_TRAIN"
TRAIN_SUBJECTS = ["01", "02", "03", "04", "05", "06"]
FPS = 30


def write_ffv1_from_pngs(pngs, path):
    im0 = cv2.imread(pngs[0]); h, w = im0.shape[:2]
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-c:v", "ffv1", "-level", "3", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in pngs:
        p.stdin.write(cv2.imread(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed: " + path)


def convert_trial(trial):
    jp = os.path.join(SRC, trial, f"{trial}.json")
    j = json.load(open(jp))
    img_ts = np.array([e["Timestamp"] for e in j["/Image"]], dtype=np.int64)
    ppg_ts = np.array([e["Timestamp"] for e in j["/FullPackage"]], dtype=np.int64)
    wave = np.array([e["Value"]["waveform"] for e in j["/FullPackage"]], dtype=np.float64)
    pngs = sorted(glob.glob(os.path.join(SRC, trial, trial, "*.png")))
    n = min(len(pngs), len(img_ts))
    pngs, img_ts = pngs[:n], img_ts[:n]
    ppg = np.interp(img_ts, ppg_ts, wave)
    ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
    # 대표 HR: 제로패딩 FFT (기록용; 로더는 1행 PPG만 사용)
    fr = np.fft.rfftfreq(n * 8, 1 / FPS)
    pw = np.abs(np.fft.rfft(ppg - ppg.mean(), n=n * 8)) ** 2
    band = (fr >= 0.7) & (fr <= 4.0)
    hr = float(fr[band][pw[band].argmax()] * 60)

    sid = 90000 + int(trial[:2]) * 100 + int(trial[3:5])
    d = os.path.join(OUT, f"subject{sid}")
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(os.path.join(d, "vid.avi")):
        write_ffv1_from_pngs(pngs, os.path.join(d, "vid.avi"))
    with open(os.path.join(d, "ground_truth.txt"), "w") as f:
        f.write(" ".join(f"{v:.6f}" for v in ppg) + "\n")
        f.write(" ".join(f"{hr:.2f}" for _ in range(n)) + "\n")
        f.write(" ".join(f"{i/FPS:.4f}" for i in range(n)) + "\n")
    with open(os.path.join(d, "source.txt"), "w") as f:
        f.write(f"PURE {trial} n_frames={n} hr_fft={hr:.2f}\n")
    print(f"[pure-train] {trial} -> subject{sid} n={n} hr={hr:.1f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    trials = sorted(os.path.basename(t) for t in glob.glob(os.path.join(SRC, "??-??")))
    todo = [t for t in trials if t[:2] in TRAIN_SUBJECTS]
    print(f"converting {len(todo)} trials from subjects {TRAIN_SUBJECTS}")
    for t in todo:
        convert_trial(t)
    print("[pure-train] done")
