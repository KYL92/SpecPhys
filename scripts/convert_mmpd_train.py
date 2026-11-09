#!/usr/bin/env python3
"""MMPD 학습 분할(피험자 1~6, 120클립) → UBFC-rPPG 포맷 변환.
- 평가 held-out은 피험자 7~33 (subject-disjoint). PURE 01~06/07~10 규약과 동형.
- mat['video'] (T,320,240,3) float32 RGB [0,1] → uint8 → FFV1 무손실 vid.avi (BGR로 기록: UBFC 로더가 BGR→RGB 변환)
- GT: mat['GT_ppg'] (30Hz, 프레임 정렬) 표준화
- subject id: 7SSTT + 700000 → 700000+S*100+T (예: 700100 = subject1/p1_0). 기존 풀 id(0~9xx, 9SSTT)와 비충돌
"""
import glob, os, subprocess, sys
import numpy as np, scipy.io as sio
SRC = "/SPECPHYS_RAW/MMPD"
OUT = "/SPECPHYS_ROOT/_workspace/data/physformer/MMPD_TRAIN"
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6]
FPS = 30

def write_ffv1(frames_bgr, path):
    t, h, w, _ = frames_bgr.shape
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS),
           "-i", "-", "-c:v", "ffv1", "-level", "3", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.stdin.write(np.ascontiguousarray(frames_bgr).tobytes()); p.stdin.close()
    if p.wait() != 0: raise RuntimeError("ffmpeg failed: " + path)

def convert(mat_path, subj, trial):
    m = sio.loadmat(mat_path)
    v = (np.round(np.asarray(m["video"], np.float32) * 255)).astype(np.uint8)[..., ::-1]  # RGB→BGR
    ppg = np.asarray(m["GT_ppg"], float).reshape(-1)
    n = min(len(v), len(ppg)); v, ppg = v[:n], ppg[:n]
    ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
    fr = np.fft.rfftfreq(n * 8, 1 / FPS); pw = np.abs(np.fft.rfft(ppg - ppg.mean(), n=n * 8)) ** 2
    band = (fr >= 0.7) & (fr <= 4.0); hr = float(fr[band][pw[band].argmax()] * 60)
    sid = 700000 + subj * 100 + trial
    d = os.path.join(OUT, f"subject{sid}"); os.makedirs(d, exist_ok=True)
    if not os.path.exists(os.path.join(d, "vid.avi")): write_ffv1(v, os.path.join(d, "vid.avi"))
    with open(os.path.join(d, "ground_truth.txt"), "w") as f:
        f.write(" ".join(f"{x:.6f}" for x in ppg) + "\n")
        f.write(" ".join(f"{hr:.2f}" for _ in range(n)) + "\n")
        f.write(" ".join(f"{i/FPS:.4f}" for i in range(n)) + "\n")
    with open(os.path.join(d, "source.txt"), "w") as f:
        f.write(f"MMPD subject{subj}/p{subj}_{trial} light={m['light'][0]} motion={m['motion'][0]} "
                f"exercise={m['exercise'][0]} skin={int(m['skin_color'][0][0])} n_frames={n} hr_fft={hr:.2f}\n")
    print(f"[mmpd-train] subject{subj}/p{subj}_{trial} -> subject{sid} n={n} hr={hr:.1f}", flush=True)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for s in TRAIN_SUBJECTS:
        for mp in sorted(glob.glob(f"{SRC}/subject{s}/p{s}_*.mat"), key=lambda p: int(p.split("_")[-1][:-4])):
            convert(mp, s, int(mp.split("_")[-1][:-4]))
    print("[mmpd-train] done")
