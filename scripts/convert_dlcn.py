#!/usr/bin/env python3
"""DLCN_face → UBFC 포맷 평가셋 변환.
- 원본: /SPECPHYS_RAW/DLCN_face/P{s}_{t}/ frames(1801 png, 128² 사전 크롭 얼굴) + bvp.npy(프레임 정렬)
- fs=30 검증됨(추정 HR 63~101 BPM). 평가셋: 피험자 P1~P20의 첫 트라이얼 → DLCN_TEST/subjectN
- 사전 크롭이므로 toolbox 평가 config에서 DO_CROP_FACE: False 필요
"""
import glob, os, subprocess
import numpy as np, cv2

SRC = "/SPECPHYS_RAW/DLCN_face"
OUT = "/SPECPHYS_ROOT/_workspace/data/physformer/DLCN_TEST"
FPS = 30
N_SUBJ = 20


def write_ffv1(frames, path):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "ffv1", "-level", "3", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    assert p.wait() == 0, path


def main():
    os.makedirs(OUT, exist_ok=True)
    done = 0
    for s in range(1, 200):
        if done >= N_SUBJ:
            break
        trials = sorted(glob.glob(os.path.join(SRC, f"P{s}_*")),
                        key=lambda p: int(p.rsplit("_", 1)[1]))
        if not trials:
            continue
        tdir = trials[0]
        pngs = sorted(glob.glob(os.path.join(tdir, "frames", "*.png")))
        bvp = np.load(os.path.join(tdir, "bvp.npy")).astype(float)
        n = min(len(pngs), len(bvp))
        pngs, bvp = pngs[:n], bvp[:n]
        bvp = (bvp - bvp.mean()) / (bvp.std() + 1e-8)
        done += 1
        d = os.path.join(OUT, f"subject{done}")
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(os.path.join(d, "vid.avi")):
            write_ffv1([cv2.imread(p) for p in pngs], os.path.join(d, "vid.avi"))
        fr = np.fft.rfftfreq(n * 8, 1 / FPS)
        pw = np.abs(np.fft.rfft(bvp - bvp.mean(), n=n * 8)) ** 2
        band = (fr >= 0.7) & (fr <= 4.0)
        hr = float(fr[band][pw[band].argmax()] * 60)
        with open(os.path.join(d, "ground_truth.txt"), "w") as f:
            f.write(" ".join(f"{v:.6f}" for v in bvp) + "\n")
            f.write(" ".join(f"{hr:.2f}" for _ in range(n)) + "\n")
            f.write(" ".join(f"{i/FPS:.4f}" for i in range(n)) + "\n")
        with open(os.path.join(d, "source.txt"), "w") as f:
            f.write(f"DLCN {os.path.basename(tdir)} n={n} hr_fft={hr:.2f}\n")
        print(f"[dlcn] subject{done} <- {os.path.basename(tdir)} n={n} hr={hr:.1f}")
    print("[dlcn] done:", done)


if __name__ == "__main__":
    main()
