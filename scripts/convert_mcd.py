#!/usr/bin/env python3
"""MCD-rPPG(정면 카메라)를 rPPG-Toolbox UBFC 레이아웃으로 변환.
- 20명 × {before(안정), after(운동직후)} → MCD_B / MCD_A
- 앞 60초(1800프레임)만 사용(길이·RAM 관리), FFV1 무손실 재인코딩(원본 mpeg4 디코딩 프레임 그대로)
- ground_truth.txt 1행 = ppg_sync의 프레임당 PPG 값(이미 프레임 동기화됨)
"""
import csv, glob, os, subprocess, sys
import numpy as np
import cv2

D = "/SPECPHYS_RAW/mcd_rppg"
OUT = "/SPECPHYS_ROOT/_workspace/data/physformer"
CAM = "FullHDwebcam"
N_SUB = 20
N_FRAMES = 1800  # 60 s @ 30 fps


def write_ffv1(frames, path, fps=30):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                          "-c:v", "ffv1", "-level", "3", path],
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {path}")


def convert(pid, step, sub_idx, root):
    vid = f"{D}/video/{pid}_{CAM}_{step}.avi"
    sync = f"{D}/ppg_sync/{pid}_{CAM}_{step}.txt"
    if not (os.path.exists(vid) and os.path.exists(sync)):
        return False
    gt = np.array([float(l.split()[0]) for l in open(sync) if l.strip()])
    cap = cv2.VideoCapture(vid)
    frames = []
    while len(frames) < N_FRAMES:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    n = min(len(frames), len(gt), N_FRAMES)
    if n < 900:  # 30초 미만이면 제외
        return False
    d = os.path.join(root, f"subject{sub_idx}")
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(os.path.join(d, "vid.avi")):
        write_ffv1(frames[:n], os.path.join(d, "vid.avi"))
    with open(os.path.join(d, "ground_truth.txt"), "w") as f:
        f.write(" ".join(f"{x:.4f}" for x in gt[:n]) + "\n")
        f.write(" ".join("0" for _ in range(n)) + "\n")
        f.write(" ".join(f"{i/30:.4f}" for i in range(n)) + "\n")
    with open(os.path.join(d, "source.txt"), "w") as f:
        f.write(f"{pid}_{CAM}_{step}  frames={n}\n")
    return True


if __name__ == "__main__":
    # 정면 카메라·양 상태 파일이 모두 있는 환자 중 앞 N_SUB명 (정렬 → 결정론)
    pids = sorted({os.path.basename(p).split("_")[0]
                   for p in glob.glob(f"{D}/video/*_{CAM}_before.avi")})
    chosen = []
    for pid in pids:
        if (os.path.exists(f"{D}/video/{pid}_{CAM}_after.avi")
                and os.path.exists(f"{D}/ppg_sync/{pid}_{CAM}_before.txt")
                and os.path.exists(f"{D}/ppg_sync/{pid}_{CAM}_after.txt")):
            chosen.append(pid)
        if len(chosen) == N_SUB:
            break
    print("subjects:", chosen)
    for step, root in [("before", f"{OUT}/MCD_B"), ("after", f"{OUT}/MCD_A")]:
        os.makedirs(root, exist_ok=True)
        ok = 0
        for i, pid in enumerate(chosen):
            if convert(pid, step, i + 1, root):
                ok += 1
                print(f"[{step}] subject{i+1} <- {pid}")
        print(f"{step}: {ok} videos")
    print("done")
