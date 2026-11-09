#!/usr/bin/env python3
"""make_noisy_real.py — held-out 실데이터(UBFC) 영상에 통제된 노이즈를 입혀
'실제 생리신호 + 통제 노이즈' 평가셋을 만든다. 정답(ground_truth.txt)은 실제
산소포화도계 신호 그대로 복사(노이즈는 라벨을 건드리지 않음).

조건 3종 (합성 실험과 동일 파라미터, 새 시드):
  illum  : 조명 드리프트 0.15 + 플리커 0.03 @ 1.1Hz(66 BPM, 심박 대역 내)
  motion : 강체 움직임 지터 10px / 2°
  combo  : illum + motion + 센서 σ2
"""
import glob, os, shutil, subprocess, sys
import cv2
import numpy as np

SKILLS = "/SPECPHYS_ROOT"
sys.path.insert(0, os.path.join(SKILLS, "generator"))
from add_noise import smooth_walk  # noqa

SRC = "/SPECPHYS_ROOT/_workspace/data/physformer/REALTEST"
OUT = "/SPECPHYS_ROOT/_workspace/data/physformer"
FPS = 30.0

CONDS = {
    "RTNI": dict(illum_drift=0.15, flicker_amp=0.03, flicker_hz=1.1),
    "RTNM": dict(motion_px=10, rot_deg=2.0),
    "RTNC": dict(illum_drift=0.15, flicker_amp=0.03, flicker_hz=1.1,
                 motion_px=10, rot_deg=2.0, sensor_std=2),
}


def write_ffv1(frames, path, fps):
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


def corrupt(vid, out_vid, nz, seed):
    cap = cv2.VideoCapture(vid)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    n = len(frames)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FPS
    drift = (smooth_walk(n, nz.get("illum_drift", 0) / max(np.sqrt(n), 1), int(FPS), rng)
             if nz.get("illum_drift", 0) > 0 else np.zeros(n))
    drift = drift - drift.mean()
    flick = (nz.get("flicker_amp", 0) * np.sin(2 * np.pi * nz.get("flicker_hz", 1.1) * t)
             if nz.get("flicker_amp", 0) > 0 else np.zeros(n))
    gain = np.clip(1 + drift + flick, 0.3, 1.7)
    mp, rd = nz.get("motion_px", 0), nz.get("rot_deg", 0)
    tx = smooth_walk(n, mp / max(np.sqrt(n), 1), int(FPS // 2), rng) if mp > 0 else np.zeros(n)
    ty = smooth_walk(n, mp / max(np.sqrt(n), 1), int(FPS // 2), rng) if mp > 0 else np.zeros(n)
    rot = smooth_walk(n, rd / max(np.sqrt(n), 1), int(FPS // 2), rng) if rd > 0 else np.zeros(n)
    sensor = nz.get("sensor_std", 0)
    h, w = frames[0].shape[:2]
    ctr = (w / 2, h / 2)
    out = []
    for i, img in enumerate(frames):
        img = img.astype(np.float32)
        if mp > 0 or rd > 0:
            M = cv2.getRotationMatrix2D(ctr, float(rot[i]), 1.0)
            M[:, 2] += (tx[i], ty[i])
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        img = img * gain[i]
        if sensor > 0:
            img = img + rng.normal(0, sensor, img.shape)
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    write_ffv1(out, out_vid, FPS)
    return n


if __name__ == "__main__":
    subs = sorted(glob.glob(os.path.join(SRC, "subject*")))
    for cname, nz in CONDS.items():
        root = os.path.join(OUT, cname)
        os.makedirs(root, exist_ok=True)
        for si, s in enumerate(subs):
            d = os.path.join(root, os.path.basename(s))
            os.makedirs(d, exist_ok=True)
            if os.path.exists(os.path.join(d, "vid.avi")):
                continue
            n = corrupt(os.path.join(s, "vid.avi"), os.path.join(d, "vid.avi"),
                        nz, seed=700 + list(CONDS).index(cname) * 10 + si)
            shutil.copy(os.path.join(s, "ground_truth.txt"),
                        os.path.join(d, "ground_truth.txt"))
            print(f"[{cname}] {os.path.basename(s)}: {n} frames")
    print("done")
