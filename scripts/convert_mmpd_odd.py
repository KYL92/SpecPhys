#!/usr/bin/env python3
"""[체인 52, 34 문서] MMPD 피험자 7~33 의 홀수 트라이얼 270 클립을 UBFC 포맷으로 추가 변환(convert_mmpd_all.py 는 짝수만) → MMPD_ALL 이 7~33 전 540 클립.
v2(10-17): 메모리 절약판 — mat 의 video(float32 (T,320,240,3) ≈1.7 GB)를 통째로 복사하지 않고 300 프레임 블록씩 uint8 BGR 로 바꿔 ffmpeg 에 스트리밍(피크 ≈ mat 1회분).
사용: python scripts/convert_mmpd_odd.py  (이미 있는 클립은 건너뜀, 멱등)"""
import glob, os, subprocess, sys
import numpy as np, scipy.io as sio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_mmpd_all as C


def convert_stream(mat_path, subj, trial, block=300):
    m = sio.loadmat(mat_path); v = m["video"]; ppg = np.asarray(m["GT_ppg"], float).reshape(-1)
    n = min(len(v), len(ppg)); ppg = ppg[:n]; ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
    fr = np.fft.rfftfreq(n * 8, 1 / C.FPS); pw = np.abs(np.fft.rfft(ppg - ppg.mean(), n=n * 8)) ** 2
    band = (fr >= 0.7) & (fr <= 4.0); hr = float(fr[band][pw[band].argmax()] * 60)
    sid = 700000 + subj * 100 + trial; d = os.path.join(C.OUT, f"subject{sid}"); os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "vid.avi")
    if not os.path.exists(out):
        t, h, w = n, v.shape[1], v.shape[2]
        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(C.FPS), "-i", "-", "-c:v", "ffv1", "-level", "3", out + ".tmp.avi"]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for s in range(0, t, block):
            blk = (np.round(np.asarray(v[s:min(s + block, t)], np.float32) * 255)).astype(np.uint8)[..., ::-1]
            p.stdin.write(np.ascontiguousarray(blk).tobytes()); del blk
        p.stdin.close()
        if p.wait() != 0: raise RuntimeError("ffmpeg failed: " + out)
        os.replace(out + ".tmp.avi", out)
    with open(os.path.join(d, "ground_truth.txt"), "w") as f:
        f.write(" ".join(f"{x:.6f}" for x in ppg) + "\n"); f.write(" ".join(f"{hr:.2f}" for _ in range(n)) + "\n"); f.write(" ".join(f"{i/C.FPS:.4f}" for i in range(n)) + "\n")
    with open(os.path.join(d, "source.txt"), "w") as f:
        f.write(f"MMPD subject{subj}/p{subj}_{trial} light={m['light'][0]} motion={m['motion'][0]} exercise={m['exercise'][0]} skin={int(m['skin_color'][0][0])} n_frames={n} hr_fft={hr:.2f}\n")
    del v, m
    print(f"[mmpd-odd] subject{subj}/p{subj}_{trial} -> subject{sid} n={n} hr={hr:.1f}", flush=True)


if __name__ == "__main__":
    os.makedirs(C.OUT, exist_ok=True); n = 0
    for s in C.TRAIN_SUBJECTS:
        for mp in sorted(glob.glob(f"{C.SRC}/subject{s}/p{s}_*.mat"), key=lambda p: int(p.split("_")[-1][:-4])):
            t = int(mp.split("_")[-1][:-4])
            if t % 2 == 0: continue
            d = os.path.join(C.OUT, f"subject{700000 + s * 100 + t}")
            if os.path.exists(os.path.join(d, "ground_truth.txt")): continue
            for junk in ("vid.avi", "vid.avi.tmp.avi"):
                if os.path.exists(os.path.join(d, junk)): os.remove(os.path.join(d, junk))  # 중단된 이전 실행의 불완전 영상
            convert_stream(mp, s, t); n += 1
    print(f"[mmpd-odd] converted {n}")
