#!/usr/bin/env python3
"""MR-NIRP Car → UBFC 포맷 평가셋 변환 (975nm 세션의 RGB 카메라만 사용).
- RGB.zip: 640² uint16 Bayer(P5) — BayerBG 디모자이크 검증됨(육안 확인). /256→8bit, 320²로 축소
- PulseOX.zip cam0_full.pkl: pulseOxRecord/pulseOxTime(epoch초, ~53.5Hz) → 프레임시각(t0+i/30)에 선형보간
- 조건별 출력: NIRP_STILL / NIRP_SMALL / NIRP_LARGE (subject당 앞 60초 = 1800프레임)
- 화이트밸런스 미적용(원시): DiffNormalized·POS/CHROM 모두 정적 채널 게인 상쇄 — README와 함께 문서화
"""
import io, os, pickle, subprocess, zipfile
import numpy as np, cv2

SRC = "/SPECPHYS_RAW/MR-NIRP/MR-NIRP_Car"
PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
FPS, N_FRAMES, SIZE = 30, 1800, 320
CONDS = {"driving_still_975": "NIRP_STILL",
         "driving_small_motion_975": "NIRP_SMALL",
         "driving_large_motion_975": "NIRP_LARGE"}


def write_ffv1(frames, path):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "ffv1", "-level", "3", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    assert p.wait() == 0, path


def convert(subj_num, cond, out_root):
    trial = f"subject{subj_num}_{cond}"
    base = os.path.join(SRC, f"Subject{subj_num}", trial)
    rgb_zip, pox_zip = os.path.join(base, "RGB.zip"), None
    for cand in ("PulseOX.zip", "PulseOx.zip"):
        if os.path.exists(os.path.join(base, cand)):
            pox_zip = os.path.join(base, cand)
    if not (os.path.exists(rgb_zip) and pox_zip):
        print(f"[nirp] skip {trial} (missing zip)")
        return
    d = os.path.join(out_root, f"subject{subj_num}")
    if os.path.exists(os.path.join(d, "vid.avi")):
        print(f"[nirp] exists {trial}")
        return
    with zipfile.ZipFile(pox_zip) as z:
        name = [n for n in z.namelist() if n.endswith("cam0_full.pkl")][0]
        pk = pickle.loads(z.read(name), encoding="latin1")
    t = np.asarray(pk["pulseOxTime"], dtype=float)
    rec = np.asarray(pk["pulseOxRecord"], dtype=float)
    frames = []
    with zipfile.ZipFile(rgb_zip) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".pgm"))[:N_FRAMES]
        for n in names:
            buf = np.frombuffer(z.read(n), dtype=np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            rgb = cv2.cvtColor((im / 256).astype(np.uint8), cv2.COLOR_BayerBG2BGR)
            frames.append(cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
    n = len(frames)
    ft = t[0] + np.arange(n) / FPS          # 프레임 시각 (동기 시작 가정)
    ppg = np.interp(ft, t, rec)
    ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
    os.makedirs(d, exist_ok=True)
    write_ffv1(frames, os.path.join(d, "vid.avi"))
    fr = np.fft.rfftfreq(n * 8, 1 / FPS)
    pw = np.abs(np.fft.rfft(ppg - ppg.mean(), n=n * 8)) ** 2
    band = (fr >= 0.7) & (fr <= 4.0)
    hr = float(fr[band][pw[band].argmax()] * 60)
    with open(os.path.join(d, "ground_truth.txt"), "w") as f:
        f.write(" ".join(f"{v:.6f}" for v in ppg) + "\n")
        f.write(" ".join(f"{hr:.2f}" for _ in range(n)) + "\n")
        f.write(" ".join(f"{i/FPS:.4f}" for i in range(n)) + "\n")
    with open(os.path.join(d, "source.txt"), "w") as f:
        f.write(f"MR-NIRP_Car {trial} n={n} hr_fft={hr:.2f} bayer=BG wb=none\n")
    print(f"[nirp] {trial} n={n} hr={hr:.1f}")


if __name__ == "__main__":
    for cond, out in CONDS.items():
        os.makedirs(os.path.join(PF, out), exist_ok=True)
        for s in range(1, 20):
            convert(s, cond, os.path.join(PF, out))
    print("[nirp] done")
