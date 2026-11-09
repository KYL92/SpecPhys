#!/usr/bin/env python3
"""MR-NIRP Car 전 trial → 학습용 UBFC 포맷 (31 문서 트랙 N, in-domain 셀 전용; zero-shot 축 NIRP_STILL/SMALL/LARGE 와 별도).
- 19명 × {driving, garage} × {still, small_motion, large_motion} × 세션 {975, 940}, 전장(driving ≈4분, garage ≈2분), convert_nirp.py 와 동일 디모자이크/축소/PPG 보간.
- 출력 NIRP_TRAIN/subject{code}: code = 30000 + S*100 + cond*10 + sess (cond: d-still 1, d-small 2, d-large 3, g-still 4, g-small 5, g-large 6; sess 975=1, 940=2)
  → 피험자 S = (code-30000)//100 로 fold 배타 필터. 프레임은 ffmpeg 파이프로 스트리밍(메모리 < 1 GB/프로세스).
사용: python scripts/convert_nirp_train.py [WORKERS=3]"""
import os, pickle, subprocess, sys, zipfile
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, cv2
cv2.setNumThreads(1)
SRC = "/SPECPHYS_RAW/MR-NIRP/MR-NIRP_Car"
OUT = "/SPECPHYS_ROOT/_workspace/data/physformer/NIRP_TRAIN"
FPS, SIZE = 30, 320
CONDS = {"driving_still": 1, "driving_small_motion": 2, "driving_large_motion": 3, "garage_still": 4, "garage_small_motion": 5, "garage_large_motion": 6}
SESS = {"975": 1, "940": 2}


def convert(job):
    s, cond, sess = job
    code = 30000 + s * 100 + CONDS[cond] * 10 + SESS[sess]
    trial = f"subject{s}_{cond}_{sess}"; base = os.path.join(SRC, f"Subject{s}", trial)
    d = os.path.join(OUT, f"subject{code}")
    if os.path.exists(os.path.join(d, "ground_truth.txt")): return f"exists {trial}"
    rgb_zip = os.path.join(base, "RGB.zip"); pox = [os.path.join(base, c) for c in ("PulseOX.zip", "PulseOx.zip") if os.path.exists(os.path.join(base, c))]
    if not (os.path.exists(rgb_zip) and pox): return f"skip {trial} (missing zip)"
    try:
        with zipfile.ZipFile(pox[0]) as z:
            name = [n for n in z.namelist() if n.endswith("cam0_full.pkl")][0]
            pk = pickle.loads(z.read(name), encoding="latin1")
        t = np.asarray(pk["pulseOxTime"], dtype=float); rec = np.asarray(pk["pulseOxRecord"], dtype=float)
        os.makedirs(d, exist_ok=True); tmp = os.path.join(d, "vid.avi.tmp")
        with zipfile.ZipFile(rgb_zip) as z:
            names = sorted(n for n in z.namelist() if n.endswith(".pgm"))
            n = len(names)
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "-", "-c:v", "ffv1", "-level", "3", "-f", "avi", tmp]
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for nm in names:
                im = cv2.imdecode(np.frombuffer(z.read(nm), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                rgb = cv2.cvtColor((im / 256).astype(np.uint8), cv2.COLOR_BayerBG2BGR)
                p.stdin.write(cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA).tobytes())
            p.stdin.close(); assert p.wait() == 0, "ffmpeg"
        os.rename(tmp, os.path.join(d, "vid.avi"))
        ft = t[0] + np.arange(n) / FPS; ppg = np.interp(ft, t, rec); ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
        fr = np.fft.rfftfreq(n * 8, 1 / FPS); pw = np.abs(np.fft.rfft(ppg - ppg.mean(), n=n * 8)) ** 2; band = (fr >= 0.7) & (fr <= 4.0)
        hr = float(fr[band][pw[band].argmax()] * 60)
        with open(os.path.join(d, "ground_truth.txt"), "w") as f:
            f.write(" ".join(f"{v:.6f}" for v in ppg) + "\n"); f.write(" ".join(f"{hr:.2f}" for _ in range(n)) + "\n"); f.write(" ".join(f"{i/FPS:.4f}" for i in range(n)) + "\n")
        with open(os.path.join(d, "source.txt"), "w") as f:
            f.write(f"MR-NIRP_Car {trial} code={code} n={n} hr_fft={hr:.2f} bayer=BG wb=none full_length\n")
        return f"ok {trial} code={code} n={n} hr={hr:.1f}"
    except Exception as e:
        return f"FAIL {trial}: {e!r}"


if __name__ == "__main__":
    from multiprocessing import Pool
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    jobs = [(s, c, se) for s in range(1, 20) for c in CONDS for se in SESS]
    os.makedirs(OUT, exist_ok=True)
    with Pool(w) as pool:
        for r in pool.imap_unordered(convert, jobs): print(r, flush=True)
    print("NIRP_TRAIN_DONE", flush=True)
