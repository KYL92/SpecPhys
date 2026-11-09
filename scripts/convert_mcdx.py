#!/usr/bin/env python3
"""[60 문서 §25] MCD-rPPG 미사용 피험자(정렬 순서 45번째 이후)의 실측 극단 심박 영상 → UBFC 레이아웃(학습 전용).
선택: after(운동 직후) 메타 맥박 ≥120, before(안정) 메타 맥박 <62. 앞 900프레임(30초), FFV1 무손실, ground_truth 1행 = ppg_sync PPG.
  python scripts/convert_mcdx.py [N_WORKERS=4]  → _workspace/data/physformer/MCDX/subject4001.. + MCDX/manifest.csv"""
import csv, glob, os, subprocess, sys
from multiprocessing import Pool
import numpy as np, cv2
D = "/SPECPHYS_RAW/mcd_rppg"; OUT = "/SPECPHYS_ROOT/_workspace/data/physformer/MCDX"; CAM = "FullHDwebcam"; NF = 900
def job(a):
    sid, pid, step, pulse = a; d = f"{OUT}/subject{sid}"; os.makedirs(d, exist_ok=True)
    if os.path.exists(f"{d}/vid.avi") and os.path.exists(f"{d}/ground_truth.txt"): return sid, "skip"
    gt = np.array([float(l.split()[0]) for l in open(f"{D}/ppg_sync/{pid}_{CAM}_{step}.txt") if l.strip()])
    cap = cv2.VideoCapture(f"{D}/video/{pid}_{CAM}_{step}.avi"); fr = []
    while len(fr) < NF:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    cap.release(); n = min(len(fr), len(gt), NF)
    if n < 600: return sid, f"short {n}"
    h, w = fr[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", "ffv1", "-level", "3", f"{d}/vid.avi"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in fr[:n]: p.stdin.write(f.tobytes())
    p.stdin.close(); assert p.wait() == 0
    with open(f"{d}/ground_truth.txt", "w") as f:
        f.write(" ".join(f"{x:.4f}" for x in gt[:n]) + "\n"); f.write(" ".join("0" for _ in range(n)) + "\n"); f.write(" ".join(f"{i/30:.4f}" for i in range(n)) + "\n")
    open(f"{d}/source.txt", "w").write(f"{pid}_{CAM}_{step}  frames={n}  meta_pulse={pulse}\n"); return sid, "ok"
if __name__ == "__main__":
    pids = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(f"{D}/video/*_{CAM}_before.avi")})
    both = [p for p in pids if os.path.exists(f"{D}/video/{p}_{CAM}_after.avi") and os.path.exists(f"{D}/ppg_sync/{p}_{CAM}_before.txt") and os.path.exists(f"{D}/ppg_sync/{p}_{CAM}_after.txt")]
    used = set(both[:44]); rest = [p for p in pids if p not in used]   # convert_mcd.py 의 평가 20 + 학습 24 와 동일한 결정론적 선택을 제외
    pulse = {(r["patient_id"], r["step"]): float(r["pulse"]) for r in csv.DictReader(open(f"{D}/db.csv")) if r["camera"] == CAM and r["pulse"]}
    sel = [(p, "after") for p in rest if pulse.get((p, "after"), 0) >= 120] + [(p, "before") for p in rest if pulse.get((p, "before"), 999) < 62]
    sel = [(p, s) for p, s in sel if os.path.exists(f"{D}/video/{p}_{CAM}_{s}.avi") and os.path.exists(f"{D}/ppg_sync/{p}_{CAM}_{s}.txt")]
    jobs = [(4001 + i, p, s, pulse[(p, s)]) for i, (p, s) in enumerate(sel)]
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/manifest.csv", "w") as f:
        f.write("subject,patient,step,meta_pulse\n"); [f.write(f"{a},{b},{c},{d}\n") for a, b, c, d in jobs]
    print(f"[mcdx] {len(jobs)} clips (after>=120: {sum(s == 'after' for _, s in sel)}, before<62: {sum(s == 'before' for _, s in sel)}); excluded {len(used)} split patients", flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pl:
        res = pl.map(job, jobs)
    bad = [r for r in res if r[1] not in ("ok", "skip")]; print(f"[mcdx] done ok={sum(r[1] == 'ok' for r in res)} skip={sum(r[1] == 'skip' for r in res)} bad={bad}", flush=True)
