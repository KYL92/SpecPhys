#!/usr/bin/env python3
"""MCD-rPPG 학습 풀: 평가에 쓰지 않은 정체성 21~44(24명) x 2상태 -> MCD_TRAIN (in-domain co-training용)."""
import glob, os, sys
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
import convert_mcd as C

D = C.D; CAM = C.CAM
OUT = "/SPECPHYS_ROOT/_workspace/data/physformer/MCD_TRAIN"
if __name__ == "__main__":
    pids = sorted({os.path.basename(p).split("_")[0]
                   for p in glob.glob(f"{D}/video/*_{CAM}_before.avi")})
    chosen = []
    for pid in pids:
        if (os.path.exists(f"{D}/video/{pid}_{CAM}_after.avi")
                and os.path.exists(f"{D}/ppg_sync/{pid}_{CAM}_before.txt")
                and os.path.exists(f"{D}/ppg_sync/{pid}_{CAM}_after.txt")):
            chosen.append(pid)
    train_ids = chosen[20:44]   # 평가 20명 제외, 다음 24명
    print("train identities:", len(train_ids))
    os.makedirs(OUT, exist_ok=True)
    k = 0
    for step, base in [("before", 701), ("after", 801)]:
        for i, pid in enumerate(train_ids):
            if C.convert(pid, step, base + i - 700 if False else base + i, OUT):
                k += 1
    print("converted:", k)

