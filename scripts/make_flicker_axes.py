#!/usr/bin/env python3
"""[체인 68, 49 문서] 띠 밖 플리커 평가 축: REALTEST(UBFC held-out 8명) + 조명 드리프트 0.15 + 플리커 0.03 @ 1.7 Hz(RTNI17) / 2.4 Hz(RTNI24).
RTNI(1.1 Hz, 학습 띠 안)와 같은 생성 함수·같은 드리프트 시드 구조(seed 700 + cond_idx*10 + si; cond_idx 3,4 → 730+, 740+). 마커 /tmp/flicker_axes.done"""
import glob, os, shutil, sys
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from make_noisy_real import corrupt, SRC, OUT  # noqa
CONDS = {"RTNI17": dict(illum_drift=0.15, flicker_amp=0.03, flicker_hz=1.7), "RTNI24": dict(illum_drift=0.15, flicker_amp=0.03, flicker_hz=2.4)}
subs = sorted(glob.glob(os.path.join(SRC, "subject*")))
for ci, (cname, nz) in enumerate(CONDS.items(), 3):
    root = os.path.join(OUT, cname); os.makedirs(root, exist_ok=True)
    for si, s in enumerate(subs):
        d = os.path.join(root, os.path.basename(s)); os.makedirs(d, exist_ok=True)
        if os.path.exists(os.path.join(d, "vid.avi")): continue
        n = corrupt(os.path.join(s, "vid.avi"), os.path.join(d, "vid.avi"), nz, seed=700 + ci * 10 + si)
        shutil.copy(os.path.join(s, "ground_truth.txt"), os.path.join(d, "ground_truth.txt")); print(f"[{cname}] {os.path.basename(s)}: {n} frames", flush=True)
open("/tmp/flicker_axes.done", "w").write("ok\n"); print("done")
