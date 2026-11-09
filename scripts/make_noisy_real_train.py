#!/usr/bin/env python3
"""co-training용 실데이터(34명)의 노이즈 증강 사본 생성 — 조건을 순환·랜덤화(시드 800+).
정답은 실제 산소포화도계 신호 그대로(디노이징 원칙: 오염 입력, 청정 감독)."""
import glob, os, shutil, sys
import numpy as np

sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from make_noisy_real import corrupt  # noqa

CO = "/SPECPHYS_ROOT/_workspace/data/physformer/COTRAIN"
OUT = "/SPECPHYS_ROOT/_workspace/data/physformer/COTRAIN_NOISY_REAL"

# 학습 그리드 범위 내에서 조건 순환 (평가 시드 700대와 분리된 800대)
VARIANTS = [
    dict(illum_drift=0.12, flicker_amp=0.025, flicker_hz=1.3),
    dict(motion_px=8, rot_deg=1.5),
    dict(illum_drift=0.18, flicker_amp=0.035, flicker_hz=1.0, sensor_std=2),
    dict(illum_drift=0.10, flicker_amp=0.02, flicker_hz=1.2, motion_px=6, rot_deg=1.0),
    dict(illum_drift=0.20, flicker_amp=0.04, flicker_hz=0.9, motion_px=12,
         rot_deg=2.5, sensor_std=3),
]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    # COTRAIN의 실데이터(subject번호 <200)만 대상
    subs = [s for s in sorted(glob.glob(f"{CO}/subject*"))
            if int(os.path.basename(s).replace("subject", "")) < 200]
    print(f"noising {len(subs)} real co-training subjects")
    for i, s in enumerate(subs):
        name = os.path.basename(s)
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        if os.path.exists(os.path.join(d, "vid.avi")):
            continue
        nz = VARIANTS[i % len(VARIANTS)]
        n = corrupt(os.path.realpath(os.path.join(s, "vid.avi")),
                    os.path.join(d, "vid.avi"), nz, seed=800 + i)
        shutil.copy(os.path.realpath(os.path.join(s, "ground_truth.txt")),
                    os.path.join(d, "ground_truth.txt"))
        print(f"[{i+1}/{len(subs)}] {name}: {n} frames, variant {i % len(VARIANTS)}")
    print("done")
