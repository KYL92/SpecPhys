#!/usr/bin/env python3
"""TRAIN_BAL 합성 config 생성 — BPM 40~160을 10 BPM 빈마다 3클립(총 36)으로 균등 커버.
- 조건: clean : light : medium = 1:1:1, 빈 내 (BPM 오프셋, 조건) 순환으로 교락 방지
- 정체성: SFHQ 5명(01,02,03,05,06) 순환(학습 전용; 평가 C/E 세트는 sfhq04·diff_* 사용)
- 파형: train_h2_proposal.json 의 실분포 정합 규약 — a2 = 실분포 분위수 함수(p5..p95 보간)의
  36 균등 분위, 셔플; φ2~N(0.084,0.415), a3~N(0.293,0.091) clip[0,0.5], φ3~N(-0.279,0.602)
- 공간: uniform / patches(coverage 0.4~0.6) 교대
- 시드: 1001~1036 (기존 학습 101~120/501~566, 평가 301~424/700대, 실노이즈 800대와 분리)
출력: _workspace/data/physformer/configs/train_bal.json
"""
import json, os
import numpy as np

ROOT = "/SPECPHYS_ROOT"
OUT_CFG = f"{ROOT}/_workspace/data/physformer/configs/train_bal.json"
FACES = ["sfhq_01_frames", "sfhq_02_frames", "sfhq_03_frames", "sfhq_05_frames", "sfhq_06_frames"]
# 실 PPG a2 분위수 (12 보고서 / train_h2_proposal._proposal_note)
A2_Q = [0.05, 0.25, 0.50, 0.75, 0.95]
A2_V = [0.3419, 0.4202, 0.5018, 0.6273, 0.8420]
LIGHT = [  # 경노이즈: 기존 그리드 최소값의 ~절반
    dict(illum_drift=0.06, flicker_amp=0.012, flicker_hz=1.2),
    dict(motion_px=4, rot_deg=0.8),
    dict(illum_drift=0.05, flicker_amp=0.01, flicker_hz=1.4, motion_px=3, rot_deg=0.6, sensor_std=1),
]
MEDIUM = [  # 중노이즈: 기존 학습 그리드 중앙값 수준
    dict(illum_drift=0.15, flicker_amp=0.03, flicker_hz=1.1, sensor_std=2),
    dict(motion_px=10, rot_deg=2.0, sensor_std=2),
    dict(illum_drift=0.12, flicker_amp=0.025, flicker_hz=0.95, motion_px=8, rot_deg=1.5, sensor_std=2),
]


def main():
    rng = np.random.default_rng(19)
    n = 36
    a2 = np.interp(np.linspace(0.05, 0.95, n), A2_Q, A2_V)
    rng.shuffle(a2)
    phi2 = rng.normal(0.084, 0.415, n)
    a3 = np.clip(rng.normal(0.293, 0.091, n), 0.0, 0.5)
    phi3 = rng.normal(-0.279, 0.602, n)
    subs = []
    k = 0
    for b, lo in enumerate(range(40, 160, 10)):
        for j, off in enumerate((2, 5, 8)):
            cond = (j + b) % 3          # 0 clean, 1 light, 2 medium — BPM 오프셋과 교락 방지
            bpm = lo + off
            spatial = "uniform" if k % 2 == 0 else "patches"
            s = {"id": k + 1, "frames_dir": f"_workspace/data/faces/{FACES[k % 5]}",
                 "bpm": bpm, "alpha": 0.01, "seed": 1001 + k,
                 "spatial": spatial, "coverage": 1.0 if spatial == "uniform" else [0.4, 0.5, 0.6][k % 3],
                 "patch_scale": 40,
                 "wave_a2": round(float(a2[k]), 3), "wave_phi2": round(float(phi2[k]), 3),
                 "wave_a3": round(float(a3[k]), 3), "wave_phi3": round(float(phi3[k]), 3),
                 "cond": ["clean", "light", "medium"][cond],
                 "noise": {} if cond == 0 else (LIGHT if cond == 1 else MEDIUM)[b % 3]}
            subs.append(s)
            k += 1
    cfg = {"fps": 30, "size": 256, "out": "_workspace/data/physformer/TRAIN_BAL", "subjects": subs,
           "_note": {"design": "12 bins x 3 clips (BPM lo+2/lo+5/lo+8); cond=(j+bin)%3 -> clean/light/medium 12:12:12",
                     "waveform": "a2 real-quantile 36 uniform (p5..p95), phi2~N(0.084,0.415), a3~N(0.293,0.091), phi3~N(-0.279,0.602), rng seed 19",
                     "seeds": "1001-1036", "faces": FACES}}
    os.makedirs(os.path.dirname(OUT_CFG), exist_ok=True)
    json.dump(cfg, open(OUT_CFG, "w"), indent=1, ensure_ascii=False)
    from collections import Counter
    print("wrote", OUT_CFG, len(subs), "subjects")
    print("cond:", Counter(s["cond"] for s in subs), "spatial:", Counter(s["spatial"] for s in subs))
    print("a2 range %.3f-%.3f  phi2 %.2f..%.2f  a3 %.2f..%.2f" % (a2.min(), a2.max(), phi2.min(), phi2.max(), a3.min(), a3.max()))
    for s in subs:
        print(f"  id{s['id']:2d} bpm{s['bpm']:4d} {s['cond']:6s} {s['spatial']:8s} a2={s['wave_a2']:.3f} {os.path.basename(s['frames_dir'])} noise={s['noise']}")


if __name__ == "__main__":
    main()
