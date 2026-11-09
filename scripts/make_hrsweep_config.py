#!/usr/bin/env python3
"""[68 문서 §4] 심박 추종 통제 시험 HRSWEEP: 학습 풀 밖 정체성 4개 × 심박 10개(45~180 bpm, 15 간격). 정체성마다 시드를 고정해 움직임·조명·센서 잡음이 심박과 무관하게 똑같고,
같은 뱅크 구간(nearest 아님)을 각 심박으로 재표본해 파형 모양도 같다 — 심박만 다르다. 맥파 마스크는 깨끗한 판(skin_mask_parser), 잡음은 약하게(드리프트 0.1, 이동 3 px, 회전 1°, 센서 2).
  python scripts/make_hrsweep_config.py → configs/hrsweep40.json (출력 HRSWEEP40, 피험자 1~40)"""
import json
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; BANK = f"{PFA}/ppg_bank.npz"
FACES = [("diff_female_01", 9501, 11), ("diff_female_02", 9502, 23), ("diff_male_01", 9503, 37), ("sfhq04", 9504, 52)]
subs = []; i = 0
for face, seed, seg in FACES:
    for bpm in range(45, 181, 15):
        i += 1
        subs.append({"id": i, "frames_dir": f"_workspace/data/faces/{face}_frames", "bpm": bpm, "alpha": 0.01, "seed": seed, "spatial": "uniform", "coverage": 1.0, "patch_scale": 40,
                     "wave_bank": BANK, "wave_bank_idx": seg, "noise": {"illum_drift": 0.1, "motion_px": 3, "rot_deg": 1.0, "sensor_std": 2}, "skin_mask_parser": True, "hrsweep_face": face})
json.dump({"fps": 30, "size": 256, "out": f"{PFA}/HRSWEEP40", "subjects": subs}, open(f"{PFA}/configs/hrsweep40.json", "w"), indent=1)
print("hrsweep40:", len(subs), "clips")
