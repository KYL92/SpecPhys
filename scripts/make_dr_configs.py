#!/usr/bin/env python3
"""[65 문서, 체인 124] 이중 렌더 픽셀·색 정답 실험의 합성 클립 설정 생성 (build_synth_dataset.py 입력).
  ① train24dn_nv / train24dn2_nv : 기존 이중 렌더 48클립(피험자 261~308)과 같은 설정 + noise_video (vid.avi 가 기존과 비트 단위로 같아야 한다 — 같은 시드·같은 난수 순서)
  ② train48dr  : 새 이중 렌더 학습 48클립(피험자 2001~2048, 풀의 합성 정체성 sfhq_01·02·03·05·06, 뱅크 파형, 진폭 0.008), 잡음 4종 × 12
       A 전역 플리커가 심박 ±4 bpm 안 · B 피부 국소 플리커가 심박 ±4 bpm 안 · C 피부 국소 플리커가 심박에서 8~40 bpm · D 주기 머리 움직임이 심박 ±4 bpm 안
  ③ stress_sepx24 : 풀 밖 시험 24클립(정체성 diff_female_01·02, diff_male_01, sfhq04), 전역 플리커 12 + 피부 국소 플리커 12, 심박과의 거리 0·+3·−3 bpm
  python scripts/make_dr_configs.py → _workspace/data/physformer/configs/{train24dn_nv,train24dn2_nv,train48dr,stress_sepx24}.json"""
import json, copy
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; C = f"{PFA}/configs"; BANK = f"{PFA}/ppg_bank.npz"
for src, dst in (("train24dn", "TRAIN24DN_NV"), ("train24dn2", "TRAIN24DN2_NV")):
    d = json.load(open(f"{C}/{src}.json")); d = copy.deepcopy(d); d["out"] = f"{PFA}/{dst}"
    for s in d["subjects"]: s["noise_video"] = True
    json.dump(d, open(f"{C}/{src}_nv.json", "w"), indent=1)
FACES = ["sfhq_01", "sfhq_02", "sfhq_03", "sfhq_05", "sfhq_06"]
NEAR = [0, 2, -2, 4, -4, 3, -3, 1, -1, 0, 2, -2]; FAR = [8, -8, 12, -12, 20, -20, 30, -30, 40, -40, 16, -16]
subs = []
for k in range(48):
    typ = "ABCD"[k // 12]; j = k % 12; bpm = int(round(48 + (k * 37 % 48) * 2.6))   # 48~170 bpm, 유형마다 고르게 섞음
    d_ = (NEAR if typ != "C" else FAR)[j]; nb = (bpm + d_) / 60.0
    nz = {"illum_drift": 0.15, "motion_px": 6, "rot_deg": 1.5, "sensor_std": 2}
    if typ == "A": nz.update(flicker_amp=0.03, flicker_hz=round(nb, 4))
    elif typ in "BC": nz["local_flicker"] = {"amp": 0.08, "hz": round(nb, 4), "frac": 0.35}
    else: nz.update(periodic_motion_amp=6, periodic_motion_hz=round(nb, 4))
    subs.append({"id": 2001 + k, "frames_dir": f"_workspace/data/faces/{FACES[k % 5]}_frames", "bpm": bpm, "alpha": 0.008, "seed": 6001 + k,
                 "spatial": "uniform" if k % 2 == 0 else "patches", "coverage": 1.0 if k % 2 == 0 else 0.55, "patch_scale": 40,
                 "wave_bank": BANK, "wave_bank_nearest": True, "wave_bank_idx": k, "noise": nz, "noise_gt": True, "noise_video": True, "dr_type": typ, "dr_gap": d_})
json.dump({"fps": 30, "size": 256, "out": f"{PFA}/TRAIN48DR", "subjects": subs}, open(f"{C}/train48dr.json", "w"), indent=1)
HO = [("diff_female_01", 64), ("diff_female_02", 70), ("diff_male_01", 91), ("sfhq04", 77)]
subs = []; i = 0
for typ in ("G", "L"):
    for gap in (0, 3, -3):
        for face, bpm in HO:
            i += 1; nb = (bpm + gap) / 60.0; nz = {"illum_drift": 0.1, "sensor_std": 2}
            if typ == "G": nz.update(flicker_amp=0.03, flicker_hz=round(nb, 4))
            else: nz["local_flicker"] = {"amp": 0.08, "hz": round(nb, 4), "frac": 0.35}
            subs.append({"id": i, "frames_dir": f"_workspace/data/faces/{face}_frames", "bpm": bpm, "alpha": 0.01, "seed": 9100 + i, "spatial": "uniform", "coverage": 1.0, "patch_scale": 40,
                         "wave_bank": BANK, "wave_bank_nearest": True, "wave_bank_idx": 40 + i, "noise": nz, "sepx_type": typ, "sepx_gap": gap})
json.dump({"fps": 30, "size": 256, "out": f"{PFA}/STRESS_SEPX24", "subjects": subs}, open(f"{C}/stress_sepx24.json", "w"), indent=1)
print("configs written")
