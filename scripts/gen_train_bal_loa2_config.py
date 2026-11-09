#!/usr/bin/env python3
"""TRAIN_BAL_LOA2 합성 config 생성 — 균형 풀(COTRAIN_BAL)에 저 a2(0~0.3) 파형 구간 보강(v2).
근거(17 보고서 §6c): 실분포 정합 파형만 쓴 CoTNH2가 안정 MCD에서 CoTNH(a2 0~0.75 그리드, a2<0.3 클립 11/20)보다
유의하게 악화(18.90 vs 5.19, p=.049) → 저 a2 클립이 기본파-고조파 대비 학습에 필요. TRAIN_BAL(a2 0.34~0.84)에는 없는 구간.
- 12클립: TRAIN_BAL 과 같은 12빈(40~160) 하나씩, BPM = 빈하한 + {4,6,3,7} 순환(TRAIN_BAL 의 +2/+5/+8 과 빈 내 불일치)
- a2 ∈ {0.0, 0.1, 0.2, 0.3} 순환(각 3클립), φ2~N(0.084,0.415) 실분포 표본, a3 ∈ {0, 0.1} 교대, φ3~N(-0.279,0.602)
- 조건 clean 8 : light 4 (k%3==1 → light, 경노이즈 3종 순환; a2 4종이 light 에 각 1회 → a2-조건 교락 없음)
- 정체성 SFHQ 5명(01,02,03,05,06) 순환, 공간 uniform/patches 교대
- 시드 1201~1212 (기존 configs/*.json 시드 101~120 / 301~354 / 401~424 / 501~566 / 1001~1036, 실노이즈 700·800대와 분리)
출력: _workspace/data/physformer/configs/train_bal_loa2.json
"""
import json, os
from collections import Counter
import numpy as np

ROOT = "/SPECPHYS_ROOT"
PF = f"{ROOT}/_workspace/data/physformer"
OUT_CFG = f"{PF}/configs/train_bal_loa2.json"
FACES = ["sfhq_01_frames", "sfhq_02_frames", "sfhq_03_frames", "sfhq_05_frames", "sfhq_06_frames"]
LIGHT = [  # gen_train_bal_config.LIGHT 와 동일
    dict(illum_drift=0.06, flicker_amp=0.012, flicker_hz=1.2),
    dict(motion_px=4, rot_deg=0.8),
    dict(illum_drift=0.05, flicker_amp=0.01, flicker_hz=1.4, motion_px=3, rot_deg=0.6, sensor_std=1),
]
A2_CYCLE = [0.0, 0.1, 0.2, 0.3]
OFFS = [4, 6, 3, 7]
SEED0 = 1201


def existing_seeds():
    s = set()
    for c in sorted(os.listdir(f"{PF}/configs")):
        if c.endswith(".json") and c != os.path.basename(OUT_CFG):
            for x in json.load(open(f"{PF}/configs/{c}")).get("subjects", []):
                s.add(int(x["seed"]))
    return s


def main():
    rng = np.random.default_rng(23)
    n = 12
    phi2 = rng.normal(0.084, 0.415, n)
    phi3 = rng.normal(-0.279, 0.602, n)
    used = existing_seeds()
    subs = []
    tb_bpms = {s["bpm"] for s in json.load(open(f"{PF}/configs/train_bal.json"))["subjects"]}
    for k, lo in enumerate(range(40, 160, 10)):
        bpm = lo + OFFS[k % 4]
        assert bpm not in tb_bpms, bpm
        cond = "light" if k % 3 == 1 else "clean"
        spatial = "uniform" if k % 2 == 0 else "patches"
        seed = SEED0 + k
        assert seed not in used, f"seed {seed} collides with existing config"
        subs.append({"id": k + 1, "frames_dir": f"_workspace/data/faces/{FACES[k % 5]}",
                     "bpm": bpm, "alpha": 0.01, "seed": seed,
                     "spatial": spatial, "coverage": 1.0 if spatial == "uniform" else [0.4, 0.5, 0.6][k % 3],
                     "patch_scale": 40,
                     "wave_a2": A2_CYCLE[k % 4], "wave_phi2": round(float(phi2[k]), 3),
                     "wave_a3": [0.0, 0.1][k % 2], "wave_phi3": round(float(phi3[k]), 3),
                     "cond": cond, "noise": {} if cond == "clean" else LIGHT[(k // 3) % 3]})  # light 3종 순환
    cfg = {"fps": 30, "size": 256, "out": "_workspace/data/physformer/TRAIN_BAL_LOA2", "subjects": subs,
           "_note": {"design": "low-a2 supplement for COTRAIN_BAL v2: 12 bins x 1 clip (BPM lo+{4,6,3,7}), clean 8 : light 4",
                     "waveform": "a2 cycle {0,0.1,0.2,0.3}, phi2~N(0.084,0.415), a3 alternate {0,0.1}, phi3~N(-0.279,0.602), rng seed 23",
                     "seeds": f"{SEED0}-{SEED0 + n - 1}", "faces": FACES,
                     "rationale": "17_harmonic_diversity_analysis §6c: CoTNH2 (a2 0.34-0.85 only) worse than CoTNH (a2 0-0.75 incl. 11/20 a2<0.3) on stable MCD, p=.049"}}
    json.dump(cfg, open(OUT_CFG, "w"), indent=1, ensure_ascii=False)
    print("wrote", OUT_CFG, len(subs), "subjects; existing seeds checked:", len(used))
    print("cond:", Counter(s["cond"] for s in subs), "a2:", Counter(s["wave_a2"] for s in subs),
          "a3:", Counter(s["wave_a3"] for s in subs), "faces:", Counter(os.path.basename(s["frames_dir"]) for s in subs))
    for s in subs:
        print(f"  id{s['id']:2d} bpm{s['bpm']:4d} {s['cond']:5s} {s['spatial']:8s} a2={s['wave_a2']:.1f} phi2={s['wave_phi2']:+.3f} a3={s['wave_a3']:.1f} seed={s['seed']} {os.path.basename(s['frames_dir'])} noise={s['noise']}")


if __name__ == "__main__":
    main()
