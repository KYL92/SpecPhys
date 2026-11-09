#!/usr/bin/env python3
"""[68 문서 §5] 정체성–심박 분리를 강화하는 합성 학습 클립 두 갈래(대조 = 깨끗한 마스크 80클립, 67 문서). 클립 수·정체성·잡음 분포·마스크는 대조와 같다.
 R(클립 안 심박 변화): 대조 80클립과 같은 설정(같은 시드·잡음·정체성·평균 심박) + bpm_curve — 심박이 클립 안에서 평균 ±20 bpm 범위로 변함(오르막·내리막 번갈아, 4개 중 1개는 V자). 폴더 끝 _R.
 W(같은 장면 짝): 대조 80클립 중 짝수 번째 40개 장면을 같은 시드(같은 잡음·움직임)·같은 뱅크 구간으로 두 번 렌더 — 원래 심박과 55 bpm 떨어진 짝 심박(45~180 안). 폴더 TRAINPAIR80(피험자 1~80, 장면 k → 2k−1 원래, 2k 짝).
  python scripts/make_idhr_train_configs.py → configs/{train20h_tw,train6hi_tw,train6lo_tw,train24dn,train24dn2}_r.json, configs/trainpair80.json, configs/idhr_slots.json(풀 칸 대응)"""
import json, copy
import numpy as np
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; C = f"{PFA}/configs"
SETS = [("train20h_tw", 200), ("train6hi_tw", 220), ("train6lo_tw", 230), ("train24dn", 0), ("train24dn2", 0)]
bank = np.load(f"{PFA}/ppg_bank.npz"); f0 = np.asarray(bank["f0"], float)
clips = []   # (set, pool_slot, config)
for c, base in SETS:
    d = json.load(open(f"{C}/{c}_c.json"))
    for s in d["subjects"]: clips.append((c, base + int(s["id"]), s))
assert len(clips) == 80
# --- R ---
j = 0
for c, base in SETS:
    d = json.load(open(f"{C}/{c}_c.json")); d["out"] = d["out"].replace("_C", "_R")
    for s in d["subjects"]:
        b = float(s["bpm"]); lo, hi = max(42.0, b - 20), min(185.0, b + 20)
        s["bpm_curve"] = [[0, lo], [0.5, hi], [1, lo]] if j % 4 == 3 else ([[0, lo], [1, hi]] if j % 2 == 0 else [[0, hi], [1, lo]]); j += 1
    json.dump(d, open(f"{C}/{c}_r.json", "w"), indent=1)
# --- W ---
subs = []; slots = {}
for k, (c, slot, s) in enumerate(clips[0::2]):
    a = copy.deepcopy(s); b_ = copy.deepcopy(s)
    if a.get("wave_bank_nearest"):
        order = np.argsort(np.abs(np.log(f0 / (a["bpm"] / 60.0)))); seg = int(order[int(a.get("wave_bank_idx", 0)) % max(1, min(20, len(order)))])
    else:
        seg = int(a["wave_bank_idx"]) % len(f0)
    for x in (a, b_): x["wave_bank_nearest"] = False; x["wave_bank_idx"] = seg; x["pair_scene"] = k + 1; x["pair_src"] = f"{c}/{s['id']}"
    bp = a["bpm"] + 55 if a["bpm"] + 55 <= 180 else a["bpm"] - 55; b_["bpm"] = int(max(45, min(180, bp)))
    a["id"], b_["id"] = 2 * k + 1, 2 * k + 2; subs += [a, b_]
json.dump({"fps": 30, "size": 256, "out": f"{PFA}/TRAINPAIR80", "subjects": subs}, open(f"{C}/trainpair80.json", "w"), indent=1)
slot_list = [slot for _, slot, _ in clips]
json.dump({"R": {str(slot): f"{c.upper().replace('TRAIN20H_TW', 'TRAIN20H_TW')}" for c, slot, _ in clips}, "W": {str(slot_list[i]): i + 1 for i in range(80)}}, open(f"{C}/idhr_slots.json", "w"), indent=1)
bp = np.array([[x["bpm"] for x in subs[0::2]], [x["bpm"] for x in subs[1::2]]]); print("W pairs: mean |ΔHR| %.1f, HR range %d-%d" % (np.abs(bp[0] - bp[1]).mean(), bp.min(), bp.max()))
print("R written for", len(clips), "clips; W written", len(subs), "clips")
# --- K(결합 대조): 대조 80클립과 같은 설정·같은 심박 집합이지만 심박을 정체성마다 좁은 대역으로 다시 배정(낮은 심박 → sfhq_01 … 높은 심박 → sfhq_06). 정체성이 심박을 결정(η² 높음), 전체 심박 분포는 대조와 같다. 폴더 끝 _K.
rng = np.random.default_rng(68)
hrs = sorted(s["bpm"] for _, _, s in clips); ids_order = ["sfhq_01", "sfhq_02", "sfhq_03", "sfhq_05", "sfhq_06"]
by_id = {i: [k for k, (_, _, s) in enumerate(clips) if s["frames_dir"].split("/")[-1].replace("_frames", "") == i] for i in ids_order}
newbpm = {}; pos = 0
for i in ids_order:
    band = hrs[pos:pos + len(by_id[i])]; pos += len(by_id[i]); rng.shuffle(band)
    for k, b in zip(by_id[i], band): newbpm[k] = int(b)
k = 0
for c, base in SETS:
    d = json.load(open(f"{C}/{c}_c.json")); d["out"] = d["out"].replace("_C", "_K")
    for s in d["subjects"]: s["bpm"] = newbpm[k]; k += 1
    json.dump(d, open(f"{C}/{c}_k.json", "w"), indent=1)
grp = {i: [newbpm[k] for k in by_id[i]] for i in ids_order}; allv = [x for v in grp.values() for x in v]; m = np.mean(allv)
eta = sum(len(v) * (np.mean(v) - m) ** 2 for v in grp.values()) / sum((x - m) ** 2 for x in allv)
print("K: identity bands", {i: (min(v), max(v)) for i, v in grp.items()}, "eta^2 %.3f" % eta)
