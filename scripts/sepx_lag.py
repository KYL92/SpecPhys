#!/usr/bin/env python3
"""[65 문서 §6, 탐색 — 사전 등록 밖] 분리 시험 r 의 위상 교란 점검. 모델(시드)마다 출력과 정답 PPG 사이의 고정 시간 지연 L* 를
심박과 잡음이 먼 기존 클립(STRESS_OFFBAND6, 모든 셀이 심박을 97 % 이상 맞힘)에서 ±15 프레임 안의 평균 r 최대로 정하고, 그 지연에서 새 분리 시험 24클립의 r 을 다시 잰다.
지연을 고정하므로 잡음을 따라간 출력은 맞춰지지 않는다. pro6000 컨테이너에서 실행: python scripts/sepx_lag.py → results/sepx_lag.json"""
import glob, json, pickle, sys
import numpy as np
sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
PFA = "/SPECPHYS_ROOT/_workspace/data/physformer"; RES = "/SPECPHYS_ROOT/_workspace/data/results"
src = open("/SPECPHYS_ROOT/scripts/sepx_metrics.py").read(); src = src[:src.index("for M in sys.argv[1:]:")]
g = {}; exec(compile(src, "sm", "exec"), g); META = g["META"]
def z(a): a = np.asarray(a, np.float64); return (a - a.mean()) / (a.std() + 1e-12)
def r_lag(p, y, L):
    if L > 0: p, y = p[L:], y[:-L]
    elif L < 0: p, y = p[:L], y[-L:]
    return float(np.mean(z(p) * z(y)))
CELLS = {"F": ("NF_NSP2DR_L", [2600, 2700, 2800, 2900, 3000]), "LS": ("NF_LSUB_L", [2600, 2700, 2800, 2900, 3000]), "L0": ("NF_LSUB0_L", [2600, 2700, 2800, 2900, 3000]), "REF": ("NF_NSP2_L", [2100, 2200, 2300, 2400, 2500]), "BG": ("NF_BGREF_L", [2600, 2700, 2800, 2900, 3000]), "BGT": ("NF_NSP2DR_BGT_L", [2600, 2700, 2800, 2900, 3000]), "C": ("NF_NSP2C_L", [2100, 2200, 2300, 2400, 2500]), "NS": ("NF_NSP2NS_L", [2100, 2200, 2300, 2400, 2500])}   # [67 문서]   # [66 문서] 입력 단계 배경 기준 보정 셀 추가
LAGS = range(-15, 16); OUT = {}
for arm, (cell, seeds) in CELLS.items():
    for s in seeds:
        pk = glob.glob(f"{PFA}/runs/ev_{cell}_s{s}_SEPEVAL{'C' if g['CLEAN'] else ''}/*/saved_test_outputs/*.pickle")
        if not pk: continue
        d = pickle.load(open(pk[0], "rb")); P, Y = d["predictions"], d["labels"]
        get = lambda t: t.cpu().numpy().ravel() if hasattr(t, "cpu") else np.asarray(t).ravel()
        off = [sub for sub in P if META.get(sub, {}).get("set") == "STRESS_OFFBAND6"]
        score = {L: np.mean([r_lag(get(P[sub][c]), get(Y[sub][c]), L) for sub in off for c in P[sub]]) for L in LAGS}
        Ls = max(score, key=score.get); clips = {}
        for sub in P:
            m = META.get(sub)
            if m is None: continue
            clips[sub] = dict(set=m["set"], kind=m["kind"], gap=m["gap"], r0=float(np.mean([r_lag(get(P[sub][c]), get(Y[sub][c]), 0) for c in P[sub]])),
                              rL=float(np.mean([r_lag(get(P[sub][c]), get(Y[sub][c]), Ls) for c in P[sub]])))
        OUT[f"{cell}_s{s}"] = dict(arm=arm, lag=int(Ls), off_r_at_lag=float(score[Ls]), off_r0=float(score[0]), clips=clips)
        print(arm, cell, s, "lag", Ls, "OFFBAND r0 %.2f rL %.2f" % (score[0], score[Ls]), flush=True)
json.dump(OUT, open(f"{RES}/sepx_lag{'_c' if g['CLEAN'] else ''}.json", "w"), indent=1); print("written")
