#!/usr/bin/env python3
"""체인 41: 학습된 신뢰도(예측 노이즈 비율) 라우팅 평가 — GT-모션 라우팅(9.59)과 비교.
MMPD에서 clip별 conf(청크 평균, ev_*_MMPD_B*/conf_outputs.json) > τ 이면 HYB5, 아니면 TW.
τ는 MMPD 1~6(학습 분할) conf 분포에서 보정(HYB5가 1~6을 학습했으므로 과발화 편향 가능 — 명시).
사용: python conf_routing.py [MODEL_BASE=CoTAM_HYB5_L]
"""
import glob, json, re, sys
import numpy as np
from scipy.stats import wilcoxon

ROOT = "/SPECPHYS_ROOT"
R = f"{ROOT}/_workspace/data/results"
PF = f"{ROOT}/_workspace/data/physformer"
BASE = sys.argv[1] if len(sys.argv) > 1 else "CoTAM_HYB5_L"

def mm(m):
    return {r["vid"]: (r["err"], r["subject"], r["motion"])
            for r in json.load(open(f"{R}/agg_{m}_MMPD.json"))}

def confs(model):
    out = {}
    for f in glob.glob(f"{PF}/runs/ev_{model}_MMPD_B*/conf_outputs.json"):
        for vid, d in json.load(open(f)).items():
            out[vid] = float(np.mean(list(d.values())))
    return out

def wp(x, y):
    if len(x) < 8 or np.allclose(x, y): return float("nan")
    return float(wilcoxon(x, y).pvalue)

seeds = ("", "_s200", "_s300")
tw = [mm(f"CoTAF_TW_L{s}") for s in seeds]
hy = [mm(f"{BASE}{s}") for s in seeds]
cf = [confs(f"{BASE}{s}") for s in seeds]
ks = sorted(set(tw[0]) & set(hy[0]) & set(cf[0]))
sub = np.array([tw[0][k][1] for k in ks]); mo = np.array([tw[0][k][2] for k in ks])
ho = sub >= 7
T = np.array([np.mean([d[k][0] for d in tw]) for k in ks])
H = np.array([np.mean([d[k][0] for d in hy]) for k in ks])
C = np.array([np.mean([d[k] for d in cf]) for k in ks])

print(f"conf 분포: 전체 p10/50/90={np.percentile(C,[10,50,90]).round(3)} | "
      f"정지 {C[ho&(mo==1)].mean():.3f} vs 움직임 {C[ho&(mo>1)].mean():.3f} (분리도 확인)")

# τ 보정: 학습분할(1~6)에서 composite 오차 최소화 (편향 주의 명시)
tr = ~ho
taus = np.percentile(C[tr], np.arange(10, 91, 5))
best = min(taus, key=lambda t: np.where(C[tr] > t, H[tr], T[tr]).mean())
print(f"τ*(1~6 보정)={best:.3f}")

X, Yh = T[ho], H[ho]
gt_route = np.where(mo[ho] > 1, Yh, X)
cf_route = np.where(C[ho] > best, Yh, X)
print(f"held-out n={int(ho.sum())}: TW {X.mean():.2f} | HYB {Yh.mean():.2f} | "
      f"GT-모션 라우팅 {gt_route.mean():.2f} (p={wp(X, gt_route):.4f}) | "
      f"학습된-신뢰도 라우팅 {cf_route.mean():.2f} (p={wp(X, cf_route):.4f})")
agree = float((( C[ho] > best) == (mo[ho] > 1)).mean())
print(f"신뢰도-라우팅과 GT-모션 일치율: {agree:.2f}")
print("== τ 스윕(held-out, 참고) ==")
for q in (30, 40, 50, 60, 70):
    t = np.percentile(C[ho], q)
    r = np.where(C[ho] > t, Yh, X)
    print(f"  p{q} τ={t:.3f}: {r.mean():.2f} (HYB 사용 {float((C[ho]>t).mean()):.2f})")
print("CONF_ROUTING_DONE")
