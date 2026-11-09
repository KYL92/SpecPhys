#!/usr/bin/env python3
"""체인 42: 5시드(100~500) 풀링 재판정 — 논문 통계 보강. TW vs HYB4, 움직임 분할 + 조건부 배치."""
import json
import numpy as np
from scipy.stats import wilcoxon

R = "/SPECPHYS_ROOT/_workspace/data/results"
SEEDS = ("", "_s200", "_s300", "_s400", "_s500")

def mm(m):
    return {(r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]): (r["err"], r["motion"])
            for r in json.load(open(f"{R}/agg_{m}_MMPD.json")) if r["subject"] >= 7}

def stack(base):
    ds = []
    for s in SEEDS:
        try: ds.append(mm(f"{base}{s}"))
        except FileNotFoundError: print(f"경고: {base}{s} 없음")
    ks = sorted(set.intersection(*[set(d) for d in ds]))
    return ks, ds

ks, tw = stack("CoTAF_TW_L")
_, hb = stack("CoTAM_HYB4_L")
MO = np.array([tw[0][k][1] for k in ks])
X = np.array([np.mean([d[k][0] for d in tw]) for k in ks])
Y = np.array([np.mean([d[k][0] for d in hb]) for k in ks])

def w(a, b): return wilcoxon(a, b).pvalue

print(f"=== 5시드 풀링 (n={len(ks)}, 시드 {len(tw)}/{len(hb)}) ===")
for nm, sel in (("전체", MO > 0), ("정지", MO == 1), ("움직임", MO > 1), ("Walking", MO == 4)):
    print(f"{nm:8s} TW {X[sel].mean():6.2f} -> HYB4 {Y[sel].mean():6.2f} (p={w(X[sel], Y[sel]):.4f})")
C = np.where(MO > 1, Y, X)
print(f"조건부배치 TW {X.mean():.2f} -> {C.mean():.2f} (p={w(X, C):.4f})")
print("시드별 (움직임): ", end="")
for i, s in enumerate(SEEDS):
    if i < len(tw) and i < len(hb):
        xs = np.array([tw[i][k][0] for k in ks if tw[i][k][1] > 1])
        ys = np.array([hb[i][k][0] for k in ks if hb[i][k][1] > 1])
        print(f"{s or '_s100'}: {xs.mean():.2f}->{ys.mean():.2f} (p={w(xs, ys):.3f})", end="  ")
print("\nPOOL5_DONE")
