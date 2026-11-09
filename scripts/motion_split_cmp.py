#!/usr/bin/env python3
"""움직임 분할 판정(사용자 표준, 2026-09-03): 모델 M을 정지/움직임 환경으로 나눠 TW·POS 대비 평가.
- MMPD held-out: 정지(MO1) vs 움직임(MO2~4, Rotation/Talking/Walking) — TW·POS 쌍대
- MR-NIRP: STILL vs SMALL+LARGE — TW·POS(agg_POS_NIRP 있으면) 쌍대
사용: python motion_split_cmp.py <MODEL> [REF=CoTAF_TW_L]
"""
import json, sys
import numpy as np
from scipy.stats import wilcoxon

R = "/SPECPHYS_ROOT/_workspace/data/results"
M = sys.argv[1]; REF = sys.argv[2] if len(sys.argv) > 2 else "CoTAF_TW_L"

def wp(x, y):
    if len(x) < 8 or np.allclose(x, y): return float("nan")
    return float(wilcoxon(x, y).pvalue)

def mmpd(m):
    return {(r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]): (r["err"], r["motion"])
            for r in json.load(open(f"{R}/agg_{m}_MMPD.json")) if r["subject"] >= 7}

print(f"=== 움직임 분할: {M} (기준 {REF}, POS) ===")
try:
    a, b, p = mmpd(REF), mmpd(M), mmpd("POS")
    ks = sorted(set(a) & set(b) & set(p))
    for name, sel in (("MMPD 정지(MO1)", lambda mo: mo == 1), ("MMPD 움직임(MO2~4)", lambda mo: mo > 1),
                      ("MMPD Walking(MO4)", lambda mo: mo == 4)):
        kk = [k for k in ks if sel(a[k][1])]
        X = np.array([a[k][0] for k in kk]); Y = np.array([b[k][0] for k in kk]); P = np.array([p[k][0] for k in kk])
        print(f"{name:18s} n={len(kk)}  {REF} {X.mean():6.2f} | {M} {Y.mean():6.2f} (p={wp(X, Y):.3f}) | "
              f"POS {P.mean():6.2f} (M vs POS p={wp(P, Y):.3f})")
except FileNotFoundError as e:
    print("MMPD 생략:", e)

def ax(m, a):
    d = json.load(open(f"{R}/agg_{m}.json"))
    return {r["vid"]: r["err"] for r in d.get(a, [])}

try:
    pos_n = {}
    try:
        for r in json.load(open(f"{R}/agg_POS_NIRP.json")):
            pos_n.setdefault(r["axis"], {})[r["vid"]] = r["err"]
    except FileNotFoundError:
        pass
    for name, axes in (("NIRP 정지(STILL)", ["NIRP_STILL"]), ("NIRP 움직임(SMALL+LARGE)", ["NIRP_SMALL", "NIRP_LARGE"])):
        Xs, Ys, Ps = [], [], []
        for axn in axes:
            ta, tb = ax(REF, axn), ax(M, axn)
            kk = sorted(set(ta) & set(tb))
            Xs += [ta[k] for k in kk]; Ys += [tb[k] for k in kk]
            if axn in pos_n:
                Ps += [pos_n[axn].get(k, np.nan) for k in kk]
        X, Y = np.array(Xs), np.array(Ys)
        line = f"{name:22s} n={len(X)}  {REF} {X.mean():6.2f} | {M} {Y.mean():6.2f} (p={wp(X, Y):.3f})"
        if Ps and not np.isnan(Ps).all():
            P = np.array(Ps); ok = ~np.isnan(P)
            line += f" | POS {P[ok].mean():6.2f} (M vs POS p={wp(P[ok], Y[ok]):.3f})"
        print(line)
except FileNotFoundError as e:
    print("NIRP 생략:", e)
print("MOTIONSPLIT_DONE")
