#!/usr/bin/env python3
"""체인 38: LODO(Leave-One-Dataset-Out) 뱅크 확장 평가 집계 — 폴드별 TW(뱅크 v1) vs TW2(뱅크 v2).

각 폴드의 held-out 데이터셋 클립에서 per-clip 쌍대 Wilcoxon. MMPD 폴드는 전 661클립(전부 미학습)이
LODO 표본이며 참고로 표준 held-out(피험자 7~33, 540)도 병기. 마지막에 5폴드 풀링(전 클립 쌍) 검정.
"""
import json
import numpy as np
from scipy.stats import wilcoxon

R = "/SPECPHYS_ROOT/_workspace/data/results"
FOLD_AXES = {
    "DLCN": ["DLCN"],
    "MCD": ["MCD_A", "MCD_B"],
    "PURE": ["PURE"],
    "NIRP": ["NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE"],
    "MMPD": None,
}

def axis_pairs(fold, ax):
    a = json.load(open(f"{R}/agg_LD_{fold}_TW_L.json"))
    b = json.load(open(f"{R}/agg_LD_{fold}_TW2_L.json"))
    ka = {r["vid"]: r["err"] for r in a.get(ax, [])}
    kb = {r["vid"]: r["err"] for r in b.get(ax, [])}
    ks = sorted(set(ka) & set(kb))
    return np.array([ka[k] for k in ks]), np.array([kb[k] for k in ks])

def mmpd_pairs(sub_min=1):
    a = json.load(open(f"{R}/agg_LD_MMPD_TW_L_MMPD.json"))
    b = json.load(open(f"{R}/agg_LD_MMPD_TW2_L_MMPD.json"))
    ka = {r["vid"]: r["err"] for r in a if r["subject"] >= sub_min}
    kb = {r["vid"]: r["err"] for r in b if r["subject"] >= sub_min}
    ks = sorted(set(ka) & set(kb))
    return np.array([ka[k] for k in ks]), np.array([kb[k] for k in ks])

def wtest(x, y):
    if len(x) < 8 or np.allclose(x, y): return float("nan")
    return float(wilcoxon(x, y).pvalue)

pool_x, pool_y = [], []
print("=== LODO 폴드별: TW(v1) -> TW2(v2), held-out 데이터셋 클립 ===")
for fold, axes in FOLD_AXES.items():
    try:
        if fold == "MMPD":
            x, y = mmpd_pairs(1)
            xh, yh = mmpd_pairs(7)
            print(f"[{fold}] n={len(x)} {x.mean():.2f} -> {y.mean():.2f} p={wtest(x, y):.4f} "
                  f"(참고 held-out 7~33 n={len(xh)}: {xh.mean():.2f} -> {yh.mean():.2f} p={wtest(xh, yh):.4f})")
        else:
            fx, fy = [], []
            parts = []
            for ax in axes:
                x, y = axis_pairs(fold, ax)
                if not len(x): parts.append(f"{ax}: 없음"); continue
                parts.append(f"{ax} n={len(x)} {x.mean():.2f}->{y.mean():.2f} p={wtest(x, y):.3f}")
                fx.append(x); fy.append(y)
            x = np.concatenate(fx) if fx else np.array([]); y = np.concatenate(fy) if fy else np.array([])
            print(f"[{fold}] n={len(x)} {x.mean():.2f} -> {y.mean():.2f} p={wtest(x, y):.4f} | " + "  ".join(parts))
        pool_x.append(x); pool_y.append(y)
    except FileNotFoundError as e:
        print(f"[{fold}] agg 없음: {e}")
if pool_x:
    X = np.concatenate(pool_x); Y = np.concatenate(pool_y)
    d = Y - X
    print(f"=== 풀링(5폴드 전 클립) n={len(X)} TW {X.mean():.3f} -> TW2 {Y.mean():.3f} "
          f"p={wtest(X, Y):.4f} | 개선 {int((d < 0).sum())} 악화 {int((d > 0).sum())} ===")
print("LODOCMP_DONE")
