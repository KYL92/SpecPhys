#!/usr/bin/env python3
"""[60 문서 §30] MMPD 동적 얼굴 검출 판정. 1차(Holm 2, 동적 조건): D1 풀 고심박(VIPL 장치 ≥120 ∪ MMPD ≥110) 비열등(2 bpm, 단측), D2 MMPD 전체 비열등(0.3, 단측) — 채택 셀 vs 공개 RhythmFormer.
2차: 같은 비교의 정적 u8 조건, 공개 FactorizePhys 대비, MMPD 대역·보행 조건. VIPL 은 재전처리하지 않으므로 두 조건 공통(기존 VIPL 출력).
  python scripts/mmpd_dyn_judge.py → results/mmpd_dyn_judge.{txt,json}"""
import glob, json, os, re
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"; SEEDS = (600, 700, 800, 900, 1000); LINES = []; OUT = {}
def out(s): print(s); LINES.append(s)
def key(r): return (r["subject"], r["light"], r["motion"], r["exercise"], r["skin"])
def vipl(names):
    acc = {}
    for n in names:
        for r in json.load(open(f"{R}/agg_{n}_VIPL.json")): acc.setdefault(r["vid"], []).append(r)
    return {k: dict(gt=v[0]["gt_lit"], err=float(np.mean([x["err_lit"] for x in v]))) for k, v in acc.items()}
def mmpd(names):
    acc = {}
    for n in names:
        for r in json.load(open(f"{R}/agg_{n}_MMPD.json")):
            if 7 <= r["subject"] <= 33: acc.setdefault(key(r), []).append(r)
    return {k: dict(gt=float(np.median([x["gt"] for x in v])), err=float(np.mean([x["err"] for x in v])), motion=v[0]["motion"]) for k, v in acc.items()}
OURS_V = vipl([f"NF_REP_D_TA2_L_s{s}" for s in SEEDS]); OURS_S = mmpd([f"NF_REP_D_TA2_L_s{s}" for s in SEEDS]); OURS_D = mmpd([f"NF_REPDYN_D_TA2_L_s{s}" for s in SEEDS])
RF_V = vipl(["RFbase"]); FP_V = vipl(["FPbase"])
REF = {"RF_S": mmpd(["RFbase_U8S"]), "RF_D": mmpd(["RFbase_U8D"]), "FP_S": mmpd(["FPbase_U8S"]), "FP_D": mmpd(["FPbase_U8D"])}
def pair(a, b, sel=lambda g: True):
    k = [x for x in a if x in b and sel(a[x]["gt"])]; return np.array([a[x]["err"] for x in k]), np.array([b[x]["err"] for x in k])
def ni(x, y, m):
    d = x - y - m; return 1.0 if np.allclose(d, 0) else float(wilcoxon(d, alternative="less").pvalue)
def sup(x, y): d = x - y; return 1.0 if np.allclose(d, 0) else float(wilcoxon(x, y).pvalue)
def block(tag, ours_m, ref_m, ref_v):
    xv, yv = pair(OURS_V, ref_v, lambda g: g >= 120); xm, ym = pair(ours_m, ref_m, lambda g: g >= 110)
    xh, yh = np.r_[xv, xm], np.r_[yv, ym]; xa, ya = pair(ours_m, ref_m)
    p1, p2 = ni(xh, yh, 2.0), ni(xa, ya, 0.3); adj = sorted([p1, p2]); h = {p1: min(1, 2 * adj[0]) if p1 == adj[0] else max(min(1, 2 * adj[0]), adj[1]), p2: min(1, 2 * adj[0]) if p2 == adj[0] else max(min(1, 2 * adj[0]), adj[1])}
    out(f"\n=== {tag}")
    out(f"  D1 풀 고심박(VIPL ≥120 ∪ MMPD ≥110) n={len(xh)}: 기존 {yh.mean():.2f} → 채택 {xh.mean():.2f} Δ{xh.mean()-yh.mean():+.2f} | 비열등(2) p={p1:.4f} Holm={h[p1]:.4f} {'✓' if h[p1] < .05 else '✗'} | 우세 p={sup(xh, yh):.4f}")
    out(f"  D2 MMPD 전체 n={len(xa)}: 기존 {ya.mean():.2f} → 채택 {xa.mean():.2f} Δ{xa.mean()-ya.mean():+.2f} | 비열등(0.3) p={p2:.4f} Holm={h[p2]:.4f} {'✓' if h[p2] < .05 else '✗'} | 우세 p={sup(xa, ya):.4f}")
    for nm, lo, hi in (("MMPD <70", 0, 70), ("MMPD 70–110", 70, 110), ("MMPD ≥110", 110, 999)):
        x, y = pair(ours_m, ref_m, lambda g: lo <= g < hi); out(f"    (서술) {nm:12s} n={len(x):3d} 기존 {y.mean():6.2f} → 채택 {x.mean():6.2f}")
    k = [z for z in ours_m if z in ref_m and ours_m[z]["motion"] == 4]; out(f"    (서술) MMPD 보행 n={len(k)} 기존 {np.mean([ref_m[z]['err'] for z in k]):.2f} → 채택 {np.mean([ours_m[z]['err'] for z in k]):.2f}")
    OUT[tag] = dict(D1=dict(ref=float(yh.mean()), ours=float(xh.mean()), p_ni=p1, holm=h[p1]), D2=dict(ref=float(ya.mean()), ours=float(xa.mean()), p_ni=p2, holm=h[p2]))
block("1차: 동적 검출 조건 — 채택 셀 vs 공개 RhythmFormer", OURS_D, REF["RF_D"], RF_V)
block("2차: 정적 u8 조건 — 채택 셀 vs 공개 RhythmFormer", OURS_S, REF["RF_S"], RF_V)
block("2차: 교차 — 채택 셀(동적) vs 공개 RhythmFormer(정적, 공개 RF 의 더 나은 조건)", OURS_D, REF["RF_S"], RF_V)
block("2차: 동적 — 채택 셀 vs 공개 FactorizePhys", OURS_D, REF["FP_D"], FP_V)
block("2차: 정적 u8 — 채택 셀 vs 공개 FactorizePhys", OURS_S, REF["FP_S"], FP_V)
x, y = pair(OURS_D, OURS_S); out(f"\n(서술) 채택 셀 정적 → 동적: MMPD 전체 {y.mean():.2f} → {x.mean():.2f} (p={sup(x, y):.4f}); ≥110: {pair(OURS_S, OURS_S, lambda g: g >= 110)[0].mean():.2f} → {pair(OURS_D, OURS_D, lambda g: g >= 110)[0].mean():.2f}")
open(f"{R}/mmpd_dyn_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/mmpd_dyn_judge.json", "w"), indent=1, ensure_ascii=False)
