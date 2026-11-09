#!/usr/bin/env python3
"""44 문서 통합 판정: VIPL 5-fold in-domain 고심박 가중 셀(TWgW) — 시드 풀링 vs 앵커(TWg)·VIPL 단독(BASEAV/BASEAVW)·FactorizePhys zero-shot; zero-shot 가중 셀(CoTAF_HW_L) vs 채택 모델 + 13축·MMPD 비열등.
  python scripts/final_vipl_judge.py [a1|all]   (a1: A1 게이트 — TWgW 가용 시드 풀링 vs TWg 가용 시드 풀링, 전체 Δ<0·p<.05 이면 'GO')"""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"; PF = "/SPECPHYS_ROOT/_workspace/data/physformer"
mode = sys.argv[1] if len(sys.argv) > 1 else "all"
def wp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); return float("nan") if len(x) < 6 or np.allclose(x, y) else float(wilcoxon(x, y).pvalue)
def cv_rows(tag, seeds=("", "_s200", "_s300")):
    """fold 모델 VF{k}_{tag}{seed} 의 자기 fold 출력을 모아 영상별 시드 평균 (agg_VF{k}_{tag}{s}_VIPL.json)"""
    per = {}
    for s in seeds:
        rows = []
        for k in range(1, 6):
            f = f"{R}/agg_VF{k}_{tag}{s}_VIPL.json"
            if not os.path.exists(f): rows = None; break
            rows += [r for r in json.load(open(f)) if r["fold"] == k and r["err_lit"] is not None]
        if rows is None: continue
        for r in rows: per.setdefault(r["vid"], []).append(r)
    if not per: return None, 0
    n = min(len(v) for v in per.values())
    return {k: dict(err_lit=np.mean([r["err_lit"] for r in v]), err=np.mean([r["err"] for r in v]), gt=v[0]["gt_lit"], sc=v[0]["scenario"]) for k, v in per.items() if len(v) == n}, n
def zs_rows(m, seeds=("", "_s200", "_s300")):
    per = {}
    for s in seeds:
        f = f"{R}/agg_{m}{s}_VIPL.json"
        if not os.path.exists(f): continue
        for r in json.load(open(f)):
            if r["err_lit"] is not None: per.setdefault(r["vid"], []).append(r)
    if not per: return None, 0
    n = min(len(v) for v in per.values())
    return {k: dict(err_lit=np.mean([r["err_lit"] for r in v]), err=np.mean([r["err"] for r in v]), gt=v[0]["gt_lit"], sc=v[0]["scenario"]) for k, v in per.items() if len(v) == n}, n
sel = {"all": lambda r: True, "60-90": lambda r: 60 <= r["gt"] < 90, "90-120": lambda r: 90 <= r["gt"] < 120, ">=120": lambda r: r["gt"] >= 120, "<60": lambda r: r["gt"] < 60, "v7": lambda r: r["sc"] == "v7", "v2+v9": lambda r: r["sc"] in ("v2", "v9")}
def cmp(name, A, B, na, nb, s="all", key="err_lit"):
    if not A or not B: return None
    ks = [k for k in A if k in B and sel[s](A[k])]
    if not ks: return None
    x = np.array([A[k][key] for k in ks]); y = np.array([B[k][key] for k in ks])
    return dict(cmp=name, bin=s, key=key, n=len(ks), seeds=[na, nb], a=float(x.mean()), b=float(y.mean()), delta=float(y.mean() - x.mean()), p=wp(x, y))
def holm(ts):
    ps = [(i, t["p"]) for i, t in enumerate(ts) if t and not np.isnan(t["p"])]; ps.sort(key=lambda z: z[1]); m = len(ps); prev = 0.0
    for r, (i, p) in enumerate(ps): adj = min(1.0, max(prev, (m - r) * p)); ts[i]["p_holm"] = adj; prev = adj
    return ts
def show(t):
    if t: print(f"  {t['cmp']:34s} [{t['bin']:6s}] n={t['n']:4d} ({t['seeds'][0]}s→{t['seeds'][1]}s) {t['a']:6.2f} → {t['b']:6.2f} Δ {t['delta']:+6.2f} p={t['p']:.4f}" + (f" Holm={t['p_holm']:.4f}" if 'p_holm' in t else ""))
TWgW, nW = cv_rows("TWgW_L"); TWg, nA = cv_rows("TWg_L"); BAV, nB = cv_rows("BASEAV_L", ("",)); BAVW, nBW = cv_rows("BASEAVW_L", ("",)); FP, _ = zs_rows("FPbase", ("",))
HW, nHW = zs_rows("CoTAF_HW_L"); TW, nTW = zs_rows("CoTAF_TW_L")
res = {"mode": mode, "primary": [], "secondary": []}
print(f"=== 44 문서 통합 판정 mode={mode}: TWgW {nW}s, TWg {nA}s, BASEAVW {nBW}s, CoTAF_HW {nHW}s")
P = [cmp("TWg→TWgW", TWg, TWgW, nA, nW), cmp("FPbase(zero-shot)→TWgW", FP, TWgW, 1, nW), cmp("BASEAVW→TWgW", BAVW, TWgW, nBW, nW), cmp("CoTAF_TW→CoTAF_HW (zero-shot)", TW, HW, nTW, nHW)]
res["primary"] = holm([t for t in P if t]); print("--- primary (Holm)"); [show(t) for t in res["primary"]]
S = []
for s in ("<60", "60-90", "90-120", ">=120", "v7", "v2+v9"):
    S += [cmp("TWg→TWgW", TWg, TWgW, nA, nW, s), cmp("FPbase→TWgW", FP, TWgW, 1, nW, s), cmp("BASEAVW→TWgW", BAVW, TWgW, nBW, nW, s), cmp("CoTAF_TW→CoTAF_HW", TW, HW, nTW, nHW, s)]
S += [cmp("BASEAV→TWgW", BAV, TWgW, nB, nW), cmp("BASEAV→BASEAVW", BAV, BAVW, nB, nBW), cmp("TWg→TWgW", TWg, TWgW, nA, nW, "all", "err")]
res["secondary"] = [t for t in S if t]; print("--- secondary"); [show(t) for t in res["secondary"]]
json.dump(res, open(f"{R}/final_vipl_judge_{mode}.json", "w"), indent=1, ensure_ascii=False)
if mode == "a1":
    t = res["primary"][0] if res["primary"] else None
    print("GO" if t and t["cmp"].startswith("TWg→TWgW") and t["delta"] < 0 and (np.isnan(t["p"]) or t["p"] < 0.05) else "STOP")
