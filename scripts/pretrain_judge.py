#!/usr/bin/env python3
"""체인 62·63 판정(45 문서 §3): 코퍼스 사전학습 출발(PT) vs 공개 출발(BASE/PUB) — MMPD 5-fold(+부분집합 곡선)·VIPL 5-fold. 영상 단위 쌍대 Wilcoxon, 시드 풀링, Holm.
  python scripts/pretrain_judge.py [gate62|all]"""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"; mode = sys.argv[1] if len(sys.argv) > 1 else "all"
def wp(x, y): x, y = np.asarray(x, float), np.asarray(y, float); return float("nan") if len(x) < 6 or np.allclose(x, y) else float(wilcoxon(x, y).pvalue)
def pool(ds):
    ds = [d for d in ds if d]
    if not ds: return {}, 0
    ks = set.intersection(*[set(d) for d in ds]); return {k: float(np.mean([d[k] for d in ds])) for k in ks}, len(ds)
def mmpd_cv(tag, seeds=("", "_s200", "_s300")):
    ds = []
    for s in seeds:
        f = f"{R}/agg_MCV5_{tag}{s}_MMPD.json"
        if os.path.exists(f): ds.append({r["vid"]: float(r["err"]) for r in json.load(open(f))})
    return pool(ds)
def mmpd_fold(k, tag, seeds=("", "_s200", "_s300")):
    ds = []
    for s in seeds:
        f = f"{R}/agg_MF{k}_{tag}{s}_MMPD.json"
        if os.path.exists(f): ds.append({r["vid"]: float(r["err"]) for r in json.load(open(f))})
    return pool(ds)
def vipl_cv(tag, seeds=("",)):
    ds = []
    for s in seeds:
        f = f"{R}/agg_CV_{tag}{s}_VIPL.json"
        if os.path.exists(f): ds.append({r["vid"]: float(r["err_lit"]) for r in json.load(open(f)) if r["err_lit"] is not None})
    return pool(ds)
def cmp(name, A, B):
    (x, na), (y, nb) = A, B; ks = sorted(set(x) & set(y), key=str)
    if not ks: return None
    xa = np.array([x[k] for k in ks]); yb = np.array([y[k] for k in ks]); t = dict(cmp=name, n=len(ks), seeds=[na, nb], a=float(xa.mean()), b=float(yb.mean()), delta=float(yb.mean() - xa.mean()), p=wp(xa, yb))
    print(f"  {name:44s} n={t['n']:4d} ({na}s→{nb}s) {t['a']:6.2f} → {t['b']:6.2f} Δ {t['delta']:+6.2f} p={t['p']:.4f}"); return t
res = {"mode": mode, "c62": [], "c63": []}
print("=== 체인 62: 2단계(코퍼스 출발) vs 공개 출발 / joint")
PT = mmpd_cv("PT_L"); res["c62"] += [cmp("MMPD BASE(공개 출발)→PT(코퍼스 출발)", mmpd_cv("BASE_L"), PT), cmp("MMPD TW(joint)→PT", mmpd_cv("TW_L"), PT), cmp("MMPD HYB(분해)→PT", mmpd_cv("HYB_L"), PT)]
PTAV = vipl_cv("PTAV_L"); res["c62"] += [cmp("VIPL BASEAV(공개 출발)→PTAV(코퍼스 출발)", vipl_cv("BASEAV_L"), PTAV), cmp("VIPL TWg(joint)→PTAV", vipl_cv("TWg_L"), PTAV)]
if mode == "gate62":
    t = res["c62"][0]; print("GO" if t and t["delta"] < 0 and (np.isnan(t["p"]) or t["p"] < 0.10) else "STOP"); sys.exit()
print("=== 체인 63: 데이터 효율 곡선 (fold 1·2, 학습 피험자 25/50/100%)")
for k in (1, 2):
    for frac in (25, 50):
        res["c63"].append(cmp(f"fold{k} {frac}% PUB→PT", mmpd_fold(k, f"SUB{frac}_PUB_L"), mmpd_fold(k, f"SUB{frac}_PT_L")))
    res["c63"].append(cmp(f"fold{k} 100% BASE→PT", mmpd_fold(k, "BASE_L"), mmpd_fold(k, "PT_L", ("",))))
res = {k: ([t for t in v if t] if isinstance(v, list) else v) for k, v in res.items()}; json.dump(res, open(f"{R}/pretrain_judge.json", "w"), indent=1, ensure_ascii=False)
