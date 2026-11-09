#!/usr/bin/env python3
"""[60 문서 §32] 용량 가설 판정: 폭 k 배 셀 vs 폭 1 채택 셀(NF_HYB_D_TA2_L) — 가족 Holm 3: H-C1 VIPL 장치 ≥120, H-C2 VIPL 파형 ≥110, H-C3 MMPD ≥110 (Δ<0 지지); 가드 VIPL·MMPD 전체·VIPL <60 Δ≤+0.3.
  python scripts/capacity_judge.py → results/capacity_judge.{txt,json}"""
import glob, json, os, re
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"; LINES = []; OUT = {}
def out(s): print(s); LINES.append(s)
def files(m, suf):
    pat = re.compile(rf"agg_{re.escape(m)}_s\d+{suf}\.json$")
    return [f for f in [f"{R}/agg_{m}{suf}.json"] + sorted(x for x in glob.glob(f"{R}/agg_{m}_s[0-9]*{suf}.json") if pat.search(os.path.basename(x))) if os.path.exists(f)]
W = {}
for m in ("NF_HYB_D_L", "RF_SPEC_L", "NF_HYB_U8_L"):
    for f in files(m, "_VIPL"):
        for r in json.load(open(f)): W.setdefault(r["vid"], []).append(r["gt"])
W = {k: float(np.median(v)) for k, v in W.items()}
def vipl(m):
    acc = {}
    for f in files(m, "_VIPL"):
        for r in json.load(open(f)): acc.setdefault(r["vid"], []).append(r)
    return {k: dict(dev=v[0]["gt_lit"], e=float(np.mean([x["err_lit"] for x in v])), p=float(np.mean([x["pred"] for x in v]))) for k, v in acc.items()}
def mmpd(m):
    acc = {}
    for f in files(m, "_MMPD"):
        for r in json.load(open(f)):
            if 7 <= r["subject"] <= 33: acc.setdefault((r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]), []).append(r)
    return {k: dict(gt=float(np.median([x["gt"] for x in v])), e=float(np.mean([x["err"] for x in v]))) for k, v in acc.items()}
def cmp(a, b):
    k = sorted(set(a) & set(b)); x = np.array([a[i] for i in k]); y = np.array([b[i] for i in k]); d = x - y
    return dict(n=len(k), a=float(x.mean()), b=float(y.mean()), d=float(d.mean()), p=1.0 if np.allclose(d, 0) else float(wilcoxon(x, y).pvalue))
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj
BV, BM = vipl("NF_HYB_D_TA2_L"), mmpd("NF_HYB_D_TA2_L")
for cell, params in (("NF_W4_D_TA2_L", "0.82 M"), ("NF_W8_D_TA2_L", "3.29 M")):
    V, Mm = vipl(cell), mmpd(cell)
    if not V or not Mm: out(f"\n=== {cell}: 결과 없음"); continue
    T = [("H-C1 VIPL 장치 ≥120", cmp({k: v["e"] for k, v in V.items() if v["dev"] >= 120}, {k: v["e"] for k, v in BV.items() if v["dev"] >= 120})),
         ("H-C2 VIPL 파형 ≥110", cmp({k: abs(v["p"] - W[k]) for k, v in V.items() if k in W and W[k] >= 110}, {k: abs(v["p"] - W[k]) for k, v in BV.items() if k in W and W[k] >= 110})),
         ("H-C3 MMPD ≥110", cmp({k: v["e"] for k, v in Mm.items() if v["gt"] >= 110}, {k: v["e"] for k, v in BM.items() if v["gt"] >= 110}))]
    G = [("가드 VIPL 전체", cmp({k: v["e"] for k, v in V.items()}, {k: v["e"] for k, v in BV.items()})), ("가드 MMPD 전체", cmp({k: v["e"] for k, v in Mm.items()}, {k: v["e"] for k, v in BM.items()})),
         ("가드 VIPL <60", cmp({k: v["e"] for k, v in V.items() if v["dev"] < 60}, {k: v["e"] for k, v in BV.items() if v["dev"] < 60}))]
    ps = holm([t[1]["p"] for t in T]); out(f"\n=== {cell} ({params}, 시드 {len(files(cell, '_VIPL'))}) vs 폭 1 채택 셀(0.052 M, 시드 {len(files('NF_HYB_D_TA2_L', '_VIPL'))})")
    res = {}
    for (nm, r), ph in zip(T, ps):
        v = "지지" if r["d"] < 0 and ph < .05 else ("역방향" if r["d"] > 0 and ph < .05 else "ns"); out(f"  {nm:20s} n={r['n']:4d} {r['b']:6.2f} -> {r['a']:6.2f} Δ{r['d']:+6.2f} p={r['p']:.4f} Holm={ph:.4f} → {v}"); res[nm] = dict(r, holm=ph, verdict=v)
    for nm, r in G:
        v = "위반" if r["d"] > 0.3 else "유지"; out(f"  {nm:20s} n={r['n']:4d} {r['b']:6.2f} -> {r['a']:6.2f} Δ{r['d']:+6.2f} p={r['p']:.4f} → {v}"); res[nm] = dict(r, verdict=v)
    OUT[cell] = res
open(f"{R}/capacity_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/capacity_judge.json", "w"), indent=1, ensure_ascii=False)
