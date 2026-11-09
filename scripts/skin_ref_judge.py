#!/usr/bin/env python3
"""[61 문서 §5·§6] 피부 가중·배경 조명 제거·잡음 참조 셀 판정: 셀 vs 채택 셀(NF_HYB_D_TA2_L) — 가족 Holm 4: H-K1 MMPD ≥110, H-K2 MMPD 보행, H-K3 풀 고심박(VIPL 장치 ≥120 ∪ MMPD ≥110), H-K4 RTNI (Δ<0 지지);
가드 VIPL 전체·MMPD 전체·VIPL <60 Δ≤+0.3; 서술 MR-NIRP 3축·VIPL 파형 ≥110; 2차 공개 RhythmFormer 대비 풀 고심박 비열등(2 bpm).
  python scripts/skin_ref_judge.py → results/skin_ref_judge.{txt,json}"""
import glob, json, os, re
import numpy as np
from scipy.stats import wilcoxon
import sys as _sys; _sys.path.insert(0, "/SPECPHYS_ROOT/scripts"); import mmpd_common_gt as _MGC   # [61 §9.3] MMPD 공통 정답
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
        for r in _MGC.load(f):
            if 7 <= r["subject"] <= 33: acc.setdefault((r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]), []).append(r)
    return {k: dict(gt=float(np.median([x["gt"] for x in v])), e=float(np.mean([x["err"] for x in v])), mo=v[0]["motion"]) for k, v in acc.items()}
def axis(m, ax):
    acc = {}
    for f in files(m, ""):
        d = json.load(open(f))
        for r in (d.get(ax) or []) if isinstance(d, dict) else []: acc.setdefault(r["vid"], []).append(r["err"])
    return {k: float(np.mean(v)) for k, v in acc.items()}
def cmp(a, b):
    k = sorted(set(a) & set(b))
    if len(k) < 5: return None
    x = np.array([a[i] for i in k]); y = np.array([b[i] for i in k]); d = x - y
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(x, y).pvalue)
    return dict(n=len(k), a=float(x.mean()), b=float(y.mean()), d=float(d.mean()), p=p if np.isfinite(p) else 1.0)
def ni(a, b, m=2.0):
    k = sorted(set(a) & set(b)); d = np.array([a[i] - b[i] for i in k]) - m
    return 1.0 if np.allclose(d, 0) else float(wilcoxon(d, alternative="less").pvalue)
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj
def pooled(V, M): return {**{("V", k): v["e"] for k, v in V.items() if v["dev"] >= 120}, **{("M",) + k: v["e"] for k, v in M.items() if v["gt"] >= 110}}
BASE = "NF_HYB_D_TA2_L"; BV, BM = vipl(BASE), mmpd(BASE)
RFV, RFM = vipl("RFbase"), mmpd("RhythmFormer")
CELLS = [("NF_SKW_L", "피부 가중"), ("NF_BGDC_L", "배경 조명 DC 정규화"), ("NF_SKBG_L", "피부 가중 + 배경 DC"), ("NF_SKREF_L", "피부 가중 + 잡음 참조 배제"), ("NF_BGRG_L", "회귀형 배경 조명 제거"), ("NF_SKBGRG_L", "피부 가중 + 회귀형 배경 제거")]
for cell, desc in CELLS:
    V, M = vipl(cell), mmpd(cell)
    if not V or not M: out(f"\n=== {cell} ({desc}): 결과 없음"); continue
    T = [("H-K1 MMPD ≥110", cmp({k: v["e"] for k, v in M.items() if v["gt"] >= 110}, {k: v["e"] for k, v in BM.items() if v["gt"] >= 110})),
         ("H-K2 MMPD 보행", cmp({k: v["e"] for k, v in M.items() if v["mo"] == 4}, {k: v["e"] for k, v in BM.items() if v["mo"] == 4})),
         ("H-K3 풀 고심박", cmp(pooled(V, M), pooled(BV, BM))),
         ("H-K4 RTNI", cmp(axis(cell, "RTNI"), axis(BASE, "RTNI")))]
    ps = holm([t[1]["p"] if t[1] else 1.0 for t in T]); res = {}
    out(f"\n=== {cell} ({desc}; 시드 {len(files(cell, '_VIPL'))}) vs 채택 셀 (시드 {len(files(BASE, '_VIPL'))}) — Holm 4")
    for (nm, r), ph in zip(T, ps):
        if r is None: out(f"  {nm:16s} (missing)"); continue
        v = "지지" if r["d"] < 0 and ph < .05 else ("역방향" if r["d"] > 0 and ph < .05 else "ns"); out(f"  {nm:16s} n={r['n']:4d} {r['b']:6.2f} -> {r['a']:6.2f} Δ{r['d']:+6.2f} p={r['p']:.4f} Holm={ph:.4f} → {v}"); res[nm] = dict(r, holm=ph, verdict=v)
    for nm, a, b in (("가드 VIPL 전체", {k: v["e"] for k, v in V.items()}, {k: v["e"] for k, v in BV.items()}), ("가드 MMPD 전체", {k: v["e"] for k, v in M.items()}, {k: v["e"] for k, v in BM.items()}),
                     ("가드 VIPL <60", {k: v["e"] for k, v in V.items() if v["dev"] < 60}, {k: v["e"] for k, v in BV.items() if v["dev"] < 60})):
        r = cmp(a, b); v = "위반" if r["d"] > 0.3 else "유지"; out(f"  {nm:16s} n={r['n']:4d} {r['b']:6.2f} -> {r['a']:6.2f} Δ{r['d']:+6.2f} p={r['p']:.4f} → {v}"); res[nm] = dict(r, verdict=v)
    for nm, ax in (("(서술) MR-NIRP 정지", "NIRP_STILL"), ("(서술) MR-NIRP 소동작", "NIRP_SMALL"), ("(서술) MR-NIRP 대동작", "NIRP_LARGE"), ("(서술) PURE", "PURE")):
        r = cmp(axis(cell, ax), axis(BASE, ax))
        if r: out(f"  {nm:16s} n={r['n']:4d} {r['b']:6.2f} -> {r['a']:6.2f} Δ{r['d']:+6.2f} p={r['p']:.4f}")
    r = cmp({k: abs(v["p"] - W[k]) for k, v in V.items() if k in W and W[k] >= 110}, {k: abs(v["p"] - W[k]) for k, v in BV.items() if k in W and W[k] >= 110})
    out(f"  (서술) VIPL 파형 ≥110 n={r['n']:4d} {r['b']:6.2f} -> {r['a']:6.2f} Δ{r['d']:+6.2f} p={r['p']:.4f}")
    a, b = pooled(V, M), pooled(RFV, RFM); k = sorted(set(a) & set(b)); pn = ni(a, b)
    out(f"  2차 공개 RhythmFormer 대비 풀 고심박: {np.mean([b[i] for i in k]):.2f} -> {np.mean([a[i] for i in k]):.2f} (n={len(k)}) 비열등(2 bpm) p={pn:.4f} → {'✓' if pn < .05 else '✗'}")
    res["S_vs_RF_pooled_high_NI_p"] = pn; OUT[cell] = res
open(f"{R}/skin_ref_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/skin_ref_judge.json", "w"), indent=1, ensure_ascii=False)
