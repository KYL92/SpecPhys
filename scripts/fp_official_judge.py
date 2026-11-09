#!/usr/bin/env python3
"""[체인 54, 39 문서] FactorizePhys + 공식 코퍼스 zero-shot 판정: FP_OFF_TW / FP_OFF_HYB (시드 ""/_s200/_s300 영상별 평균 풀링)
vs 공개 FactorizePhys(FPbase VIPL / FactorizePhys MMPD), POS, 같은 풀의 PhysFormer(CoTAF_TW_L / CoTAM_HYB4_L 시드 풀링), 소프.
1차 가족 Holm 4건: TW×VIPL, TW×MMPD, HYB×VIPL, HYB×MMPD. 출력 표 + results/fp_official_judge.json."""
import json, glob, sys, os, numpy as np
MMPD_REF = "FactorizePhys"
if "--mmpd_ref" in sys.argv: i = sys.argv.index("--mmpd_ref"); MMPD_REF = sys.argv[i + 1]   # 10-21: u8 경로 내 비교용 기준(FPBASE_U8CHK)
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"
def vipl(m):
    acc = {}
    for f in [f"{R}/agg_{m}_VIPL.json"] + sorted(glob.glob(f"{R}/agg_{m}_s[0-9]*_VIPL.json")):
        try: rows = json.load(open(f))
        except FileNotFoundError: continue
        rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
        for r in rows:
            if r.get("gt_lit") is None: continue
            acc.setdefault(r["vid"], []).append((abs(r["pred"] - r["gt_lit"]), r.get("scenario")))
    return {k: (float(np.mean([e for e, _ in v])), v[0][1]) for k, v in acc.items()}
def mmpd(m, subs=set(range(7, 34))):
    acc = {}
    for f in [f"{R}/agg_{m}_MMPD.json"] + sorted(glob.glob(f"{R}/agg_{m}_s[0-9]*_MMPD.json")):
        try: rows = json.load(open(f))
        except FileNotFoundError: continue
        for r in rows:
            if r["subject"] in subs: acc.setdefault((r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]), []).append(r["err"])
    return {k: float(np.mean(v)) for k, v in acc.items()}
def cmp(a, b, sel=lambda k, v: True):
    ks = [k for k in a if k in b and sel(k, a[k])]
    if len(ks) < 5: return None
    x = np.array([a[k][0] if isinstance(a[k], tuple) else a[k] for k in ks]); y = np.array([b[k][0] if isinstance(b[k], tuple) else b[k] for k in ks])
    return {"n": len(ks), "a": float(x.mean()), "b": float(y.mean()), "d": float(x.mean() - y.mean()), "p": float(wilcoxon(x, y).pvalue)}
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [None] * len(ps); m = len(ps); run = 0
    for rank, i in enumerate(order): run = max(run, min(1.0, (m - rank) * ps[i])); out[i] = run
    return out
def line(name, r, flag=""):
    if r is None: print(f"  {name:48s} (missing)"); return
    print(f"  {name:48s} n={r['n']:5d} {r['b']:6.2f} -> {r['a']:6.2f}  Δ{r['d']:+6.2f} p={r['p']:.4f}{flag}")
out = {"cells": {}, "pairs": {}, "primary": []}
for cell, pf in (("FP_OFF_TW", "CoTAF_TW_L"), ("FP_OFF_HYB", "CoTAM_HYB4_L")):
    v = vipl(cell); m = mmpd(cell)
    if not v and not m: print(f"=== {cell}: no results yet"); continue
    ns = len([f for f in [f"{R}/agg_{cell}_VIPL.json"] + glob.glob(f"{R}/agg_{cell}_s[0-9]*_VIPL.json") if __import__('os').path.exists(f)])
    print(f"=== {cell} (시드 {ns}) : VIPL 문헌 {np.mean([e for e,_ in v.values()]):.2f} (n={len(v)}) | MMPD held-out {np.mean(list(m.values())):.2f} (n={len(m)}) | MMPD 움직임 {np.mean([e for k,e in m.items() if k[2]!=1]):.2f}")
    out["cells"][cell] = {"seeds": ns, "vipl": float(np.mean([e for e,_ in v.values()])) if v else None, "mmpd": float(np.mean(list(m.values()))) if m else None}
    for rn, rv in (("공개 FactorizePhys", vipl("FPbase")), ("POS", vipl("POS")), (f"PhysFormer 같은 풀 {pf}(시드 풀링)", vipl(pf)), ("소프", vipl("CoTAM_HYB4_SOUP_L"))):
        r = cmp(v, rv); line(f"{cell} vs {rn} [VIPL 전체]", r); out["pairs"][f"{cell}|{rn}|VIPL"] = r
        if rn == "공개 FactorizePhys":
            out["primary"].append((f"{cell}×VIPL", r))
            for sc, nm in (("v2", "큰 움직임 v2"), ("v9", "휴대폰 움직임 v9")):
                r2 = cmp(v, rv, lambda k, val, sc=sc: val[1] == sc); line(f"{cell} vs {rn} [VIPL {nm}]", r2); out["pairs"][f"{cell}|{rn}|VIPL_{sc}"] = r2
    for rn, rm in (("공개 FactorizePhys", mmpd(MMPD_REF)), ("POS", mmpd("POS")), (f"PhysFormer 같은 풀 {pf}(시드 풀링)", mmpd(pf)), ("소프", mmpd("CoTAM_HYB4_SOUP_L"))):
        r = cmp(m, rm); line(f"{cell} vs {rn} [MMPD 7~33 전체]", r); out["pairs"][f"{cell}|{rn}|MMPD"] = r
        r2 = cmp(m, rm, lambda k, val: k[2] != 1); line(f"{cell} vs {rn} [MMPD 움직임 합산]", r2); out["pairs"][f"{cell}|{rn}|MMPD_motion"] = r2
        if rn == "공개 FactorizePhys": out["primary"].append((f"{cell}×MMPD", r))
prim = [(n, r) for n, r in out["primary"] if r]
if prim:
    hp = holm([r["p"] for _, r in prim]); print("=== 1차 가족 Holm (vs 공개 FactorizePhys)")
    out["holm"] = {}
    for (n, r), h in zip(prim, hp):
        verdict = "유의 개선" if (r["d"] < 0 and h < .05) else ("유의 악화" if (r["d"] > 0 and h < .05) else "ns")
        print(f"  {n:16s} Δ{r['d']:+6.2f} p={r['p']:.4f} holm={h:.4f} {verdict}"); out["holm"][n] = {"d": r["d"], "p": r["p"], "holm": h, "verdict": verdict}
out["mmpd_ref"] = MMPD_REF
json.dump(out, open(f"{R}/fp_official_judge.json", "w"), indent=1, ensure_ascii=False)
