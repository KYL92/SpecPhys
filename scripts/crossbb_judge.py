#!/usr/bin/env python3
"""[체인 75, 59 문서] 스펙트럼 손실 백본 교차 검증 + PhysMamba VIPL 악화 진단 판정 — 영상 단위 쌍대 Wilcoxon, 시드 영상별 평균 풀링, 가족별 Holm.
  python scripts/crossbb_judge.py  → 표 + results/crossbb_judge.{txt,json}
X1 RF_SPEC_L vs RF_POOL_L × {VIPL, MMPD 7~33} | X2 PM_SPEC_L vs PM_POOL_L × {VIPL, MMPD} | X3 RF_SPEC_L vs NF_HYB_L(SpecFP) × {VIPL, MMPD} 양측(동급 |Δ|≤0.3)
D1 PM_POOL_EP_L epoch 0/2/4/6/9 VIPL 곡선 vs PMbase(서술) | D2 PM_POOL_LR5_L vs PMbase VIPL(비열등 Δ≤+0.3 / 우세) · vs PM_POOL_L VIPL (Holm 2)
2차(서술): 실 축 9개, 반감(hlv)·배가(dbl) 건수."""
import glob, json, os, re
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"
def files(m, suffix):
    pat = re.compile(rf"agg_{re.escape(m)}_s\d+{re.escape(suffix)}\.json$")
    return [f for f in [f"{R}/agg_{m}{suffix}.json"] + sorted(x for x in glob.glob(f"{R}/agg_{m}_s[0-9]*{suffix}.json") if pat.search(os.path.basename(x))) if os.path.exists(f)]
def vipl(m):
    acc, meta = {}, {}
    for f in files(m, "_VIPL"):
        rows = json.load(open(f)); rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
        for r in rows:
            if r.get("err_lit") is None: continue
            acc.setdefault(r["vid"], []).append(r["err_lit"]); meta[r["vid"]] = (r.get("scenario"), r.get("gt_lit"), r.get("ratio"))
    return {k: (float(np.mean(v)),) + meta[k] for k, v in acc.items()}
def axis(m, ax):
    acc = {}
    for f in files(m, ""):
        d = json.load(open(f)); d = (d.get(ax) or []) if isinstance(d, dict) else []
        for r in d: acc.setdefault(r["vid"], []).append(r["err"])
    return {k: (float(np.mean(v)), None, None) for k, v in acc.items()}
def mmpd(m, subs=set(range(7, 34))):
    acc = {}
    for f in files(m, "_MMPD"):
        for r in json.load(open(f)):
            if r["subject"] in subs: acc.setdefault((r["subject"], r["light"], r["motion"], r["exercise"], r["skip"] if "skip" in r else r["skin"]), []).append(r["err"])
    return {k: (float(np.mean(v)), None, None) for k, v in acc.items()}
def counts(m, suffix):   # 반감/배가 건수(시드 평균): ratio 기준 |r-0.5|<0.1 / |r-2|<0.2
    hl = db = n = 0
    for f in files(m, suffix):
        rows = json.load(open(f)); rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
        for r in rows:
            rt = r.get("ratio")
            if rt is None: continue
            n += 1; hl += abs(rt - 0.5) < 0.1; db += abs(rt - 2.0) < 0.2
    k = max(1, len(files(m, suffix))); return (hl / k, db / k, n // k)
def cmp(a, b, sel=lambda k, v: True):
    ks = [k for k in a if k in b and sel(k, a[k])]
    if len(ks) < 5: return None
    x = np.array([a[k][0] for k in ks]); y = np.array([b[k][0] for k in ks])
    try: p = float(wilcoxon(x, y).pvalue)
    except ValueError: p = 1.0
    return {"n": len(ks), "a": float(x.mean()), "b": float(y.mean()), "d": float(x.mean() - y.mean()), "p": p}
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [None] * len(ps); m = len(ps); run = 0.0
    for rank, i in enumerate(order): run = max(run, min(1.0, (m - rank) * ps[i])); out[i] = run
    return out
LINES = []
def head(s): print(s); LINES.append(s)
def line(name, r, flag=""):
    s = f"  {name:64s} (missing)" if r is None else f"  {name:64s} n={r['n']:5d} {r['b']:6.2f} -> {r['a']:6.2f}  Δ{r['d']:+6.2f} p={r['p']:.4f}{flag}"
    print(s); LINES.append(s)
def mae(d): return (float(np.mean([v[0] for v in d.values()])), len(d)) if d else (None, 0)
SEC = [("MR-NIRP small", "NIRP_SMALL", None), ("MR-NIRP large", "NIRP_LARGE", None), ("MR-NIRP still", "NIRP_STILL", None), ("RTNI flicker", "RTNI", None), ("RTNC", "RTNC", None),
       ("PURE held-out 07-10", "PURE", lambda k, v: k >= 701), ("DLCN", "DLCN", None), ("MCD A", "MCD_A", None), ("MCD B", "MCD_B", None)]
MODELS = ["RF_POOL_L", "RF_SPEC_L", "PM_POOL_L", "PM_SPEC_L", "PMbase", "PM_POOL_LR5_L", "CoTAM_L", "PF_SPEC_L", "NF_HYB_L"]
V = {m: vipl(m) for m in MODELS}; M = {m: mmpd(m) for m in MODELS}
out = {"table": {}, "X1": {}, "X2": {}, "X3": {}, "X4": {}, "D1": {}, "D2": {}, "secondary": {}}
head("=== 표: 영상 단위 MAE(시드 영상별 평균 풀링) + 반감/배가 건수(시드 평균)")
for m in MODELS:
    mv, nv = mae(V[m]); mm, nm = mae(M[m]); hv = counts(m, "_VIPL"); hm = counts(m, "_MMPD")
    sec = {ax: mae({k: v for k, v in axis(m, ax).items() if sel is None or sel(k, v)})[0] for _, ax, sel in SEC}
    out["table"][m] = {"VIPL": mv, "MMPD": mm, "seeds_VIPL": len(files(m, "_VIPL")), "seeds_MMPD": len(files(m, "_MMPD")), "hlv_dbl_VIPL": hv[:2], "hlv_dbl_MMPD": hm[:2], **sec}
    head(f"  {m:14s} VIPL {('%.2f' % mv) if mv is not None else '  -  '} (n={nv}, seeds {len(files(m, '_VIPL'))}, hlv={hv[0]:.1f} dbl={hv[1]:.1f}) | MMPD {('%.2f' % mm) if mm is not None else '  -  '} (n={nm}, hlv={hm[0]:.1f} dbl={hm[1]:.1f}) | " + " ".join(f"{ax}={('%.2f' % v) if v is not None else '-'}" for ax, v in sec.items()))
def family(name, tests, rule):
    head(f"\n=== 가족 {name}")
    ps = holm([t[1]["p"] if t[1] else 1.0 for t in tests])
    for (nm, r), ph in zip(tests, ps):
        verdict = rule(r, ph) if r else "missing"; line(nm, r, f" Holm={ph:.4f} → {verdict}" if r else ""); out[name.split()[0]][nm] = dict(r or {}, holm=ph, verdict=verdict)
rule_lt = lambda r, ph: "지지" if (r["d"] < 0 and ph < .05) else ("기각(역방향)" if (r["d"] > 0 and ph < .05) else "ns")
family("X1 스펙트럼 손실 이식(RhythmFormer): RF_SPEC_L vs RF_POOL_L (Δ<0 → 지지)", [("H-X1 VIPL", cmp(V["RF_SPEC_L"], V["RF_POOL_L"])), ("H-X1 MMPD 7~33", cmp(M["RF_SPEC_L"], M["RF_POOL_L"]))], rule_lt)
family("X2 스펙트럼 손실 이식(PhysMamba): PM_SPEC_L vs PM_POOL_L (Δ<0 → 지지)", [("H-X2 VIPL", cmp(V["PM_SPEC_L"], V["PM_POOL_L"])), ("H-X2 MMPD 7~33", cmp(M["PM_SPEC_L"], M["PM_POOL_L"]))], rule_lt)
rule_par = lambda r, ph: ("동급(|Δ|≤0.3, ns)" if (ph >= .05 and abs(r["d"]) <= 0.3) else ("ns" if ph >= .05 else ("RhythmFormer 우세" if r["d"] < 0 else "SpecFP 우세 유지")))
family("X4 스펙트럼 손실 이식(PhysFormer, Alpha 체인 76): PF_SPEC_L vs CoTAM_L (Δ<0 → 지지)", [("H-X4 VIPL", cmp(V["PF_SPEC_L"], V["CoTAM_L"])), ("H-X4 MMPD 7~33", cmp(M["PF_SPEC_L"], M["CoTAM_L"]))], rule_lt)
family("X3 백본 비의존성: RF_SPEC_L vs SpecFP(NF_HYB_L), 양측", [("H-X3 VIPL", cmp(V["RF_SPEC_L"], V["NF_HYB_L"])), ("H-X3 MMPD 7~33", cmp(M["RF_SPEC_L"], M["NF_HYB_L"]))], rule_par)
head("\n=== D1 PhysMamba 풀 학습 epoch 곡선(VIPL 문헌 GT; 공개 PMbase 기준)")
pb = mae(V["PMbase"])[0]
for k in (0, 2, 4, 6, 9):
    d = vipl(f"PM_POOL_EP_L_E{k}"); v, n = mae(d); out["D1"][f"E{k}"] = v
    head(f"  epoch {k}: VIPL MAE {('%.2f' % v) if v is not None else '-'} (n={n})" + (f"  vs PMbase {pb:.2f} Δ{v - pb:+.2f}" if v is not None and pb is not None else ""))
rule_d2 = lambda r, ph: ("우세" if (r["d"] < 0 and ph < .05) else ("비열등(Δ≤+0.3)" if r["d"] <= 0.3 else "기각(열세)"))
family("D2 낮은 학습률: PM_POOL_LR5_L vs PMbase(VIPL) · vs PM_POOL_L(VIPL)", [("H-D2 VIPL: LR 1e-5 vs 공개 PhysMamba", cmp(V["PM_POOL_LR5_L"], V["PMbase"])), ("H-D2 VIPL: LR 1e-5 vs LR 1e-4 풀 셀", cmp(V["PM_POOL_LR5_L"], V["PM_POOL_L"]))], rule_d2)
r = cmp(M["PM_POOL_LR5_L"], M["PMbase"]); line("  (서술) MMPD 7~33: LR 1e-5 vs PMbase", r); out["D2"]["MMPD_vs_PMbase"] = r
r = cmp(M["PM_POOL_LR5_L"], M["PM_POOL_L"]); line("  (서술) MMPD 7~33: LR 1e-5 vs LR 1e-4", r); out["D2"]["MMPD_vs_POOL"] = r
head("\n=== 2차(서술, 비조정): 실 축")
for a, b in (("RF_SPEC_L", "RF_POOL_L"), ("PM_SPEC_L", "PM_POOL_L"), ("PF_SPEC_L", "CoTAM_L"), ("RF_SPEC_L", "NF_HYB_L"), ("PM_SPEC_L", "NF_HYB_L"), ("PF_SPEC_L", "NF_HYB_L")):
    for nm, ax, sel in SEC:
        r = cmp(axis(a, ax), axis(b, ax), sel or (lambda k, v: True)); line(f"  {nm}: {a} vs {b}", r); out["secondary"][f"{nm}|{a}|{b}"] = r
os.makedirs(R, exist_ok=True); open(f"{R}/crossbb_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(out, open(f"{R}/crossbb_judge.json", "w"), indent=1, ensure_ascii=False, default=float)
print(f"\n-> {R}/crossbb_judge.txt / .json")
