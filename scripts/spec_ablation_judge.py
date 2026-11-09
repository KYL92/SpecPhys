#!/usr/bin/env python3
"""[체인 77·78, 59 문서 §6] 스펙트럼 손실 절제·가중치 스윕 판정 — 영상 단위 쌍대 Wilcoxon, 시드 영상별 평균 풀링, 가족별 Holm.
  python scripts/spec_ablation_judge.py → 표 + results/spec_ablation_judge.{txt,json}
NH: RF_SPEC_NH_L(반감 항 0) vs RF_SPEC_L × {MR-NIRP 정지/소/대동작} (Holm 3; Δ<0 → 반감 항이 악화 원인) + 가드 VIPL·MMPD Δ≤+0.3 ; H-NH2 VIPL·MMPD 양측 |Δ|≤0.3 ns → 반감 항 불필요
W : 각 백본 {고유, 0.5, 1.0, 2.0} 표(시드 1, 서술) — VIPL·MMPD·MR-NIRP 소/대동작·RTNI"""
import glob, json, os, re
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"
def files(m, suffix):
    pat = re.compile(rf"agg_{re.escape(m)}_s\d+{re.escape(suffix)}\.json$")
    return [f for f in [f"{R}/agg_{m}{suffix}.json"] + sorted(x for x in glob.glob(f"{R}/agg_{m}_s[0-9]*{suffix}.json") if pat.search(os.path.basename(x))) if os.path.exists(f)]
def vipl(m):
    acc = {}
    for f in files(m, "_VIPL"):
        rows = json.load(open(f)); rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
        for r in rows:
            if r.get("err_lit") is not None: acc.setdefault(r["vid"], []).append(r["err_lit"])
    return {k: (float(np.mean(v)),) for k, v in acc.items()}
def axis(m, ax, sel=None):
    acc = {}
    for f in files(m, ""):
        d = json.load(open(f)); d = (d.get(ax) or []) if isinstance(d, dict) else []
        for r in d:
            if sel is None or sel(r["vid"]): acc.setdefault(r["vid"], []).append(r["err"])
    return {k: (float(np.mean(v)),) for k, v in acc.items()}
def mmpd(m, subs=set(range(7, 34))):
    acc = {}
    for f in files(m, "_MMPD"):
        for r in json.load(open(f)):
            if r["subject"] in subs: acc.setdefault((r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]), []).append(r["err"])
    return {k: (float(np.mean(v)),) for k, v in acc.items()}
def cmp(a, b):
    ks = [k for k in a if k in b]
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
    s = f"  {name:62s} (missing)" if r is None else f"  {name:62s} n={r['n']:5d} {r['b']:6.2f} -> {r['a']:6.2f}  Δ{r['d']:+6.2f} p={r['p']:.4f}{flag}"
    print(s); LINES.append(s)
def mae(d): return float(np.mean([v[0] for v in d.values()])) if d else None
out = {"NH": {}, "W": {}}
head("=== 가족 NH: 반감 항 절제 RF_SPEC_NH_L vs RF_SPEC_L (Δ = NH − SPEC)")
tests = [(f"H-NH1 {nm}", cmp(axis("RF_SPEC_NH_L", ax), axis("RF_SPEC_L", ax))) for nm, ax in (("MR-NIRP 정지", "NIRP_STILL"), ("MR-NIRP 소동작", "NIRP_SMALL"), ("MR-NIRP 대동작", "NIRP_LARGE"))]
ps = holm([t[1]["p"] if t[1] else 1.0 for t in tests])
for (nm, r), ph in zip(tests, ps):
    v = "missing" if r is None else ("지지(반감 항이 악화 원인)" if (r["d"] < 0 and ph < .05) else ("역방향" if (r["d"] > 0 and ph < .05) else "ns")); line(nm, r, f" Holm={ph:.4f} → {v}" if r else ""); out["NH"][nm] = dict(r or {}, holm=ph, verdict=v)
for nm, a, b in (("가드/H-NH2 VIPL", vipl("RF_SPEC_NH_L"), vipl("RF_SPEC_L")), ("가드/H-NH2 MMPD 7~33", mmpd("RF_SPEC_NH_L"), mmpd("RF_SPEC_L"))):
    r = cmp(a, b); v = "missing" if r is None else ("동급(반감 항 불필요)" if (abs(r["d"]) <= 0.3 and r["p"] >= .05) else ("가드 위반(악화)" if r["d"] > 0.3 else ("개선" if r["d"] < -0.3 else "ns"))); line(nm, r, f" → {v}" if r else ""); out["NH"][nm] = dict(r or {}, verdict=v)
r = cmp(axis("RF_SPEC_NH_L", "RTNI"), axis("RF_SPEC_L", "RTNI")); line("  (서술) RTNI", r); out["NH"]["RTNI"] = r
for nm, a, b in (("(서술) NH vs 고유 RF_POOL_L: VIPL", vipl("RF_SPEC_NH_L"), vipl("RF_POOL_L")), ("(서술) NH vs 고유 RF_POOL_L: MMPD", mmpd("RF_SPEC_NH_L"), mmpd("RF_POOL_L")), ("(서술) NH vs 고유: MR-NIRP 소동작", axis("RF_SPEC_NH_L", "NIRP_SMALL"), axis("RF_POOL_L", "NIRP_SMALL")), ("(서술) NH vs 고유: MR-NIRP 대동작", axis("RF_SPEC_NH_L", "NIRP_LARGE"), axis("RF_POOL_L", "NIRP_LARGE"))):
    r = cmp(a, b); line(nm, r); out["NH"][nm] = r
head("\n=== 가족 W: 가중치 스윕(SPEC_W_SPEC; 시드 1 — 1.0 은 3시드 평균) — VIPL / MMPD 7~33 / NIRP 소 / NIRP 대 / RTNI")
for bb, native, cells in (("RhythmFormer", "RF_POOL_L", [("0.5", "RF_SPEC_W05_L"), ("1.0", "RF_SPEC_L"), ("2.0", "RF_SPEC_W20_L"), ("1.0, half=0", "RF_SPEC_NH_L")]),
                          ("PhysMamba", "PM_POOL_L", [("0.5", "PM_SPEC_W05_L"), ("1.0", "PM_SPEC_L"), ("2.0", "PM_SPEC_W20_L")]),
                          ("PhysFormer", "CoTAM_L", [("0.5", "PF_SPEC_W05_L"), ("1.0", "PF_SPEC_L"), ("2.0", "PF_SPEC_W20_L")])):
    rows = [("native", native)] + cells
    for w, m in rows:
        vals = [mae(vipl(m)), mae(mmpd(m)), mae(axis(m, "NIRP_SMALL")), mae(axis(m, "NIRP_LARGE")), mae(axis(m, "RTNI"))]
        out["W"][f"{bb}|{w}"] = dict(zip(["VIPL", "MMPD", "NIRP_SMALL", "NIRP_LARGE", "RTNI"], vals), seeds=len(files(m, "_VIPL")))
        head(f"  {bb:13s} w={w:11s} ({m:15s} seeds {len(files(m, '_VIPL'))}) " + " ".join(f"{('%.2f' % v) if v is not None else '  -  ':>6s}" for v in vals))
open(f"{R}/spec_ablation_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(out, open(f"{R}/spec_ablation_judge.json", "w"), indent=1, ensure_ascii=False, default=float)
print(f"\n-> {R}/spec_ablation_judge.txt / .json")
