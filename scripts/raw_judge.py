#!/usr/bin/env python3
"""체인 57 판정(42 문서): 원시(Standardized) 입력 PhysFormer 셀의 VIPL zero-shot — 고심박 구간 진단 + 공개 FactorizePhys 대비.
  python scripts/raw_judge.py [phase1|final|gate]   (phase1/gate: 시드 100 vs 앵커 시드 100; final: 가용 시드 영상별 평균 풀링, 앵커 3시드)
1차: H-R1 90–120 bpm 구간 MAE(문헌 GT) 셀 < 앵커 CoTAF_TW_L; H-R2 전체 VIPL 문헌 GT 셀 < FPbase 6.88 (Holm 2 × 셀). 2차: 60–90 구간(악화 ≤1 가드), ≥120, v7, 정본 GT."""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"
CELLS = ["RAW_TW_L", "RAWS_TW_L"]; ERM = "CoTAF_TW_L"
mode = sys.argv[1] if len(sys.argv) > 1 else "phase1"
SEEDS = [""] if mode in ("phase1", "gate") else ["", "_s200", "_s300"]
def wp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); return float("nan") if len(x) < 6 or np.allclose(x, y) else float(wilcoxon(x, y).pvalue)
def rows(m, seeds):
    out = {}
    for s in seeds:
        f = f"{R}/agg_{m}{s}_VIPL.json"
        if not os.path.exists(f): continue
        for r in json.load(open(f)):
            if r["err_lit"] is None: continue
            out.setdefault(r["vid"], []).append(r)
    return {k: dict(err_lit=np.mean([r["err_lit"] for r in v]), err=np.mean([r["err"] for r in v]), gt=v[0]["gt_lit"], sc=v[0]["scenario"], n=len(v)) for k, v in out.items()}
A = rows(ERM, SEEDS); FP = rows("FPbase", [""])
sel = {"all": lambda r: True, "60-90": lambda r: 60 <= r["gt"] < 90, "90-120": lambda r: 90 <= r["gt"] < 120, ">=120": lambda r: r["gt"] >= 120, "<60": lambda r: r["gt"] < 60, "v7": lambda r: r["sc"] == "v7", "v2+v9": lambda r: r["sc"] in ("v2", "v9")}
def cmp(name, a, b, key, s):
    ks = [k for k in a if k in b and sel[s](a[k])]
    if not ks: return None
    x = np.array([a[k][key] for k in ks]); y = np.array([b[k][key] for k in ks])
    return dict(cmp=name, bin=s, key=key, n=len(ks), a=float(x.mean()), b=float(y.mean()), delta=float(y.mean() - x.mean()), p=wp(x, y))
def holm(ts):
    ps = [(i, t["p"]) for i, t in enumerate(ts) if t and not np.isnan(t["p"])]; ps.sort(key=lambda z: z[1]); m = len(ps); prev = 0.0
    for r, (i, p) in enumerate(ps): adj = min(1.0, max(prev, (m - r) * p)); ts[i]["p_holm"] = adj; prev = adj
    return ts
def show(t):
    if t: print(f"  {t['cmp']:26s} [{t['bin']:6s}] n={t['n']:4d} {t['a']:6.2f} → {t['b']:6.2f} Δ {t['delta']:+6.2f} p={t['p']:.4f}" + (f" Holm={t['p_holm']:.4f}" if 'p_holm' in t else ""))
res = {"mode": mode, "primary": [], "secondary": [], "guard": {}}
cells = {c: rows(c, SEEDS) for c in CELLS}; cells = {c: v for c, v in cells.items() if v}
print(f"=== 체인 57 판정 mode={mode} cells={list(cells)} (문헌 GT MAE)")
for c, B in cells.items():
    res["primary"].append(cmp(f"{ERM}→{c}", A, B, "err_lit", "90-120")); res["primary"].append(cmp(f"FPbase→{c}", FP, B, "err_lit", "all"))
    for s in ("all", "<60", "60-90", ">=120", "v7", "v2+v9"): res["secondary"].append(cmp(f"{ERM}→{c}", A, B, "err_lit", s))
    for s in ("90-120", ">=120", "60-90", "v7"): res["secondary"].append(cmp(f"FPbase→{c}", FP, B, "err_lit", s))
    res["secondary"].append(cmp(f"{ERM}→{c}", A, B, "err", "all"))
    g = cmp("guard60-90", A, B, "err_lit", "60-90"); res["guard"][c] = dict(delta=g["delta"], pass_=bool(g["delta"] <= 1.0)) if g else {"pass_": False}
res["primary"] = holm([t for t in res["primary"] if t]); print("--- primary (Holm)"); [show(t) for t in res["primary"]]
print("--- guard (60-90 악화 ≤1.0):", json.dumps(res["guard"])); print("--- secondary"); [show(t) for t in res["secondary"] if t]
res["secondary"] = [t for t in res["secondary"] if t]
json.dump(res, open(f"{R}/raw_judge_{mode if mode != 'gate' else 'phase1'}.json", "w"), indent=1, ensure_ascii=False)
if mode == "gate":
    go = [t["cmp"].split("→")[1] for t in res["primary"] if t["bin"] == "90-120" and t["delta"] < 0 and (np.isnan(t["p"]) or t["p"] < 0.10) and res["guard"].get(t["cmp"].split("→")[1], {}).get("pass_")]
    print(" ".join(go) if go else "NONE")
