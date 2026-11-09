#!/usr/bin/env python3
"""체인 58 판정(43 문서): VIPL 5-fold in-domain — 원시 입력 / 고심박 가중 셀 vs 앵커(장치 라벨 공동학습 TWg, VIPL 단독 증강+검증 BASEAV) + 공개 FactorizePhys zero-shot.
  python scripts/raw5_judge.py phase1|gate   # fold 1: VF1_<cell> vs VF1_TWg_L / VF1_BASEAV_L (fold 1 영상만), vs FPbase(같은 영상)
  python scripts/raw5_judge.py final         # 5-fold 합산 CV_<cell>_VIPL.json vs CV_TWg_L / CV_BASEAV_L / FPbase
1차(Holm, 셀별 2): VIPL 전체 문헌 GT vs 자기 앵커; 전체 vs FPbase 6.88. 2차: 90–120·≥120·60–90·v7 구간, 정본 GT. 게이트: 전체 Δ<0·p<.10 vs 자기 앵커."""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"
CELLS = {"RAWg_L": "TWg_L", "RAWgW_L": "TWg_L", "TWgW_L": "TWg_L", "RAWV_L": "BASEAV_L"}
mode = sys.argv[1] if len(sys.argv) > 1 else "phase1"
def wp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); return float("nan") if len(x) < 6 or np.allclose(x, y) else float(wilcoxon(x, y).pvalue)
def rows(name):
    f = f"{R}/agg_{name}_VIPL.json"
    if not os.path.exists(f): return None
    return {r["vid"]: dict(err_lit=r["err_lit"], err=r["err"], gt=r["gt_lit"], sc=r["scenario"]) for r in json.load(open(f)) if r["err_lit"] is not None}
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
    if t: print(f"  {t['cmp']:30s} [{t['bin']:6s}] n={t['n']:4d} {t['a']:6.2f} → {t['b']:6.2f} Δ {t['delta']:+6.2f} p={t['p']:.4f}" + (f" Holm={t['p_holm']:.4f}" if 'p_holm' in t else ""))
pre = "VF1_" if mode in ("phase1", "gate") else "CV_"
FP = rows("FPbase"); res = {"mode": mode, "primary": [], "secondary": []}
print(f"=== 체인 58 판정 mode={mode} (문헌 GT MAE; {'fold 1 영상' if pre == 'VF1_' else '5-fold 합산 2,378편'})")
go = []
for c, anc in CELLS.items():
    B = rows(pre + c); A = rows(pre + anc)
    if not B or not A: continue
    t1 = cmp(f"{anc}→{c}", A, B, "err_lit", "all"); t2 = cmp(f"FPbase→{c}", FP, B, "err_lit", "all")
    res["primary"] += [t1, t2]
    for s in ("90-120", ">=120", "60-90", "<60", "v7", "v2+v9"): res["secondary"].append(cmp(f"{anc}→{c}", A, B, "err_lit", s))
    for s in ("90-120", ">=120", "60-90", "v7"): res["secondary"].append(cmp(f"FPbase→{c}", FP, B, "err_lit", s))
    res["secondary"].append(cmp(f"{anc}→{c}", A, B, "err", "all"))
    if t1 and t1["delta"] < 0 and (np.isnan(t1["p"]) or t1["p"] < 0.10): go.append(c)
res["primary"] = holm([t for t in res["primary"] if t]); print("--- primary (Holm)"); [show(t) for t in res["primary"]]
print("--- secondary"); [show(t) for t in res["secondary"] if t]; res["secondary"] = [t for t in res["secondary"] if t]
json.dump(res, open(f"{R}/raw5_judge_{mode if mode != 'gate' else 'phase1'}.json", "w"), indent=1, ensure_ascii=False)
if mode == "gate": print(" ".join(go) if go else "NONE")
