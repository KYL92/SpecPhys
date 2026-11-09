#!/usr/bin/env python3
"""체인 61 판정(45 문서 ⑥): 잡음 라벨 절제 — 정확(이중 렌더, CoTAM_HYB4_L 3시드) vs 관측 trace(HYBX_TOT_L) vs 대응 파괴(HYBX_SHUF_L) vs 분해 없음(CoTAM_L).
  python scripts/noise_label_judge.py   → results/noise_label_judge.{txt,json}; 영상 단위 쌍대 Wilcoxon, 시드 풀링, 1차 가족(정확 vs 각 변형, MMPD 움직임) Holm."""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"; SEEDS = ("", "_s200", "_s300")
CELLS = {"CoTAM_HYB4_L": "정확(이중 렌더)", "HYBX_TOT_L": "관측 trace(맥박 포함)", "HYBX_SHUF_L": "대응 파괴(시간 이동)", "CoTAM_L": "분해 없음"}
def wp(x, y): x, y = np.asarray(x, float), np.asarray(y, float); return float("nan") if len(x) < 6 or np.allclose(x, y) else float(wilcoxon(x, y).pvalue)
def pool(ds):
    ds = [d for d in ds if d]
    if not ds: return {}, 0
    ks = set.intersection(*[set(d) for d in ds]); return {k: float(np.mean([d[k] for d in ds])) for k in ks}, len(ds)
def load(m, sfx=""): f = f"{R}/agg_{m}{sfx}.json"; return json.load(open(f)) if os.path.exists(f) else None
def metrics(m):
    per = {}
    for s in SEEDS:
        d = load(m + s)
        if d:
            for ax, rows in d.items():
                if isinstance(rows, list) and rows and "err" in rows[0]: per.setdefault(ax, []).append({str(r["vid"]): float(r["err"]) for r in rows})
            nm = {}
            for ax in ("NIRP_SMALL", "NIRP_LARGE"): nm.update({ax + ":" + str(r["vid"]): float(r["err"]) for r in d.get(ax, [])})
            if nm: per.setdefault("NIRP_motion", []).append(nm)
        mm = load(m + s, "_MMPD")
        if mm:
            rows = [r for r in mm if r["subject"] >= 7]
            per.setdefault("MMPD_all", []).append({r["vid"]: float(r["err"]) for r in rows}); per.setdefault("MMPD_motion", []).append({r["vid"]: float(r["err"]) for r in rows if r["motion"] != 1})
        vv = load(m + s, "_VIPL")
        if vv: per.setdefault("VIPL_lit", []).append({r["vid"]: float(r["err_lit"]) for r in vv if r["err_lit"] is not None})
    return {k: pool(v) for k, v in per.items()}
M = {c: metrics(c) for c in CELLS}
def cmp(a, b, met):
    if met not in M[a] or met not in M[b]: return None
    (x, na), (y, nb) = M[a][met], M[b][met]; ks = sorted(set(x) & set(y), key=str)
    if not ks: return None
    xa = np.array([x[k] for k in ks]); yb = np.array([y[k] for k in ks]); return dict(a=a, b=b, metric=met, n=len(ks), seeds=[na, nb], mae_a=float(xa.mean()), mae_b=float(yb.mean()), delta=float(yb.mean() - xa.mean()), p=wp(xa, yb))
def holm(ts):
    ps = [(i, t["p"]) for i, t in enumerate(ts) if t and not np.isnan(t["p"])]; ps.sort(key=lambda z: z[1]); m = len(ps); prev = 0.0
    for r, (i, p) in enumerate(ps): adj = min(1.0, max(prev, (m - r) * p)); ts[i]["p_holm"] = adj; prev = adj
    return ts
def show(t):
    if t: print(f"  {CELLS[t['a']]:14s}→{CELLS[t['b']]:16s} [{t['metric']:12s}] n={t['n']:4d} ({t['seeds'][0]}s→{t['seeds'][1]}s) {t['mae_a']:6.2f} → {t['mae_b']:6.2f} Δ {t['delta']:+6.2f} p={t['p']:.4f}" + (f" Holm={t['p_holm']:.4f}" if 'p_holm' in t else ""))
res = {"primary": [], "secondary": []}
print("=== 잡음 라벨 절제 (Δ = 변형 − 정확; 양수 = 정확 라벨이 우세)")
for c in ("HYBX_TOT_L", "HYBX_SHUF_L", "CoTAM_L"): res["primary"].append(cmp("CoTAM_HYB4_L", c, "MMPD_motion"))
res["primary"] = holm([t for t in res["primary"] if t]); print("--- primary: MMPD 움직임 (Holm 3)"); [show(t) for t in res["primary"]]
print("--- secondary")
for c in ("HYBX_TOT_L", "HYBX_SHUF_L", "CoTAM_L"):
    for met in ("MMPD_all", "RTNI", "RTNC", "NIRP_motion", "NIRP_STILL", "PURE", "MCD_A", "DLCN", "VIPL_lit"): t = cmp("CoTAM_HYB4_L", c, met); res["secondary"].append(t); show(t)
res["secondary"] = [t for t in res["secondary"] if t]; json.dump(res, open(f"{R}/noise_label_judge.json", "w"), indent=1, ensure_ascii=False)
