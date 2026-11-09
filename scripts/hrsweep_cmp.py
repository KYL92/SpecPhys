#!/usr/bin/env python3
"""[68 문서 §4] HRSWEEP40 셀 비교(Alpha): 클립 예측을 시드 평균 → 정체성 안 중심화 기울기(예측 vs 정답), 평균 절대 오차, 5 bpm 안 비율, 심박 구간별 오차. 셀끼리 클립 단위 부호 뒤집기 순열.
  python scripts/hrsweep_cmp.py CELL:SEEDS[,...] ... (예: NF_NSP2_L:2100-2500) → results/hrsweep_cmp.txt"""
import itertools, json, os, sys
import numpy as np
R = "/SPECPHYS_ROOT/_workspace/data/results"; L = []
LO = os.environ.get("HRSWEEP_SET", "") == "LO"; PRE = "hrsweeplo" if LO else "hrsweep"
def out(s): print(s); L.append(s)
def load(cell, seeds):
    acc = {}
    for s in seeds:
        p = f"{R}/{PRE}_{cell}_s{s}.json"
        if not os.path.exists(p): continue
        for k, v in json.load(open(p))["clips"].items(): acc.setdefault(k, dict(v, preds=[]))["preds"].append(v["pred"])
    for v in acc.values(): v["pm"] = float(np.mean(v["preds"])); v["ae"] = float(np.mean([abs(x - v["gt"]) for x in v["preds"]])); v["ok"] = float(np.mean([abs(x - v["gt"]) <= 5 for x in v["preds"]]))
    return acc
def slope(d):
    by = {}
    for k, v in d.items(): by.setdefault(v["face"], []).append(v)
    dp, dg = [], []
    for vs in by.values():
        gt = np.array([v["gt"] for v in vs]); pr = np.array([v["pm"] for v in vs]); dg += list(gt - gt.mean()); dp += list(pr - pr.mean())
    dp, dg = np.array(dp), np.array(dg); return float((dp * dg).sum() / (dg * dg).sum())
def flip(a, b, n=20000, rng=np.random.default_rng(0)):
    d = np.asarray(a) - np.asarray(b); obs = abs(d.mean()); return float((np.abs((rng.choice([-1.0, 1.0], size=(n, len(d))) * d).mean(1)) >= obs - 1e-12).mean())
cells = []
for a in sys.argv[1:]:
    c, sp = a.split(":"); lo, hi = sp.split("-"); cells.append((c, load(c, range(int(lo), int(hi) + 1, 100))))
out(f"=== {'HRSWEEP40LO(맥파 0.003, 잡음 강함)' if LO else 'HRSWEEP40'}: 풀 밖 정체성 4 × 심박 10(45~180 bpm), 장면 고정 — 셀별 시드 평균")
for c, d in cells:
    bins = {}
    for v in d.values(): bins.setdefault(v["target"], []).append(v["ae"])
    out(f"  {c:14s} 기울기 {slope(d):.3f}  평균 절대 오차 {np.mean([v['ae'] for v in d.values()]):6.2f}  5 bpm 안 {np.mean([v['ok'] for v in d.values()]):.2f}  | 심박별 오차 " + " ".join(f"{t}:{np.mean(x):.1f}" for t, x in sorted(bins.items())))
if len(cells) > 1:
    base_c, base = cells[0]
    for c, d in cells[1:]:
        ks = sorted(set(d) & set(base)); out(f"  {c} − {base_c}: 오차 Δ{np.mean([d[k]['ae'] - base[k]['ae'] for k in ks]):+.2f} (클립 순열 p = {flip([d[k]['ae'] for k in ks], [base[k]['ae'] for k in ks]):.4f}), 기울기 Δ{slope(d) - slope(base):+.3f}")
open(f"{R}/{PRE}_cmp.txt", "w").write("\n".join(L) + "\n")
