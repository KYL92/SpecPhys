#!/usr/bin/env python3
"""체인 51 판정(33 문서): 실-only 코퍼스 모델(기본 CoTReal_L, 실 161) vs 공식 모델(기본 CoTAF_TW_L, 실 161 + 가상얼굴 42) — 가상얼굴 42 클립의 zero-shot 순수 기여.
  python scripts/realonly_judge.py [A=CoTReal_L] [B=CoTAF_TW_L]
시드 ""/_s200/_s300 을 영상(피험자) 단위 오차 평균으로 풀링(pool5_judge 정의) 후 쌍대 Wilcoxon. 13축(agg_{M}.json) + MMPD held-out(피험자≥7, 움직임 조건별) + VIPL zero-shot(정본·문헌 GT, 시나리오·소스).
출력: 표 + results/realonly_judge.json"""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"; SEEDS = ("", "_s200", "_s300")
A = sys.argv[1] if len(sys.argv) > 1 else "CoTReal_L"; B = sys.argv[2] if len(sys.argv) > 2 else "CoTAF_TW_L"


def wp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float("nan") if len(x) < 6 or np.allclose(x, y) else float(wilcoxon(x, y).pvalue)


def load(m, suffix=""):
    f = f"{R}/agg_{m}{suffix}.json"; return json.load(open(f)) if os.path.exists(f) else None


def pool(dicts):
    """dicts: 시드별 {key: err} → 공통 key 의 시드 평균"""
    dicts = [d for d in dicts if d]
    if not dicts: return {}, 0
    keys = set.intersection(*[set(d) for d in dicts]); return {k: float(np.mean([d[k] for d in dicts])) for k in keys}, len(dicts)


def axes(m):
    out = {}
    for s in SEEDS:
        d = load(m + s)
        if d is None: continue
        for ax, rows in d.items():
            if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "err" in rows[0]:
                out.setdefault(ax, []).append({str(r["vid"]): float(r["err"]) for r in rows})
    return out


def mmpd(m):
    out = []
    for s in SEEDS:
        d = load(m + s, "_MMPD")
        if d is None: continue
        out.append({(r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]): (float(r["err"]), r["motion"]) for r in d if r["subject"] >= 7})
    return out


def vipl(m):
    out = []
    for s in SEEDS:
        d = load(m + s, "_VIPL")
        if d is None: continue
        out.append({r["vid"]: r for r in d})
    return out


def line(name, a, b, n_a, n_b, res):
    ks = sorted(set(a) & set(b))
    if not ks: return
    x = np.array([a[k] for k in ks]); y = np.array([b[k] for k in ks]); p = wp(x, y)
    res[name] = dict(n=len(ks), seeds=[n_a, n_b], mae_a=float(x.mean()), mae_b=float(y.mean()), delta=float(y.mean() - x.mean()), p=p)
    print(f"  {name:24s} n={len(ks):4d} {A}({n_a}s) {x.mean():6.2f} | {B}({n_b}s) {y.mean():6.2f} | Δ(B−A) {y.mean()-x.mean():+6.2f} p={p:.4f}")


res = {"A": A, "B": B, "axes": {}, "mmpd": {}, "vipl": {}}
print(f"=== 13축 zero-shot/in-domain (시드 풀링, 영상 단위 쌍대): A={A}(실-only) vs B={B}(+가상얼굴)")
AA, BB = axes(A), axes(B)
for ax in sorted(set(AA) & set(BB)):
    a, na = pool(AA[ax]); b, nb = pool(BB[ax]); line(ax, a, b, na, nb, res["axes"])
print("=== MMPD held-out (피험자 7~33)")
MA, MB = mmpd(A), mmpd(B)
if MA and MB:
    a, na = pool([{k: v[0] for k, v in d.items()} for d in MA]); b, nb = pool([{k: v[0] for k, v in d.items()} for d in MB])
    line("MMPD all", a, b, na, nb, res["mmpd"])
    mot = {k: v[1] for d in MA for k, v in d.items()}
    for cond in sorted(set(mot.values()), key=str):
        ks = [k for k in a if mot.get(k) == cond]; line(f"MMPD {cond}", {k: a[k] for k in ks}, {k: b[k] for k in ks if k in b}, na, nb, res["mmpd"])
    stat = [k for k in a if mot.get(k) in (1, "1", "Stationary", "stationary")]; mov = [k for k in a if k not in stat]
    line("MMPD motion(non-stationary)", {k: a[k] for k in mov}, {k: b[k] for k in mov if k in b}, na, nb, res["mmpd"])
print("=== VIPL zero-shot (2,378 영상)")
VA, VB = vipl(A), vipl(B)
if VA and VB:
    for gt, key in (("정본", "err"), ("문헌GT", "err_lit")):
        a, na = pool([{k: r[key] for k, r in d.items() if r[key] is not None} for d in VA]); b, nb = pool([{k: r[key] for k, r in d.items() if r[key] is not None} for d in VB])
        meta = {k: r for d in VA for k, r in d.items()}
        line(f"VIPL all [{gt}]", a, b, na, nb, res["vipl"])
        for nm, sel in (("source2 휴대폰", lambda r: r["source"] == 2), ("v2 motion", lambda r: r["scenario"] == "v2"), ("v7 exercise", lambda r: r["scenario"] == "v7"), ("v9 phone-motion", lambda r: r["scenario"] == "v9"), ("v1 stable", lambda r: r["scenario"] == "v1")):
            ks = [k for k in a if sel(meta[k])]; line(f"VIPL {nm} [{gt}]", {k: a[k] for k in ks}, {k: b[k] for k in ks if k in b}, na, nb, res["vipl"])
json.dump(res, open(f"{R}/realonly_judge{'' if (A, B) == ('CoTReal_L', 'CoTAF_TW_L') else '_' + A + '_vs_' + B}.json", "w"), indent=1, ensure_ascii=False)
