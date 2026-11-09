#!/usr/bin/env python
"""MMPD 인트라 5-fold(체인 52) 셀 하나의 held-out fold 쌍대 비교 — 실행 로그 전사용(정본 판정은 mmpd_judge.py).

usage: mmpd_fold_cmp.py MF4_TW_L [--refs POS FactorizePhys CoTAF_TW_L CoTAM_HYB4_L] [--pool]
  * 이름 MF{k}_{TAG}_L 에서 fold k 를 읽고, 같은 fold 의 다른 셀(BASE/TW/HYB 중 집계된 것)과 참조 모델에 대해
    held-out 피험자(mmpd_folds.json) 영상 단위 Wilcoxon 을 전체/정지/움직임/보행 별로 출력한다.
  * --pool: 같은 TAG 의 완료된 모든 fold 를 합산(선행 fold 들)하여 같은 비교를 한다.
motion 코드는 agg_mmpd.py 규약(1 정지 2 회전 3 발화 4 보행)."""
import json, re, sys, numpy as np
from scipy.stats import wilcoxon
R = "_workspace/data/results"; FD = json.load(open("_workspace/data/physformer/mmpd_folds.json"))
KEY = ("subject", "light", "motion", "exercise", "skin")
SUB = (("전체", lambda k: True), ("정지", lambda k: k[2] == 1), ("움직임", lambda k: k[2] != 1), ("보행", lambda k: k[2] == 4))

def load(model, subs):
    try: rows = json.load(open(f"{R}/agg_{model}_MMPD.json"))
    except FileNotFoundError: return None
    return {tuple(r[c] for c in KEY): r["err"] for r in rows if r["subject"] in subs}

def pval(v): return "p<.001" if v < 0.001 else f"p={v:.3f}"[:-0 or None].replace("p=0.", "p=.")

def compare(name, a, ref_name, b):
    ks = [k for k in a if k in b]; parts = []
    for sn, sel in SUB:
        kk = [k for k in ks if sel(k)]
        if len(kk) < 5: continue
        x = np.array([a[k] for k in kk]); y = np.array([b[k] for k in kk]); pv = wilcoxon(x, y).pvalue
        parts.append(f"{sn} {x.mean():.2f} vs {y.mean():.2f} ({x.mean()-y.mean():+.2f}, {pval(pv) if pv < .05 else 'ns'})")
    print(f"  {name} vs {ref_name}: " + ", ".join(parts))

def main():
    args = sys.argv[1:]; pool = "--pool" in args; args = [a for a in args if a != "--pool"]
    refs = ["POS", "FactorizePhys", "CoTAF_TW_L", "CoTAM_HYB4_L"]
    if "--refs" in args: i = args.index("--refs"); refs = args[i+1:]; args = args[:i]
    model = args[0]; m = re.match(r"MF(\d)_(\w+)_L$", model); k, tag = m.group(1), m.group(2)
    subs = FD[k]; a = load(model, subs); assert a, model
    v = np.array(list(a.values())); print(f"{model} fold {k}: n={len(a)} MAE {v.mean():.2f}  " +
        "  ".join(f"{sn} {np.mean([a[q] for q in a if sel(q)]):.2f}" for sn, sel in SUB[1:]))
    for other in ("BASE", "TW", "HYB"):
        if other != tag and (b := load(f"MF{k}_{other}_L", subs)): compare(model, a, f"MF{k}_{other}_L", b)
    for r in refs:
        if (b := load(r, subs)): compare(model, a, f"{r} zero-shot" if r != "POS" else r, b)
    if pool:
        folds = [f for f in "12345" if load(f"MF{f}_{tag}_L", FD[f])]
        subs = sum((FD[f] for f in folds), []); a = {}
        for f in folds: a.update(load(f"MF{f}_{tag}_L", FD[f]))
        print(f"\n{tag} pooled folds {''.join(folds)}: n={len(a)} ({len(subs)}명) MAE {np.mean(list(a.values())):.2f}")
        for other in ("BASE", "TW", "HYB"):
            if other == tag: continue
            b = {}; ok = True
            for f in folds:
                bb = load(f"MF{f}_{other}_L", FD[f]); ok = ok and bool(bb); b.update(bb or {})
            if ok: compare(f"{tag} pooled", a, f"{other} pooled", b)
        for r in refs:
            if (b := load(r, subs)): compare(f"{tag} pooled", a, f"{r} zero-shot" if r != "POS" else r, b)

if __name__ == "__main__": main()
