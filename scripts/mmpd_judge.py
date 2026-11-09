#!/usr/bin/env python3
"""[체인 52, 34 문서 §5] MMPD 인트라 5-fold 판정: 영상 단위 쌍대 Wilcoxon(키 = subject,light,motion,exercise,skin).
  python scripts/mmpd_judge.py [phase1|lit]
phase1: agg_MCV5_{BASE_L,TW_L,HYB_L}_MMPD.json → BASE→TW(데이터 기여), TW→HYB(목적함수 기여), BASE→HYB; 각 셀 vs POS/CHROM/공식 가중치(base=PhysFormer-UBFC, RhythmFormer, FactorizePhys);
        held-out 7~33 부분집합에서 zero-shot CoTAF_TW_L·CoTAM_HYB4_L(MMPD 1~6 학습)·체인 28 3-fold CoTAF_MCV_L 대비; 층(정지/회전/대화/보행/움직임 합산), 피부톤, 조건부 배치(정지=TW, 움직임=HYB).
        1차 가족 3건(BASE→TW 전체, TW→HYB 움직임, HYB vs POS 움직임) Holm. 출력 results/mmpd_judge.json
lit:    agg_ML_{BASE_L,HYB_L}_MMPD.json (피험자 27~33) 상호 비교 + 문헌 인트라 참고선 표기."""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"
MOTION = {1: "Stationary", 2: "Rotation", 3: "Talking", 4: "Walking"}


def wp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float("nan") if len(x) < 6 or np.allclose(x, y) else float(wilcoxon(x, y).pvalue)


def load(m):
    f = f"{R}/agg_{m}_MMPD.json"
    if not os.path.exists(f): return None
    d = {}
    for r in json.load(open(f)): d[(r["subject"], r["light"], r["motion"], r["exercise"], r["skin"])] = r["err"]
    return d


def holm(ps):
    idx = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(idx): run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


STRATA = {"전체": lambda k: True, "정지": lambda k: k[2] == 1, "회전": lambda k: k[2] == 2, "대화": lambda k: k[2] == 3, "보행": lambda k: k[2] == 4, "움직임 합산": lambda k: k[2] != 1,
          "피부톤 5·6": lambda k: k[4] >= 5, "held-out 7~33": lambda k: k[0] >= 7}


def cmp(name, a, b, sel, out):
    ks = [k for k in a if k in b and sel(k)]
    if not ks: return None
    x = np.array([a[k] for k in ks]); y = np.array([b[k] for k in ks]); p = wp(x, y)
    out[name] = dict(n=len(ks), a=float(x.mean()), b=float(y.mean()), delta=float(y.mean() - x.mean()), p=p)
    print(f"  {name:44s} n={len(ks):3d} {x.mean():6.2f} → {y.mean():6.2f} Δ{y.mean()-x.mean():+6.2f} p={p:.4f}"); return out[name]


def phase1():
    C = {t: load(f"MCV5_{t}") for t in ("BASE_L", "TW_L", "HYB_L")}
    refs = {"POS": load("POS"), "CHROM": load("CHROM"), "PhysFormer-UBFC": load("base"), "RhythmFormer-UBFC": load("RhythmFormer"), "FactorizePhys-UBFC": load("FactorizePhys"),
            "zero-shot CoTAF_TW_L": load("CoTAF_TW_L"), "CoTAM_HYB4_L(1~6 학습)": load("CoTAM_HYB4_L"), "3-fold CoTAF_MCV_L": load("CoTAF_MCV_L")}
    out = {"cells": {}, "pairs": {}, "refs": {}}
    print("=== 셀 요약(5-fold 합산, 33명 660편)")
    for t, d in C.items():
        if d is None: print(f"  {t} missing"); continue
        e = np.array(list(d.values())); out["cells"][t] = {s: float(np.mean([d[k] for k in d if f(k)])) for s, f in STRATA.items()}
        print(f"  {t:7s} " + " | ".join(f"{s} {v:5.2f}" for s, v in out["cells"][t].items()))
    print("=== 셀 간 쌍대(Δ = 뒤 − 앞)")
    prim = []
    for a, b in (("BASE_L", "TW_L"), ("TW_L", "HYB_L"), ("BASE_L", "HYB_L")):
        if C[a] is None or C[b] is None: continue
        for s, f in STRATA.items():
            r = cmp(f"{a}→{b} [{s}]", C[a], C[b], f, out["pairs"])
            if r and ((a, b, s) == ("BASE_L", "TW_L", "전체") or (a, b, s) == ("TW_L", "HYB_L", "움직임 합산")): prim.append((f"{a}→{b} [{s}]", r))
    if C.get("TW_L") and C.get("HYB_L"):  # 조건부 배치: 정지=TW, 움직임=HYB
        comb = {k: (C["TW_L"][k] if k[2] == 1 else C["HYB_L"][k]) for k in C["HYB_L"] if k in C["TW_L"]}
        cmp("TW_L→조건부 배치(정지 TW, 움직임 HYB) [전체]", C["TW_L"], comb, lambda k: True, out["pairs"])
    print("=== 기존 방법 대비(Δ = 셀 − 기준, 음수 = 셀 우세)")
    for t, d in C.items():
        if d is None: continue
        for rn, rd in refs.items():
            if rd is None: continue
            sub = (lambda k: k[0] >= 7) if ("zero-shot" in rn or "1~6" in rn) else (lambda k: True)
            for s in ("전체", "정지", "움직임 합산", "보행"):
                f = STRATA[s]; r = cmp(f"{t} vs {rn} [{s}{'·7~33' if sub(( 1, 1, 1, 1, 3)) is False else ''}]", rd, d, lambda k, f=f, sub=sub: f(k) and sub(k), out["refs"])
                if r and t == "HYB_L" and rn == "POS" and s == "움직임 합산": prim.append((f"HYB_L vs POS [움직임 합산]", r))
    if prim:
        adj = holm([r["p"] if np.isfinite(r["p"]) else 1.0 for _, r in prim])
        print("=== 1차 가족 Holm"); out["primary"] = {}
        for (nm, r), pa in zip(prim, adj): out["primary"][nm] = dict(r, p_holm=float(pa)); print(f"  {nm:44s} Δ{r['delta']:+.2f} p={r['p']:.4f} holm={pa:.4f} {'유의' if pa < 0.05 and r['delta'] < 0 else 'ns'}")
    json.dump(out, open(f"{R}/mmpd_judge.json", "w"), indent=1, ensure_ascii=False)


def lit():
    A, B = load("ML_BASE_L"), load("ML_HYB_L"); out = {}
    print("=== 문헌 분할 참고(학습 1~23, 테스트 27~33; 문헌 인트라 참고선: RhythmFormer 3.07, PhysFormer++ 4.88, PhysNet 4.80, PhysFormer 11.99 — mini-MMPD·순차 7:1:2·다른 읽기)")
    for nm, d in (("ML_BASE_L(MMPD 만)", A), ("ML_HYB_L(+코퍼스+분해)", B)):
        if d: print(f"  {nm:24s} n={len(d)} MAE {np.mean(list(d.values())):.2f} | 움직임 {np.mean([v for k, v in d.items() if k[2] != 1]):.2f} | 정지 {np.mean([v for k, v in d.items() if k[2] == 1]):.2f}")
    if A and B:
        for s, f in STRATA.items(): cmp(f"ML_BASE_L→ML_HYB_L [{s}]", A, B, f, out)
    for rn in ("POS", "base", "RhythmFormer"):
        rd = load(rn)
        if rd and B: cmp(f"ML_HYB_L vs {rn} [27~33 전체]", rd, B, lambda k: k[0] >= 27, out)
    json.dump(out, open(f"{R}/mmpd_judge_lit.json", "w"), indent=1, ensure_ascii=False)


def pooled(tag, seeds=("", "_s200", "_s300")):
    """MCV5_{tag}{seed} 들의 영상 단위 평균 오차(있는 시드만)."""
    ds = [d for d in (load(f"MCV5_{tag}{sd}") for sd in seeds) if d]
    if not ds: return None, 0
    keys = set(ds[0]);  [keys.intersection_update(d) for d in ds[1:]]
    return {k: float(np.mean([d[k] for d in ds])) for k in keys}, len(ds)


def phase3():
    """Phase 3(34 §4·§8): TW·HYB 시드 100·200·300 영상 단위 평균 vs BASE(시드 100) — phase1 과 같은 1차 가족 3건 Holm + 2차."""
    C = {}; ns = {}
    C["BASE_L"], ns["BASE_L"] = pooled("BASE_L"); C["TW_L"], ns["TW_L"] = pooled("TW_L"); C["HYB_L"], ns["HYB_L"] = pooled("HYB_L")
    refs = {"POS": load("POS"), "CHROM": load("CHROM"), "PhysFormer-UBFC": load("base"), "RhythmFormer-UBFC": load("RhythmFormer"), "FactorizePhys-UBFC": load("FactorizePhys"),
            "zero-shot CoTAF_TW_L": load("CoTAF_TW_L"), "CoTAM_HYB4_L(1~6 학습)": load("CoTAM_HYB4_L")}
    out = {"seeds": ns, "cells": {}, "pairs": {}, "refs": {}}
    print(f"=== Phase 3 시드 풀링(시드 수 BASE={ns['BASE_L']} TW={ns['TW_L']} HYB={ns['HYB_L']}; 영상 단위 평균, 33명 660편)")
    for t, d in C.items():
        if d is None: print(f"  {t} missing"); continue
        out["cells"][t] = {s: float(np.mean([d[k] for k in d if f(k)])) for s, f in STRATA.items()}
        print(f"  {t:7s} " + " | ".join(f"{s} {v:5.2f}" for s, v in out["cells"][t].items()))
    for sd in ("", "_s200", "_s300"):  # 시드별 전체 MAE
        line = []
        for t in ("TW_L", "HYB_L"):
            d = load(f"MCV5_{t}{sd}"); line.append(f"{t}{sd or '_s100'} {np.mean(list(d.values())):.2f}" if d else f"{t}{sd} -")
        print("  시드별: " + " | ".join(line))
    print("=== 셀 간 쌍대(Δ = 뒤 − 앞)"); prim = []
    for a, b in (("BASE_L", "TW_L"), ("TW_L", "HYB_L"), ("BASE_L", "HYB_L")):
        if C[a] is None or C[b] is None: continue
        for s, f in STRATA.items():
            r = cmp(f"{a}→{b} [{s}]", C[a], C[b], f, out["pairs"])
            if r and ((a, b, s) == ("BASE_L", "TW_L", "전체") or (a, b, s) == ("TW_L", "HYB_L", "움직임 합산")): prim.append((f"{a}→{b} [{s}]", r))
    print("=== 기존 방법 대비(Δ = 셀 − 기준)")
    for t in ("TW_L", "HYB_L"):
        d = C[t]
        if d is None: continue
        for rn, rd in refs.items():
            if rd is None: continue
            sub = (lambda k: k[0] >= 7) if ("zero-shot" in rn or "1~6" in rn) else (lambda k: True)
            for s in ("전체", "정지", "움직임 합산", "보행"):
                f = STRATA[s]; r = cmp(f"{t} vs {rn} [{s}]", rd, d, lambda k, f=f, sub=sub: f(k) and sub(k), out["refs"])
                if r and t == "HYB_L" and rn == "POS" and s == "움직임 합산": prim.append(("HYB_L vs POS [움직임 합산]", r))
    if prim:
        adj = holm([r["p"] if np.isfinite(r["p"]) else 1.0 for _, r in prim]); print("=== 1차 가족 Holm"); out["primary"] = {}
        for (nm, r), pa in zip(prim, adj): out["primary"][nm] = dict(r, p_holm=float(pa)); print(f"  {nm:44s} Δ{r['delta']:+.2f} p={r['p']:.4f} holm={pa:.4f} {'유의' if pa < 0.05 and r['delta'] < 0 else 'ns'}")
    json.dump(out, open(f"{R}/mmpd_judge_phase3.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "phase1"
    {"lit": lit, "phase3": phase3}.get(m, phase1)()
