#!/usr/bin/env python3
"""[68 문서 §3, 파악] 같은 사람의 심박이 바뀔 때 예측이 따라가는가(실제 영상, 기존 예측, 학습 없음).
VIPL-HR(장치 심박, 시나리오 v1~v9: 운동 직후 v7 포함)·MMPD 7~33(공통 정답, 운동 전후 시도 포함)에서 셀마다 영상별 예측을 시드 평균한 뒤 피험자 안에서 중심화:
  기울기 b = Σ Δpred·Δgt / Σ Δgt² (1 이면 완전 추종, 작으면 피험자 평균 쪽으로 끌림), 피험자 안 상관 r_w, 피험자 안 오차(|Δpred−Δgt| 평균).
피험자 단위 부호 뒤집기 순열(셀끼리 짝, 2만 회)로 b 차이 검정. 셀: 합성 클립 포함 최종 NF_NSP2_L, 깨끗한 합성 클립 NF_NSP2C_L, 합성 클립 없음 NF_NSP2NS_L, 시드 2100~2500.
  python scripts/idhr_within_subject.py → results/idhr_within_subject.{txt,json}"""
import json, os, sys
import numpy as np
src = open("/SPECPHYS_ROOT/scripts/nsg_judge.py").read(); src = src[:src.index('if "--screen2" in sys.argv:')]
g = {}; exec(compile(src, "nj", "exec"), g); R, f_of, MGC = g["R"], g["f_of"], g["_MGC"]
S = [2100, 2200, 2300, 2400, 2500]; CELLS = {"합성 포함(최종)": "NF_NSP2_L", "깨끗한 합성": "NF_NSP2C_L", "합성 없음": "NF_NSP2NS_L"}; L = []; OUT = {}
def out(s): print(s); L.append(s)
def vipl(cell):
    acc = {}
    for s in S:
        f = f_of(cell, s, "_VIPL")
        for r in json.load(open(f)): acc.setdefault(r["vid"], dict(sub=r["subject"], gt=r["gt_lit"], sc=r["scenario"], p=[]))["p"].append(r["pred"])
    return {k: dict(v, pred=float(np.mean(v["p"]))) for k, v in acc.items()}
def mmpd(cell):
    acc = {}
    for s in S:
        for r in MGC.load(f_of(cell, s, "_MMPD")):
            if 7 <= r["subject"] <= 33: acc.setdefault((r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]), dict(sub=r["subject"], gt=[], ex=r["exercise"], p=[])); a = acc[(r["subject"], r["light"], r["motion"], r["exercise"], r["skin"])]; a["p"].append(r["pred"]); a["gt"].append(r["gt"])
    return {k: dict(v, pred=float(np.mean(v["p"])), gt=float(np.median(v["gt"]))) for k, v in acc.items()}
def center(d, keys):
    by = {}
    for k in keys: by.setdefault(d[k]["sub"], []).append(k)
    dp, dg, sub = [], [], []
    for s_, ks in by.items():
        if len(ks) < 2: continue
        gt = np.array([d[k]["gt"] for k in ks]); pr = np.array([d[k]["pred"] for k in ks])
        dg += list(gt - gt.mean()); dp += list(pr - pr.mean()); sub += [s_] * len(ks)
    return np.array(dp), np.array(dg), np.array(sub)
def stats(dp, dg):
    b = float((dp * dg).sum() / (dg * dg).sum()); r = float(np.corrcoef(dp, dg)[0, 1]); e = float(np.abs(dp - dg).mean()); return b, r, e
def perm_b(dpA, dpB, dg, sub, n=20000, rng=np.random.default_rng(0)):
    us = np.unique(sub); idx = [np.where(sub == u)[0] for u in us]; obs = stats(dpA, dg)[0] - stats(dpB, dg)[0]; den = (dg * dg).sum(); hit = 0
    contrib = np.array([((dpA[i] - dpB[i]) * dg[i]).sum() for i in idx])
    for _ in range(n // 1000):
        sg = rng.choice([-1.0, 1.0], size=(1000, len(us))); hit += int((np.abs(sg @ contrib / den) >= abs(obs) - 1e-12).sum())
    return obs, (hit + 1) / (n + 1)
for ds, loader, sel_name, selector in (("VIPL-HR", vipl, "모든 영상", lambda v: True), ("VIPL-HR", vipl, "피험자 안 심박 폭 ≥ 20 bpm 인 피험자", None), ("MMPD 7~33", mmpd, "모든 시도", lambda v: True)):
    D = {nm: loader(c) for nm, c in CELLS.items()}; keys = sorted(set.intersection(*[set(d) for d in D.values()]), key=str)
    if selector is None:
        by = {}
        for k in keys: by.setdefault(D["합성 포함(최종)"][k]["sub"], []).append(D["합성 포함(최종)"][k]["gt"])
        good = {s_ for s_, v in by.items() if max(v) - min(v) >= 20}; keys = [k for k in keys if D["합성 포함(최종)"][k]["sub"] in good]
    res = {}; out(f"=== {ds} — {sel_name}: 영상 {len(keys)}")
    C0 = {}
    for nm, d in D.items():
        dp, dg, sub = center(d, keys); b, r, e = stats(dp, dg); res[nm] = dict(slope=b, r_within=r, mae_within=e, n=len(dp), subjects=int(len(np.unique(sub)))); C0[nm] = (dp, dg, sub)
        out(f"  {nm:10s} 기울기 {b:.3f}  피험자 안 상관 {r:.3f}  피험자 안 오차 {e:5.2f} bpm  (영상 {len(dp)}, 피험자 {len(np.unique(sub))})")
    for a, b_ in (("합성 포함(최종)", "합성 없음"), ("깨끗한 합성", "합성 없음"), ("깨끗한 합성", "합성 포함(최종)")):
        d, p = perm_b(C0[a][0], C0[b_][0], C0[a][1], C0[a][2]); res[f"{a} vs {b_}"] = dict(d_slope=d, p=p); out(f"  기울기 차 {a} − {b_}: {d:+.3f} (피험자 단위 순열 p = {p:.4f})")
    OUT[f"{ds}|{sel_name}"] = res
open(f"{R}/idhr_within_subject.txt", "w").write("\n".join(L) + "\n"); json.dump(OUT, open(f"{R}/idhr_within_subject.json", "w"), indent=1, ensure_ascii=False)
