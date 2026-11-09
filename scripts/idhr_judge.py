#!/usr/bin/env python3
"""[68 문서 §5, 사전 등록] 체인 127 판정. 대조 = 깨끗한 마스크 NF_NSP2C_L, 시드 2100~2500.
 가족 D(분리냐 범위냐, 결합 대조 K vs 대조, Holm 2): M1 어려운 통제 시험(HRSWEEP40LO) 클립 평균 오차(시드 평균, 클립 단위 부호 뒤집기), M2 VIPL-HR 피험자 안 추종 기울기(피험자 단위 순열).
 가족 E(보강, R·W 각각 vs 대조, 셀마다 Holm 2): 같은 M1·M2.
 가족 P(실제 정확도, K·R·W 각각 vs 대조, 셀마다 Holm 5) + 가드 7. 서술: MMPD 피험자 안 기울기, 쉬운 통제 시험, 심박 구간별 오차, 합성 없음 셀.
  python scripts/idhr_judge.py → results/idhr_judge.{txt,json}"""
import json, os
import numpy as np
src = open("/SPECPHYS_ROOT/scripts/lsub_judge.py").read(); src = src[:src.index('OUT["M"] = family(')].replace("S = [2600, 2700, 2800, 2900, 3000]", "S = [2100, 2200, 2300, 2400, 2500]")
g = {}; exec(compile(src, "lj", "exec"), g); pfam, holm, R, S = g["pfam"], g["holm"], g["R"], g["S"]; PRIM, GUARD = g["PRIM"], g["GUARD"]
w = open("/SPECPHYS_ROOT/scripts/idhr_within_subject.py").read(); w = w[:w.index("for ds, loader, sel_name, selector in")].replace('S = [2100, 2200, 2300, 2400, 2500]; CELLS = {"합성 포함(최종)": "NF_NSP2_L", "깨끗한 합성": "NF_NSP2C_L", "합성 없음": "NF_NSP2NS_L"}; L = []; OUT = {}', "S = [2100, 2200, 2300, 2400, 2500]; L = []; OUT = {}")
h = {}; exec(compile(w, "iw", "exec"), h); vipl, mmpd, center, stats, perm_b = h["vipl"], h["mmpd"], h["center"], h["stats"], h["perm_b"]
CTRL, NS = "NF_NSP2C_L", "NF_NSP2NS_L"; ARMS = {"K 결합 대조": "NF_NSP2K_L", "R 클립 안 심박 변화": "NF_NSP2R_L", "W 같은 장면 짝": "NF_NSP2W_L"}; LINES = []; OUT = {}
def out(s): print(s); LINES.append(s)
def sweep(cell, pre):
    acc = {}
    for s in S:
        p = f"{R}/{pre}_{cell}_s{s}.json"
        if not os.path.exists(p): continue
        for k, v in json.load(open(p))["clips"].items(): acc.setdefault(k, dict(v, ae=[]))["ae"].append(abs(v["pred"] - v["gt"]))
    return {k: dict(v, mae=float(np.mean(v["ae"]))) for k, v in acc.items()}
def flip(d, n=20000, rng=np.random.default_rng(0)):
    d = np.asarray(d, float); obs = abs(d.mean()); return float((np.abs((rng.choice([-1.0, 1.0], size=(n, len(d))) * d).mean(1)) >= obs - 1e-12).mean())
def mech(cell):
    A, B = sweep(cell, "hrsweeplo"), sweep(CTRL, "hrsweeplo"); ks = sorted(set(A) & set(B)); dM1 = [A[k]["mae"] - B[k]["mae"] for k in ks]
    VA, VB = vipl(cell), vipl(CTRL); keys = sorted(set(VA) & set(VB), key=str); dpA, dg, sub = center(VA, keys); dpB, _, _ = center(VB, keys)
    d2, p2 = perm_b(dpA, dpB, dg, sub)
    return dict(M1=dict(ctrl=float(np.mean([B[k]["mae"] for k in ks])), cell=float(np.mean([A[k]["mae"] for k in ks])), d=float(np.mean(dM1)), p=flip(dM1), n=len(ks)),
                M2=dict(ctrl=stats(dpB, dg)[0], cell=stats(dpA, dg)[0], d=d2, p=p2, n=int(len(np.unique(sub)))))
for lab, cell in ARMS.items():
    m = mech(cell); hs = holm([m["M1"]["p"], m["M2"]["p"]]); m["M1"]["holm"], m["M2"]["holm"] = hs
    fam = "가족 D(분리냐 범위냐)" if cell.endswith("K_L") else "가족 E(보강)"
    out(f"=== {fam}: {lab} {cell} vs 대조 {CTRL}, 시드 {S}")
    out(f"  M1 어려운 통제 시험 평균 오차  대조 {m['M1']['ctrl']:6.2f} / {m['M1']['cell']:6.2f}  Δ{m['M1']['d']:+6.2f} bpm  클립 순열 p={m['M1']['p']:.4f}  Holm={m['M1']['holm']:.4f}  (클립 {m['M1']['n']})")
    out(f"  M2 VIPL-HR 피험자 안 기울기    대조 {m['M2']['ctrl']:.3f} / {m['M2']['cell']:.3f}  Δ{m['M2']['d']:+.3f}  피험자 순열 p={m['M2']['p']:.4f}  Holm={m['M2']['holm']:.4f}  (피험자 {m['M2']['n']})")
    if cell.endswith("K_L"): v = "분리가 중요(결합이 나쁨)" if ((m["M1"]["d"] > 0 and m["M1"]["holm"] < .05) or (m["M2"]["d"] < 0 and m["M2"]["holm"] < .05)) else "범위 덮기로 설명됨(결합해도 차이 없음)" if all(m[k]["holm"] >= .05 for k in ("M1", "M2")) else "결합이 오히려 나음"
    else: v = "보강 지지" if ((m["M1"]["d"] < 0 and m["M1"]["holm"] < .05) or (m["M2"]["d"] > 0 and m["M2"]["holm"] < .05)) else "보강 측정되지 않음" if all(m[k]["holm"] >= .05 for k in ("M1", "M2")) else "역효과"
    out(f"  판정: {v}"); P = pfam(cell, CTRL); ps = [P[k]["perm"] for k in PRIM]; hs = holm(ps); sup, adv, gv = [], [], []
    for k, h_ in zip(PRIM, hs):
        r = P[k]; r["holm"] = h_; sup += [k] if (r["d"] < 0 and h_ < .05) else []; adv += [k] if (r["d"] > 0 and h_ < .05) else []
        out(f"  (P) {k:20s} n={r['n']:5d} 대조 {r['base']:6.2f} / {r['cell']:6.2f} Δ{r['d']:+6.2f} perm={r['perm']:.4f} Holm={h_:.4f}")
    for k in GUARD:
        r = P.get(k)
        if r is None: continue
        bad = r["d"] > 0.3 and r["perm"] < .05; gv += [k] if bad else []
        out(f"  (가드) {k:14s} {r['base']:6.2f} / {r['cell']:6.2f} Δ{r['d']:+6.2f} perm={r['perm']:.4f}" + ("  위반" if bad else ""))
    out(f"  실제 정확도: 지지 {sup}  역방향 {adv}  가드 위반 {gv}"); OUT[cell] = dict(mech=m, verdict=v, P=P, supported=sup, adverse=adv, guard_violations=gv)
out("=== 서술: 셀별 시드 평균(대조·합성 없음 포함)")
for lab, cell in [("합성 없음", NS), ("대조(깨끗한 합성)", CTRL)] + list(ARMS.items()):
    lo, hi = sweep(cell, "hrsweeplo"), sweep(cell, "hrsweep"); Vm = mmpd(cell); km = sorted(Vm, key=str); dp, dg, _ = center(Vm, km)
    bins = {}
    for v in lo.values(): bins.setdefault(v["target"], []).append(v["mae"])
    out(f"  {lab:14s} 어려운 통제 오차 {np.mean([v['mae'] for v in lo.values()]):6.2f} | 쉬운 통제 오차 {np.mean([v['mae'] for v in hi.values()]) if hi else float('nan'):5.2f} | MMPD 피험자 안 기울기 {stats(dp, dg)[0]:.3f} | 어려운 판 심박별 " + " ".join(f"{t}:{np.mean(x):.1f}" for t, x in sorted(bins.items())))
open(f"{R}/idhr_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/idhr_judge.json", "w"), indent=1, ensure_ascii=False, default=float)
