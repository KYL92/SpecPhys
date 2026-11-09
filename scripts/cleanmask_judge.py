#!/usr/bin/env python3
"""[67 문서, 사전 등록] 체인 126 판정: 맥파 마스크를 고친 풀의 최종 모델 NF_NSP2C_L vs 현재 최종 NF_NSP2_L(가족 C1, Holm 5, 가드 7) · vs 합성 클립 뺀 NF_NSP2NS_L(가족 C2, Holm 5). 시드 2100~2500, 영상 단위 부호 뒤집기 순열(양측).
 서술: 깨끗한 분리 시험(sepxc_*) 심박 판독·지연 보정 r(sepx_lag_c.json).
  python scripts/cleanmask_judge.py → results/cleanmask_judge.{txt,json}"""
import json, os
import numpy as np
src = open("/SPECPHYS_ROOT/scripts/lsub_judge.py").read(); src = src[:src.index('OUT["M"] = family(')]
src = src.replace("S = [2600, 2700, 2800, 2900, 3000]", "S = [2100, 2200, 2300, 2400, 2500]")
g = {}; exec(compile(src, "lj", "exec"), g)
pfam, holm, R, S = g["pfam"], g["holm"], g["R"], g["S"]; PRIM, GUARD, DESC = g["PRIM"], g["GUARD"], g["DESC"]
C, F, NS = "NF_NSP2C_L", "NF_NSP2_L", "NF_NSP2NS_L"; L = []; OUT = {}
def out(s): print(s); L.append(s)
def fam(title, a, b, guards):
    out(f"=== {title}: {a} vs {b}, 시드 {S} (Δ = 앞 − 뒤)"); P = pfam(a, b); ps = [P[k]["perm"] for k in PRIM]; hs = holm(ps); better, worse = [], []
    for k, h in zip(PRIM, hs):
        r = P[k]; r["holm"] = h; better += [k] if (r["d"] < 0 and h < .05) else []; worse += [k] if (r["d"] > 0 and h < .05) else []
        out(f"  {k:20s} n={r['n']:5d} {b} {r['base']:6.2f} / {a} {r['cell']:6.2f} Δ{r['d']:+6.2f} perm={r['perm']:.4f} Holm={h:.4f} W={r['p']:.4f}")
    gv = []
    for k in guards:
        r = P.get(k)
        if r is None: continue
        bad = r["d"] > 0.3 and r["perm"] < .05; gv += [k] if bad else []
        out(f"  (가드) {k:14s} n={r['n']:5d} {r['base']:6.2f} / {r['cell']:6.2f} Δ{r['d']:+6.2f} perm={r['perm']:.4f}" + ("  위반" if bad else ""))
    for k in (GUARD if not guards else ()) + DESC:
        r = P.get(k)
        if r is not None: out(f"  (서술) {k:14s} n={r['n']:5d} {r['base']:6.2f} / {r['cell']:6.2f} Δ{r['d']:+6.2f} perm={r['perm']:.4f}")
    v = "앞이 나음" if better and not worse else "앞이 나쁨" if worse and not better else "혼재" if better and worse else "측정되지 않음"
    out(f"  판정: {v} (앞이 유의하게 낮음 {better}, 높음 {worse}, 가드 위반 {gv})"); return dict(results=P, better=better, worse=worse, guard_violations=gv, verdict=v)
OUT["C1_fix_vs_current"] = fam("가족 C1 누출을 고친 효과", C, F, GUARD)
OUT["C2_clean_synthetic_contribution"] = fam("가족 C2 깨끗한 합성 클립의 몫", C, NS, ())
out("=== 서술: 깨끗한 분리 시험 심박 판독(±6 bpm)·지연 보정 r, 셀별 5시드·클립 평균")
lag = json.load(open(f"{R}/sepx_lag_c.json")) if os.path.exists(f"{R}/sepx_lag_c.json") else {}
def grp(m): return m["set"].replace("STRESS_", "") + (":" + m["kind"] if m["set"] == "STRESS_SEPX24" else "")
SETS = ["SEPX24:G", "INBAND6", "NEAR6", "TEST_C3", "OFFBAND6", "PMOT6", "SEPX24:L"]
for cell in (F, C, NS):
    hr, rl = {}, {}
    for s in S:
        p = f"{R}/sepxc_{cell}_s{s}.json"
        if os.path.exists(p):
            for sub, v in json.load(open(p))["clips"].items(): hr.setdefault(grp(v), []).append(v["hr_ok"])
        v = lag.get(f"{cell}_s{s}")
        if v:
            for sub, c in v["clips"].items(): rl.setdefault(grp(c), []).append(c["rL"])
    out(f"  {cell:12s} 판독 " + " ".join(f"{k}={np.mean(hr[k]):.2f}" for k in SETS if k in hr) + ("  | r " + " ".join(f"{k}={np.mean(rl[k]):.2f}" for k in SETS if k in rl) if rl else ""))
open(f"{R}/cleanmask_judge.txt", "w").write("\n".join(L) + "\n"); json.dump(OUT, open(f"{R}/cleanmask_judge.json", "w"), indent=1, ensure_ascii=False, default=float)
