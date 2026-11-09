#!/usr/bin/env python3
"""[66 문서, 사전 등록] 체인 125 판정: 입력 단계 배경 기준 보정(물리 규칙 게이트). 1차 = 학습·평가 모두 적용(NF_BGREF_L), 2차 = 기준 체크포인트에 평가 때만 적용(NF_NSP2DR_BGT_L). 기준 NF_NSP2DR_L(체인 124, 같은 풀·같은 시드 2600~3000).
 가족 M(기전, Holm 2): 지연 보정 r(모델·시드마다 STRESS_OFFBAND6 에서 고른 고정 지연, scripts/sepx_lag.py) — M1 새 분리 시험 전역 플리커 근접 12클립, M2 기존 전역 플리커 12클립(STRESS_INBAND6·NEAR6·TEST_C3).
   클립별 5시드 평균의 쌍대 차이, 정확 부호 뒤집기 순열(양측). 지지: Δ>0, Holm<.05. 음성 대조(서술): 피부 국소 플리커 근접 12클립(배경에 보이지 않아 변화가 없어야 함).
 가족 P(실제 정확도, Holm 5)·가드 7: 체인 124 와 같다.
  python scripts/bgref_judge.py → results/bgref_judge.{txt,json}"""
import itertools, json
import numpy as np
from scipy.stats import wilcoxon
src = open("/SPECPHYS_ROOT/scripts/lsub_judge.py").read(); src = src[:src.index('OUT["M"] = family(')]
g = {}; exec(compile(src, "lj", "exec"), g)
pfam, holm, R, S = g["pfam"], g["holm"], g["R"], g["S"]; PRIM, GUARD, DESC = g["PRIM"], g["GUARD"], g["DESC"]
F = "NF_NSP2DR_L"; L = []; OUT = {}
def out(s): print(s); L.append(s)
lag = json.load(open(f"{R}/sepx_lag.json")); meta = {s: c for v in lag.values() for s, c in v["clips"].items()}
def clip_mean(cell, key):
    acc = {}
    for k, v in lag.items():
        if k.rsplit("_s", 1)[0] == cell and int(k.rsplit("_s", 1)[1]) in S:
            for sub, c in v["clips"].items(): acc.setdefault(sub, []).append(c[key])
    return {s: float(np.mean(x)) for s, x in acc.items()}, len({k for k in lag if k.rsplit("_s", 1)[0] == cell})
def exact(dd):
    dd = np.asarray(dd, float); obs = abs(dd.mean()); hit = tot = 0
    for sg in itertools.product((-1.0, 1.0), repeat=len(dd)): tot += 1; hit += abs((np.array(sg) * dd).mean()) >= obs - 1e-12
    return hit / tot
SEL = {"M1 새 시험 전역 플리커 근접": lambda m: m["set"] == "STRESS_SEPX24" and m["kind"] == "G",
       "M2 기존 전역 플리커(INBAND6·NEAR6·TEST_C3)": lambda m: m["set"] in ("STRESS_INBAND6", "STRESS_NEAR6", "TEST_C3"),
       "(음성 대조) 피부 국소 플리커 근접": lambda m: m["set"] == "STRESS_SEPX24" and m["kind"] == "L",
       "(서술) 기존 플리커 먼 거리": lambda m: m["set"] == "STRESS_OFFBAND6", "(서술) 기존 주기 움직임": lambda m: m["set"] == "STRESS_PMOT6"}
def arm(BG, title):
    O = {}; out(f"######## {title}: {BG} vs {F}")
    for key, nm in (("rL", "지연 보정 r (1차)"), ("r0", "지연 0 r (서술)")):
        A, na = clip_mean(BG, key); B, nb = clip_mean(F, key); out(f"=== 가족 M: {nm} — {BG}({na}시드) vs {F}({nb}시드), Δ = 보정 − 기준")
        ps, names, res = [], [], {}
        for lab, f in SEL.items():
            subs = sorted(s for s in A if s in B and f(meta[s])); dd = [A[s] - B[s] for s in subs]
            r = dict(n=len(subs), base=float(np.mean([B[s] for s in subs])), cell=float(np.mean([A[s] for s in subs])), d=float(np.mean(dd)), perm=exact(dd) if len(dd) <= 16 else None,
                     w=float(wilcoxon(dd).pvalue) if len(dd) >= 5 and not np.allclose(dd, 0) else 1.0); res[lab] = r
            if key == "rL" and lab.startswith("M"): ps.append(r["perm"]); names.append(lab)
        hs = dict(zip(names, holm(ps))) if ps else {}
        for lab, r in res.items():
            h = hs.get(lab); r["holm"] = h
            out(f"  {lab:34s} n={r['n']:2d} 기준 {r['base']:.3f} / 보정 {r['cell']:.3f} Δ{r['d']:+.3f} 정확 순열 p={r['perm']:.4f}" + (f" Holm={h:.4f}" if h is not None else ""))
        if key == "rL": O["M"] = dict(results=res, supported=[k for k in names if res[k]["d"] > 0 and res[k]["holm"] < .05], adverse=[k for k in names if res[k]["d"] < 0 and res[k]["holm"] < .05]); out(f"  지지 {O['M']['supported']}  역방향 {O['M']['adverse']}")
    out(f"=== 가족 P: 실제 정확도 — {BG} vs {F}, 시드 {S}")
    P = pfam(BG, F); ps = [P[k]["perm"] for k in PRIM]; hs = holm(ps); sup, adv = [], []
    for k, h in zip(PRIM, hs):
        r = P[k]; r["holm"] = h; sup += [k] if (r["d"] < 0 and h < .05) else []; adv += [k] if (r["d"] > 0 and h < .05) else []
        out(f"  {k:20s} n={r['n']:5d} 기준 {r['base']:6.2f} / 보정 {r['cell']:6.2f} Δ{r['d']:+6.2f} perm={r['perm']:.4f} Holm={h:.4f} W={r['p']:.4f}")
    gv = []
    for k in GUARD:
        r = P.get(k)
        if r is None: continue
        bad = r["d"] > 0.3 and r["perm"] < .05; gv += [k] if bad else []
        out(f"  (가드) {k:14s} n={r['n']:5d} {r['base']:6.2f} / {r['cell']:6.2f} Δ{r['d']:+6.2f} perm={r['perm']:.4f}" + ("  위반" if bad else ""))
    for k in DESC:
        r = P.get(k)
        if r is not None: out(f"  (서술) {k:14s} n={r['n']:5d} {r['base']:6.2f} / {r['cell']:6.2f} Δ{r['d']:+6.2f} perm={r['perm']:.4f}")
    O["P"] = dict(results=P, supported=sup, adverse=adv, guard_violations=gv); out(f"  지지 {sup}  역방향 유의 {adv}  가드 위반 {gv}")
    return O
OUT["primary_train_and_test"] = arm("NF_BGREF_L", "1차 — 학습·평가 모두 게이트 보정")
OUT["secondary_test_time_only"] = arm("NF_NSP2DR_BGT_L", "2차 — 기준 체크포인트에 평가 때만 게이트 보정")
out("=== 서술: 새 분리 시험 심박 판독(±6 bpm) 거리별, 셀별 시드 평균")
for cell in (F, "NF_BGREF_L", "NF_NSP2DR_BGT_L"):
    acc = {}
    for s in S:
        try: d = json.load(open(f"{R}/sepx_{cell}_s{s}.json"))["clips"]
        except FileNotFoundError: continue
        for sub, v in d.items():
            if v["set"] == "STRESS_SEPX24": acc.setdefault(f"{v['kind']}:gap{int(round(v['gap']))}", []).append(v["hr_ok"])
    out(f"  {cell:12s} " + "  ".join(f"{k}={np.mean(x):.2f}" for k, x in sorted(acc.items())))
open(f"{R}/bgref_judge.txt", "w").write("\n".join(L) + "\n"); json.dump(OUT, open(f"{R}/bgref_judge.json", "w"), indent=1, ensure_ascii=False, default=float)
