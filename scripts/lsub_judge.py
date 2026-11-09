#!/usr/bin/env python3
"""[65 문서, 사전 등록] 체인 124 판정: 학습된 잡음 빼기 + 이중 렌더 픽셀·색 정답(NF_LSUB_L) vs 기준(새 풀, NF_NSP2DR_L) vs 정답 없는 빼기(NF_LSUB0_L). 시드 2600~3000.
 가족 M(기전, Holm 2): 풀 밖 합성 STRESS_SEPX24 — M1 전역 플리커 근접 12클립, M2 피부 국소 플리커 근접 12클립, 클립별 청크 Pearson r(출력, 정답 PPG) 5시드 평균의 쌍대 차이, 부호 뒤집기 순열(정확, 양측).
 가족 P(실제 정확도, Holm 5): VIPL-HR 전체, MMPD 7~33 전체(공통 정답), 풀 저심박, 풀 고심박, MR-NIRP 소동작 — 영상별 오차 5시드 평균, 부호 뒤집기 순열(양측), Wilcoxon 병기.
 가족 L(정답의 몫, Holm 7): LS vs L0 의 M1·M2·P 5개. 가드(LS vs F): VIPL <60·≥120, MMPD <110·≥110, DLCN, RTNI, PURE — Δ>+0.3 bpm 이고 비보정 p<.05 면 위반.
  python scripts/lsub_judge.py → results/lsub_judge.{txt,json}"""
import itertools, json, os
import numpy as np
from scipy.stats import wilcoxon
src = open("/SPECPHYS_ROOT/scripts/nsg_judge.py").read(); src = src[:src.index('if "--screen2" in sys.argv:')]
g = {}; exec(compile(src, "nj", "exec"), g); bundle, cmp, flip_p, holm, R = g["bundle"], g["cmp"], g["flip_p"], g["holm"], g["R"]
S = [2600, 2700, 2800, 2900, 3000]; SR = [2100, 2200, 2300, 2400, 2500]; LINES = []; OUT = {}
F, LS, L0, REF = "NF_NSP2DR_L", "NF_LSUB_L", "NF_LSUB0_L", "NF_NSP2_L"
def out(s): print(s); LINES.append(s)
PRIM = ("P2 VIPL-HR all", "P3 MMPD 7-33 all", "pooled low", "pooled high", "NIRP small")
GUARD = ("VIPL <60", "VIPL >=120", "MMPD <110", "MMPD >=110", "DLCN", "RTNI", "PURE 07-10")
DESC = ("VIPL v2", "VIPL v7", "VIPL v9", "MMPD walking", "MMPD rotation", "NIRP still", "NIRP large", "RTNC", "MCD_A", "MCD_B", "UBFC 42-49")
def sepx(cell, seeds, key="r_ppg"):
    """{subject: 시드 평균 값} (없는 시드는 건너뜀) + 메타."""
    acc, meta = {}, {}
    for s in seeds:
        p = f"{R}/sepx_{cell}_s{s}.json"
        if not os.path.exists(p): continue
        for sub, v in json.load(open(p))["clips"].items():
            if v.get(key) is None: continue
            acc.setdefault(sub, []).append(v[key]); meta[sub] = v
    return {k: float(np.mean(v)) for k, v in acc.items()}, meta
def exact_flip(d):
    d = np.asarray(d, float); obs = abs(d.mean()); n = len(d)
    if n == 0: return 1.0
    tot = hit = 0
    for sg in itertools.product((-1.0, 1.0), repeat=n): tot += 1; hit += abs((np.array(sg) * d).mean()) >= obs - 1e-12
    return hit / tot
def mfam(a, b):
    ra, meta = sepx(a, S); rb, _ = sepx(b, S); res = {}
    for k, kind in (("M1 전역 플리커 근접", "G"), ("M2 피부 국소 플리커 근접", "L")):
        subs = sorted(s for s in ra if s in rb and meta[s]["set"] == "STRESS_SEPX24" and meta[s]["kind"] == kind)
        d = [ra[s] - rb[s] for s in subs]; res[k] = dict(n=len(subs), base=float(np.mean([rb[s] for s in subs])) if subs else None, cell=float(np.mean([ra[s] for s in subs])) if subs else None,
                                                         d=float(np.mean(d)) if d else None, perm=exact_flip(d), w=float(wilcoxon(d).pvalue) if len(d) >= 5 and not np.allclose(d, 0) else 1.0)
    return res
def pfam(a, b):
    A, B = bundle(a, S), bundle(b, S); res = {}
    for k in PRIM + GUARD + DESC:
        r = cmp(A[k], B[k])
        if r is None: continue
        r["perm"] = flip_p(A[k], B[k]); res[k] = r
    return res
def family(title, a, b, keys_m, keys_p, guards):
    out(f"=== {title}: {a} vs {b}, 시드 {S} (Δ = 앞 − 뒤)"); M = mfam(a, b) if keys_m else {}; P = pfam(a, b); ps, names = [], []
    for k in keys_m:
        if not M[k]["n"]: out(f"  {k}: 자료 없음"); continue
        ps.append(M[k]["perm"]); names.append(k)
    for k in keys_p:
        if k not in P: out(f"  {k}: 자료 없음"); continue
        ps.append(P[k]["perm"]); names.append(k)
    hs = holm(ps) if ps else []; sup, adv = [], []
    for k, h in zip(names, hs):
        r = M.get(k) or P.get(k); r["holm"] = h
        good = (r["d"] > 0) if k in keys_m else (r["d"] < 0); sup += [k] if (good and h < .05) else []; adv += [k] if ((not good) and h < .05) else []
        unit = "r" if k in keys_m else "bpm"
        out(f"  {k:22s} n={r['n']:5d} {b} {r['base']:7.3f} / {a} {r['cell']:7.3f} Δ{r['d']:+7.3f} {unit} perm={r['perm']:.4f} Holm={h:.4f}" + (f" W={r['p']:.4f}" if 'p' in r else f" W={r['w']:.4f}"))
    gv = []
    for k in guards:
        r = P.get(k)
        if r is None: continue
        bad = r["d"] > 0.3 and r["perm"] < .05; gv += [k] if bad else []
        out(f"  (가드) {k:14s} n={r['n']:5d} {r['base']:6.2f} / {r['cell']:6.2f} Δ{r['d']:+6.2f} perm={r['perm']:.4f}" + ("  위반" if bad else ""))
    for k in DESC:
        r = P.get(k)
        if r is not None: out(f"  (서술) {k:14s} n={r['n']:5d} {r['base']:6.2f} / {r['cell']:6.2f} Δ{r['d']:+6.2f} perm={r['perm']:.4f}")
    out(f"  지지 {sup}  역방향 유의 {adv}  가드 위반 {gv}"); return dict(M=M, P=P, supported=sup, adverse=adv, guard_violations=gv)
OUT["M"] = family("가족 M + P: 학습된 빼기 + 정답 vs 기준(새 풀)", LS, F, ("M1 전역 플리커 근접", "M2 피부 국소 플리커 근접"), (), ())
OUT["P"] = family("가족 P: 실제 정확도", LS, F, (), PRIM, GUARD)
OUT["L"] = family("가족 L: 정답의 몫(학습된 빼기 정답 있음 vs 없음)", LS, L0, ("M1 전역 플리커 근접", "M2 피부 국소 플리커 근접"), PRIM, ())
out("=== 서술: 분리 시험 전체(청크 r, 심박 판독 ±6 bpm, 잡음 결맞음), 셀별 시드 평균")
for cell, seeds in ((F, S), (LS, S), (L0, S), (REF, SR)):
    for key in ("r_ppg", "hr_ok", "flick"):
        v, meta = sepx(cell, seeds, key)
        if not v: continue
        grp = {}
        for s, x in v.items():
            m = meta[s]; gname = f"{m['set']}:{m['kind']}" + (f":gap{int(round(m['gap']))}" if m["set"] == "STRESS_SEPX24" and m.get("gap") is not None else ""); grp.setdefault(gname, []).append(x)
        out(f"  {cell:12s} {key:5s} " + "  ".join(f"{k}={np.mean(x):.2f}" for k, x in sorted(grp.items())))
open(f"{R}/lsub_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/lsub_judge.json", "w"), indent=1, ensure_ascii=False, default=float)
