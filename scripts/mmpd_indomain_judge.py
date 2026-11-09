#!/usr/bin/env python3
"""[61 문서 §14, 사전 등록] 설정 C 의 MMPD in-domain 축(피험자 배타 5-fold 2단계, 피험자 7~33 판정, 공통 정답) — in-domain 결과는 zero-shot 과 섞지 않는다.
(공용 도움 함수 출처) 피부 가중 + 잡음 참조 배제의 확인 재현 — NF_SKREF_REP_L(새 시드 600~1000) vs 같은 시드의 채택 셀 재현 NF_REP_D_TA2_L(체인 104).
1차 R1(단일 검정, α=.05): 풀 고심박(VIPL 장치 ≥120 ∪ MMPD ≥110) Δ<0 — 영상 단위 양측 Wilcoxon(시드별 오차의 영상 평균, skin_ref_judge 방식) + 방향.
핵심 2차(Holm 2): 공개 RhythmFormer 대비 풀 고심박 비열등(여유 2 bpm, 단측) — R2 영상 구간 PPG 파형 정답(시드 평균 예측, vipl_wave_gt_judge 방식), R3 장치 정답(시드별 오차 평균).
가드(같은 시드 기준 대비): VIPL 전체·MMPD 전체·VIPL <60 Δ ≤ +0.3. 서술: 하위 대역·MR-NIRP 3축·RTNI·PURE, 공개 FactorizePhys 대비 저·고심박과 6축, 시드별 짝 차이, 10시드(100~1000) 합산.
판정은 새 시드만 쓴다(시드 100~500 은 서술용 10시드 합산에만).
  python scripts/mmpd_indomain_judge.py → results/mmpd_indomain_judge.{txt,json}"""
import glob, json, os, re
import numpy as np
from scipy.stats import wilcoxon
import sys as _sys; _sys.path.insert(0, "/SPECPHYS_ROOT/scripts"); import mmpd_common_gt as _MGC   # [61 §9.3] MMPD 공통 정답
R = "/SPECPHYS_ROOT/_workspace/data/results"; LINES = []; OUT = {}
def out(s): print(s); LINES.append(s)
def files(ms, suf):
    if isinstance(ms, tuple):   # (셀, 시드) → 그 시드의 파일 하나만
        nm = ms[0] if ms[1] == 100 else f"{ms[0]}_s{ms[1]}"; f = f"{R}/agg_{nm}{suf}.json"; return [f] if os.path.exists(f) else []
    ms = [ms] if isinstance(ms, str) else ms; res = []
    for m in ms:
        pat = re.compile(rf"agg_{re.escape(m)}(_s\d+)?{suf}\.json$")
        res += [f for f in [f"{R}/agg_{m}{suf}.json"] + sorted(glob.glob(f"{R}/agg_{m}_s[0-9]*{suf}.json")) if os.path.exists(f) and pat.fullmatch(os.path.basename(f))]
    return res
def seeds(ms):
    s = []
    for f in files(ms, "_VIPL"):
        g = re.search(r"_s(\d+)_VIPL\.json$", f); s.append(int(g.group(1)) if g else 100)
    return sorted(s)
def key(r): return (r["subject"], r["light"], r["motion"], r["exercise"], r["skin"])
def vipl(ms):
    acc = {}
    for f in files(ms, "_VIPL"):
        for r in json.load(open(f)): acc.setdefault(r["vid"], []).append(r)
    return {k: dict(dev=v[0]["gt_lit"], e=float(np.mean([x["err_lit"] for x in v])), p=float(np.mean([x["pred"] for x in v])), sc=v[0].get("scenario")) for k, v in acc.items()}
def mmpd(ms):
    acc = {}
    for f in files(ms, "_MMPD"):
        for r in _MGC.load(f):
            if 7 <= r["subject"] <= 33: acc.setdefault(key(r), []).append(r)
    return {k: dict(gt=float(np.median([x["gt"] for x in v])), e=float(np.mean([x["err"] for x in v])), p=float(np.mean([x["pred"] for x in v])), mo=v[0]["motion"]) for k, v in acc.items()}
def axis(ms, ax, sel=None):
    acc = {}
    for f in files(ms, ""):
        d = json.load(open(f))
        for r in (d.get(ax) or []) if isinstance(d, dict) else []:
            if sel is None or sel(r): acc.setdefault(r["vid"], []).append(r["err"])
    return {k: float(np.mean(v)) for k, v in acc.items()}
def cmp(a, b):
    k = sorted(set(a) & set(b))
    if len(k) < 5: return None
    x = np.array([a[i] for i in k]); y = np.array([b[i] for i in k]); d = x - y
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(x, y).pvalue)
    return dict(n=len(k), base=float(y.mean()), cell=float(x.mean()), d=float(d.mean()), p=p if np.isfinite(p) else 1.0)
def ni_p(a, b, m=2.0):
    k = sorted(set(a) & set(b)); d = np.array([a[i] - b[i] for i in k]) - m
    if len(k) < 5 or np.allclose(d, 0): return 1.0
    p = float(wilcoxon(d, alternative="less").pvalue); return p if np.isfinite(p) else 1.0
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj
def hi_dev(V, M): return {**{("V", k): v["e"] for k, v in V.items() if v["dev"] >= 120}, **{("M",) + k: v["e"] for k, v in M.items() if v["gt"] >= 110}}
def lo_dev(V, M): return {**{("V", k): v["e"] for k, v in V.items() if v["dev"] < 60}, **{("M",) + k: v["e"] for k, v in M.items() if v["gt"] < 60}}
# 파형 정답(60 §26 과 같은 정의): 우리 셀·시드 라벨 심박의 영상별 중앙값
WV, WM = {}, {}
for m in ("NF_HYB_D_L", "RF_SPEC_L", "NF_HYB_U8_L"):
    for f in files(m, "_VIPL"):
        for r in json.load(open(f)): WV.setdefault(r["vid"], []).append(r["gt"])
    for f in files(m, "_MMPD"):
        for r in json.load(open(f)):
            if 7 <= r["subject"] <= 33: WM.setdefault(key(r), []).append(r["gt"])
WV = {k: float(np.median(v)) for k, v in WV.items()}; WM = {k: float(np.median(v)) for k, v in WM.items()}
def hi_wave(V, M): return {**{("V", k): abs(v["p"] - WV[k]) for k, v in V.items() if k in WV and WV[k] >= 110}, **{("M",) + k: abs(v["p"] - WM[k]) for k, v in M.items() if k in WM and WM[k] >= 110}}
def lo_wave(V, M): return {**{("V", k): abs(v["p"] - WV[k]) for k, v in V.items() if k in WV and WV[k] < 60}, **{("M",) + k: abs(v["p"] - WM[k]) for k, v in M.items() if k in WM and WM[k] < 60}}
def line(nm, r, tail=""):
    if r is None: out(f"  {nm:30s} (missing)"); return
    out(f"  {nm:30s} n={r['n']:4d} {r['base']:6.2f} -> {r['cell']:6.2f} Δ{r['d']:+6.2f} p={r['p']:.4f}{tail}")

SEEDS = (600, 700, 800)
def mm_seeds(m, seeds=SEEDS):   # 시드 지정 MMPD 적재(같은 시드 짝 비교용)
    acc = {}
    for s in seeds:
        for f in files((m, s), "_MMPD"):
            for r in _MGC.load(f):
                if 7 <= r["subject"] <= 33: acc.setdefault(key(r), []).append(r)
    return {k: dict(gt=float(np.median([x["gt"] for x in v])), e=float(np.mean([x["err"] for x in v])), p=float(np.mean([x["pred"] for x in v])), mo=v[0]["motion"], n=len(v)) for k, v in acc.items()}
sel = {"MMPD ≥110": lambda v: v["gt"] >= 110, "MMPD 전체": lambda v: True, "MMPD <110": lambda v: v["gt"] < 110, "MMPD <60": lambda v: v["gt"] < 60, "MMPD 보행": lambda v: v["mo"] == 4}
def pick(X, nm): return {k: v["e"] for k, v in X.items() if sel[nm](v)}
IA, IB, IC = mm_seeds("MCV5_SKA_L"), mm_seeds("MCV5_SKB_L"), mm_seeds("MCV5_SKC_L")
ZA, ZB, ZC = mm_seeds("NF_SKREF_REP_L"), mm_seeds("NF_SKREF_X15_L"), mm_seeds("NF_ENS_AB_L")
out(f"=== 61 §14 MMPD in-domain 축(피험자 7~33, 공통 정답, 시드 {SEEDS}) — in-domain 결과는 zero-shot 축과 별도")
out(f"  영상 수: in-domain C {len(IC)} (시드당 영상 평균 {np.mean([v['n'] for v in IC.values()]) if IC else 0:.1f}) · zero-shot C {len(ZC)}")
if not IC or len(IC) < 500: out("  !! in-domain C 결과가 불완전 — 판정 보류")
T = [("I1 MMPD ≥110 (in-domain C vs zero-shot C)", cmp(pick(IC, "MMPD ≥110"), pick(ZC, "MMPD ≥110"))), ("I2 MMPD 전체 (in-domain C vs zero-shot C)", cmp(pick(IC, "MMPD 전체"), pick(ZC, "MMPD 전체")))]
hs = holm([r["p"] if r and r["d"] < 0 else 1.0 for _, r in T])
out("--- 1차(Holm 2)")
for (nm, r), h in zip(T, hs):
    v = "지지" if r and r["d"] < 0 and h < .05 else ("역방향" if r and r["d"] > 0 and r["p"] < .05 else "ns"); line(nm, r, f" Holm={h:.4f} → {v}"); OUT[nm] = dict(r or {}, holm=h, verdict=v)
out("--- 2차: in-domain 설정끼리(같은 시드) — 라우팅의 in-domain 가치")
for nm in ("MMPD ≥110", "MMPD 전체", "MMPD <110", "MMPD <60", "MMPD 보행"):
    for lab, X in (("C vs A", IA), ("C vs B", IB)):
        r = cmp(pick(IC, nm), pick(X, nm)); line(f"{lab} {nm}", r); OUT[f"{lab} {nm}"] = r
out("--- 2차: in-domain vs zero-shot (같은 설정·같은 시드)")
for nm in ("MMPD ≥110", "MMPD 전체", "MMPD <110", "MMPD <60", "MMPD 보행"):
    for lab, I, Z in (("A", IA, ZA), ("B", IB, ZB), ("C", IC, ZC)):
        r = cmp(pick(I, nm), pick(Z, nm)); line(f"in-domain {lab} vs zero-shot {lab} {nm}", r); OUT[f"I{lab} vs Z{lab} {nm}"] = r
out("--- 서술: in-domain C vs 공개 모델(공개 모델은 zero-shot — 같은 표에 섞지 말 것) · 이전 in-domain SpecFP(MCV5_NF*_L, 이전 레시피·시드 100~300)")
RF, FP = mmpd("RhythmFormer"), mmpd("FactorizePhys"); OLD = {}
for t in ("MCV5_NF_L", "MCV5_NFs2_L", "MCV5_NFs3_L"):
    for f in files(t, "_MMPD"):
        for r in _MGC.load(f):
            if 7 <= r["subject"] <= 33: OLD.setdefault(key(r), []).append(r)
OLD = {k: dict(gt=float(np.median([x["gt"] for x in v])), e=float(np.mean([x["err"] for x in v])), p=float(np.mean([x["pred"] for x in v])), mo=v[0]["motion"]) for k, v in OLD.items()}
for nm in ("MMPD ≥110", "MMPD 전체", "MMPD <110", "MMPD <60", "MMPD 보행"):
    for lab, X in (("공개 RhythmFormer", RF), ("공개 FactorizePhys", FP), ("이전 in-domain SpecFP", OLD)):
        a, b = pick(IC, nm), pick(X, nm); r = cmp(a, b)
        if r: out(f"  C(in) vs {lab:22s} {nm:9s} n={r['n']:3d} {r['base']:6.2f} -> {r['cell']:6.2f} Δ{r['d']:+6.2f} p={r['p']:.4f} | 순위 비열등(2 bpm) p={ni_p(a, b):.4f}"); OUT[f"C(in) vs {lab} {nm}"] = dict(r, ni_p=ni_p(a, b))
open(f"{R}/mmpd_indomain_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/mmpd_indomain_judge.json", "w"), indent=1, ensure_ascii=False, default=float)
