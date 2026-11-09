#!/usr/bin/env python3
"""[62 문서 §11] 잡음 부분공간 투영 셀의 기존 방법 대비 표(학습 없음). final_config_tables 와 같은 정의: MMPD 공통 정답, VIPL 장치 정답, 시드 600~1000.
  장치 정답: 저심박 우세(풀, Wilcoxon, Holm 5: VIPL <60·VIPL ≥120·VIPL v7·풀 저심박·풀 고심박), 고심박 순위 비열등(2 bpm, Holm 2: VIPL ≥120·풀 고심박), 평균 차와 평균 기준 t 검정.
  파형 정답: 풀 저심박 우세, 고심박 비열등. 6개 집합 + 나머지 집합의 쌍대 비교. MMPD ≥110 단독 비열등.
  python scripts/nsp_vs_existing.py → results/nsp_vs_existing.{md,json}"""
import json, numpy as np
from scipy.stats import ttest_1samp, wilcoxon
src = open("/SPECPHYS_ROOT/scripts/final_config_tables.py").read(); src = src[:src.index("NAMES = [")]
g = {}; exec(compile(src, "fct", "exec"), g)
vipl, mmpd, axis, cmp, ni_p, holm, hi_dev, lo_dev, hi_wave, lo_wave, R, seeds = [g[k] for k in ("vipl", "mmpd", "axis", "cmp", "ni_p", "holm", "hi_dev", "lo_dev", "hi_wave", "lo_wave", "R", "seeds")]
S5 = [600, 700, 800, 900, 1000]
def vs(cell): return [f"{cell}" if s == 100 else f"{cell}_s{s}" for s in S5]
EXIST = [("public FactorizePhys", "FPbase", "FactorizePhys"), ("public RhythmFormer", "RFbase", "RhythmFormer"), ("public PhysMamba", "PMbase", "PMbase"), ("previous PhysFormer + corpus", "CoTAF_TW_L", "CoTAF_TW_L")]
CELLS = [("SpecPhys-R (current final)", "NF_ENS_AB_L"), ("single model + projection, parser at input", "NF_NSP2_L"), ("single model + projection, no parser", "NF_NSP2P_L"), ("routed + projection", "NF_ENS_N_L"), ("SpecPhys-L (no projection)", "NF_SKREF_REP_L")]
ve = lambda V, c=None: {k: v["e"] for k, v in V.items() if c is None or c(v)}
me = lambda M, c=None: {k: v["e"] for k, v in M.items() if c is None or c(v)}
mean = lambda d: float(np.mean(list(d.values()))) if d else float("nan")
md = []; J = {}
def emit(s=""): md.append(s); print(s)
def load(cell):
    n = vs(cell); return vipl(n), mmpd(n), n
def fp(p): return "<.0001" if p < 1e-4 else (f"{p:.4f}".lstrip("0") if p < .01 else f"{p:.3f}".lstrip("0"))
PUB = {nm: (vipl(v), mmpd(m), m) for nm, v, m in EXIST}
emit("# Noise-subspace projection against existing methods (zero-shot; seeds 600-1000; MMPD common labels; VIPL-HR device labels)\n")
emit("## A. Levels"); emit("| Model | VIPL all | VIPL <60 | VIPL ≥120 | v7 | MMPD all | MMPD <110 | MMPD ≥110 | walking | pooled low | pooled high | NIRP still | small | large | DLCN | RTNI | PURE 07–10 |"); emit("|---|" + "---|" * 16)
SETS = [("VIPL-HR", None), ("MMPD 7–33", None), ("MR-NIRP still", "NIRP_STILL"), ("MR-NIRP small", "NIRP_SMALL"), ("MR-NIRP large", "NIRP_LARGE"), ("DLCN", "DLCN"), ("RTNI", "RTNI"), ("PURE 07–10", "PURE"), ("UBFC 42–49", "REALTEST"), ("MCD-A", "MCD_A"), ("MCD-B", "MCD_B")]
def ax(name, key):
    return axis(name, key, (lambda r: int(r["vid"]) >= 701) if key == "PURE" else None)
def levels(nm, V, M, m):
    r = [mean(ve(V)), mean(ve(V, lambda x: x["dev"] < 60)), mean(ve(V, lambda x: x["dev"] >= 120)), mean(ve(V, lambda x: x["sc"] == "v7")), mean(me(M)), mean(me(M, lambda x: x["gt"] < 110)), mean(me(M, lambda x: x["gt"] >= 110)), mean(me(M, lambda x: x["mo"] == 4)), mean(lo_dev(V, M)), mean(hi_dev(V, M))] + [mean(ax(m, k)) for k in ("NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE", "DLCN", "RTNI", "PURE")]
    emit(f"| {nm} | " + " | ".join("–" if not np.isfinite(x) else f"{x:.2f}" for x in r) + " |"); J.setdefault("levels", {})[nm] = r
for nm, (V, M, m) in PUB.items(): levels(nm, V, M, m)
LOAD = {}
for nm, c in CELLS:
    V, M, n = load(c); LOAD[nm] = (V, M, n); levels(nm, V, M, n)
emit("\n## B. Paired tests against each existing method (Holm over five heart-rate tests; rank non-inferiority 2 bpm Holm over two; waveform labels: Holm over two)")
for nm, c in CELLS[:4]:
    V, M, n = LOAD[nm]; emit(f"\n### {nm}"); emit("| Existing method | pooled low (Holm) | pooled high (rank NI Holm) | VIPL ≥120 | mean gap (t p) | MMPD ≥110 alone (rank NI p) | MMPD <110 (NI 0.3 p) | waveform pooled low (p) | waveform pooled high (NI p) |"); emit("|---|" + "---|" * 8)
    for en, (XV, XM, xm) in PUB.items():
        T = {"v60": (ve(V, lambda x: x["dev"] < 60), ve(XV, lambda x: x["dev"] < 60)), "v120": (ve(V, lambda x: x["dev"] >= 120), ve(XV, lambda x: x["dev"] >= 120)), "v7": (ve(V, lambda x: x["sc"] == "v7"), ve(XV, lambda x: x["sc"] == "v7")), "low": (lo_dev(V, M), lo_dev(XV, XM)), "high": (hi_dev(V, M), hi_dev(XV, XM))}
        res = {k: cmp(*v) for k, v in T.items()}; hs = dict(zip(res, holm([r["p"] for r in res.values()])))
        n1, n2 = ni_p(*T["v120"]), ni_p(*T["high"]); hn = holm([n1, n2])
        a, b = T["high"]; ks = sorted(set(a) & set(b), key=str); d = np.array([a[i] - b[i] for i in ks]); tp = float(ttest_1samp(d - 2, 0, alternative="less").pvalue)
        mh = (me(M, lambda x: x["gt"] >= 110), me(XM, lambda x: True)); ml = (me(M, lambda x: x["gt"] < 110), me(XM, lambda x: True))
        mh_b = {k: mh[1][k] for k in mh[0] if k in mh[1]}; ml_b = {k: ml[1][k] for k in ml[0] if k in ml[1]}; pmh = ni_p(mh[0], mh_b, 2.0); pml = ni_p(ml[0], ml_b, 0.3)
        wl = cmp(lo_wave(V, M), lo_wave(XV, XM)); wh = cmp(hi_wave(V, M), hi_wave(XV, XM)); wni = ni_p(hi_wave(V, M), hi_wave(XV, XM))
        emit(f"| {en} | {res['low']['base']:.2f} → {res['low']['cell']:.2f} ({fp(hs['low'])}) | {res['high']['base']:.2f} → {res['high']['cell']:.2f} ({fp(hn[1])}) | {res['v120']['base']:.2f} → {res['v120']['cell']:.2f} | {d.mean():+.2f} ({fp(tp)}) | {cmp(*mh)['base']:.2f} → {cmp(*mh)['cell']:.2f} ({fp(pmh)}) | {cmp(*ml)['base']:.2f} → {cmp(*ml)['cell']:.2f} ({fp(pml)}) | {wl['base']:.2f} → {wl['cell']:.2f} ({fp(wl['p'])}) | {wh['base']:.2f} → {wh['cell']:.2f} ({fp(wni)}) |")
        J.setdefault("guarantee", {}).setdefault(nm, {})[en] = dict(low=res["low"], low_holm=hs["low"], high=res["high"], high_holm=hs["high"], ni_holm=hn, mean_gap=float(d.mean()), mean_ni_p=tp, mmpd_hi_ni=pmh, mmpd_lo_ni=pml, wave_low=wl, wave_high=wh, wave_ni=wni)
emit("\n## C. Evaluation sets against each existing method (Δ = cell − reference; ✓ significantly better, Holm over six on VIPL-HR, MMPD, MR-NIRP small and large, RTNI, PURE, unadjusted elsewhere; ✗ significantly worse)")
SIX = {"VIPL-HR", "MMPD 7–33", "MR-NIRP small", "MR-NIRP large", "RTNI", "PURE 07–10"}
for nm, c in CELLS[:4]:
    V, M, n = LOAD[nm]; emit(f"\n### {nm}"); emit("| Set | " + " | ".join(e for e, _, _ in EXIST) + " |"); emit("|---|" + "---|" * 4); table = {}
    for en, (XV, XM, xm) in PUB.items():
        rows = [(a, cmp(ve(V), ve(XV)) if a == "VIPL-HR" else (cmp(me(M), me(XM)) if a == "MMPD 7–33" else cmp(ax(n, k), ax(xm, k)))) for a, k in SETS]; av = [(a, r) for a, r in rows if r]
        A6 = [(a, r) for a, r in av if a in SIX]; hs = dict(zip([a for a, _ in A6], holm([r["p"] for _, r in A6])))
        for a, r in av:
            ph = hs.get(a, r["p"]); table.setdefault(a, {})[en] = f"{r['base']:.2f} ({r['d']:+.2f}{' ✓' if r['d'] < 0 and ph < .05 else (' ✗' if r['d'] > 0 and ph < .05 else '')})"
        J.setdefault("sets", {}).setdefault(nm, {})[en] = {a: dict(r, holm=hs.get(a)) for a, r in av}
    for a, _ in SETS: emit(f"| {a}{'*' if a in SIX else ''} | " + " | ".join(table.get(a, {}).get(e, "–") for e, _, _ in EXIST) + " |")
open(f"{R}/nsp_vs_existing.md", "w").write("\n".join(md) + "\n"); json.dump(J, open(f"{R}/nsp_vs_existing.json", "w"), indent=1, ensure_ascii=False, default=float)
