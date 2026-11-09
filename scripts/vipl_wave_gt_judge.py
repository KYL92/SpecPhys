#!/usr/bin/env python3
"""[60 문서 §26, 탐색적 — 사전 등록 아님] VIPL 정답 타당성: 장치 심박(gt_HR.csv, 문헌 정답) vs 영상 구간에 기록된 PPG 파형 심박(정본 readout).
운동 직후(v7) 영상에서 장치 심박이 파형 심박보다 높게 기록되는 문제를 정량화하고, 모든 모델을 같은 파형 정답(우리 셀들의 라벨 심박 중앙값, 셀·시드 간 차이 0)으로 다시 채점,
공개 가중치 대비 저심박 우세·고심박 비열등(여유 2 bpm, 단측)을 검정한다. MMPD 는 접촉 PPG 라벨 심박(같은 방식의 공통 기준).
  python scripts/vipl_wave_gt_judge.py → results/vipl_wave_gt_judge.{txt,json}"""
import glob, json, os, re
import numpy as np
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"; LINES = []; OUT = {}
def out(s): print(s); LINES.append(s)
def files(m, suf):
    pat = re.compile(rf"agg_{re.escape(m)}_s\d+{suf}\.json$")
    return [f for f in [f"{R}/agg_{m}{suf}.json"] + sorted(x for x in glob.glob(f"{R}/agg_{m}_s[0-9]*{suf}.json") if pat.search(os.path.basename(x))) if os.path.exists(f)]
def key(r): return (r["subject"], r["light"], r["motion"], r["exercise"], r["skin"])
def vp(m):
    acc = {}
    for f in files(m, "_VIPL"):
        for r in json.load(open(f)): acc.setdefault(r["vid"], []).append(r["pred"])
    return {k: float(np.mean(v)) for k, v in acc.items()}
def mp(m):
    acc = {}
    for f in files(m, "_MMPD"):
        for r in json.load(open(f)):
            if 7 <= r["subject"] <= 33: acc.setdefault(key(r), []).append(r["pred"])
    return {k: float(np.mean(v)) for k, v in acc.items()}
W, MG = {}, {}
for m in ("NF_HYB_D_L", "RF_SPEC_L", "NF_HYB_U8_L"):
    for f in files(m, "_VIPL"):
        for r in json.load(open(f)): W.setdefault(r["vid"], []).append(r["gt"])
    for f in files(m, "_MMPD"):
        for r in json.load(open(f)):
            if 7 <= r["subject"] <= 33: MG.setdefault(key(r), []).append(r["gt"])
spr = np.array([np.ptp(v) for v in W.values()]); sp = float(np.mean(spr > 1)); W = {k: float(np.median(v)) for k, v in W.items()}; MG = {k: float(np.median(v)) for k, v in MG.items()}
rows = {r["vid"]: r for r in json.load(open(f"{R}/agg_NF_HYB_D_L_VIPL.json"))}
out(f"=== VIPL 정답 타당성(파형 정답: 우리 셀·시드 라벨 심박의 중앙값; 셀·시드 간 차이 >1 bpm 인 영상 비율 {sp:.4f})")
for nm, lo, hi in (("<60", 0, 60), ("60-90", 60, 90), ("90-120", 90, 120), (">=120", 120, 999)):
    v = [k for k, r in rows.items() if lo <= r["gt_lit"] < hi and k in W]; d = np.array([W[k] - rows[k]["gt_lit"] for k in v])
    out(f"  장치 심박 {nm:6s} n={len(v):4d}: 장치 {np.mean([rows[k]['gt_lit'] for k in v]):6.1f} · 파형 {np.mean([W[k] for k in v]):6.1f} · |차| {np.abs(d).mean():5.1f} · >10 bpm {np.mean(np.abs(d) > 10):.2f} · v7 비율 {np.mean([rows[k].get('scenario') == 'v7' for k in v]):.2f}")
    OUT.setdefault("validity", {})[nm] = dict(n=len(v), abs_diff=float(np.abs(d).mean()), frac_gt10=float(np.mean(np.abs(d) > 10)))
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj
PAIRS = [("NF_ENS_AB_L", "FPbase", "FactorizePhys"), ("NF_ENS_AB_L", "RFbase", "RhythmFormer"), ("NF_ENS_AB_L", "PMbase", "PMbase"), ("NF_ENS_AB_L", "CoTAF_TW_L", "CoTAF_TW_L"),   # [61 §11] 설정 C
         ("NF_SKREF_X15_L", "FPbase", "FactorizePhys"), ("NF_SKREF_X15_L", "RFbase", "RhythmFormer"), ("NF_SKREF_X15_L", "PMbase", "PMbase"), ("NF_SKREF_X15_L", "CoTAF_TW_L", "CoTAF_TW_L"),   # [61 §10] 1.5배속 셀
         ("NF_SKREF_REP_L", "FPbase", "FactorizePhys"), ("NF_SKREF_REP_L", "RFbase", "RhythmFormer"), ("NF_SKREF_REP_L", "PMbase", "PMbase"), ("NF_SKREF_REP_L", "CoTAF_TW_L", "CoTAF_TW_L"),   # [61 §8] 확인 재현(새 시드 600~1000 만)
         ("NF_SKREF_L", "FPbase", "FactorizePhys"), ("NF_SKREF_L", "RFbase", "RhythmFormer"),   # [61 문서] 피부 가중 + 잡음 참조 배제
         ("NF_W4_D_TA2_L", "RFbase", "RhythmFormer"), ("NF_W8_D_TA2_L", "RFbase", "RhythmFormer"), ("NF_W4_D_TA2_L", "FPbase", "FactorizePhys"), ("NF_W8_D_TA2_L", "FPbase", "FactorizePhys"),
         ("NF_REP_D_TA2_L", "FPbase", "FactorizePhys"),   # [§28] 확인 재현(새 시드 600~1000 만)
         ("NF_REP_D_TA2_L", "RFbase", "RhythmFormer"), ("NF_REP_D_TA2_L", "PMbase", "PMbase"), ("NF_REP_D_TA2_L", "CoTAF_TW_L", "CoTAF_TW_L"),
         ("RF_ENS_G_L", "RFbase", "RhythmFormer"), ("RF_ENS_DTA_L", "RFbase", "RhythmFormer"), ("RF_SPEC_D_L", "RFbase", "RhythmFormer"),
         ("NF_HYB_D_TA2_L", "FPbase", "FactorizePhys"), ("NF_HYB_D_L", "FPbase", "FactorizePhys"), ("NF_HYB_TA_L", "FPbase", "FactorizePhys"), ("NF_MCDX_D_L", "FPbase", "FactorizePhys"),
         ("RF_ENS_G_L", "CoTAF_TW_L", "CoTAF_TW_L"), ("NF_HYB_D_TA2_L", "CoTAF_TW_L", "CoTAF_TW_L")]
for cell, pv, pm in PAIRS:
    a, b, am, bm = vp(cell), vp(pv), mp(cell), mp(pm)
    if not a: continue
    T = (("VIPL 파형 <60", lambda g: g < 60, None), ("VIPL 파형 ≥110", lambda g: g >= 110, None), ("풀 저심박(VIPL 파형 <60 ∪ MMPD <60)", lambda g: g < 60, lambda g: g < 60), ("풀 고심박(VIPL 파형 ≥110 ∪ MMPD ≥110)", lambda g: g >= 110, lambda g: g >= 110))
    res = []
    for nm, sv, sm in T:
        ka = [k for k in a if k in b and k in W and sv(W[k])]; km = [k for k in am if k in bm and k in MG and sm(MG[k])] if sm else []
        x = np.array([abs(a[k] - W[k]) for k in ka] + [abs(am[k] - MG[k]) for k in km]); y = np.array([abs(b[k] - W[k]) for k in ka] + [abs(bm[k] - MG[k]) for k in km]); d = x - y
        p = 1.0 if np.allclose(d, 0) else float(wilcoxon(x, y).pvalue); pni = float(wilcoxon(d - 2.0, alternative="less").pvalue)
        res.append(dict(test=nm, n=len(x), ref=float(y.mean()), cell=float(x.mean()), d=float(d.mean()), p=p if np.isfinite(p) else 1.0, p_ni=pni))
    for r, ph in zip(res, holm([r["p"] for r in res])): r["holm"] = ph; r["verdict"] = "우세" if r["d"] < 0 and ph < .05 else ("열세" if r["d"] > 0 and ph < .05 else "ns")
    nih = holm([res[1]["p_ni"], res[3]["p_ni"]]); lo_ok = res[0]["verdict"] == "우세" and res[2]["verdict"] == "우세"; hi_ni = max(nih) < .05
    out(f"\n=== {cell} vs {pv} (탐색적, 파형 정답; 우세 Holm 4 · 고심박 비열등 여유 2 bpm Holm 2)")
    for r in res: out(f"  {r['test']:38s} n={r['n']:4d} {r['ref']:6.2f} -> {r['cell']:6.2f} Δ{r['d']:+6.2f} p={r['p']:.4f} Holm={r['holm']:.4f} → {r['verdict']} | 비열등 p={r['p_ni']:.4f}")
    out(f"  판정: 저심박 우세 {'✓' if lo_ok else '✗'} · 고심박 비열등 {'✓' if hi_ni else '✗'} (Holm {nih[0]:.4f}/{nih[1]:.4f})")
    OUT[f"{cell}|{pv}"] = dict(tests=res, low_superior=lo_ok, high_noninferior=hi_ni, ni_holm=nih)
open(f"{R}/vipl_wave_gt_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/vipl_wave_gt_judge.json", "w"), indent=1, ensure_ascii=False)
