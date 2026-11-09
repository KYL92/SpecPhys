#!/usr/bin/env python3
"""[60 문서 §19(a)] 채택 셀 vs 기존 방법(공개 가중치·이전 공식 모델)의 극단 심박 검정 — 영상 단위 쌍대 Wilcoxon, 시드 영상별 평균 풀링, 셀×기준마다 Holm 5.
  python scripts/hr_vs_public_judge.py → results/hr_vs_public_judge.{txt,json}"""
import glob, json, os, re
import numpy as np
from scipy.stats import wilcoxon
import sys as _sys; _sys.path.insert(0, "/SPECPHYS_ROOT/scripts"); import mmpd_common_gt as _MGC   # [61 §9.3] MMPD 공통 정답
R = "/SPECPHYS_ROOT/_workspace/data/results"
def files(m, suffix):
    pat = re.compile(rf"agg_{re.escape(m)}_s\d+{re.escape(suffix)}\.json$")
    return [f for f in [f"{R}/agg_{m}{suffix}.json"] + sorted(x for x in glob.glob(f"{R}/agg_{m}_s[0-9]*{suffix}.json") if pat.search(os.path.basename(x))) if os.path.exists(f)]
def rows(f):
    d = json.load(open(f)); return d.get("rows", d) if isinstance(d, dict) else d
def vipl(m, sel):
    acc = {}
    for f in files(m, "_VIPL"):
        for r in rows(f):
            if sel(r): acc.setdefault(r["vid"], []).append(r["err_lit"])
    return {("V",) + (k,): float(np.mean(v)) for k, v in acc.items()}
def mmpd(m, sel):
    acc = {}
    for f in files(m, "_MMPD"):
        for r in _MGC.load(f):
            if 7 <= r["subject"] <= 33 and sel(r): acc.setdefault(("M", r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]), []).append(r["err"])
    return {k: float(np.mean(v)) for k, v in acc.items()}
def cmp(a, b):
    k = sorted(set(a) & set(b))
    if len(k) < 6: return None
    x = np.array([a[i] for i in k]); y = np.array([b[i] for i in k]); d = x - y
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(x, y).pvalue)
    if not np.isfinite(p): p = 1.0
    return dict(n=len(k), a=float(x.mean()), b=float(y.mean()), d=float(d.mean()), p=p)
def ni(a, b, margin=2.0):   # [§23(b)] 단측 비열등: 셀 − 기존 − margin < 0
    k = sorted(set(a) & set(b))
    if len(k) < 6: return None
    d = np.array([a[i] for i in k]) - np.array([b[i] for i in k]) - margin
    if np.allclose(d, 0): return 1.0
    p = float(wilcoxon(d, alternative="less").pvalue); return p if np.isfinite(p) else 1.0
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj
# (셀, 기준 VIPL 이름, 기준 MMPD 이름, 설명)
PAIRS = [("NF_ENS_AB_L", "FPbase", "FactorizePhys", "[61 §11] 설정 C(A·B 청크 라우팅) vs 공개 FactorizePhys"), ("NF_ENS_AB_L", "RFbase", "RhythmFormer", "[61 §11] 설정 C vs 공개 RhythmFormer"),
         ("NF_ENS_AB_L", "PMbase", "PMbase", "[61 §11] 설정 C vs 공개 PhysMamba"), ("NF_ENS_AB_L", "CoTAF_TW_L", "CoTAF_TW_L", "[61 §11] 설정 C vs 이전 공식 PhysFormer"),
         ("NF_SKREF_X15_L", "FPbase", "FactorizePhys", "[61 §10] 1.5배속 MMPD 학습 클립 셀 vs 공개 FactorizePhys"), ("NF_SKREF_X15_L", "RFbase", "RhythmFormer", "[61 §10] 1.5배속 셀 vs 공개 RhythmFormer"),
         ("NF_SKREF_X15_L", "PMbase", "PMbase", "[61 §10] 1.5배속 셀 vs 공개 PhysMamba"), ("NF_SKREF_X15_L", "CoTAF_TW_L", "CoTAF_TW_L", "[61 §10] 1.5배속 셀 vs 이전 공식 PhysFormer"),
         ("NF_SKREF_REP_L", "FPbase", "FactorizePhys", "[61 §8] 확인 재현: 피부 가중 + 잡음 참조 배제(새 시드 600~1000) vs 공개 FactorizePhys"), ("NF_SKREF_REP_L", "RFbase", "RhythmFormer", "[61 §8] 확인 재현 vs 공개 RhythmFormer"),
         ("NF_SKREF_REP_L", "PMbase", "PMbase", "[61 §8] 확인 재현 vs 공개 PhysMamba"), ("NF_SKREF_REP_L", "CoTAF_TW_L", "CoTAF_TW_L", "[61 §8] 확인 재현 vs 이전 공식 PhysFormer"),
         ("NF_SKREF_L", "FPbase", "FactorizePhys", "[61 문서] 피부 가중 + 잡음 참조 배제 vs 공개 FactorizePhys"), ("NF_SKREF_L", "RFbase", "RhythmFormer", "[61 문서] 피부 가중 + 잡음 참조 배제 vs 공개 RhythmFormer"),
         ("NF_W4_D_TA2_L", "RFbase", "RhythmFormer", "폭 4배(0.82 M) 채택 레시피 vs 공개 RhythmFormer"), ("NF_W8_D_TA2_L", "RFbase", "RhythmFormer", "폭 8배(3.29 M) 채택 레시피 vs 공개 RhythmFormer"),
         ("NF_W4_D_TA2_L", "FPbase", "FactorizePhys", "폭 4배 vs 공개 FactorizePhys"), ("NF_W8_D_TA2_L", "FPbase", "FactorizePhys", "폭 8배 vs 공개 FactorizePhys"),
         ("NF_REP_D_TA2_L", "FPbase", "FactorizePhys", "확인 재현: SpecFP + 배가 + ×2 전용 증강(새 시드 600~1000) vs 공개 FactorizePhys"),
         ("NF_REP_D_TA2_L", "RFbase", "RhythmFormer", "확인 재현 셀 vs 공개 RhythmFormer(백본 교차)"),
         ("NF_REP_D_TA2_L", "PMbase", "PMbase", "확인 재현 셀 vs 공개 PhysMamba(백본 교차)"),
         ("NF_REP_D_TA2_L", "CoTAF_TW_L", "CoTAF_TW_L", "확인 재현 셀 vs 이전 공식 PhysFormer"),
         ("NF_MCDX_D_L", "FPbase", "FactorizePhys", "SpecFP + 배가 + MCD 실측 극단 심박 확장 풀 vs 공개 FactorizePhys"),
         ("RF_MCDX_D_L", "RFbase", "RhythmFormer", "RhythmFormer + 스펙트럼 + 배가 + MCD 확장 풀 vs 공개 RhythmFormer"),
         ("NF_MCDX_D_L", "CoTAF_TW_L", "CoTAF_TW_L", "SpecFP + 배가 + MCD 확장 풀 vs 이전 공식 PhysFormer"),
         ("RF_MCDX_D_L", "CoTAF_TW_L", "CoTAF_TW_L", "RhythmFormer + 배가 + MCD 확장 풀 vs 이전 공식 PhysFormer"),
         ("RF_ENS_GP_L", "RFbase", "RhythmFormer", "RhythmFormer 라우팅 앙상블 + 공개 RF 고심박 폴백 vs 공개 RhythmFormer"),
         ("RF_ENS_GP_L", "CoTAF_TW_L", "CoTAF_TW_L", "RhythmFormer 라우팅 앙상블 + 공개 RF 고심박 폴백 vs 이전 공식 PhysFormer"),
         ("RF_TTA_D_L", "RFbase", "RhythmFormer", "RhythmFormer + 스펙트럼 + 배가 + 반속 TTA vs 공개 RhythmFormer"),
         ("NF_TTA_D_L", "FPbase", "FactorizePhys", "SpecFP + 배가 + 반속 TTA vs 공개 FactorizePhys"),
         ("RF_TTA_D_L", "CoTAF_TW_L", "CoTAF_TW_L", "RhythmFormer + 배가 + 반속 TTA vs 이전 공식 모델 PhysFormer CoTAF_TW_L"),
         ("NF_TTA_D_L", "CoTAF_TW_L", "CoTAF_TW_L", "SpecFP + 배가 + 반속 TTA vs 이전 공식 모델 PhysFormer CoTAF_TW_L"),
         ("RF_ENS_DTA_L", "RFbase", "RhythmFormer", "RhythmFormer 앙상블(배가 ⊕ 증강) vs 공개 RhythmFormer"),
         ("RF_ENS_G_L", "RFbase", "RhythmFormer", "RhythmFormer 라우팅 앙상블 vs 공개 RhythmFormer"),
         ("RF_SPEC_D_L", "RFbase", "RhythmFormer", "RhythmFormer + 스펙트럼 + 배가 vs 공개 RhythmFormer"),
         ("RF_SPEC_TAD25_L", "RFbase", "RhythmFormer", "RhythmFormer 절충 셀 vs 공개 RhythmFormer"),
         ("NF_HYB_D_L", "FPbase", "FactorizePhys", "SpecFP + 배가 vs 공개 FactorizePhys"),
         ("NF_ENS_DHRX_L", "FPbase", "FactorizePhys", "SpecFP 앙상블(배가 ⊕ 고심박 합성) vs 공개 FactorizePhys"),
         ("NF_ENS_G_L", "FPbase", "FactorizePhys", "SpecFP 라우팅 앙상블 vs 공개 FactorizePhys"),
         ("NF_HYB_D_TA2_L", "FPbase", "FactorizePhys", "SpecFP + 배가 + ×2 전용 증강 vs 공개 FactorizePhys"),
         ("NF_HRX_L", "FPbase", "FactorizePhys", "SpecFP + 극단 심박 합성 40편 vs 공개 FactorizePhys"),
         ("NF_HRX_D_L", "FPbase", "FactorizePhys", "SpecFP + 극단 심박 합성 + 배가 vs 공개 FactorizePhys"),
         ("NF_HRX2_D_L", "FPbase", "FactorizePhys", "SpecFP + 고심박 합성 2배 + 배가 vs 공개 FactorizePhys"),
         ("NF_HYB_TA_L", "FPbase", "FactorizePhys", "SpecFP + 배속 증강 vs 공개 FactorizePhys"),
         ("NF_HYB_D_HB_L", "FPbase", "FactorizePhys", "SpecFP + 배가 + 균형 표본화 vs 공개 FactorizePhys"),
         ("NF_HYB_U8_L", "FPbase", "FactorizePhys", "SpecFP(개입 없음, u8 대조) vs 공개 FactorizePhys"),
         ("RF_SPEC_TA_L", "RFbase", "RhythmFormer", "RhythmFormer + 스펙트럼 + 증강 vs 공개 RhythmFormer"),
         ("RF_SPEC_TAD25_HB_L", "RFbase", "RhythmFormer", "RhythmFormer 절충 + 균형 표본화 vs 공개 RhythmFormer"),
         ("RF_SPEC_L", "RFbase", "RhythmFormer", "RhythmFormer + 스펙트럼(개입 없음) vs 공개 RhythmFormer"),
         ("RF_POOL_L", "RFbase", "RhythmFormer", "RhythmFormer 풀 학습(고유 손실) vs 공개 RhythmFormer"),
         ("PM_SPEC_W20_D_L", "PMbase", "PMbase", "PhysMamba + 스펙트럼 2.0 + 배가 vs 공개 PhysMamba"),
         ("RF_ENS_DTA_L", "CoTAF_TW_L", "CoTAF_TW_L", "RhythmFormer 앙상블 vs 이전 공식 모델 PhysFormer CoTAF_TW_L"),
         ("NF_HYB_D_L", "CoTAF_TW_L", "CoTAF_TW_L", "SpecFP + 배가 vs 이전 공식 모델 PhysFormer CoTAF_TW_L"),
         ("RF_ENS_G_L", "CoTAF_TW_L", "CoTAF_TW_L", "RhythmFormer 라우팅 앙상블 vs 이전 공식 모델 PhysFormer CoTAF_TW_L"),
         ("NF_HYB_D_TA2_L", "CoTAF_TW_L", "CoTAF_TW_L", "SpecFP + 배가 + ×2 증강 vs 이전 공식 모델 PhysFormer CoTAF_TW_L")]
LINES, OUT = [], {}
def out(s): print(s); LINES.append(s)
lo = lambda r: r["gt_lit"] < 60; hi = lambda r: r["gt_lit"] >= 120; v7 = lambda r: r.get("scenario") == "v7"
for cell, bv, bm, desc in PAIRS:
    if not files(cell, "_VIPL"): out(f"\n=== {desc}: {cell} 없음(건너뜀)"); continue
    pl = lambda m, vn, mn, s1, s2: {**vipl(vn, s1), **mmpd(mn, s2)}
    tests = [("VIPL <60 bpm", cmp(vipl(cell, lo), vipl(bv, lo))), ("VIPL ≥120 bpm", cmp(vipl(cell, hi), vipl(bv, hi))), ("VIPL 운동 후(v7)", cmp(vipl(cell, v7), vipl(bv, v7))),
             ("풀 저심박(VIPL <60 ∪ MMPD <60)", cmp({**vipl(cell, lo), **mmpd(cell, lambda r: r["gt"] < 60)}, {**vipl(bv, lo), **mmpd(bm, lambda r: r["gt"] < 60)})),
             ("풀 고심박(VIPL ≥120 ∪ MMPD ≥110)", cmp({**vipl(cell, hi), **mmpd(cell, lambda r: r["gt"] >= 110)}, {**vipl(bv, hi), **mmpd(bm, lambda r: r["gt"] >= 110)}))]
    ps = holm([t[1]["p"] if t[1] else 1.0 for t in tests]); ok = 0; res = {}
    out(f"\n=== {desc} (셀 시드 {len(files(cell, '_VIPL'))}) — Holm 5")
    for (nm, r), ph in zip(tests, ps):
        if r is None: out(f"  {nm:34s} (missing)"); res[nm] = None; continue
        v = "보장(우세)" if (r["d"] < 0 and ph < .05) else ("열세" if (r["d"] > 0 and ph < .05) else "ns"); ok += v == "보장(우세)"
        out(f"  {nm:34s} n={r['n']:5d} {r['b']:6.2f} -> {r['a']:6.2f}  Δ{r['d']:+6.2f} p={r['p']:.2e} Holm={ph:.2e} → {v}"); res[nm] = dict(r, holm=ph, verdict=v)
    nip = [ni(vipl(cell, hi), vipl(bv, hi)), ni({**vipl(cell, hi), **mmpd(cell, lambda r: r["gt"] >= 110)}, {**vipl(bv, hi), **mmpd(bm, lambda r: r["gt"] >= 110)})]
    nih = holm([x if x is not None else 1.0 for x in nip]); res["NI"] = dict(vipl_hi_p=nip[0], pooled_hi_p=nip[1], holm=nih, both=bool(max(nih) < .05))
    out(f"  고심박 비열등(여유 2 bpm, 단측, Holm 2): VIPL ≥120 p={nip[0]} · 풀 고심박 p={nip[1]} → {'비열등 확인' if max(nih) < .05 else '비열등 미확인'}")
    bad = sum(1 for v in res.values() if v and isinstance(v, dict) and v.get("verdict") == "열세")
    out(f"  판정: {ok}/5 보장, 열세 {bad} → {'저·고심박 모두 기존 대비 보장' if ok == 5 else ('보장 미달·열세 없음' if bad == 0 else '미달(열세 있음)')}"); OUT[f"{cell}|{bv}"] = dict(desc=desc, tests=res, n_ok=ok, n_bad=bad)
open(f"{R}/hr_vs_public_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/hr_vs_public_judge.json", "w"), indent=1, ensure_ascii=False)
print(f"\n-> {R}/hr_vs_public_judge.txt / .json")
