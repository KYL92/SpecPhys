#!/usr/bin/env python3
"""[체인 81·82, 60 문서] 심박 커버리지(시간 배속 증강) 판정 — 영상 단위 쌍대 Wilcoxon, 시드 영상별 평균 풀링, 백본별 가족 Holm.
  python scripts/hrcov_judge.py → 표 + results/hrcov_judge.{txt,json}
쌍: RF_SPEC_TA_L vs RF_SPEC_L | [체인 87·90·91] RF_SPEC_D_L·PM_SPEC_W20_D_L(배가 항) vs 스펙트럼 기준, RF_SPEC_TAD_L(증강+배가, 5시드) vs RF_SPEC_L/TA/D | PM_SPEC_W20_TA_L vs PM_SPEC_W20_L | NF_HYB_TA_L vs NF_HYB_L (+ 서술 NF_HYB_NOSUP_L) | 대조 RF_POOL_TA_L vs RF_POOL_L, PM_POOL_TA_L vs PM_POOL_L
가족 L+H(Holm 4): VIPL <60 · MR-NIRP gt<65(3축 합) · VIPL ≥120 · VIPL v7 | 가드 G: VIPL 전체·MMPD 7~33 Δ≤+0.3 | 종합 S: TA 셀 vs 공개 가중치 6축(Holm 6)."""
import glob, json, os, re
import numpy as np
from scipy.stats import wilcoxon
import sys as _sys; _sys.path.insert(0, "/SPECPHYS_ROOT/scripts"); import mmpd_common_gt as _MGC   # [61 §9.3] MMPD 공통 정답
R = "/SPECPHYS_ROOT/_workspace/data/results"
def files(m, suffix):
    pat = re.compile(rf"agg_{re.escape(m)}_s\d+{re.escape(suffix)}\.json$")
    return [f for f in [f"{R}/agg_{m}{suffix}.json"] + sorted(x for x in glob.glob(f"{R}/agg_{m}_s[0-9]*{suffix}.json") if pat.search(os.path.basename(x))) if os.path.exists(f)]
def vipl(m, sel=None):
    acc = {}
    for f in files(m, "_VIPL"):
        rows = json.load(open(f)); rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
        for r in rows:
            if r.get("err_lit") is None or (sel and not sel(r)): continue
            acc.setdefault(r["vid"], []).append(r["err_lit"])
    return {k: float(np.mean(v)) for k, v in acc.items()}
def axis(m, ax, sel=None):
    acc = {}
    for f in files(m, ""):
        d = json.load(open(f)); d = (d.get(ax) or []) if isinstance(d, dict) else []
        for r in d:
            if sel and not sel(r): continue
            acc.setdefault((ax, r["vid"]), []).append(r["err"])
    return {k: float(np.mean(v)) for k, v in acc.items()}
def nirp_low(m):
    out = {}
    for ax in ("NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE"): out.update(axis(m, ax, lambda r: r["gt"] < 65))
    return out
def mmpd_sel(m, sel):
    acc = {}
    for f in files(m, "_MMPD"):
        for r in _MGC.load(f):
            if 7 <= r["subject"] <= 33 and sel(r): acc.setdefault(("M", r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]), []).append(r["err"])
    return {k: float(np.mean(v)) for k, v in acc.items()}
def pooled(m, lo_hi):   # [§14] 고심박 풀 = VIPL ≥120 ∪ MMPD 7~33 gt≥110 (n≈71) / 저심박 풀 = VIPL <60 ∪ MMPD gt<60 (n≈126)
    if lo_hi == "hi": d = dict(vipl(m, lambda r: r["gt_lit"] >= 120)); d.update(mmpd_sel(m, lambda r: r["gt"] >= 110))
    else: d = dict(vipl(m, lambda r: r["gt_lit"] < 60)); d.update(mmpd_sel(m, lambda r: r["gt"] < 60))
    return d
def mmpd(m):
    acc = {}
    for f in files(m, "_MMPD"):
        for r in _MGC.load(f):
            if 7 <= r["subject"] <= 33: acc.setdefault((r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]), []).append(r["err"])
    return {k: float(np.mean(v)) for k, v in acc.items()}
def cmp(a, b):
    ks = [k for k in a if k in b]
    if len(ks) < 5: return None
    x = np.array([a[k] for k in ks]); y = np.array([b[k] for k in ks])
    try: p = float(wilcoxon(x, y).pvalue)
    except ValueError: p = 1.0
    if not np.isfinite(p): p = 1.0   # [11-03 수정] 차이가 전부 0(복사된 실 축 등)이면 wilcoxon 이 NaN → Holm 순서·보정 전체를 오염(NaN 이 min/max 비교를 통과)
    return {"n": len(ks), "a": float(x.mean()), "b": float(y.mean()), "d": float(x.mean() - y.mean()), "p": p}
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [None] * len(ps); m = len(ps); run = 0.0
    for rank, i in enumerate(order): run = max(run, min(1.0, (m - rank) * ps[i])); out[i] = run
    return out
LINES = []; OUT = {}
def head(s): print(s); LINES.append(s)
def line(name, r, flag=""):
    s = f"  {name:58s} (missing)" if r is None else f"  {name:58s} n={r['n']:5d} {r['b']:6.2f} -> {r['a']:6.2f}  Δ{r['d']:+6.2f} p={r['p']:.4f}{flag}"
    print(s); LINES.append(s)
def mae(d): return float(np.mean(list(d.values()))) if d else None
def counts(m):
    hl = db = 0; k = len(files(m, "_VIPL"))
    for f in files(m, "_VIPL"):
        for r in json.load(open(f)):
            rt = r.get("ratio")
            if rt is None: continue
            hl += abs(rt - 0.5) < 0.1; db += abs(rt - 2.0) < 0.2
    return (hl / k, db / k) if k else (None, None)
PUBLIC = {"RF": ("RFbase", "RhythmFormer"), "PM": ("PMbase", "PMbase"), "NF": ("FPbase", "FactorizePhys"), "NF-TA25": ("FPbase", "FactorizePhys"), "NF-HRX": ("FPbase", "FactorizePhys"), "NF-HRX-D": ("FPbase", "FactorizePhys"), "NF-D": ("FPbase", "FactorizePhys"), "NF-U8ctrl vs float": ("FPbase", "FactorizePhys"), "NF-HRX2-D vs U8ctrl": ("FPbase", "FactorizePhys"), "RF-D": ("RFbase", "RhythmFormer"), "PM-D": ("PMbase", "PMbase"), "RF-TAD": ("RFbase", "RhythmFormer"), "RF-TAD25": ("RFbase", "RhythmFormer"), "RF-TAD-D1": ("RFbase", "RhythmFormer"), "NF-D-TA2 vs U8ctrl": ("FPbase", "FactorizePhys"), "RF-D-HW": ("RFbase", "RhythmFormer"), "RF-TAD25-HW": ("RFbase", "RhythmFormer"), "NF-D-HW vs U8ctrl": ("FPbase", "FactorizePhys"), "RF-D-HB": ("RFbase", "RhythmFormer"), "RF-TAD25-HB": ("RFbase", "RhythmFormer"), "NF-D-HB vs U8ctrl": ("FPbase", "FactorizePhys"), "RF-ENS": ("RFbase", "RhythmFormer"), "NF-ENS vs U8ctrl": ("FPbase", "FactorizePhys"), "RF-ENS-G": ("RFbase", "RhythmFormer"), "NF-ENS-G vs U8ctrl": ("FPbase", "FactorizePhys"), "RF-TTA": ("RFbase", "RhythmFormer"), "NF-TTA vs U8ctrl": ("FPbase", "FactorizePhys"), "RF-ENS-GP": ("RFbase", "RhythmFormer"), "RF-MCDX-D": ("RFbase", "RhythmFormer"), "NF-MCDX-D vs U8ctrl": ("FPbase", "FactorizePhys"), "NF-REP vs U8ctrl": ("FPbase", "FactorizePhys"), "NF-W4 vs W1": ("FPbase", "FactorizePhys"), "NF-W8 vs W1": ("FPbase", "FactorizePhys"), "NF-SKW vs TA2": ("FPbase", "FactorizePhys"), "NF-BGDC vs TA2": ("FPbase", "FactorizePhys"), "NF-SKBG vs TA2": ("FPbase", "FactorizePhys"), "NF-SKREF vs TA2": ("FPbase", "FactorizePhys"), "NF-BGRG vs TA2": ("FPbase", "FactorizePhys"), "NF-SKBGRG vs TA2": ("FPbase", "FactorizePhys"), "NF-SKREF-REP vs REP": ("FPbase", "FactorizePhys"), "NF-X15 vs SKREF-REP": ("FPbase", "FactorizePhys"), "NF-ENSAB vs SKREF-REP": ("FPbase", "FactorizePhys")}   # (VIPL 이름, MMPD/축 이름)
PAIRS = [("RF", "RF_SPEC_TA_L", "RF_SPEC_L"), ("PM", "PM_SPEC_W20_TA_L", "PM_SPEC_W20_L"), ("NF", "NF_HYB_TA_L", "NF_HYB_L"), ("NF-TA25", "NF_HYB_TA25_L", "NF_HYB_L"), ("NF-HRX", "NF_HRX_L", "NF_HYB_L"), ("NF-HRX-D", "NF_HRX_D_L", "NF_HYB_L"), ("NF-D", "NF_HYB_D_L", "NF_HYB_L"), ("NF-HRX-D vs HRX", "NF_HRX_D_L", "NF_HRX_L"), ("NF-HRX2-D vs U8ctrl", "NF_HRX2_D_L", "NF_HYB_U8_L"), ("NF-HRX2-D vs HRX-D", "NF_HRX2_D_L", "NF_HRX_D_L"), ("NF-U8ctrl vs float", "NF_HYB_U8_L", "NF_HYB_L"), ("NF-HRX-D vs U8ctrl", "NF_HRX_D_L", "NF_HYB_U8_L"), ("NF-D vs U8ctrl", "NF_HYB_D_L", "NF_HYB_U8_L"), ("NF-HRX vs U8ctrl", "NF_HRX_L", "NF_HYB_U8_L"), ("RF-D", "RF_SPEC_D_L", "RF_SPEC_L"), ("PM-D", "PM_SPEC_W20_D_L", "PM_SPEC_W20_L"), ("RF-TAD", "RF_SPEC_TAD_L", "RF_SPEC_L"), ("RF-TAD vs TA", "RF_SPEC_TAD_L", "RF_SPEC_TA_L"), ("RF-TAD vs D", "RF_SPEC_TAD_L", "RF_SPEC_D_L"), ("RF-TAD25", "RF_SPEC_TAD25_L", "RF_SPEC_L"), ("RF-TAD-D1", "RF_SPEC_TAD_D1_L", "RF_SPEC_L"), ("NF-D-TA2 vs U8ctrl", "NF_HYB_D_TA2_L", "NF_HYB_U8_L"), ("NF-D-TA2 vs D", "NF_HYB_D_TA2_L", "NF_HYB_D_L"), ("RF-D-HW", "RF_SPEC_D_HW_L", "RF_SPEC_L"), ("RF-D-HW vs D", "RF_SPEC_D_HW_L", "RF_SPEC_D_L"), ("RF-TAD25-HW", "RF_SPEC_TAD25_HW_L", "RF_SPEC_L"), ("RF-TAD25-HW vs TAD25", "RF_SPEC_TAD25_HW_L", "RF_SPEC_TAD25_L"), ("NF-D-HW vs U8ctrl", "NF_HYB_D_HW_L", "NF_HYB_U8_L"), ("NF-D-HW vs D", "NF_HYB_D_HW_L", "NF_HYB_D_L"), ("RF-D-HB", "RF_SPEC_D_HB_L", "RF_SPEC_L"), ("RF-D-HB vs D", "RF_SPEC_D_HB_L", "RF_SPEC_D_L"), ("RF-TAD25-HB", "RF_SPEC_TAD25_HB_L", "RF_SPEC_L"), ("RF-TAD25-HB vs TAD25", "RF_SPEC_TAD25_HB_L", "RF_SPEC_TAD25_L"), ("NF-D-HB vs U8ctrl", "NF_HYB_D_HB_L", "NF_HYB_U8_L"), ("NF-D-HB vs D", "NF_HYB_D_HB_L", "NF_HYB_D_L"), ("RF-ENS", "RF_ENS_DTA_L", "RF_SPEC_L"), ("RF-ENS vs D", "RF_ENS_DTA_L", "RF_SPEC_D_L"), ("RF-ENS vs TA", "RF_ENS_DTA_L", "RF_SPEC_TA_L"), ("NF-ENS vs U8ctrl", "NF_ENS_DHRX_L", "NF_HYB_U8_L"), ("NF-ENS vs D", "NF_ENS_DHRX_L", "NF_HYB_D_L"), ("RF-ENS-G", "RF_ENS_G_L", "RF_SPEC_L"), ("RF-ENS-G vs ENS", "RF_ENS_G_L", "RF_ENS_DTA_L"), ("NF-ENS-G vs U8ctrl", "NF_ENS_G_L", "NF_HYB_U8_L"), ("NF-ENS-G vs D", "NF_ENS_G_L", "NF_HYB_D_L"), ("RF-TTA", "RF_TTA_D_L", "RF_SPEC_L"), ("RF-TTA vs D", "RF_TTA_D_L", "RF_SPEC_D_L"), ("NF-TTA vs U8ctrl", "NF_TTA_D_L", "NF_HYB_U8_L"), ("NF-TTA vs D", "NF_TTA_D_L", "NF_HYB_D_L"), ("RF-ENS-GP", "RF_ENS_GP_L", "RF_SPEC_L"), ("RF-ENS-GP vs ENS-G", "RF_ENS_GP_L", "RF_ENS_G_L"), ("RF-MCDX-D", "RF_MCDX_D_L", "RF_SPEC_L"), ("RF-MCDX-D vs D", "RF_MCDX_D_L", "RF_SPEC_D_L"), ("NF-MCDX-D vs U8ctrl", "NF_MCDX_D_L", "NF_HYB_U8_L"), ("NF-MCDX-D vs D", "NF_MCDX_D_L", "NF_HYB_D_L"), ("NF-REP vs U8ctrl", "NF_REP_D_TA2_L", "NF_HYB_U8_L"), ("NF-W4 vs W1", "NF_W4_D_TA2_L", "NF_HYB_D_TA2_L"), ("NF-W8 vs W1", "NF_W8_D_TA2_L", "NF_HYB_D_TA2_L"), ("NF-SKW vs TA2", "NF_SKW_L", "NF_HYB_D_TA2_L"), ("NF-BGDC vs TA2", "NF_BGDC_L", "NF_HYB_D_TA2_L"), ("NF-SKBG vs TA2", "NF_SKBG_L", "NF_HYB_D_TA2_L"), ("NF-SKREF vs TA2", "NF_SKREF_L", "NF_HYB_D_TA2_L"), ("NF-BGRG vs TA2", "NF_BGRG_L", "NF_HYB_D_TA2_L"), ("NF-SKBGRG vs TA2", "NF_SKBGRG_L", "NF_HYB_D_TA2_L"), ("NF-SKREF-REP vs REP", "NF_SKREF_REP_L", "NF_REP_D_TA2_L"), ("NF-X15 vs SKREF-REP", "NF_SKREF_X15_L", "NF_SKREF_REP_L"), ("NF-ENSAB vs SKREF-REP", "NF_ENS_AB_L", "NF_SKREF_REP_L"), ("RF(고유)", "RF_POOL_TA_L", "RF_POOL_L"), ("PM(고유)", "PM_POOL_TA_L", "PM_POOL_L")]
head("=== 표: 셀별 MAE — VIPL 전체 / <60 / ≥120 / v7 운동 후 | MMPD 7~33 | MR-NIRP gt<65 (3축 합) / 정지 / 소동작 / 대동작 | RTNI | PURE 07~10")
for bb, ta, base in PAIRS:
    for m in (base, ta):
        v = vipl(m); row = [mae(v), mae(vipl(m, lambda r: r["gt_lit"] < 60)), mae(vipl(m, lambda r: r["gt_lit"] >= 120)), mae(vipl(m, lambda r: r.get("scenario") == "v7")), mae(mmpd(m)), mae(nirp_low(m)),
                            mae(axis(m, "NIRP_STILL")), mae(axis(m, "NIRP_SMALL")), mae(axis(m, "NIRP_LARGE")), mae(axis(m, "RTNI")), mae(axis(m, "PURE", lambda r: int(r["vid"]) >= 701))]
        OUT.setdefault("table", {})[m] = dict(zip(["VIPL", "VIPL<60", "VIPL>=120", "VIPL_v7", "MMPD", "NIRP_low", "NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE", "RTNI", "PURE_HO"], row), seeds=len(files(m, "_VIPL")))
        hl, db = counts(m); head(f"  {m:18s} (seeds {len(files(m, '_VIPL'))}) " + " ".join(f"{('%.2f' % x) if x is not None else '  -  ':>6s}" for x in row) + (f" | VIPL hlv {hl:.1f} dbl {db:.1f}" if hl is not None else ""))
for bb, ta, base in PAIRS:
    head(f"\n=== {bb}: {ta} vs {base} (Δ = TA − 기준)")
    tests = [("H-L1 VIPL <60 bpm", cmp(vipl(ta, lambda r: r["gt_lit"] < 60), vipl(base, lambda r: r["gt_lit"] < 60))), ("H-L1 MR-NIRP gt<65 (3축)", cmp(nirp_low(ta), nirp_low(base))),
             ("H-H1 VIPL ≥120 bpm", cmp(vipl(ta, lambda r: r["gt_lit"] >= 120), vipl(base, lambda r: r["gt_lit"] >= 120))), ("H-H1 VIPL v7 운동 후", cmp(vipl(ta, lambda r: r.get("scenario") == "v7"), vipl(base, lambda r: r.get("scenario") == "v7")))]
    ps = holm([t[1]["p"] if t[1] else 1.0 for t in tests]); fam = {}
    for (nm, r), ph in zip(tests, ps):
        v = "missing" if r is None else ("지지" if (r["d"] < 0 and ph < .05) else ("역방향" if (r["d"] > 0 and ph < .05) else "ns")); line(nm, r, f" Holm={ph:.4f} → {v}" if r else ""); fam[nm] = dict(r or {}, holm=ph, verdict=v)
    for nm, a, b in (("H-G 가드 VIPL 전체", vipl(ta), vipl(base)), ("H-G 가드 MMPD 7~33", mmpd(ta), mmpd(base))):
        r = cmp(a, b); v = "missing" if r is None else ("우세" if (r["d"] < 0 and r["p"] < .05) else ("비열등(Δ≤+0.3)" if r["d"] <= 0.3 else "가드 위반")); line(nm, r, f" → {v}" if r else ""); fam[nm] = dict(r or {}, verdict=v)
    for nm, ax in (("(서술) MR-NIRP 정지", "NIRP_STILL"), ("(서술) MR-NIRP 소동작", "NIRP_SMALL"), ("(서술) MR-NIRP 대동작", "NIRP_LARGE"), ("(서술) RTNI", "RTNI")):
        r = cmp(axis(ta, ax), axis(base, ax)); line(nm, r); fam[nm] = r
    r = cmp(vipl(ta, lambda r: 60 <= r["gt_lit"] < 90), vipl(base, lambda r: 60 <= r["gt_lit"] < 90)); line("(서술) VIPL 60–90 bpm(중간 대역 트레이드오프)", r); fam["VIPL 60-90"] = r
    pt = [("H-P 고심박 풀(VIPL ≥120 ∪ MMPD ≥110)", cmp(pooled(ta, "hi"), pooled(base, "hi"))), ("H-P 저심박 풀(VIPL <60 ∪ MMPD <60)", cmp(pooled(ta, "lo"), pooled(base, "lo")))]
    pps = holm([t[1]["p"] if t[1] else 1.0 for t in pt])
    for (nm, r), ph in zip(pt, pps):
        v = "missing" if r is None else ("지지" if (r["d"] < 0 and ph < .05) else ("역방향" if (r["d"] > 0 and ph < .05) else "ns")); line(nm, r, f" Holm={ph:.4f} → {v}" if r else ""); fam[nm] = dict(r or {}, holm=ph, verdict=v)
    OUT[bb] = fam
    key = bb if bb in PUBLIC else bb.split("(")[0]
    if key in PUBLIC and "(고유)" not in bb:
        pv, pm = PUBLIC[key]; head(f"  --- H-S 종합: {ta} vs 공개 가중치 ({pv}/{pm}), Holm 6")
        st = [("VIPL", cmp(vipl(ta), vipl(pv))), ("MMPD 7~33", cmp(mmpd(ta), mmpd(pm))), ("MR-NIRP 소동작", cmp(axis(ta, "NIRP_SMALL"), axis(pm, "NIRP_SMALL"))), ("MR-NIRP 대동작", cmp(axis(ta, "NIRP_LARGE"), axis(pm, "NIRP_LARGE"))), ("RTNI", cmp(axis(ta, "RTNI"), axis(pm, "RTNI"))), ("PURE 07~10", cmp(axis(ta, "PURE", lambda r: int(r["vid"]) >= 701), axis(pm, "PURE", lambda r: int(r["vid"]) >= 701)))]
        ps = holm([t[1]["p"] if t[1] else 1.0 for t in st]); wins = 0; ok = 0
        for (nm, r), ph in zip(st, ps):
            v = "missing" if r is None else ("우세" if (r["d"] < 0 and ph < .05) else ("비열등" if r["d"] <= 0.3 else "열세")); wins += v == "우세"; ok += v in ("우세", "비열등"); line("  " + nm, r, f" Holm={ph:.4f} → {v}" if r else ""); OUT[bb][f"S|{nm}"] = dict(r or {}, holm=ph, verdict=v)
        head(f"  H-S 판정: 우세 {wins}/6, 비열등 포함 {ok}/6 → {'지지(대부분 축 보장)' if wins >= 4 and ok == 6 else '기각'}"); OUT[bb]["S_verdict"] = dict(wins=wins, ok=ok)
r = cmp(vipl("NF_HYB_TA_L"), vipl("NF_HYB_NOSUP_L")); line("\n(서술) NF_HYB_TA_L vs NF_HYB_NOSUP_L(교란 감독 없음, 시드 1) VIPL", r)
open(f"{R}/hrcov_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/hrcov_judge.json", "w"), indent=1, ensure_ascii=False, default=float)
print(f"\n-> {R}/hrcov_judge.txt / .json")
