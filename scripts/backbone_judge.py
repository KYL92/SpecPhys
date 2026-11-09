#!/usr/bin/env python3
"""[체인 74, 57 문서] 백본 일치 비교 + MMPD 문헌 분할 판정 — 영상 단위 쌍대 Wilcoxon, 시드(""/_s200/_s300) 영상별 평균 풀링, 가족별 Holm (nf_judge.py 규약).
  python scripts/backbone_judge.py   → 표 + results/backbone_judge.{txt,json}
가족 B1: SpecFP(NF_HYB_L) vs 같은 풀 RhythmFormer(RF_POOL_L)·PhysMamba(PM_POOL_L) × {VIPL 전체(문헌 GT), MMPD 7~33}  (Holm 4; Δ=SpecFP−백본 <0 → 지지)
가족 B2: 풀 학습 vs 공개 가중치 — RF_POOL_L vs RFbase(VIPL)/RhythmFormer(MMPD), PM_POOL_L vs PMbase × {VIPL, MMPD}  (Holm 4; Δ<0 → 지지)
가족 B3: MMPD 문헌 분할(테스트 27~33, in-domain) — H-B3a ML_NF_L vs ML_NFO_L(양측) | H-B3b ML_NF_L vs ML_BASE_L(비열등 Δ≤+0.3 / 우세 Δ<0) | H-B3c ML_NF_L vs zero-shot NF_HYB_L 같은 140 영상  (Holm 3)
서술: 2차 축(MR-NIRP 소/대동작·RTNI·PURE held-out·DLCN·MCD·RTNC), 같은 풀 PhysFormer(CoTAM_L)·FactorizePhys 분해 손실(FP_OFF_HYB), 청크 길이 점검 PM_POOL_L(160) vs PM_CoTAM_L(128) MMPD, ML_NFO_L 의 B3b/B3c 대응."""
import json, os, sys
import numpy as np
import glob, re
from scipy.stats import wilcoxon
R = "/SPECPHYS_ROOT/_workspace/data/results"
# nf_judge.py 의 보조 함수 사본(nf_judge 는 import 시 본문을 실행해 results/nf_judge.* 를 덮어쓰므로 import 하지 않는다)
def files(m, suffix):
    pat = re.compile(rf"agg_{re.escape(m)}_s\d+{re.escape(suffix)}\.json$")
    return [f for f in [f"{R}/agg_{m}{suffix}.json"] + sorted(x for x in glob.glob(f"{R}/agg_{m}_s[0-9]*{suffix}.json") if pat.search(os.path.basename(x))) if os.path.exists(f)]
def vipl(m):
    acc, meta = {}, {}
    for f in files(m, "_VIPL"):
        rows = json.load(open(f)); rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
        for r in rows:
            if r.get("err_lit") is None: continue
            acc.setdefault(r["vid"], []).append(r["err_lit"]); meta[r["vid"]] = (r.get("scenario"), r.get("gt_lit"))
    return {k: (float(np.mean(v)),) + meta[k] for k, v in acc.items()}, len(files(m, "_VIPL"))
def axis(m, ax):
    acc = {}
    for f in files(m, ""):
        d = json.load(open(f)); d = (d.get(ax) or []) if isinstance(d, dict) else []
        for r in d: acc.setdefault(r["vid"], []).append(r["err"])
    return {k: (float(np.mean(v)), None, None) for k, v in acc.items()}
def mmpd(m, subs=set(range(7, 34))):
    acc = {}
    for f in files(m, "_MMPD"):
        for r in json.load(open(f)):
            if r["subject"] in subs: acc.setdefault((r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]), []).append(r["err"])
    return {k: (float(np.mean(v)), None, None) for k, v in acc.items()}
def cmp(a, b, sel=lambda k, v: True):
    ks = [k for k in a if k in b and sel(k, a[k])]
    if len(ks) < 5: return None
    x = np.array([a[k][0] for k in ks]); y = np.array([b[k][0] for k in ks])
    try: p = float(wilcoxon(x, y).pvalue)
    except ValueError: p = 1.0
    return {"n": len(ks), "a": float(x.mean()), "b": float(y.mean()), "d": float(x.mean() - y.mean()), "p": p}
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [None] * len(ps); m = len(ps); run = 0.0
    for rank, i in enumerate(order): run = max(run, min(1.0, (m - rank) * ps[i])); out[i] = run
    return out

SEC = [("MR-NIRP small", "NIRP_SMALL", None), ("MR-NIRP large", "NIRP_LARGE", None), ("MR-NIRP still", "NIRP_STILL", None), ("RTNI flicker", "RTNI", None), ("RTNC", "RTNC", None),
       ("PURE held-out 07-10", "PURE", lambda k, v: k >= 701), ("DLCN", "DLCN", None), ("MCD A", "MCD_A", None), ("MCD B", "MCD_B", None)]
LINES = []
def head(s): print(s); LINES.append(s)
def line(name, r, flag=""):
    s = f"  {name:64s} (missing)" if r is None else f"  {name:64s} n={r['n']:5d} {r['b']:6.2f} -> {r['a']:6.2f}  Δ{r['d']:+6.2f} p={r['p']:.4f}{flag}"
    print(s); LINES.append(s)
def mae(d): return (float(np.mean([v[0] for v in d.values()])), len(d)) if d else (None, 0)
def nseeds(m, suffix): return len(files(m, suffix))

out = {"B1": {}, "B2": {}, "B3": {}, "secondary": {}, "table": {}}
V = {m: vipl(m)[0] for m in ["NF_HYB_L", "RF_POOL_L", "PM_POOL_L", "RFbase", "PMbase", "CoTAM_L", "FP_OFF_HYB", "FPbase", "PFbase", "CoTAF_TW_L", "POS"]}
M = {m: mmpd(m) for m in ["NF_HYB_L", "RF_POOL_L", "PM_POOL_L", "RhythmFormer", "PMbase", "CoTAM_L", "FP_OFF_HYB", "FactorizePhys", "CoTAF_TW_L", "PM_CoTAM_L", "POS"]}
head("=== 표: 영상 단위 MAE(시드 영상별 평균 풀링) — VIPL 전체(문헌 GT) / MMPD held-out 7~33 / 2차 축")
for m in ["RFbase", "RF_POOL_L", "PMbase", "PM_POOL_L", "PFbase", "CoTAM_L", "FPbase", "FP_OFF_HYB", "NF_HYB_L", "CoTAF_TW_L", "POS"]:
    mv, nv = mae(V.get(m, {})); mm, nm = mae(M.get(m, {}) if m not in ("PFbase", "FPbase") else M.get({"PFbase": "PFbase", "FPbase": "FactorizePhys"}[m], {}))
    sec = {ax: mae({k: v for k, v in axis(m, ax).items() if sel is None or sel(k, v)}) for _, ax, sel in SEC}
    out["table"][m] = {"VIPL": mv, "n_VIPL": nv, "seeds_VIPL": nseeds(m, "_VIPL"), "MMPD": mm, "n_MMPD": nm, "seeds_MMPD": nseeds(m, "_MMPD"), **{ax: s[0] for ax, s in sec.items()}}
    head(f"  {m:12s} VIPL {('%.2f' % mv) if mv is not None else '  -  '} (n={nv}, seeds {nseeds(m, '_VIPL')}) | MMPD {('%.2f' % mm) if mm is not None else '  -  '} (n={nm}) | " + " ".join(f"{ax}={('%.2f' % s[0]) if s[0] is not None else '-'}" for ax, s in sec.items()))

head("\n=== 가족 B1: SpecFP(NF_HYB_L) vs 같은 풀 다른 백본 (Δ = SpecFP − 백본; <0 이고 Holm p<.05 → 지지)")
tests = []
for bb in ["RF_POOL_L", "PM_POOL_L"]:
    tests.append((f"H-B1 VIPL 전체: SpecFP vs {bb}", cmp(V["NF_HYB_L"], V[bb]))); tests.append((f"H-B1 MMPD 7~33: SpecFP vs {bb}", cmp(M["NF_HYB_L"], M[bb])))
ps = holm([t[1]["p"] if t[1] else 1.0 for t in tests])
for (nm, r), ph in zip(tests, ps):
    verdict = "지지" if (r and r["d"] < 0 and ph < .05) else ("기각(역방향)" if (r and r["d"] > 0 and ph < .05) else "ns")
    line(nm, r, f" Holm={ph:.4f} → {verdict}" if r else ""); out["B1"][nm] = dict(r or {}, holm=ph, verdict=verdict)
head("  2차(서술, 비조정 p): SpecFP vs 같은 풀 백본·분해 손실 FactorizePhys·PhysFormer 고유 손실")
for bb in ["RF_POOL_L", "PM_POOL_L", "CoTAM_L", "FP_OFF_HYB"]:
    for nm, ax, sel in SEC:
        r = cmp(axis("NF_HYB_L", ax), axis(bb, ax), sel or (lambda k, v: True)); line(f"  {nm}: SpecFP vs {bb}", r); out["secondary"][f"{nm}|{bb}"] = r
    for nm, ref in (("VIPL", V), ("MMPD 7~33", M)):
        if bb in ("CoTAM_L", "FP_OFF_HYB"): r = cmp(ref["NF_HYB_L"], ref[bb]); line(f"  {nm}: SpecFP vs {bb}", r); out["secondary"][f"{nm}|{bb}"] = r

head("\n=== 가족 B2: 풀 학습 vs 공개 가중치 (Δ = 풀 − 공개; <0 이고 Holm p<.05 → 지지)")
tests = [("H-B2 VIPL: RF_POOL_L vs RFbase(공개 RhythmFormer)", cmp(V["RF_POOL_L"], V["RFbase"])), ("H-B2 MMPD 7~33: RF_POOL_L vs 공개 RhythmFormer", cmp(M["RF_POOL_L"], M["RhythmFormer"])),
         ("H-B2 VIPL: PM_POOL_L vs PMbase(공개 PhysMamba)", cmp(V["PM_POOL_L"], V["PMbase"])), ("H-B2 MMPD 7~33: PM_POOL_L vs PMbase", cmp(M["PM_POOL_L"], M["PMbase"]))]
ps = holm([t[1]["p"] if t[1] else 1.0 for t in tests])
for (nm, r), ph in zip(tests, ps):
    verdict = "지지" if (r and r["d"] < 0 and ph < .05) else ("기각(역방향)" if (r and r["d"] > 0 and ph < .05) else "ns")
    line(nm, r, f" Holm={ph:.4f} → {verdict}" if r else ""); out["B2"][nm] = dict(r or {}, holm=ph, verdict=verdict)
head("  서술: 같은 풀 PhysMamba 청크 길이 점검 — PM_POOL_L(청크 160, u8) vs PM_CoTAM_L(청크 128, float) MMPD 7~33")
r = cmp(M["PM_POOL_L"], M["PM_CoTAM_L"]); line("  MMPD 7~33: PM_POOL_L vs PM_CoTAM_L", r); out["secondary"]["chunk_check"] = r
r = cmp(V["CoTAM_L"], V["PFbase"]); line("  (참고) VIPL: CoTAM_L(PhysFormer 풀) vs PFbase(공개 PhysFormer)", r); out["secondary"]["PF_pool_vs_public_VIPL"] = r

head("\n=== 가족 B3: MMPD 문헌 분할(학습 1~23, 테스트 27~33, n=140; in-domain — zero-shot 표와 분리)")
S27 = set(range(27, 34)); L = {m: mmpd(m, S27) for m in ["ML_NF_L", "ML_NFO_L", "ML_BASE_L", "ML_HYB_L", "NF_HYB_L", "CoTAF_TW_L", "RhythmFormer", "FactorizePhys", "POS", "RF_POOL_L", "PM_POOL_L"]}
for m, d in L.items():
    v, n = mae(d); head(f"  {m:12s} MAE {('%.2f' % v) if v is not None else '  -  '} (n={n}, seeds {nseeds(m, '_MMPD')})"); out["B3"][f"mae_{m}"] = v
tests = [("H-B3a ML_NF_L(2단계, MMPD+풀) vs ML_NFO_L(MMPD 단독 SpecFP) [양측]", cmp(L["ML_NF_L"], L["ML_NFO_L"])),
         ("H-B3b ML_NF_L vs ML_BASE_L(PhysFormer MMPD 단독 5.68) [비열등 Δ≤+0.3 / 우세 Δ<0]", cmp(L["ML_NF_L"], L["ML_BASE_L"])),
         ("H-B3c ML_NF_L vs zero-shot NF_HYB_L 같은 140 영상 [Δ<0]", cmp(L["ML_NF_L"], L["NF_HYB_L"]))]
ps = holm([t[1]["p"] if t[1] else 1.0 for t in tests])
for i, ((nm, r), ph) in enumerate(zip(tests, ps)):
    if r is None: verdict = "missing"
    elif i == 0: verdict = ("코퍼스 도움" if r["d"] < 0 else "코퍼스 역방향") if ph < .05 else "ns"
    elif i == 1: verdict = ("우세" if (r["d"] < 0 and ph < .05) else ("비열등" if r["d"] <= 0.3 else "기각(열세)"))
    else: verdict = "지지" if (r["d"] < 0 and ph < .05) else "기각"
    line(nm, r, f" Holm={ph:.4f} → {verdict}" if r else ""); out["B3"][nm] = dict(r or {}, holm=ph, verdict=verdict)
head("  서술(비조정): MMPD 단독 SpecFP·PhysFormer 참고 셀·zero-shot 비교")
for nm, a, b in [("ML_NFO_L vs ML_BASE_L", "ML_NFO_L", "ML_BASE_L"), ("ML_NFO_L vs zero-shot NF_HYB_L", "ML_NFO_L", "NF_HYB_L"), ("ML_HYB_L(PhysFormer+코퍼스+분해) vs ML_BASE_L", "ML_HYB_L", "ML_BASE_L"),
                 ("zero-shot NF_HYB_L vs 공개 RhythmFormer (27~33)", "NF_HYB_L", "RhythmFormer"), ("zero-shot RF_POOL_L vs NF_HYB_L (27~33)", "RF_POOL_L", "NF_HYB_L"), ("zero-shot PM_POOL_L vs NF_HYB_L (27~33)", "PM_POOL_L", "NF_HYB_L")]:
    r = cmp(L[a], L[b]); line(f"  {nm}", r); out["B3"][f"desc|{nm}"] = r
head("  문헌 참고(다른 readout·분할, 검정 없음): RhythmFormer 논문 MMPD 인트라 MAE — PhysFormer 11.99 · PhysFormer++ 4.88 · RhythmFormer 3.07")

os.makedirs(R, exist_ok=True); open(f"{R}/backbone_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(out, open(f"{R}/backbone_judge.json", "w"), indent=1, ensure_ascii=False, default=float)
print(f"\n-> {R}/backbone_judge.txt / .json")
