#!/usr/bin/env python3
"""[61 문서 §9, 사전 등록] 잡음 스펙트럼 효과 최대화 — 배제 가중 3배(NF_SKREF_X3_L)·머리카락·옷 참조(NF_SKREF_HC_L) vs 같은 시드 NF_SKREF_REP_L.
(아래 공용 도움 함수는 skref_rep_judge.py 와 같다) 원 설명: 피부 가중 + 잡음 참조 배제의 확인 재현 — NF_SKREF_REP_L(새 시드 600~1000) vs 같은 시드의 채택 셀 재현 NF_REP_D_TA2_L(체인 104).
1차 R1(단일 검정, α=.05): 풀 고심박(VIPL 장치 ≥120 ∪ MMPD ≥110) Δ<0 — 영상 단위 양측 Wilcoxon(시드별 오차의 영상 평균, skin_ref_judge 방식) + 방향.
핵심 2차(Holm 2): 공개 RhythmFormer 대비 풀 고심박 비열등(여유 2 bpm, 단측) — R2 영상 구간 PPG 파형 정답(시드 평균 예측, vipl_wave_gt_judge 방식), R3 장치 정답(시드별 오차 평균).
가드(같은 시드 기준 대비): VIPL 전체·MMPD 전체·VIPL <60 Δ ≤ +0.3. 서술: 하위 대역·MR-NIRP 3축·RTNI·PURE, 공개 FactorizePhys 대비 저·고심박과 6축, 시드별 짝 차이, 10시드(100~1000) 합산.
판정은 새 시드만 쓴다(시드 100~500 은 서술용 10시드 합산에만).
  python scripts/skref_max_judge.py → results/skref_max_judge.{txt,json}"""
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

from scipy.stats import ttest_1samp
def ni_mean_p(a, b, m=2.0):   # [§9 P2] 평균 기준 비열등: 영상별 (셀 − 기존) − m 의 평균 < 0 단측 대응 t 검정
    k = sorted(set(a) & set(b)); d = np.array([a[i] - b[i] for i in k]) - m
    return float(ttest_1samp(d, 0.0, alternative="less").pvalue), float(d.mean() + m), len(k)
BASE = "NF_SKREF_REP_L"; ARMS = [("NF_SKREF_X3_L", "배제 가중 3배"), ("NF_SKREF_HC_L", "머리카락·옷 참조 추가")]
BV, BM = vipl(BASE), mmpd(BASE); RFV, RFM = vipl("RFbase"), mmpd("RhythmFormer"); FPV, FPM = vipl("FPbase"), mmpd("FactorizePhys")
p0, m0, n0 = ni_mean_p(hi_dev(BV, BM), hi_dev(RFV, RFM)); out(f"=== 61 §9 판정 — 기준 {BASE} 시드 {seeds(BASE)}; 기준 자신의 P2(평균 기준 비열등) p={p0:.4f}, 평균 차 {m0:+.2f} (n={n0})")
fam = []
for arm, desc in ARMS:
    s = seeds(arm); V, M = vipl(arm), mmpd(arm)
    if not V or not M: out(f"  {arm}: 결과 없음"); continue
    if s != [600, 700, 800, 900, 1000]: out(f"  !! {arm} 시드 {s} — 사전 등록 600~1000 과 다름(부분 결과)")
    r1 = cmp(hi_dev(V, M), hi_dev(BV, BM)); p2, md, n2 = ni_mean_p(hi_dev(V, M), hi_dev(RFV, RFM))
    fam += [(arm, "P1", r1["p"] if r1 and r1["d"] < 0 else 1.0, r1), (arm, "P2", p2, dict(mean_diff=md, n=n2))]
hs = holm([f[2] for f in fam]); res = {}
for (arm, t, p, r), h in zip(fam, hs): res.setdefault(arm, {})[t] = dict(r or {}, p_dir=p, holm=h, ok=bool(h < .05))
for arm, desc in ARMS:
    if arm not in res: continue
    V, M = vipl(arm), mmpd(arm); R1 = res[arm]["P1"]; R2 = res[arm]["P2"]
    out(f"\n=== {arm} ({desc}; 시드 {seeds(arm)}) vs {BASE} — 1차 가족 Holm 4")
    out(f"  P1 풀 고심박(장치)            n={R1['n']:3d} {R1['base']:6.2f} -> {R1['cell']:6.2f} Δ{R1['d']:+6.2f} p={R1['p']:.4f} Holm={R1['holm']:.4f} → {'지지' if R1['ok'] else 'ns'}")
    out(f"  P2 공개 RF 대비 평균 기준 비열등(장치) n={R2['n']:3d} 평균 차 {R2['mean_diff']:+.2f} (여유 2) p={R2['p_dir']:.4f} Holm={R2['holm']:.4f} → {'✓' if R2['ok'] else '✗'}")
    gok = True
    for nm, a, b in (("VIPL 전체", {k: v["e"] for k, v in V.items()}, {k: v["e"] for k, v in BV.items()}), ("MMPD 전체", {k: v["e"] for k, v in M.items()}, {k: v["e"] for k, v in BM.items()}),
                     ("VIPL <60", {k: v["e"] for k, v in V.items() if v["dev"] < 60}, {k: v["e"] for k, v in BV.items() if v["dev"] < 60}), ("풀 저심박", lo_dev(V, M), lo_dev(BV, BM))):
        r = cmp(a, b); v = "위반" if r["d"] > 0.3 else "유지"; gok &= v == "유지"; line("가드 " + nm, r, f" → {v}"); res[arm][f"guard {nm}"] = dict(r, verdict=v)
    res[arm]["guards_ok"] = bool(gok)
    for nm, a, b in (("MMPD ≥110", {k: v["e"] for k, v in M.items() if v["gt"] >= 110}, {k: v["e"] for k, v in BM.items() if v["gt"] >= 110}),
                     ("VIPL 장치 ≥120", {k: v["e"] for k, v in V.items() if v["dev"] >= 120}, {k: v["e"] for k, v in BV.items() if v["dev"] >= 120}),
                     ("풀 고심박(파형)", hi_wave(V, M), hi_wave(BV, BM)), ("MMPD 보행", {k: v["e"] for k, v in M.items() if v["mo"] == 4}, {k: v["e"] for k, v in BM.items() if v["mo"] == 4})):
        r = cmp(a, b); line("(서술) " + nm, r); res[arm][f"desc {nm}"] = r
    for ax in ("PURE", "NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE", "DLCN", "MCD_A", "MCD_B", "RTNI", "RTNC"):
        r = cmp(axis(arm, ax), axis(BASE, ax)); line("(서술) " + ax, r); res[arm][f"axis {ax}"] = r
    half = sum(1 for k, v in M.items() if v["gt"] >= 110 and abs(v["p"] / v["gt"] - 0.5) < 0.12); half0 = sum(1 for k, v in BM.items() if v["gt"] >= 110 and abs(v["p"] / v["gt"] - 0.5) < 0.12)
    pw, mw, nw = ni_mean_p(hi_wave(V, M), hi_wave(RFV, RFM)); pf, mf, nf = ni_mean_p(hi_dev(V, M), hi_dev(FPV, FPM))
    out(f"  (서술) MMPD ≥110 반감 오판 {half0} -> {half} · 공개 RF 대비 평균 기준 비열등(파형) 평균 차 {mw:+.2f} p={pw:.4f} · 공개 FP 대비(장치) 평균 차 {mf:+.2f} p={pf:.4f}")
    dec = R1["ok"] and gok; res[arm]["decision"] = dict(adopt=bool(dec), mean_ni=bool(R2["ok"]))
    out(f"  판정: P1 {'지지' if R1['ok'] else 'ns'} · 가드 {'유지' if gok else '위반'} · P2 {'✓' if R2['ok'] else '✗'} → {'권장 레시피에 추가' if dec else '추가하지 않음'}")
OUT.update(res)
open(f"{R}/skref_max_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/skref_max_judge.json", "w"), indent=1, ensure_ascii=False, default=float)
