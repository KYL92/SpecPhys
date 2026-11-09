#!/usr/bin/env python3
"""[61 문서 §8, 사전 등록] 피부 가중 + 잡음 참조 배제의 확인 재현 — NF_SKREF_REP_L(새 시드 600~1000) vs 같은 시드의 채택 셀 재현 NF_REP_D_TA2_L(체인 104).
1차 R1(단일 검정, α=.05): 풀 고심박(VIPL 장치 ≥120 ∪ MMPD ≥110) Δ<0 — 영상 단위 양측 Wilcoxon(시드별 오차의 영상 평균, skin_ref_judge 방식) + 방향.
핵심 2차(Holm 2): 공개 RhythmFormer 대비 풀 고심박 비열등(여유 2 bpm, 단측) — R2 영상 구간 PPG 파형 정답(시드 평균 예측, vipl_wave_gt_judge 방식), R3 장치 정답(시드별 오차 평균).
가드(같은 시드 기준 대비): VIPL 전체·MMPD 전체·VIPL <60 Δ ≤ +0.3. 서술: 하위 대역·MR-NIRP 3축·RTNI·PURE, 공개 FactorizePhys 대비 저·고심박과 6축, 시드별 짝 차이, 10시드(100~1000) 합산.
판정은 새 시드만 쓴다(시드 100~500 은 서술용 10시드 합산에만).
  python scripts/skref_rep_judge.py → results/skref_rep_judge.{txt,json}"""
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
NEW, BASE = "NF_SKREF_REP_L", "NF_REP_D_TA2_L"
sn, sb = seeds(NEW), seeds(BASE); out(f"=== 61 §8 확인 재현: {NEW} 시드 {sn} vs {BASE} 시드 {sb}")
if sn != [600, 700, 800, 900, 1000] or sb != [600, 700, 800, 900, 1000]: out("  !! 시드 집합이 사전 등록(600~1000)과 다름 — 판정 보류(부분 결과)")
OUT["seeds"] = dict(new=sn, base=sb, complete=sn == sb == [600, 700, 800, 900, 1000])
V, M, BV, BM = vipl(NEW), mmpd(NEW), vipl(BASE), mmpd(BASE)
RFV, RFM, FPV, FPM = vipl("RFbase"), mmpd("RhythmFormer"), vipl("FPbase"), mmpd("FactorizePhys")
r1 = cmp(hi_dev(V, M), hi_dev(BV, BM)); ok1 = bool(r1 and r1["d"] < 0 and r1["p"] < .05)
out("--- 1차 R1(단일, α=.05): 풀 고심박(장치 정답) Δ<0"); line("R1 풀 고심박(VIPL ≥120 ∪ MMPD ≥110)", r1, f" → {'확인(지지)' if ok1 else ('역방향' if r1 and r1['d'] > 0 and r1['p'] < .05 else '미확인')}")
OUT["R1"] = dict(r1 or {}, confirmed=ok1)
a2, b2 = hi_wave(V, M), hi_wave(RFV, RFM); a3, b3 = hi_dev(V, M), hi_dev(RFV, RFM)
p2, p3 = ni_p(a2, b2), ni_p(a3, b3); h2 = holm([p2, p3])
k2 = sorted(set(a2) & set(b2)); k3 = sorted(set(a3) & set(b3))
out("--- 핵심 2차(Holm 2): 공개 RhythmFormer 대비 풀 고심박 비열등(2 bpm, 단측)")
out(f"  R2 파형 정답  n={len(k2):3d} 공개 RF {np.mean([b2[i] for i in k2]):6.2f} -> 셀 {np.mean([a2[i] for i in k2]):6.2f}  p={p2:.4f} Holm={h2[0]:.4f} → {'비열등 ✓' if h2[0] < .05 else '✗'}")
out(f"  R3 장치 정답  n={len(k3):3d} 공개 RF {np.mean([b3[i] for i in k3]):6.2f} -> 셀 {np.mean([a3[i] for i in k3]):6.2f}  p={p3:.4f} Holm={h2[1]:.4f} → {'비열등 ✓' if h2[1] < .05 else '✗'}")
OUT["R2"] = dict(n=len(k2), p=p2, holm=h2[0], ok=bool(h2[0] < .05)); OUT["R3"] = dict(n=len(k3), p=p3, holm=h2[1], ok=bool(h2[1] < .05))
out("--- 가드(같은 시드 기준 대비 Δ ≤ +0.3)"); gok = True
for nm, a, b in (("VIPL 전체", {k: v["e"] for k, v in V.items()}, {k: v["e"] for k, v in BV.items()}), ("MMPD 전체", {k: v["e"] for k, v in M.items()}, {k: v["e"] for k, v in BM.items()}),
                 ("VIPL <60", {k: v["e"] for k, v in V.items() if v["dev"] < 60}, {k: v["e"] for k, v in BV.items() if v["dev"] < 60})):
    r = cmp(a, b); v = "missing" if r is None else ("위반" if r["d"] > 0.3 else "유지"); gok &= v == "유지"; line(nm, r, f" → {v}"); OUT[f"guard {nm}"] = dict(r or {}, verdict=v)
OUT["guards_ok"] = bool(gok)
out("--- 서술(같은 시드 기준 대비)")
for nm, a, b in (("MMPD ≥110", {k: v["e"] for k, v in M.items() if v["gt"] >= 110}, {k: v["e"] for k, v in BM.items() if v["gt"] >= 110}),
                 ("VIPL 장치 ≥120", {k: v["e"] for k, v in V.items() if v["dev"] >= 120}, {k: v["e"] for k, v in BV.items() if v["dev"] >= 120}),
                 ("VIPL 파형 ≥110", {k: abs(v["p"] - WV[k]) for k, v in V.items() if k in WV and WV[k] >= 110}, {k: abs(v["p"] - WV[k]) for k, v in BV.items() if k in WV and WV[k] >= 110}),
                 ("풀 고심박(파형 정답)", hi_wave(V, M), hi_wave(BV, BM)), ("MMPD 보행", {k: v["e"] for k, v in M.items() if v["mo"] == 4}, {k: v["e"] for k, v in BM.items() if v["mo"] == 4}),
                 ("풀 저심박(장치 정답)", lo_dev(V, M), lo_dev(BV, BM)),
                 ("MR-NIRP 정지", axis(NEW, "NIRP_STILL"), axis(BASE, "NIRP_STILL")), ("MR-NIRP 소동작", axis(NEW, "NIRP_SMALL"), axis(BASE, "NIRP_SMALL")),
                 ("MR-NIRP 대동작", axis(NEW, "NIRP_LARGE"), axis(BASE, "NIRP_LARGE")), ("RTNI", axis(NEW, "RTNI"), axis(BASE, "RTNI")), ("PURE", axis(NEW, "PURE"), axis(BASE, "PURE"))):
    r = cmp(a, b); line(nm, r); OUT[f"desc {nm}"] = r
out("--- 시드별 풀 고심박(장치 정답; 같은 시드 짝)"); pairs = []
for s in sn:
    if s not in sb: continue
    a, b = hi_dev(vipl((NEW, s)), mmpd((NEW, s))), hi_dev(vipl((BASE, s)), mmpd((BASE, s))); k = sorted(set(a) & set(b))
    pa, pb = float(np.mean([a[i] for i in k])), float(np.mean([b[i] for i in k])); pairs.append(pa - pb); out(f"  시드 {s:4d}: 기준 {pb:6.2f} -> 셀 {pa:6.2f} Δ{pa - pb:+6.2f}")
if pairs: out(f"  시드 짝 차이 평균 {np.mean(pairs):+.2f}, 셀이 낮은 시드 {sum(d < 0 for d in pairs)}/{len(pairs)}")
OUT["seed_pairs"] = pairs
out("--- 공개 FactorizePhys 대비(서술; 저심박 우세 Holm 2 · 고심박 비열등 2 bpm Holm 2 · 6축 Holm 6)")
lo = [cmp({k: v["e"] for k, v in V.items() if v["dev"] < 60}, {k: v["e"] for k, v in FPV.items() if v["dev"] < 60}), cmp(lo_dev(V, M), lo_dev(FPV, FPM))]; hl = holm([x["p"] for x in lo])
for nm, r, h in zip(("VIPL <60", "풀 저심박"), lo, hl): line(nm, r, f" Holm={h:.4f} → {'우세' if r['d'] < 0 and h < .05 else 'ns'}")
nh = holm([ni_p({k: v["e"] for k, v in V.items() if v["dev"] >= 120}, {k: v["e"] for k, v in FPV.items() if v["dev"] >= 120}), ni_p(hi_dev(V, M), hi_dev(FPV, FPM))])
out(f"  고심박 비열등(2 bpm): VIPL ≥120 Holm={nh[0]:.4f} · 풀 고심박 Holm={nh[1]:.4f} → {'✓' if max(nh) < .05 else '✗'}")
st = [("VIPL", cmp({k: v["e"] for k, v in V.items()}, {k: v["e"] for k, v in FPV.items()})), ("MMPD 7~33", cmp({k: v["e"] for k, v in M.items()}, {k: v["e"] for k, v in FPM.items()})),
      ("MR-NIRP 소동작", cmp(axis(NEW, "NIRP_SMALL"), axis("FactorizePhys", "NIRP_SMALL"))), ("MR-NIRP 대동작", cmp(axis(NEW, "NIRP_LARGE"), axis("FactorizePhys", "NIRP_LARGE"))),
      ("RTNI", cmp(axis(NEW, "RTNI"), axis("FactorizePhys", "RTNI"))), ("PURE 07~10", cmp(axis(NEW, "PURE", lambda r: int(r["vid"]) >= 701), axis("FactorizePhys", "PURE", lambda r: int(r["vid"]) >= 701)))]
hs = holm([r["p"] if r else 1.0 for _, r in st]); wins = ok6 = 0
for (nm, r), h in zip(st, hs):
    v = "missing" if r is None else ("우세" if r["d"] < 0 and h < .05 else ("비열등" if r["d"] <= 0.3 else "열세")); wins += v == "우세"; ok6 += v in ("우세", "비열등"); line("6축 " + nm, r, f" Holm={h:.4f} → {v}")
out(f"  6축: 우세 {wins}/6, 비열등 포함 {ok6}/6"); OUT["vs_FP"] = dict(low_holm=hl, high_ni_holm=nh, six_wins=wins, six_ok=ok6)
out("--- 10시드 합산(시드 100~1000, 서술 — 확인 판정 아님)")
N10, B10 = ["NF_SKREF_L", NEW], ["NF_HYB_D_TA2_L", BASE]; V10, M10, BV10, BM10 = vipl(N10), mmpd(N10), vipl(B10), mmpd(B10)
out(f"  시드: 셀 {seeds(N10)} · 기준 {seeds(B10)}")
for nm, a, b in (("풀 고심박(장치)", hi_dev(V10, M10), hi_dev(BV10, BM10)), ("풀 고심박(파형)", hi_wave(V10, M10), hi_wave(BV10, BM10)), ("MMPD ≥110", {k: v["e"] for k, v in M10.items() if v["gt"] >= 110}, {k: v["e"] for k, v in BM10.items() if v["gt"] >= 110}),
                 ("VIPL 전체", {k: v["e"] for k, v in V10.items()}, {k: v["e"] for k, v in BV10.items()}), ("MMPD 전체", {k: v["e"] for k, v in M10.items()}, {k: v["e"] for k, v in BM10.items()})):
    r = cmp(a, b); line(nm, r); OUT[f"10seed {nm}"] = r
a, b = hi_wave(V10, M10), hi_wave(RFV, RFM); k = sorted(set(a) & set(b)); out(f"  공개 RF 대비 풀 고심박(파형) 비열등 p={ni_p(a, b):.4f} · (장치) p={ni_p(hi_dev(V10, M10), hi_dev(RFV, RFM)):.4f}")
dec = ok1 and gok
out(f"\n판정(사전 등록 결정 규칙): R1 {'확인' if ok1 else '미확인'} · 가드 {'유지' if gok else '위반'} → {'권장 레시피 갱신: 채택 레시피 + 피부 가중 + 잡음 참조 배제' if dec else '채택 레시피 유지(잡음 참조 배제는 방향 일치·미확인)'}")
OUT["decision"] = dict(update_recipe=bool(dec))
open(f"{R}/skref_rep_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/skref_rep_judge.json", "w"), indent=1, ensure_ascii=False, default=float)
