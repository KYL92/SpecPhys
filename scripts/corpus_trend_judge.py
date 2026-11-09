#!/usr/bin/env python3
"""[체인 69, 50 문서] 코퍼스 교란 설정 경향 판정 — 변형 코퍼스 셀(시드 풀링) vs 공식 코퍼스, 표적 축별 영상 단위 Wilcoxon + 단계열 경향.
  python scripts/corpus_trend_judge.py [--select]
    --select : 사전 등록 선택 규칙(50 §3)으로 Phase 2(시드 200·300) 대상 변형을 고르고 results/corpus_trend_selected.txt 에 기록
출력: 표 + results/corpus_trend.{txt,json}"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon, spearmanr

R = "/SPECPHYS_ROOT/_workspace/data/results"
VARIANTS = [("공식", "CoTAF_TW_L"), ("플리커 제거", "CoTAF_NOFLK_L"), ("광대역 0.02", "CoTAF_FR_L"), ("광대역 0.03", "CoTAF_FR03_L"), ("혼합", "CoTAF_MIX_L"),
            ("좁은 띠 0.02", "CoTAF_N02_L"), ("광대역 0.04", "CoTAF_FR04_L"), ("움직임 ×2", "CoTAF_WIDE_L"), ("움직임 ×0.5", "CoTAF_NARROW_L"),
            ("복합 광대역", "CoTAF_COMBO_L"), ("복합 좁은 띠", "CoTAF_COMBON_L")]   # [체인 71, 51 문서] 복합 = 플리커 0.02 + 움직임·드리프트 ×0.5
NEW = ["CoTAF_FR_L", "CoTAF_FR03_L", "CoTAF_MIX_L", "CoTAF_N02_L", "CoTAF_FR04_L", "CoTAF_WIDE_L", "CoTAF_NARROW_L"]
SINGLE = [("복합 광대역", "CoTAF_COMBO_L"), ("복합 좁은 띠", "CoTAF_COMBON_L")]
SINGLE_TARGETS = ["VIPL 전체", "MMPD 움직임 합산", "MR-NIRP 움직임 합산"]; SINGLE_GUARDS = [("플리커 1.1 Hz", 5.0), ("플리커 2.4 Hz", 5.0), ("PURE", None), ("MCD_A", None)]


def files(m, suffix):
    import re
    pat = re.compile(rf"agg_{re.escape(m)}_s\d+{re.escape(suffix)}\.json$")
    return [f for f in [f"{R}/agg_{m}{suffix}.json"] + sorted(x for x in glob.glob(f"{R}/agg_{m}_s[0-9]*{suffix}.json") if pat.search(os.path.basename(x))) if os.path.exists(f)]


def vipl(m, sel=lambda r: True):
    acc = {}
    for f in files(m, "_VIPL"):
        for r in json.load(open(f)):
            if r.get("err_lit") is None or not sel(r): continue
            acc.setdefault(r["vid"], []).append(r["err_lit"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def mmpd(m, sel=lambda r: True):
    acc = {}
    for f in files(m, "_MMPD"):
        for r in json.load(open(f)):
            if r["subject"] < 7 or not sel(r): continue
            acc.setdefault((r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]), []).append(r["err"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def axis(m, ax):
    acc = {}
    for f in files(m, ""):
        for r in (json.load(open(f)).get(ax) or []): acc.setdefault(r["vid"], []).append(r["err"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def nirp_motion(m):
    a = axis(m, "NIRP_SMALL"); b = axis(m, "NIRP_LARGE")
    return {**{("S", k): v for k, v in a.items()}, **{("L", k): v for k, v in b.items()}}


_fx = json.load(open(f"{R}/flicker_axes.json")) if os.path.exists(f"{R}/flicker_axes.json") else {}
def flick(m, ax):
    return {k: float(v) for k, v in _fx.get(m, {}).get(ax, {}).items()}


_ex = json.load(open(f"{R}/extra_axes.json")) if os.path.exists(f"{R}/extra_axes.json") else {}   # [체인 70] 사후 축(MR-NIRP 주차장·940 nm RGB)
def extra(m, ax):
    return {k: float(v) for k, v in _ex.get(m, {}).get(ax, {}).items()}
def extra_sum(m, axes):
    return {**{(ax, k): v for ax in axes for k, v in extra(m, ax).items()}}
TARGETS = [("VIPL 전체", lambda m: vipl(m)), ("VIPL 큰 움직임 v2", lambda m: vipl(m, lambda r: r["scenario"] == "v2")), ("VIPL 휴대폰 움직임 v9", lambda m: vipl(m, lambda r: r["scenario"] == "v9")),
           ("VIPL 운동 후 v7", lambda m: vipl(m, lambda r: r["scenario"] == "v7")), ("MMPD 전체", lambda m: mmpd(m)), ("MMPD 움직임 합산", lambda m: mmpd(m, lambda r: r["motion"] != 1)),
           ("MMPD 보행", lambda m: mmpd(m, lambda r: r["motion"] == 4)), ("MMPD 정지", lambda m: mmpd(m, lambda r: r["motion"] == 1)),
           ("MR-NIRP 정지", lambda m: axis(m, "NIRP_STILL")), ("MR-NIRP 소동작", lambda m: axis(m, "NIRP_SMALL")), ("MR-NIRP 대동작", lambda m: axis(m, "NIRP_LARGE")), ("MR-NIRP 움직임 합산", nirp_motion),
           ("플리커 1.1 Hz", lambda m: flick(m, "RTNI")), ("플리커 1.7 Hz", lambda m: flick(m, "RTNI17")), ("플리커 2.4 Hz", lambda m: flick(m, "RTNI24")),
           ("PURE", lambda m: axis(m, "PURE")), ("MCD_A", lambda m: axis(m, "MCD_A")), ("DLCN", lambda m: axis(m, "DLCN")),
           ("(사후) 주차장 정지 975", lambda m: extra(m, "NIRP_GSTILL")), ("(사후) 주차장 소동작 975", lambda m: extra(m, "NIRP_GSMALL")), ("(사후) 주차장 대동작 975", lambda m: extra(m, "NIRP_GLARGE")),
           ("(사후) 주차장 움직임 합산", lambda m: extra_sum(m, ["NIRP_GSMALL", "NIRP_GLARGE"])), ("(사후) 주행 정지 940", lambda m: extra(m, "NIRP_STILL940")), ("(사후) 주행 소동작 940", lambda m: extra(m, "NIRP_SMALL940")),
           ("(사후) 주차장 정지 940", lambda m: extra(m, "NIRP_GSTILL940")), ("(사후) 주차장 소동작 940", lambda m: extra(m, "NIRP_GSMALL940"))]
SELECT_TARGETS = ["MMPD 움직임 합산", "VIPL 전체", "MR-NIRP 움직임 합산"]
SERIES = {"광대역 진폭 (0 → 0.02 → 0.03 → 0.04)": [("CoTAF_NOFLK_L", 0.0), ("CoTAF_FR_L", 0.02), ("CoTAF_FR03_L", 0.03), ("CoTAF_FR04_L", 0.04)],
          "좁은 띠 진폭 (0 → 0.02 → 공식≈0.03)": [("CoTAF_NOFLK_L", 0.0), ("CoTAF_N02_L", 0.02), ("CoTAF_TW_L", 0.03)],
          "움직임 폭 (×0.5 → ×1 → ×2)": [("CoTAF_NARROW_L", 0.5), ("CoTAF_TW_L", 1.0), ("CoTAF_WIDE_L", 2.0)]}


def cmp(a, b):
    ks = [k for k in a if k in b]
    if len(ks) < 5: return None
    x = np.array([a[k] for k in ks]); y = np.array([b[k] for k in ks])
    try: p = float(wilcoxon(x, y).pvalue)
    except ValueError: p = 1.0
    return {"n": len(ks), "a": float(x.mean()), "b": float(y.mean()), "d": float(x.mean() - y.mean()), "p": p}


LINES = []
def out(s): print(s); LINES.append(s)


res = {"pairs": {}, "series": {}, "selected": []}
data = {m: {t: f(m) for t, f in TARGETS} for _, m in VARIANTS}
off = data["CoTAF_TW_L"]
out("=== 변형 코퍼스 vs 공식 코퍼스 (영상 단위 쌍대 Wilcoxon, 시드 풀링; 값 = MAE, Δ = 변형 − 공식)")
hdr = f"{'표적':20s}" + "".join(f"{lab:>16s}" for lab, m in VARIANTS)
out(hdr)
for t, _ in TARGETS:
    cells = []
    for lab, m in VARIANTS:
        d = data[m][t]
        if not d: cells.append(f"{'–':>16s}"); continue
        if m == "CoTAF_TW_L": cells.append(f"{np.mean(list(d.values())):>16.2f}"); continue
        r = cmp(d, off[t]); res["pairs"][f"{m}|{t}"] = r
        cells.append(f"{r['a']:7.2f}({r['d']:+.2f}{'*' if r['p'] < 0.05 else ' '})".rjust(16) if r else f"{np.mean(list(d.values())):>16.2f}")
    out(f"{t:20s}" + "".join(cells))
out("  (* p<.05 vs 공식; 변형 셀은 시드 1 또는 풀링 — 시드 수는 results/agg_* 파일로 확인)")
out("=== 단계열 경향 (표적별 MAE 열, Spearman ρ; 점 3~4개라 기술적)")
for sname, seq in SERIES.items():
    res["series"][sname] = {}
    out(f"-- {sname}")
    for t, _ in TARGETS:
        vals = [(lv, np.mean(list(data[m][t].values()))) for m, lv in seq if data[m][t]]
        if len(vals) < 3: continue
        rho = spearmanr([v[0] for v in vals], [v[1] for v in vals]).correlation
        mono = "단조↓" if all(vals[i][1] >= vals[i + 1][1] for i in range(len(vals) - 1)) else ("단조↑" if all(vals[i][1] <= vals[i + 1][1] for i in range(len(vals) - 1)) else "비단조")
        res["series"][sname][t] = {"vals": vals, "rho": None if np.isnan(rho) else float(rho), "mono": mono}
        out(f"   {t:20s} " + " → ".join(f"{v:.2f}" for _, v in vals) + f"   ρ={rho:+.2f} {mono}")
if "--select" in sys.argv:
    out("=== Phase 2 선택 (50 §3: 표적별 최저 MAE 변형이 공식 대비 Δ<0·p<.05 일 때)")
    sel = []
    for t in SELECT_TARGETS:
        best = None
        for m in NEW:
            r = res["pairs"].get(f"{m}|{t}")
            if r and (best is None or r["a"] < best[1]["a"]): best = (m, r)
        if best and best[1]["d"] < 0 and best[1]["p"] < 0.05:
            out(f"   {t:20s} → {best[0]} ({best[1]['b']:.2f} → {best[1]['a']:.2f}, Δ{best[1]['d']:+.2f} p={best[1]['p']:.4f}) 선택")
            v = best[0].replace("CoTAF_", "").replace("_L", "")
            if v not in sel and v != "FR": sel.append(v)
        else: out(f"   {t:20s} → 선택 없음" + (f" (최저 {best[0]} Δ{best[1]['d']:+.2f} p={best[1]['p']:.3f})" if best else ""))
    res["selected"] = sel[:3]
    open(f"{R}/corpus_trend_selected.txt", "w").write(" ".join(res["selected"]) + "\n")
    out(f"   선택 변형(최대 3): {res['selected']}")
if "--single" in sys.argv:  # [51 문서 §3] 단일 코퍼스 전 데이터셋 최적 판정: 세 표적 모두 비열등(Δ ≤ +0.3 이고 공식 우세 p ≥ .05) + ≥1 표적 우세(Δ<0, Holm p<.05) + 플리커 가드
    out("=== 단일 코퍼스 판정 (H-S1: 세 표적 동시 비열등 + 1개 이상 우세 + 플리커 1.1/2.4 Hz < 5)")
    for lab, m in SINGLE:
        rs = {t: cmp(data[m][t], off[t]) for t in SINGLE_TARGETS}
        if any(r is None for r in rs.values()): out(f"   {lab}: 결과 없음"); continue
        ps = [rs[t]["p"] for t in SINGLE_TARGETS]; order = sorted(range(3), key=lambda i: ps[i]); run = 0.0; hol = [None] * 3
        for rank, i in enumerate(order): run = max(run, min(1.0, (3 - rank) * ps[i])); hol[i] = run
        noninf = all(rs[t]["d"] <= 0.3 and not (rs[t]["d"] > 0 and rs[t]["p"] < 0.05) for t in SINGLE_TARGETS)
        sup = [t for i, t in enumerate(SINGLE_TARGETS) if rs[t]["d"] < 0 and hol[i] < 0.05]
        g = {k: (np.mean(list(data[m][k].values())) if data[m][k] else None) for k, _ in SINGLE_GUARDS}
        guard_ok = all(g[k] is not None and g[k] < lim for k, lim in SINGLE_GUARDS if lim is not None)
        for i, t in enumerate(SINGLE_TARGETS): out(f"   {lab:10s} {t:16s} {rs[t]['b']:6.2f} → {rs[t]['a']:6.2f}  Δ{rs[t]['d']:+6.2f} p={rs[t]['p']:.4f} Holm={hol[i]:.4f}")
        out(f"   {lab:10s} 가드: " + ", ".join(f"{k}={g[k]:.2f}" if g[k] is not None else f"{k}=–" for k, _ in SINGLE_GUARDS))
        verdict = "지지" if (noninf and sup and guard_ok) else "기각"
        out(f"   {lab:10s} → 비열등 {noninf} / 우세 {sup or '없음'} / 가드 {guard_ok} → **H-S1 {verdict}**")
        res.setdefault("single", {})[m] = {"pairs": rs, "holm": dict(zip(SINGLE_TARGETS, hol)), "noninferior": bool(noninf), "superior": sup, "guards": g, "verdict": verdict}
    open(f"{R}/single_corpus_judge.txt", "w").write("\n".join(l for l in LINES if l.startswith("   ") or l.startswith("=== 단일")) + "\n")
open(f"{R}/corpus_trend.txt", "w").write("\n".join(LINES) + "\n")
json.dump(res, open(f"{R}/corpus_trend.json", "w"), indent=1, ensure_ascii=False, default=str)
print(f"-> {R}/corpus_trend.txt, corpus_trend.json")
