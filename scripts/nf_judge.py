#!/usr/bin/env python3
"""[체인 66, 48 문서] NF-Phys 판정 — 영상 단위 쌍대 Wilcoxon, 시드(""/_s200/_s300) 영상별 평균 풀링, 가족별 Holm.
  python scripts/nf_judge.py [--phase2] [--phase3]        (둘 다 생략 = 둘 다) → 표 + results/nf_judge.{txt,json}
Phase 2 zero-shot: NF_HYB_L vs 공개 FactorizePhys(VIPL=FPbase, 13축·MMPD=FactorizePhys), FP_OFF_HYB(rank-1·같은 풀·분해 손실), CoTAF_TW_L(채택 PhysFormer), CoTAM_HYB4_L, POS;
  절제 NF_HYB_NOSUP_L(교란 감독 없음)·NF_HYB_R1_L(rank-1 + 시간 배치 + 스펙트럼 손실) 은 NF_HYB_L 과 쌍대 비교.
  1차 가족(Holm 5, 48 §3): H-B2 VIPL 전체 vs 공개 FP | H-B1 VIPL 90–120 bpm vs 공개 FP | H-B3 RTNI vs 공개 FP | H-B3 MR-NIRP 소동작 vs 공개 FP | H-B3 MMPD held-out vs 공개 FP(비열등 Δ≤+0.5)
Phase 3 in-domain: CV_NF_L(5-fold 2단계) vs CV_BASEAV_L(VIPL 단독 최선 6.21)·CV_TWg_L(장치 라벨 공동학습 6.51)·FPbase zero-shot(6.88) — 1차: vs CV_BASEAV_L."""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

R = "/SPECPHYS_ROOT/_workspace/data/results"
AXES = ["RTNI", "RTNC", "NIRP_STILL", "NIRP_SMALL", "NIRP_LARGE", "DLCN", "PURE", "MCD_A", "MCD_B", "REALTEST", "C3", "C6", "E3"]
BANDS = [("<60", 0, 60), ("60-90", 60, 90), ("90-120", 90, 120), (">=120", 120, 999)]


def files(m, suffix):
    import re as _re  # [10-28] suffix "" 일 때 glob 이 agg_{m}_s200_MMPD.json 등(리스트 형식)까지 잡던 결함 → 정확 일치만
    pat = _re.compile(rf"agg_{_re.escape(m)}_s\d+{_re.escape(suffix)}\.json$")
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
        d = json.load(open(f)).get(ax) or []
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


LINES = []
def line(name, r, flag=""):
    s = f"  {name:60s} (missing)" if r is None else f"  {name:60s} n={r['n']:5d} {r['b']:6.2f} -> {r['a']:6.2f}  Δ{r['d']:+6.2f} p={r['p']:.4f}{flag}"
    print(s); LINES.append(s)
def head(s): print(s); LINES.append(s)


out = {"phase2": {}, "phase3": {}, "primary": {}}
want2 = "--phase2" in sys.argv or "--phase3" not in sys.argv
want3 = "--phase3" in sys.argv or "--phase2" not in sys.argv
if want2:
    refs_v = {"공개 FactorizePhys": vipl("FPbase")[0], "FP_OFF_HYB(rank-1·같은 풀·분해)": vipl("FP_OFF_HYB")[0], "CoTAF_TW_L(채택 PhysFormer)": vipl("CoTAF_TW_L")[0], "CoTAM_HYB4_L": vipl("CoTAM_HYB4_L")[0], "POS": vipl("POS")[0]}
    refs_a = {"공개 FactorizePhys": "FactorizePhys", "FP_OFF_HYB(rank-1·같은 풀·분해)": "FP_OFF_HYB", "CoTAF_TW_L(채택 PhysFormer)": "CoTAF_TW_L", "CoTAM_HYB4_L": "CoTAM_HYB4_L"}
    prim = []
    for cell in ("NF_HYB_L", "NF_HYB_NOSUP_L", "NF_HYB_R1_L", "NF_HYB_ORIG_L"):   # [10-28] ORIG = 원본 FSAM 배치 + 스펙트럼 손실(사후 절제, 48 §6)
        v, ns = vipl(cell)
        if not v and not axis(cell, "PURE"): head(f"=== {cell}: no results yet"); continue
        head(f"=== Phase 2 {cell} (시드 {ns}) VIPL 문헌 {np.mean([x[0] for x in v.values()]):.2f} (n={len(v)})" if v else f"=== Phase 2 {cell}: VIPL missing")
        out["phase2"][cell] = {"seeds": ns, "vipl": float(np.mean([x[0] for x in v.values()])) if v else None, "pairs": {}}
        for rn, rv in refs_v.items():
            r = cmp(v, rv); line(f"{cell} vs {rn} [VIPL 전체]", r); out["phase2"][cell]["pairs"][f"{rn}|VIPL"] = r
            if rn == "공개 FactorizePhys" and cell == "NF_HYB_L": prim.append(("H-B2 VIPL 전체 vs 공개 FP", r))
            for bn, lo, hi in BANDS:
                rb = cmp(v, rv, lambda k, val, lo=lo, hi=hi: val[2] is not None and lo <= val[2] < hi); line(f"{cell} vs {rn} [VIPL {bn} bpm]", rb); out["phase2"][cell]["pairs"][f"{rn}|VIPL_{bn}"] = rb
                if rn == "공개 FactorizePhys" and cell == "NF_HYB_L" and bn == "90-120": prim.append(("H-B1 VIPL 90–120 vs 공개 FP", rb))
            for sc, nm in (("v7", "운동 후 v7"), ("v2", "큰 움직임 v2"), ("v9", "휴대폰 움직임 v9")):
                rs = cmp(v, rv, lambda k, val, sc=sc: val[1] == sc); line(f"{cell} vs {rn} [VIPL {nm}]", rs); out["phase2"][cell]["pairs"][f"{rn}|VIPL_{sc}"] = rs
        for ax in AXES:
            a = axis(cell, ax)
            if not a: continue
            for rn, rm in refs_a.items():
                r = cmp(a, axis(rm, ax)); line(f"{cell} vs {rn} [{ax}]", r); out["phase2"][cell]["pairs"][f"{rn}|{ax}"] = r
                if rn == "공개 FactorizePhys" and cell == "NF_HYB_L" and ax in ("RTNI", "NIRP_SMALL"): prim.append((f"H-B3 {ax} vs 공개 FP", r))
        mm = mmpd(cell)
        if mm:
            for rn, rm in list(refs_a.items()) + [("POS", "POS")]:
                r = cmp(mm, mmpd(rm)); line(f"{cell} vs {rn} [MMPD 7~33 전체]", r); out["phase2"][cell]["pairs"][f"{rn}|MMPD"] = r
                r2 = cmp(mm, mmpd(rm), lambda k, val: k[2] != 1); line(f"{cell} vs {rn} [MMPD 움직임 합산]", r2); out["phase2"][cell]["pairs"][f"{rn}|MMPD_motion"] = r2
                if rn == "공개 FactorizePhys" and cell == "NF_HYB_L": prim.append(("H-B3 MMPD held-out vs 공개 FP (비열등 Δ≤+0.5)", r))
        if cell != "NF_HYB_L":  # 절제: NF_HYB_L 대비
            base_v = vipl("NF_HYB_L")[0]; r = cmp(v, base_v); line(f"{cell} vs NF_HYB_L [VIPL 전체] (H-B4 절제)", r); out["phase2"][cell]["pairs"]["NF_HYB_L|VIPL"] = r
            for ax in ("RTNI", "NIRP_SMALL", "MCD_A", "PURE"):
                r = cmp(axis(cell, ax), axis("NF_HYB_L", ax)); line(f"{cell} vs NF_HYB_L [{ax}] (H-B4 절제)", r); out["phase2"][cell]["pairs"][f"NF_HYB_L|{ax}"] = r
    prim = [(n, r) for n, r in prim if r]
    if prim:
        hp = holm([r["p"] for _, r in prim]); head("=== Phase 2 1차 가족 Holm (NF_HYB_L vs 공개 FactorizePhys)")
        for (n, r), h in zip(prim, hp):
            ok = (r["d"] < 0 and h < 0.05) if "비열등" not in n else (r["d"] <= 0.5)
            line(n, r, f" Holm={h:.4f} {'지지' if ok else '기각'}"); out["primary"][n] = dict(r, holm=h, supported=bool(ok))
if want3:
    cv, ncv = vipl("CV_NF_L")
    if not cv: head("=== Phase 3 CV_NF_L: no results yet")
    else:
        head(f"=== Phase 3 CV_NF_L (5-fold 2단계) VIPL 문헌 {np.mean([x[0] for x in cv.values()]):.2f} (n={len(cv)})")
        out["phase3"]["vipl"] = float(np.mean([x[0] for x in cv.values()])); out["phase3"]["pairs"] = {}
        for rn, rv in (("CV_BASEAV_L(VIPL 단독 최선)", vipl("CV_BASEAV_L")[0]), ("CV_TWg_L(장치 라벨 공동학습)", vipl("CV_TWg_L")[0]), ("공개 FactorizePhys zero-shot", vipl("FPbase")[0]), ("NF_HYB_L zero-shot", vipl("NF_HYB_L")[0])):
            r = cmp(cv, rv); line(f"CV_NF_L vs {rn} [VIPL 전체]", r); out["phase3"]["pairs"][f"{rn}|VIPL"] = r
            for bn, lo, hi in BANDS:
                rb = cmp(cv, rv, lambda k, val, lo=lo, hi=hi: val[2] is not None and lo <= val[2] < hi); line(f"CV_NF_L vs {rn} [VIPL {bn} bpm]", rb); out["phase3"]["pairs"][f"{rn}|VIPL_{bn}"] = rb
            if "BASEAV" in rn: out["primary"]["H-B5 CV_NF_L vs CV_BASEAV_L"] = dict(r, supported=bool(r and r["d"] < 0 and r["p"] < 0.05)) if r else None
open(f"{R}/nf_judge.txt", "w").write("\n".join(LINES) + "\n")
json.dump(out, open(f"{R}/nf_judge.json", "w"), indent=1, ensure_ascii=False)
print(f"-> {R}/nf_judge.txt, nf_judge.json")
