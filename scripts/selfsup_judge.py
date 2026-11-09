#!/usr/bin/env python3
"""[체인 72, 53 문서] 자기 지도 분리 신호 판정 — 영상 단위 쌍대 Wilcoxon, 시드(""/_s200/_s300) 영상별 평균 풀링, 1차 가족 Holm(3).
  python scripts/selfsup_judge.py → 표 + results/selfsup_judge.{txt,json}
셀: NF_SS_BOTH_L(1차, 전역 trace 자기 감독 + 어텐션 공간 정칙화) · NF_SS_TR_L(전역 trace 만) · NF_SS_AT_L(어텐션 만)  vs 기준 NF_HYB_L(스펙트럼 감독 FactorizePhys, 3시드).
1차 가족(결합 셀 vs 기준, Holm 3): H-L1 VIPL 운동 후 v7 | H-L2 MMPD held-out 보행(motion 4) | H-L3 MR-NIRP 대동작.   지지 = Δ<0 & Holm p<.05.
2차(비보정 병기): VIPL ≥90·≥120·v7&≥90, MMPD 운동 후(exercise 1)·움직임 합산, 단일 신호 셀의 같은 표적(귀속), 결합 vs 단일.
가드(비열등: Δ≤+0.3 이고 기준 우세 p<.05 아님): VIPL 전체, MMPD 전체, PURE, MCD_A, REALTEST, MR-NIRP 소동작; RTNI 절대값 <5(플리커 lock-in 없음).
H-L4 기제: 표적 영상에서 lock-in 분류 비율(selfsup_phase0_diag.classify, 피크 캐시 selfsup_nuis_peaks.json)이 결합 셀에서 기준보다 감소(영상별 시드 평균 lock-in 지표의 쌍대 Wilcoxon; 부호 검정 병기)."""
import glob, json, os, re, sys
import numpy as np
from scipy.stats import wilcoxon, binomtest

sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
import selfsup_phase0_diag as D  # noqa  (classify, load_preds)

R = "/SPECPHYS_ROOT/_workspace/data/results"
BASE = "NF_HYB_L"; PRIMARY = "NF_SS_BOTH_L"; CELLS = ["NF_SS_BOTH_L", "NF_SS_TR_L", "NF_SS_AT_L"]
NAMES = {"NF_SS_BOTH_L": "결합(trace 자기 감독 + 어텐션 정칙화)", "NF_SS_TR_L": "전역 trace 자기 감독", "NF_SS_AT_L": "어텐션 공간 정칙화", "NF_HYB_L": "기준(스펙트럼 감독 FactorizePhys)"}


def files(m, suffix):
    pat = re.compile(rf"agg_{re.escape(m)}_s\d+{re.escape(suffix)}\.json$")
    return [f for f in [f"{R}/agg_{m}{suffix}.json"] + sorted(x for x in glob.glob(f"{R}/agg_{m}_s[0-9]*{suffix}.json") if pat.search(os.path.basename(x))) if os.path.exists(f)]


def seeds(m):
    return [re.sub(r"^agg_", "", os.path.basename(f))[:-5] for f in files(m, "")]


def vipl(m):
    acc, meta = {}, {}
    for f in files(m, "_VIPL"):
        for r in json.load(open(f)):
            if r.get("err_lit") is None: continue
            acc.setdefault(r["vid"], []).append(r["err_lit"]); meta[r["vid"]] = (r.get("scenario"), r.get("gt_lit"))
    return {k: (float(np.mean(v)),) + meta[k] for k, v in acc.items()}, len(files(m, "_VIPL"))


def axis(m, ax):
    acc = {}
    for f in files(m, ""):
        for r in json.load(open(f)).get(ax) or []: acc.setdefault(r["vid"], []).append(r["err"])
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
    s = f"  {name:64s} (missing)" if r is None else f"  {name:64s} n={r['n']:5d} {r['b']:6.2f} -> {r['a']:6.2f}  Δ{r['d']:+6.2f} p={r['p']:.4f}{flag}"
    print(s); LINES.append(s)
def head(s): print(s); LINES.append(s)


def lockin_rates(axis_name, model):
    """영상 -> 시드 평균 lock-in 지표(0..1), Phase 0 피크 캐시 기준 (표적 영상만)"""
    PK = json.load(open(f"{R}/selfsup_nuis_peaks.json")).get(axis_name, {})
    mlist = [m for m in ([model] + [f"{model}_s{s}" for s in (200, 300, 400, 500)]) if (os.path.exists(f"{R}/agg_{m}_VIPL.json") if axis_name == "VIPL" else os.path.exists(f"{R}/agg_{m}_MMPD.json") if axis_name == "MMPD" else os.path.exists(f"{R}/agg_{m}.json"))]
    preds = D.load_preds(axis_name, mlist); out = {}
    for vid, k in PK.items():
        pg = preds.get(vid)
        if not pg: continue
        gt_ax = k.get("gt")
        vals = [1.0 if D.classify(pred, gt if gt_ax is None else gt_ax, k["pk_g"], k["pk_m"])[0] == "lockin" else 0.0 for pred, gt in pg]
        out[vid] = (float(np.mean(vals)), k)
    return out


def main():
    out = {"cells": {}, "primary": {}, "guards": {}, "mechanism": {}}
    base_v = vipl(BASE)[0]; base_m = mmpd(BASE)
    V_SEL = [("VIPL 운동 후 v7", lambda k, v: v[1] == "v7"), ("VIPL ≥90 bpm", lambda k, v: v[2] is not None and v[2] >= 90), ("VIPL ≥120 bpm", lambda k, v: v[2] is not None and v[2] >= 120),
             ("VIPL 운동 후 v7 & ≥90", lambda k, v: v[1] == "v7" and v[2] is not None and v[2] >= 90), ("VIPL 전체", lambda k, v: True)]
    M_SEL = [("MMPD 보행 m4", lambda k, v: k[2] == 4), ("MMPD 운동 후 e1", lambda k, v: k[3] == 1), ("MMPD 움직임 합산 m2-4", lambda k, v: k[2] != 1), ("MMPD 전체", lambda k, v: True)]
    prim = []
    for cell in CELLS:
        v, ns = vipl(cell)
        if not v and not axis(cell, "PURE"): head(f"=== {cell} ({NAMES[cell]}): no results yet"); continue
        head(f"=== {cell} ({NAMES[cell]}, 시드 {ns}) vs {BASE}" + (f" — VIPL 문헌 {np.mean([x[0] for x in v.values()]):.2f}" if v else ""))
        out["cells"][cell] = {"seeds": ns, "pairs": {}}
        for nm, sel in V_SEL:
            r = cmp(v, base_v, sel); line(f"{nm}", r); out["cells"][cell]["pairs"][nm] = r
            if cell == PRIMARY and nm == "VIPL 운동 후 v7": prim.append(("H-L1 VIPL 운동 후 v7", r))
        mm = mmpd(cell)
        for nm, sel in M_SEL:
            r = cmp(mm, base_m, sel); line(f"{nm}", r); out["cells"][cell]["pairs"][nm] = r
            if cell == PRIMARY and nm == "MMPD 보행 m4": prim.append(("H-L2 MMPD 보행 m4", r))
        for ax in ("NIRP_LARGE", "NIRP_SMALL", "NIRP_STILL", "RTNI", "RTNC", "PURE", "MCD_A", "MCD_B", "REALTEST", "DLCN", "C3", "C6", "E3"):
            r = cmp(axis(cell, ax), axis(BASE, ax)); line(f"{ax}", r); out["cells"][cell]["pairs"][ax] = r
            if cell == PRIMARY and ax == "NIRP_LARGE": prim.append(("H-L3 MR-NIRP 대동작", r))
        if cell != PRIMARY and PRIMARY in out["cells"]:   # 결합 vs 단일(간섭/귀속)
            pv = vipl(PRIMARY)[0]; r = cmp(pv, v, lambda k, val: val[1] == "v7"); line(f"결합 vs {NAMES[cell]} [VIPL 운동 후 v7]", r); out["cells"][cell]["pairs"]["BOTH_vs|v7"] = r
            r = cmp(mmpd(PRIMARY), mm, lambda k, val: k[2] == 4); line(f"결합 vs {NAMES[cell]} [MMPD 보행]", r); out["cells"][cell]["pairs"]["BOTH_vs|m4"] = r
    prim = [(n, r) for n, r in prim if r]
    if prim:
        hp = holm([r["p"] for _, r in prim]); head(f"=== 1차 가족 Holm({len(prim)}) — {PRIMARY} vs {BASE}")
        for (n, r), h in zip(prim, hp):
            ok = r["d"] < 0 and h < 0.05; line(n, r, f" Holm={h:.4f} {'지지' if ok else '기각'}"); out["primary"][n] = dict(r, holm=h, supported=bool(ok))
    if PRIMARY in out["cells"]:
        head(f"=== 가드(비열등 Δ≤+0.3 & 기준 우세 p≥.05) — {PRIMARY}")
        pv = vipl(PRIMARY)[0]; pm = mmpd(PRIMARY); allok = True
        for nm, r in (("VIPL 전체", cmp(pv, base_v)), ("MMPD 전체", cmp(pm, base_m)), ("PURE", cmp(axis(PRIMARY, "PURE"), axis(BASE, "PURE"))), ("MCD_A", cmp(axis(PRIMARY, "MCD_A"), axis(BASE, "MCD_A"))),
                      ("REALTEST", cmp(axis(PRIMARY, "REALTEST"), axis(BASE, "REALTEST"))), ("MR-NIRP 소동작", cmp(axis(PRIMARY, "NIRP_SMALL"), axis(BASE, "NIRP_SMALL")))):
            ok = r is not None and r["d"] <= 0.3 and not (r["d"] > 0 and r["p"] < 0.05); allok &= ok; line(f"가드 {nm}", r, f" {'통과' if ok else '실패'}"); out["guards"][nm] = dict(r or {}, ok=bool(ok))
        rt = axis(PRIMARY, "RTNI"); rtm = float(np.mean([x[0] for x in rt.values()])) if rt else None
        ok = rtm is not None and rtm < 5.0; allok &= ok; head(f"  가드 RTNI 절대값 {rtm if rtm is None else round(rtm, 2)} <5 {'통과' if ok else '실패'}"); out["guards"]["RTNI_abs"] = {"mae": rtm, "ok": bool(ok)}
        out["guards"]["all_ok"] = bool(allok)
        # H-L4 기제: lock-in 비율
        head("=== H-L4 기제 — 표적 영상 lock-in 비율(Phase 0 분류, 시드 평균): 기준 → 결합")
        for axn, gname, sel in (("VIPL", "VIPL 운동 후 v7", lambda k: k.get("scenario") == "v7"), ("VIPL", "VIPL ≥90 bpm(v7 제외)", lambda k: k.get("scenario") != "v7"), ("MMPD", "MMPD 보행 m4", lambda k: k.get("motion") == 4), ("NIRP_LARGE", "MR-NIRP 대동작", lambda k: True)):
            try: a = lockin_rates(axn, BASE); b = lockin_rates(axn, PRIMARY)
            except Exception as e: head(f"  {gname}: 계산 불가 ({e})"); continue
            ks = [k for k in a if k in b and sel(a[k][1])]
            if len(ks) < 5: head(f"  {gname}: n<5"); continue
            x = np.array([a[k][0] for k in ks]); y = np.array([b[k][0] for k in ks]); d = y - x
            try: p = float(wilcoxon(y, x).pvalue) if np.any(d != 0) else 1.0
            except ValueError: p = 1.0
            disc = int((d != 0).sum()); fewer = int((d < 0).sum()); ps = float(binomtest(fewer, disc, 0.5).pvalue) if disc else 1.0
            s = f"  {gname:28s} n={len(ks):4d} lock-in {100 * x.mean():5.1f}% -> {100 * y.mean():5.1f}%  Δ{100 * (y.mean() - x.mean()):+5.1f}pt  Wilcoxon p={p:.4f}  부호검정(감소 {fewer}/{disc}) p={ps:.4f}"
            head(s); out["mechanism"][gname] = {"n": len(ks), "base": float(x.mean()), "cell": float(y.mean()), "p": p, "sign_p": ps, "fewer": fewer, "discordant": disc}
    open(f"{R}/selfsup_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(out, open(f"{R}/selfsup_judge.json", "w"), indent=1, ensure_ascii=False)
    print(f"-> {R}/selfsup_judge.txt, selfsup_judge.json")


if __name__ == "__main__":
    main()
