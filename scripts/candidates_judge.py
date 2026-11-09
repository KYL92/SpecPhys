#!/usr/bin/env python3
"""[체인 73, 55 문서] 남은 후보 3종 판정 — 영상 단위 쌍대 Wilcoxon, 시드 영상별 평균 풀링, 가족별 Holm.
  python scripts/candidates_judge.py → 표 + results/candidates_judge.{txt,json}
가족 ① 국소 교란 프록시   NF_SS_LOC_L(3시드) vs 기준 NF_HYB_L: H-C1 = {VIPL 운동 후 v7, MMPD 보행, MR-NIRP 대동작} Holm 3; 2차 vs NF_SS_TR_L(전역만); 가드; H-C4 lock-in.
가족 ③ 증강 일관성       NF_SS_AUG_L(3시드) vs NF_HYB_L: H-C3 = 같은 3 표적 Holm 3; 2차 NF_SS_AUGTR_L(+전역 trace) vs NF_HYB_L / vs NF_SS_AUG_L; 가드; H-C4 lock-in.
가족 ② in-domain(라벨 사용, 별도 표기) Holm 3:
   H-C2a VIPL 5-fold 2단계 NF(3시드: CV_NF_L, CV_NFs2_L, CV_NFs3_L) vs VIPL 단독 최선 CV_BASEAV_L (전체; 구간·v7 병기), 2차 CV_NFSS_L(trace 자기 감독 적응, 시드 1) vs CV_NF_L(시드 1)
   H-C2c MMPD 5-fold 2단계 NF(MCV5_NF_L[+NFs2,NFs3]) 보행(피험자 7~33) vs zero-shot NF_HYB_L 보행; 앵커 MCV5_HYB_L(3시드 joint)·MCV5_PT_L(2단계 PhysFormer)
   H-C2d MR-NIRP 5-fold 2단계 NF(NRCV_NF_L[+s2,s3]) 대동작 vs zero-shot NF_HYB_L; 앵커 NRCV_HYBN_L(체인 47 PhysFormer in-domain)
가드(zero-shot 셀): VIPL 전체·MMPD 전체·PURE·MCD_A·REALTEST·MR-NIRP 소동작 비열등(Δ≤+0.3 & 기준 우세 p≥.05), RTNI<5."""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon, binomtest

sys.path.insert(0, "/SPECPHYS_ROOT/scripts")
from selfsup_judge import files, vipl, axis, mmpd, cmp, holm, lockin_rates  # noqa
R = "/SPECPHYS_ROOT/_workspace/data/results"
BASE = "NF_HYB_L"
NAMES = {"NF_HYB_TB_L": "정본 일치 심박 라벨(패치 40, 사후 ④)", "NF_SS_LOC_L": "국소+전역 교란 프록시 trace 자기 감독", "NF_SS_AUG_L": "증강 일관성", "NF_SS_AUGTR_L": "증강 일관성 + 전역 trace 자기 감독", "NF_SS_TR_L": "전역 trace 자기 감독(53)", "NF_HYB_L": "기준(스펙트럼 감독 FactorizePhys)"}
LINES = []
def line(name, r, flag=""):
    s = f"  {name:66s} (missing)" if r is None else f"  {name:66s} n={r['n']:5d} {r['b']:6.2f} -> {r['a']:6.2f}  Δ{r['d']:+6.2f} p={r['p']:.4f}{flag}"
    print(s); LINES.append(s)
def head(s): print(s); LINES.append(s)


def vipl_list(models):
    acc, meta = {}, {}
    for m in models:
        f = f"{R}/agg_{m}_VIPL.json"
        if not os.path.exists(f): continue
        for r in json.load(open(f)):
            if r.get("err_lit") is None: continue
            acc.setdefault(r["vid"], []).append(r["err_lit"]); meta[r["vid"]] = (r.get("scenario"), r.get("gt_lit"))
    return {k: (float(np.mean(v)),) + meta[k] for k, v in acc.items()}


def mmpd_list(models, subs=set(range(1, 34))):
    acc = {}
    for m in models:
        f = f"{R}/agg_{m}_MMPD.json"
        if not os.path.exists(f): continue
        for r in json.load(open(f)):
            if r["subject"] in subs: acc.setdefault((r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]), []).append(r["err"])
    return {k: (float(np.mean(v)), None, None) for k, v in acc.items()}


def axis_list(models, ax):
    acc = {}
    for m in models:
        f = f"{R}/agg_{m}.json"
        if not os.path.exists(f): continue
        for r in json.load(open(f)).get(ax) or []: acc.setdefault(r["vid"], []).append(r["err"])
    return {k: (float(np.mean(v)), None, None) for k, v in acc.items()}


V_SEL = [("VIPL 운동 후 v7", lambda k, v: v[1] == "v7"), ("VIPL ≥90 bpm", lambda k, v: v[2] is not None and v[2] >= 90), ("VIPL ≥120 bpm", lambda k, v: v[2] is not None and v[2] >= 120), ("VIPL 운동 후 v7 & ≥90", lambda k, v: v[1] == "v7" and v[2] is not None and v[2] >= 90), ("VIPL 전체", lambda k, v: True)]
M_SEL = [("MMPD 보행 m4", lambda k, v: k[2] == 4), ("MMPD 운동 후 e1", lambda k, v: k[3] == 1), ("MMPD 움직임 합산 m2-4", lambda k, v: k[2] != 1), ("MMPD 전체", lambda k, v: True)]
HO = set(range(7, 34))


def zero_shot_family(cell, label, prim_name, extra_refs, out):
    v, ns = vipl(cell)
    if not v and not axis(cell, "PURE"): head(f"=== {cell} ({NAMES.get(cell, cell)}): no results yet"); return []
    head(f"=== {label} {cell} ({NAMES.get(cell, cell)}, 시드 {ns}) vs {BASE}" + (f" — VIPL 문헌 {np.mean([x[0] for x in v.values()]):.2f}" if v else ""))
    base_v = vipl(BASE)[0]; base_m = mmpd(BASE); o = {"seeds": ns, "pairs": {}}; prim = []
    for nm, sel in V_SEL:
        r = cmp(v, base_v, sel); line(nm, r); o["pairs"][nm] = r
        if nm == "VIPL 운동 후 v7": prim.append((f"{prim_name} VIPL 운동 후 v7", r))
    mm = mmpd(cell)
    for nm, sel in M_SEL:
        r = cmp(mm, base_m, sel); line(nm, r); o["pairs"][nm] = r
        if nm == "MMPD 보행 m4": prim.append((f"{prim_name} MMPD 보행 m4", r))
    for ax in ("NIRP_LARGE", "NIRP_SMALL", "NIRP_STILL", "RTNI", "RTNC", "PURE", "MCD_A", "MCD_B", "REALTEST", "DLCN"):
        r = cmp(axis(cell, ax), axis(BASE, ax)); line(ax, r); o["pairs"][ax] = r
        if ax == "NIRP_LARGE": prim.append((f"{prim_name} MR-NIRP 대동작", r))
    for rn in extra_refs:
        rv = vipl(rn)[0]; r = cmp(v, rv, lambda k, val: val[1] == "v7"); line(f"vs {NAMES.get(rn, rn)} [VIPL 운동 후 v7]", r); o["pairs"][f"{rn}|v7"] = r
        r = cmp(v, rv); line(f"vs {NAMES.get(rn, rn)} [VIPL 전체]", r); o["pairs"][f"{rn}|VIPL"] = r
        r = cmp(mm, mmpd(rn), lambda k, val: k[2] == 4); line(f"vs {NAMES.get(rn, rn)} [MMPD 보행]", r); o["pairs"][f"{rn}|m4"] = r
        r = cmp(axis(cell, "NIRP_LARGE"), axis(rn, "NIRP_LARGE")); line(f"vs {NAMES.get(rn, rn)} [MR-NIRP 대동작]", r); o["pairs"][f"{rn}|NIRP_LARGE"] = r
    # 가드
    allok = True; o["guards"] = {}
    for nm, r in (("VIPL 전체", cmp(v, base_v)), ("MMPD 전체", cmp(mm, base_m)), ("PURE", cmp(axis(cell, "PURE"), axis(BASE, "PURE"))), ("MCD_A", cmp(axis(cell, "MCD_A"), axis(BASE, "MCD_A"))), ("REALTEST", cmp(axis(cell, "REALTEST"), axis(BASE, "REALTEST"))), ("MR-NIRP 소동작", cmp(axis(cell, "NIRP_SMALL"), axis(BASE, "NIRP_SMALL")))):
        ok = r is not None and r["d"] <= 0.3 and not (r["d"] > 0 and r["p"] < 0.05); allok &= ok; line(f"가드 {nm}", r, f" {'통과' if ok else '실패'}"); o["guards"][nm] = dict(r or {}, ok=bool(ok))
    rt = axis(cell, "RTNI"); rtm = float(np.mean([x[0] for x in rt.values()])) if rt else None; ok = rtm is not None and rtm < 5.0; allok &= ok
    head(f"  가드 RTNI 절대값 {None if rtm is None else round(rtm, 2)} <5 {'통과' if ok else '실패'}"); o["guards"]["RTNI_abs"] = {"mae": rtm, "ok": bool(ok)}; o["guards"]["all_ok"] = bool(allok)
    # H-C4 기제
    o["mechanism"] = {}
    for axn, gname, sel in (("VIPL", "VIPL 운동 후 v7", lambda k: k.get("scenario") == "v7"), ("MMPD", "MMPD 보행 m4", lambda k: k.get("motion") == 4), ("NIRP_LARGE", "MR-NIRP 대동작", lambda k: True)):
        try: a = lockin_rates(axn, BASE); b = lockin_rates(axn, cell)
        except Exception as e: head(f"  lock-in {gname}: 계산 불가 ({e})"); continue
        ks = [k for k in a if k in b and sel(a[k][1])]
        if len(ks) < 5: continue
        x = np.array([a[k][0] for k in ks]); y = np.array([b[k][0] for k in ks]); d = y - x
        try: p = float(wilcoxon(y, x).pvalue) if np.any(d != 0) else 1.0
        except ValueError: p = 1.0
        head(f"  lock-in {gname:16s} n={len(ks):4d} {100 * x.mean():5.1f}% -> {100 * y.mean():5.1f}%  Δ{100 * (y.mean() - x.mean()):+5.1f}pt p={p:.4f}"); o["mechanism"][gname] = {"n": len(ks), "base": float(x.mean()), "cell": float(y.mean()), "p": p}
    out[cell] = o
    return [(n, r) for n, r in prim if r]


def holm_family(title, prim, out, key):
    if not prim: return
    hp = holm([r["p"] for _, r in prim]); head(f"=== {title} Holm({len(prim)})")
    for (n, r), h in zip(prim, hp):
        ok = r["d"] < 0 and h < 0.05; line(n, r, f" Holm={h:.4f} {'지지' if ok else '기각'}"); out.setdefault(key, {})[n] = dict(r, holm=h, supported=bool(ok))


def main():
    out = {"zero_shot": {}, "in_domain": {}, "primary": {}}
    # 가족 ①
    p1 = zero_shot_family("NF_SS_LOC_L", "가족 ① 국소 교란 프록시", "H-C1", ["NF_SS_TR_L"], out["zero_shot"]); holm_family("가족 ① 1차(NF_SS_LOC_L vs 기준)", p1, out["primary"], "family1")
    # 가족 ③
    p3 = zero_shot_family("NF_SS_AUG_L", "가족 ③ 증강 일관성", "H-C3", ["NF_SS_TR_L"], out["zero_shot"]); holm_family("가족 ③ 1차(NF_SS_AUG_L vs 기준)", p3, out["primary"], "family3")
    zero_shot_family("NF_SS_AUGTR_L", "가족 ③ 2차", "H-C3b", ["NF_SS_AUG_L", "NF_SS_TR_L"], out["zero_shot"])
    # 가족 ② in-domain
    prim2 = []
    cvn = vipl_list(["CV_NF_L", "CV_NFs2_L", "CV_NFs3_L"]); nseeds = sum(os.path.exists(f"{R}/agg_{m}_VIPL.json") for m in ("CV_NF_L", "CV_NFs2_L", "CV_NFs3_L"))
    if cvn:
        head(f"=== 가족 ② VIPL 5-fold 2단계 NF (시드 {nseeds}) — VIPL 문헌 {np.mean([x[0] for x in cvn.values()]):.2f} [in-domain]"); o = {"seeds": nseeds, "pairs": {}}
        for rn, rv in (("CV_BASEAV_L(VIPL 단독 최선)", vipl_list(["CV_BASEAV_L"])), ("CV_TWg_L(장치 라벨 공동학습)", vipl_list(["CV_TWg_L"])), ("NF_HYB_L zero-shot", vipl(BASE)[0]), ("CV_NF_L 시드 1", vipl_list(["CV_NF_L"]))):
            for nm, sel in V_SEL:
                r = cmp(cvn, rv, sel); line(f"vs {rn} [{nm}]", r); o["pairs"][f"{rn}|{nm}"] = r
                if rn.startswith("CV_BASEAV") and nm == "VIPL 전체": prim2.append(("H-C2a VIPL CV NF(3시드) vs CV_BASEAV_L", r))
        ss = vipl_list(["CV_NFSS_L"])
        if ss:
            for nm, sel in V_SEL: r = cmp(ss, vipl_list(["CV_NF_L"]), sel); line(f"2차 CV_NFSS_L(trace 자기 감독 적응) vs CV_NF_L 시드 1 [{nm}]", r); o["pairs"][f"NFSS|{nm}"] = r
        out["in_domain"]["VIPL"] = o
    mcv = mmpd_list(["MCV5_NF_L", "MCV5_NFs2_L", "MCV5_NFs3_L"])
    if mcv:
        ns = sum(os.path.exists(f"{R}/agg_{m}_MMPD.json") for m in ("MCV5_NF_L", "MCV5_NFs2_L", "MCV5_NFs3_L"))
        head(f"=== 가족 ② MMPD 5-fold 2단계 NF (시드 {ns}) [in-domain] — 전체 MAE {np.mean([x[0] for x in mcv.values()]):.2f} (660), held-out 7~33 보행 {np.mean([x[0] for k, x in mcv.items() if k[0] in HO and k[2] == 4]):.2f}"); o = {"seeds": ns, "pairs": {}}
        zs = mmpd(BASE)
        for nm, sel in M_SEL:
            r = cmp(mcv, zs, lambda k, v, sel=sel: k[0] in HO and sel(k, v)); line(f"vs NF_HYB_L zero-shot [{nm}, 7~33 공통 트라이얼]", r); o["pairs"][f"zs|{nm}"] = r
            if nm == "MMPD 보행 m4": prim2.append(("H-C2c MMPD CV NF 보행 vs zero-shot NF_HYB_L", r))
        for rn, rl in (("MCV5_HYB_L(PhysFormer joint 3시드)", ["MCV5_HYB_L", "MCV5_HYB_L_s200", "MCV5_HYB_L_s300"]), ("MCV5_PT_L(PhysFormer 2단계 시드 1)", ["MCV5_PT_L"]), ("MCV5_BASE_L(MMPD 단독 3시드)", ["MCV5_BASE_L", "MCV5_BASE_L_s200", "MCV5_BASE_L_s300"])):
            rv = mmpd_list(rl)
            for nm, sel in M_SEL: r = cmp(mcv, rv, sel); line(f"vs {rn} [{nm}, 660]", r); o["pairs"][f"{rn}|{nm}"] = r
        out["in_domain"]["MMPD"] = o
    ncv = axis_list(["NRCV_NF_L", "NRCV_NFs2_L", "NRCV_NFs3_L"], "NIRP_LARGE")
    if ncv:
        ns = sum(os.path.exists(f"{R}/agg_NRCV_{t}.json") for t in ("NF_L", "NFs2_L", "NFs3_L"))
        head(f"=== 가족 ② MR-NIRP 5-fold 2단계 NF (시드 {ns}) [in-domain]"); o = {"seeds": ns, "pairs": {}}
        for ax in ("NIRP_LARGE", "NIRP_SMALL", "NIRP_STILL"):
            a = axis_list(["NRCV_NF_L", "NRCV_NFs2_L", "NRCV_NFs3_L"], ax)
            r = cmp(a, axis(BASE, ax)); line(f"vs NF_HYB_L zero-shot [{ax}]", r); o["pairs"][f"zs|{ax}"] = r
            if ax == "NIRP_LARGE": prim2.append(("H-C2d MR-NIRP CV NF 대동작 vs zero-shot NF_HYB_L", r))
            r = cmp(a, axis_list(["NRCV_HYBN_L"], ax)); line(f"vs NRCV_HYBN_L(PhysFormer in-domain, 체인 47) [{ax}]", r); o["pairs"][f"HYBN|{ax}"] = r
        out["in_domain"]["NIRP"] = o
    holm_family("가족 ② in-domain 1차", prim2, out["primary"], "family2")
    # 가족 ④ (사후, 11-01 18:10 등록): 정본 일치 심박 라벨 NF_HYB_TB_L vs 기준 — H-C5 = {MMPD 보행, MMPD 전체} Holm 2 + 가드
    p4 = zero_shot_family("NF_HYB_TB_L", "가족 ④ 정본 일치 심박 라벨", "H-C5", [], out["zero_shot"])
    p4 = [(n, r) for n, r in p4 if "MMPD" in n]
    if "NF_HYB_TB_L" in out["zero_shot"]:
        r_all = out["zero_shot"]["NF_HYB_TB_L"]["pairs"].get("MMPD 전체")
        if r_all: p4.append(("H-C5 MMPD 전체", r_all))
    holm_family("가족 ④ 1차(NF_HYB_TB_L vs 기준, MMPD)", p4, out["primary"], "family4")
    open(f"{R}/candidates_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(out, open(f"{R}/candidates_judge.json", "w"), indent=1, ensure_ascii=False)
    print(f"-> {R}/candidates_judge.txt, candidates_judge.json")


if __name__ == "__main__":
    main()
