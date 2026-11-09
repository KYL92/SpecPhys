#!/usr/bin/env python3
"""[69 문서 §2, 사전 등록] 기여 근거 판정(Alpha, 학습 없음).
 A 통제 심박 시험(HRSWEEP40LO·HRSWEEP40)에서 SpecPhys vs 기존 방법 4개: 클립 평균 절대 오차, 클립 단위 부호 뒤집기(2만 회), Holm 8.
 B 같은 사람의 심박 변화 추종: VIPL-HR 기울기(B1, Holm 4)·MMPD 7~33 기울기(B2, Holm 4), 피험자 단위 부호 뒤집기(2만 회).
 C 배음 오류(배가 |r−2|<0.2 또는 반감 |r−0.5|<0.1, r = 예측/정답) 비율: VIPL-HR·MMPD, 영상 단위 부호 뒤집기, Holm 8. 다른 세트는 서술.
 D 실제 학습 영상이 적을 때 합성 클립의 몫(체인 128): D1 Q vs QN, D2 (Q−QN)−(C−NS), D3 SO vs 공개 FactorizePhys, 각 Holm 5(실제 지표 5개).
  python scripts/contrib_judge.py → results/contrib_judge.{txt,json}"""
import json, os, sys
import numpy as np
ROOT = "/SPECPHYS_ROOT"; sys.path.insert(0, f"{ROOT}/scripts"); import mmpd_common_gt as MGC
R = f"{ROOT}/_workspace/data/results"; LINES = []; OUT = {}
src = open(f"{ROOT}/scripts/nsg_judge.py").read(); src = src[:src.index('if "--screen2" in sys.argv:')]
g = {}; exec(compile(src, "nj", "exec"), g); bundle, cmp, flip_p, holm = g["bundle"], g["cmp"], g["flip_p"], g["holm"]
def out(s=""): print(s); LINES.append(s)
CONF = list(range(1600, 2501, 100)); DEC = list(range(2100, 2501, 100))
FIN = ("SpecPhys", "NF_NSP2_L", "NF_NSP2_L", CONF)
BASES = [("public FactorizePhys", "FPbase", "FactorizePhys", [100]), ("public RhythmFormer", "RFbase", "RhythmFormer", [100]),
         ("public PhysMamba", "PMbase", "PMbase", [100]), ("previous PhysFormer + corpus", "CoTAF_TW_L", "CoTAF_TW_L", [100, 200, 300, 400, 500])]
MODELS = [FIN] + BASES
fname = lambda nm, s: nm if s == 100 else f"{nm}_s{s}"
_rng = np.random.default_rng(0)
def flip(d, n=20000):
    d = np.asarray(d, float); obs = abs(d.mean())
    if np.allclose(d, 0): return 1.0
    return float(((np.abs((_rng.choice([-1.0, 1.0], size=(n, len(d))) * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n + 1))
def holm_tag(rows):
    hs = holm([r["p"] for r in rows])
    for r, h in zip(rows, hs): r["holm"] = h
# ---------- 실제 영상 로더(영상별 시드 예측 유지) ----------
def vipl_rows(nm, S):
    acc = {}; n = 0
    for s in S:
        p = f"{R}/agg_{fname(nm, s)}_VIPL.json"
        if not os.path.exists(p): continue
        n += 1
        for r in json.load(open(p)): a = acc.setdefault(r["vid"], dict(sub=r["subject"], gt=float(r["gt_lit"]), sc=r.get("scenario"), p=[])); a["p"].append(float(r["pred"]))
    for a in acc.values(): a["pm"] = float(np.mean(a["p"]))
    return acc, n
def mmpd_rows(nm, S):
    acc = {}; n = 0
    for s in S:
        p = f"{R}/agg_{fname(nm, s)}_MMPD.json"
        if not os.path.exists(p): continue
        n += 1
        for r in MGC.load(p):
            if 7 <= r["subject"] <= 33:
                k = (r["subject"], r["light"], r["motion"], r["exercise"], r["skin"]); a = acc.setdefault(k, dict(sub=r["subject"], g=[], p=[])); a["g"].append(float(r["gt"])); a["p"].append(float(r["pred"]))
    for a in acc.values(): a["gt"] = float(np.median(a["g"])); a["pm"] = float(np.mean(a["p"]))
    return acc, n
def center(acc, keys, use=None):
    by = {}
    for k in keys: by.setdefault(acc[k]["sub"], []).append(k)
    dp, dg, sub = [], [], []
    for s_, ks in sorted(by.items(), key=lambda x: str(x[0])):
        if len(ks) < 2: continue
        gt = np.array([acc[k]["gt"] for k in ks]); pr = np.array([acc[k]["pm"] if use is None else acc[k]["p"][use] for k in ks])
        dg += list(gt - gt.mean()); dp += list(pr - pr.mean()); sub += [s_] * len(ks)
    return np.array(dp), np.array(dg), np.array(sub)
def slope(dp, dg): return float((dp * dg).sum() / (dg * dg).sum())
def perm_slope(dpA, dpB, dg, sub, n=20000):
    us = np.unique(sub); idx = [np.where(sub == u)[0] for u in us]; den = (dg * dg).sum(); obs = slope(dpA, dg) - slope(dpB, dg)
    c = np.array([((dpA[i] - dpB[i]) * dg[i]).sum() for i in idx]) / den; hit = 0
    for _ in range(n // 1000): hit += int((np.abs(_rng.choice([-1.0, 1.0], size=(1000, len(us))) @ c) >= abs(obs) - 1e-12).sum())
    return obs, (hit + 1) / (n + 1)
def famB():
    for tag, loader, fam in (("VIPL-HR", vipl_rows, "B1"), ("MMPD 7-33", mmpd_rows, "B2")):
        D = {}; NS = {}
        for lab, vn, mn, S in MODELS: D[lab], NS[lab] = loader(vn if loader is vipl_rows else mn, S)
        if any(NS[l] == 0 for l in D): out(f"  {tag}: 자료 없음 {NS}"); continue
        keys = sorted(set.intersection(*[set(d) for d in D.values()]), key=str); C0 = {}
        out(f"=== 가족 {fam}: {tag} 같은 사람의 심박 변화 추종(영상 {len(keys)}, 시드 수 {NS})")
        res = {}
        for lab in D:
            dp, dg, sub = center(D[lab], keys); C0[lab] = (dp, dg, sub); r_w = float(np.corrcoef(dp, dg)[0, 1]); e_w = float(np.abs(dp - dg).mean())
            res[lab] = dict(slope=slope(dp, dg), r_within=r_w, mae_within=e_w, videos=int(len(dp)), subjects=int(len(np.unique(sub))))
            out(f"  {lab:30s} 기울기 {res[lab]['slope']:.3f}  피험자 안 상관 {r_w:.3f}  피험자 안 오차 {e_w:5.2f} bpm  (영상 {len(dp)}, 피험자 {len(np.unique(sub))})")
        rows = []
        for lab, *_ in BASES:
            d, p = perm_slope(C0[FIN[0]][0], C0[lab][0], C0[FIN[0]][1], C0[FIN[0]][2]); rows.append(dict(base=lab, d=d, p=p))
        holm_tag(rows)
        for r in rows: r["verdict"] = "지지" if r["d"] > 0 and r["holm"] < .05 else ("반대 방향 유의" if r["d"] < 0 and r["holm"] < .05 else "지지 없음"); out(f"  SpecPhys − {r['base']:28s} 기울기 Δ{r['d']:+.3f}  p={r['p']:.4f}  Holm={r['holm']:.4f}  → {r['verdict']}")
        ps = [slope(*center(D[FIN[0]], keys, use=i)[:2]) for i in range(NS[FIN[0]])]
        bmax = max(res[l]["slope"] for l, *_ in BASES); out(f"  민감도: SpecPhys 시드별 기울기 {min(ps):.3f}~{max(ps):.3f}, 기존 방법 최고 {bmax:.3f} → 시드 {sum(x > bmax for x in ps)}/{len(ps)} 개가 모두보다 높음")
        OUT[f"B_{tag}"] = dict(levels=res, tests=rows, seed_slopes=ps)
# ---------- C 배음 오류 ----------
def octv(p, gt): r = p / gt; return float(abs(r - 2) < 0.2 or abs(r - 0.5) < 0.1)
def dbl(p, gt): return float(abs(p / gt - 2) < 0.2)
def hlv(p, gt): return float(abs(p / gt - 0.5) < 0.1)
def famC():
    rows = []; lev = {}
    for tag, loader in (("VIPL-HR", vipl_rows), ("MMPD 7-33", mmpd_rows)):
        D = {lab: loader(vn if loader is vipl_rows else mn, S)[0] for lab, vn, mn, S in MODELS}
        if any(not d for d in D.values()): out(f"  {tag}: 자료 없음"); continue
        keys = sorted(set.intersection(*[set(d) for d in D.values()]), key=str)
        share = {lab: {k: float(np.mean([octv(p, d[k]["gt"]) for p in d[k]["p"]])) for k in keys} for lab, d in D.items()}
        lev[tag] = {lab: dict(octave=100 * float(np.mean(list(share[lab].values()))), doubling=100 * float(np.mean([np.mean([dbl(p, D[lab][k]["gt"]) for p in D[lab][k]["p"]]) for k in keys])),
                              halving=100 * float(np.mean([np.mean([hlv(p, D[lab][k]["gt"]) for p in D[lab][k]["p"]]) for k in keys])), n=len(keys)) for lab in D}
        out(f"=== 가족 C: {tag} 배음 오류 비율(영상 {len(keys)}; % = 배가 + 반감)")
        for lab in D: L_ = lev[tag][lab]; out(f"  {lab:30s} 배음 {L_['octave']:5.2f} %  (배가 {L_['doubling']:5.2f}, 반감 {L_['halving']:5.2f})")
        for lab, *_ in BASES:
            d = [share[FIN[0]][k] - share[lab][k] for k in keys]; rows.append(dict(set=tag, base=lab, d=100 * float(np.mean(d)), p=flip(d, 100000)))
    if rows:
        holm_tag(rows)
        for r in rows: r["verdict"] = "지지" if r["d"] < 0 and r["holm"] < .05 else ("반대 방향 유의" if r["d"] > 0 and r["holm"] < .05 else "지지 없음"); out(f"  {r['set']:10s} SpecPhys − {r['base']:28s} Δ{r['d']:+6.2f} %p  p={r['p']:.4f}  Holm={r['holm']:.4f}  → {r['verdict']}")
    # 서술: 다른 세트(정본 집계의 ratio = 예측/라벨 파형 심박)
    AX = [("MR-NIRP still", "NIRP_STILL"), ("MR-NIRP small", "NIRP_SMALL"), ("MR-NIRP large", "NIRP_LARGE"), ("PURE 07-10", "PURE"), ("UBFC 42-49", "REALTEST"), ("DLCN", "DLCN"), ("MCD-A", "MCD_A"), ("MCD-B", "MCD_B")]
    out("  서술(다른 세트, 배음 오류 %, 시드 평균):"); desc = {}
    for lab, vn, mn, S in MODELS:
        cells = []
        for nm_, ax in AX:
            v = []
            for s in S:
                p = f"{R}/agg_{fname(mn, s)}.json"
                if not os.path.exists(p): continue
                rr = json.load(open(p)).get(ax) or []
                if ax == "PURE": rr = [x for x in rr if int(x["vid"]) >= 701]
                if rr: v.append(100 * np.mean([float(abs(x["ratio"] - 2) < 0.2 or abs(x["ratio"] - 0.5) < 0.1) for x in rr]))
            cells.append(f"{nm_} {np.mean(v):4.1f}" if v else f"{nm_} -"); desc.setdefault(lab, {})[nm_] = float(np.mean(v)) if v else None
        out(f"   {lab:30s} " + " | ".join(cells))
    OUT["C"] = dict(levels=lev, tests=rows, descriptive=desc)
# ---------- A 통제 심박 시험 ----------
def sweep(nm, S, lo, bp=False):
    pre = "hrsweeplo" if lo else "hrsweep"; acc = {}; n = 0
    for s in S:
        p = f"{R}/{pre}_{fname(nm, s)}{'_bp' if bp else ''}.json"
        if not os.path.exists(p): continue
        n += 1
        for k, v in json.load(open(p))["clips"].items(): a = acc.setdefault(k, dict(face=v["face"], target=v["target"], gt=v["gt"], preds=[])); a["preds"].append(v["pred"])
    for a in acc.values(): a["pm"] = float(np.mean(a["preds"])); a["ae"] = float(np.mean([abs(x - a["gt"]) for x in a["preds"]])); a["ok"] = float(np.mean([abs(x - a["gt"]) <= 5 for x in a["preds"]]))
    return acc, n
def sw_slope(d):
    by = {}
    for v in d.values(): by.setdefault(v["face"], []).append(v)
    dp, dg = [], []
    for vs in by.values(): gt = np.array([v["gt"] for v in vs]); pr = np.array([v["pm"] for v in vs]); dg += list(gt - gt.mean()); dp += list(pr - pr.mean())
    return slope(np.array(dp), np.array(dg))
EXTRA = [("SpecPhys without synthetic clips", "NF_NSP2NS_L", DEC), ("SpecPhys, clean-mask clips", "NF_NSP2C_L", DEC), ("quarter real + synthetic (Q)", "NF_NSP2Q_L", DEC), ("quarter real only (QN)", "NF_NSP2QN_L", DEC), ("synthetic only (SO)", "NF_NSP2SO_L", DEC)]
def famA(bp=False):
    rows = []; lev = {}
    for lo in (True, False):
        tag = ("hard (pulse 0.003, strong noise)" if lo else "easy (pulse 0.01)") + (" [band-pass readout]" if bp else "")
        D, NS = {}, {}
        for lab, vn, mn, S in MODELS + [(a, b, b, c) for a, b, c in EXTRA]: D[lab], NS[lab] = sweep(vn, S, lo, bp)
        out(f"=== {'가족 A' if not bp else '민감도 A'}: 통제 심박 시험 {tag} — 클립 40개, 시드 수 {NS}")
        for lab, d in D.items():
            if not d: out(f"  {lab:34s} 자료 없음"); continue
            bins = {}
            for v in d.values(): bins.setdefault(v["target"], []).append(v["ae"])
            L_ = dict(mae=float(np.mean([v["ae"] for v in d.values()])), slope=sw_slope(d), within5=float(np.mean([v["ok"] for v in d.values()])), per_hr={int(t): float(np.mean(x)) for t, x in sorted(bins.items())}, seeds=NS[lab]); lev.setdefault(tag, {})[lab] = L_
            out(f"  {lab:34s} 오차 {L_['mae']:6.2f}  기울기 {L_['slope']:.3f}  5 bpm 안 {L_['within5']:.2f} | " + " ".join(f"{t}:{x:.1f}" for t, x in L_["per_hr"].items()))
        if not D[FIN[0]]: continue
        for lab, *_ in BASES:
            if not D[lab]: continue
            ks = sorted(set(D[FIN[0]]) & set(D[lab])); d = [D[FIN[0]][k]["ae"] - D[lab][k]["ae"] for k in ks]
            rows.append(dict(set=tag, base=lab, ref=float(np.mean([D[lab][k]["ae"] for k in ks])), spec=float(np.mean([D[FIN[0]][k]["ae"] for k in ks])), d=float(np.mean(d)), p=flip(d), n=len(ks)))
    if rows and len(rows) == 8:
        holm_tag(rows)
        for r in rows: r["verdict"] = "지지" if r["d"] < 0 and r["holm"] < .05 else ("반대 방향 유의" if r["d"] > 0 and r["holm"] < .05 else "지지 없음"); out(f"  {r['set'][:4]:4s} SpecPhys − {r['base']:28s} {r['ref']:6.2f} → {r['spec']:6.2f}  Δ{r['d']:+6.2f}  p={r['p']:.4f}  Holm={r['holm']:.4f}  → {r['verdict']}")
    elif rows: out(f"  (기존 방법 자료 {len(rows)}/8 — 판정 보류)")
    OUT["A" + ("_bp" if bp else "")] = dict(levels=lev, tests=rows)
# ---------- D 실제 학습 영상이 적을 때 ----------
PRIM = ("P2 VIPL-HR all", "P3 MMPD 7-33 all", "pooled low", "pooled high", "NIRP small")
def bundle_pub(vname, mname):
    orig = g["f_of"]; g["f_of"] = lambda cell, seed, suf: orig(vname if suf == "_VIPL" else mname, 100, suf)
    try: return bundle("PUB", [100])
    finally: g["f_of"] = orig
def famD():
    have = all(os.path.exists(f"{R}/agg_NF_NSP2{a}_L_s{s}_VIPL.json") for a in ("Q", "QN", "SO") for s in DEC)
    if not have: out("=== 가족 D: 자료 없음(체인 128 진행 중)"); return
    T = {c: bundle(c, DEC) for c in ("NF_NSP2Q_L", "NF_NSP2QN_L", "NF_NSP2SO_L", "NF_NSP2C_L", "NF_NSP2NS_L")}; T["FP"] = bundle_pub("FPbase", "FactorizePhys")
    out("=== 가족 D 수준(시드 2100~2500 평균; 공개 FactorizePhys 는 체크포인트 하나)")
    lab = {"NF_NSP2C_L": "all real + synthetic (C)", "NF_NSP2NS_L": "all real only (NS)", "NF_NSP2Q_L": "quarter real + synthetic (Q)", "NF_NSP2QN_L": "quarter real only (QN)", "NF_NSP2SO_L": "synthetic only (SO)", "FP": "public FactorizePhys (start)"}
    for c in lab: out(f"  {lab[c]:30s} " + "  ".join(f"{k.replace('P2 ', '').replace('P3 ', '')} {np.mean(list(T[c][k].values())):6.2f}" for k in PRIM))
    res = {}
    for fam, (a, b) in (("D1 Q vs QN", ("NF_NSP2Q_L", "NF_NSP2QN_L")), ("D3 SO vs public FactorizePhys", ("NF_NSP2SO_L", "FP"))):
        rows = []
        for k in PRIM:
            r = cmp(T[a][k], T[b][k]); r["perm"] = flip_p(T[a][k], T[b][k]); r["metric"] = k; r["p_w"] = r["p"]; r["p"] = r["perm"]; rows.append(r)
        holm_tag(rows); out(f"=== 가족 {fam} (Holm 5, 영상 단위 부호 뒤집기)")
        for r in rows:
            v = ("합성 클립이 나음" if r["d"] < 0 else "합성 클립이 나쁨") if r["holm"] < .05 else "n.s."
            if fam.startswith("D3"): v = ("미세조정이 나음" if r["d"] < 0 else "미세조정이 나쁨") if r["holm"] < .05 else "n.s."
            r["verdict"] = v; out(f"  {r['metric']:18s} n={r['n']:5d} {r['base']:6.2f} → {r['cell']:6.2f}  Δ{r['d']:+6.2f}  perm={r['perm']:.4f}  Holm={r['holm']:.4f}  Wilcoxon={r['p_w']:.4f}  → {v}")
        res[fam] = rows
    rows = []
    for k in PRIM:
        A = {i: T["NF_NSP2Q_L"][k][i] - T["NF_NSP2QN_L"][k][i] for i in set(T["NF_NSP2Q_L"][k]) & set(T["NF_NSP2QN_L"][k])}
        B = {i: T["NF_NSP2C_L"][k][i] - T["NF_NSP2NS_L"][k][i] for i in set(T["NF_NSP2C_L"][k]) & set(T["NF_NSP2NS_L"][k])}
        ks = sorted(set(A) & set(B), key=str); rows.append(dict(metric=k, n=len(ks), quarter=float(np.mean([A[i] for i in ks])), full=float(np.mean([B[i] for i in ks])), d=float(np.mean([A[i] - B[i] for i in ks])), p=flip_p(A, B)))
    holm_tag(rows); out("=== 가족 D2 상호작용 (Q−QN) − (C−NS) (Holm 5): 음수 = 실제가 적을 때 합성 클립의 이득이 더 큼")
    for r in rows: r["verdict"] = "지지" if r["d"] < 0 and r["holm"] < .05 else ("반대 방향 유의" if r["d"] > 0 and r["holm"] < .05 else "지지 없음"); out(f"  {r['metric']:18s} n={r['n']:5d} 1/4 실제 {r['quarter']:+6.2f}  전체 실제 {r['full']:+6.2f}  차 {r['d']:+6.2f}  perm={r['p']:.4f}  Holm={r['holm']:.4f}  → {r['verdict']}")
    res["D2 interaction"] = rows
    for nm_, a, b in (("기전: 어려운 통제 시험 Q vs QN", "NF_NSP2Q_L", "NF_NSP2QN_L"), ("기전: 어려운 통제 시험 SO vs 공개 FactorizePhys", "NF_NSP2SO_L", "FPbase")):
        A_, _ = sweep(a, DEC, True); B_, _ = sweep(b, [100] if b == "FPbase" else DEC, True)
        if A_ and B_: ks = sorted(set(A_) & set(B_)); d = [A_[k]["ae"] - B_[k]["ae"] for k in ks]; res[nm_] = dict(ref=float(np.mean([B_[k]["ae"] for k in ks])), cell=float(np.mean([A_[k]["ae"] for k in ks])), d=float(np.mean(d)), p=flip(d)); out(f"  {nm_}: {res[nm_]['ref']:6.2f} → {res[nm_]['cell']:6.2f}  Δ{res[nm_]['d']:+6.2f}  클립 순열 p={res[nm_]['p']:.4f}")
    D_desc = {}
    for c in ("NF_NSP2Q_L", "NF_NSP2QN_L", "NF_NSP2SO_L", "NF_NSP2C_L", "NF_NSP2NS_L"):
        D_desc[c] = {k: float(np.mean(list(T[c][k].values()))) for k in ("VIPL <60", "VIPL >=120", "MMPD <110", "MMPD >=110", "PURE 07-10", "UBFC 42-49", "DLCN", "RTNI", "NIRP still", "NIRP large", "MCD_A", "MCD_B")}
    out("  서술(다른 지표): " + "; ".join(f"{lab[c]}: " + ", ".join(f"{k} {v:.2f}" for k, v in D_desc[c].items()) for c in D_desc))
    OUT["D"] = dict(tests=res, descriptive=D_desc, levels={lab[c]: {k: float(np.mean(list(T[c][k].values()))) for k in PRIM} for c in lab})
famB(); famC(); famA(); famA(bp=True); famD()
open(f"{R}/contrib_judge.txt", "w").write("\n".join(LINES) + "\n"); json.dump(OUT, open(f"{R}/contrib_judge.json", "w"), indent=1, ensure_ascii=False, default=str)
