#!/usr/bin/env python3
"""[62 문서 §14·§15, 2026-11-05] 논문 최종 설정(SpecPhys = 단일 모델 + 잡음 부분공간 투영, NF_NSP2_L) 표: 집계 파일에서 원고 숫자를 생성한다(학습 없음). MMPD 는 공통 정답, VIPL 은 장치 정답. 최종 설정의 수준값은 선택에 쓰지 않은 확인 시드 1600~2500 으로 낸다.
  python scripts/final_config_tables.py → results/final_config_tables.{md,json}"""
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

CONF = [1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]; DEC = [2100, 2200, 2300, 2400, 2500]
nm_ = lambda c, S: [f"{c}_s{x}" for x in S]
NAMES = [("public FactorizePhys", "FPbase", "FactorizePhys"), ("public RhythmFormer", "RFbase", "RhythmFormer"), ("public PhysMamba", "PMbase", "PMbase"), ("previous PhysFormer + corpus", "CoTAF_TW_L", "CoTAF_TW_L"),
         ("SpecPhys-B (same cache, control)", "NF_HYB_U8_L", "NF_HYB_U8_L"), ("original attention, same training", nm_("NF_ORIGA_L", CONF), nm_("NF_ORIGA_L", CONF)), ("SpecPhys without projection", nm_("NF_SKREF_REP_L", CONF), nm_("NF_SKREF_REP_L", CONF)),
         ("SpecPhys-R (routed pair)", nm_("NF_ENS_AB_L", DEC), nm_("NF_ENS_AB_L", DEC)), ("SpecPhys (final)", nm_("NF_NSP2_L", CONF), nm_("NF_NSP2_L", CONF))]
FIN = "SpecPhys (final)"
md = []; J = {}
def emit(s=""): md.append(s); print(s)
def ve(V, c=None): return {k: v["e"] for k, v in V.items() if c is None or c(v)}
def me(M, c=None): return {k: v["e"] for k, v in M.items() if c is None or c(v)}
mean = lambda d: float(np.mean(list(d.values()))) if d else float("nan")
_rng = np.random.default_rng(0)
def flip(a, b, n=100000):
    k = sorted(set(a) & set(b), key=str); d = np.array([a[i] - b[i] for i in k]); obs = abs(d.mean()); hit = 0
    for _ in range(n // 10000): hit += int(np.sum(np.abs((_rng.choice([-1.0, 1.0], size=(10000, len(d))) * d).mean(1)) >= obs - 1e-12))
    return (hit + 1) / (n + 1)
D = {nm: (vipl(v), mmpd(m)) for nm, v, m in NAMES}
emit("# Final configuration tables (zero-shot; generated by scripts/final_config_tables.py; MMPD common labels, VIPL-HR device labels; SpecPhys final and its same-recipe controls over confirmation seeds 1600-2500, the routed pair over 2100-2500)\n")
emit("## A. Levels")
emit("| Model | seeds | VIPL all | VIPL <60 | VIPL 60-90 | VIPL 90-120 | VIPL ≥120 | VIPL v7 | MMPD all | MMPD <60 | MMPD <110 | MMPD ≥110 | MMPD walking | pooled low | pooled high |"); emit("|---|" + "---|" * 14)
def lv(V, M): return dict(vipl=mean(ve(V)), v60=mean(ve(V, lambda x: x["dev"] < 60)), v6090=mean(ve(V, lambda x: 60 <= x["dev"] < 90)), v90120=mean(ve(V, lambda x: 90 <= x["dev"] < 120)), v120=mean(ve(V, lambda x: x["dev"] >= 120)), v7=mean(ve(V, lambda x: x["sc"] == "v7")), mmpd=mean(me(M)), m60=mean(me(M, lambda x: x["gt"] < 60)), mlo=mean(me(M, lambda x: x["gt"] < 110)), mhi=mean(me(M, lambda x: x["gt"] >= 110)), mw=mean(me(M, lambda x: x["mo"] == 4)), plo=mean(lo_dev(V, M)), phi=mean(hi_dev(V, M)))
for nm, v, m in NAMES:
    V, M = D[nm]; r = lv(V, M); r["seeds"] = len(seeds(v)); J.setdefault("levels", {})[nm] = r
    emit(f"| {nm} | {r['seeds']} | " + " | ".join(f"{r[k]:.2f}" for k in ("vipl", "v60", "v6090", "v90120", "v120", "v7", "mmpd", "m60", "mlo", "mhi", "mw", "plo", "phi")) + " |")
RV, RM = D[FIN]
def tests(V, M): return [("VIPL all", ve(V)), ("VIPL <60", ve(V, lambda x: x["dev"] < 60)), ("VIPL 60-90", ve(V, lambda x: 60 <= x["dev"] < 90)), ("VIPL ≥120", ve(V, lambda x: x["dev"] >= 120)), ("VIPL v7", ve(V, lambda x: x["sc"] == "v7")), ("MMPD all", me(M)), ("MMPD <110", me(M, lambda x: x["gt"] < 110)), ("MMPD ≥110", me(M, lambda x: x["gt"] >= 110)), ("MMPD walking", me(M, lambda x: x["mo"] == 4)), ("pooled low", lo_dev(V, M)), ("pooled high", hi_dev(V, M))]
def fam(base, A=None):
    AV, AM = (RV, RM) if A is None else A; BV, BM = D[base]; ta, tb = tests(AV, AM), dict(tests(BV, BM))
    emit(f"\n## B. SpecPhys vs {base} (paired per video; Wilcoxon and sign-flip permutation p)"); emit("| Test | n | reference | SpecPhys | Δ | Wilcoxon p | permutation p |"); emit("|---|---|---|---|---|---|---|")
    for nm, a in ta:
        r = cmp(a, tb[nm]); r["perm"] = flip(a, tb[nm]); emit(f"| {nm} | {r['n']} | {r['base']:.2f} | {r['cell']:.2f} | {r['d']:+.2f} | {r['p']:.4f} | {r['perm']:.4f} |"); J.setdefault("vs_" + base, {})[nm] = r
fam("SpecPhys without projection"); fam("original attention, same training"); fam("SpecPhys-B (same cache, control)")
fam("SpecPhys-R (routed pair)", (vipl(nm_("NF_NSP2_L", DEC)), mmpd(nm_("NF_NSP2_L", DEC))))
emit("\n## C. SpecPhys vs existing methods on the evaluation sets (paired per video; Holm 6 on the six sets pre-registered for the public FactorizePhys comparison, the others descriptive)")
FN = nm_("NF_NSP2_L", CONF)
PUBAX = [("VIPL-HR", None), ("MMPD 7–33", None), ("MR-NIRP still", "NIRP_STILL"), ("MR-NIRP small", "NIRP_SMALL"), ("MR-NIRP large", "NIRP_LARGE"), ("RTNI", "RTNI"), ("PURE 07–10", "PURE"), ("UBFC 42–49", "REALTEST"), ("DLCN", "DLCN"), ("MCD-A", "MCD_A"), ("MCD-B", "MCD_B")]
def ax_(names, key): return axis(names, key, (lambda r: int(r["vid"]) >= 701) if key == "PURE" else None)
SIX = ("VIPL-HR", "MMPD 7–33", "MR-NIRP small", "MR-NIRP large", "RTNI", "PURE 07–10")
for nm, v, m in NAMES[:4]:
    XV, XM = D[nm]; rows = []
    for ax, key in PUBAX:
        r = cmp(ve(RV), ve(XV)) if ax == "VIPL-HR" else (cmp(me(RM), me(XM)) if ax == "MMPD 7–33" else cmp(ax_(FN, key), ax_(m, key)))
        rows.append((ax, r))
    avail = [(a, r) for a, r in rows if r]; A6 = [(a, r) for a, r in avail if a in SIX]; hs = dict(zip([a for a, _ in A6], holm([r["p"] for _, r in A6]))); emit(f"\n### vs {nm}"); emit("| Set | family | n | reference | SpecPhys | Δ | p | Holm 6 | verdict |"); emit("|---|---|---|---|---|---|---|---|---|")
    w6 = o6 = sigworse = 0
    for ax, r in avail:
        h = hs.get(ax); ph = h if h is not None else r["p"]; vd = "better" if r["d"] < 0 and ph < .05 else ("worse" if r["d"] > 0 and ph < .05 else ("non-inferior" if r["d"] <= 0.3 else "higher, n.s."))
        sigworse += vd == "worse"
        if ax in SIX: w6 += vd == "better"; o6 += vd in ("better", "non-inferior")
        emit(f"| {ax} | {'primary six' if ax in SIX else 'descriptive'} | {r['n']} | {r['base']:.2f} | {r['cell']:.2f} | {r['d']:+.2f} | {r['p']:.4f} | {'–' if h is None else format(h, '.4f')} | {vd} |"); J.setdefault("axes_" + nm, {})[ax] = dict(r, holm=h, verdict=vd, family="six" if ax in SIX else "descriptive")
    emit(f"\nSix sets: better {w6}/{len(A6)}, better or within +0.3 {o6}/{len(A6)}; significantly worse on any of the {len(avail)} sets: {sigworse}"); J.setdefault("axes_summary", {})[nm] = dict(better6=w6, ok6=o6, n6=len(A6), n=len(avail), sig_worse=sigworse)
emit("\n## D. Heart-rate comparison against existing methods (device labels: Holm 5 superiority, rank non-inferiority 2 bpm Holm 2; recorded-PPG labels: Holm 4 superiority, non-inferiority Holm 2)")
emit("| Existing method | VIPL <60 | pooled low (Holm) | VIPL ≥120 | pooled high | rank NI Holm (VIPL ≥120 / pooled) | mean gap | mean NI t p | PPG <60 | PPG pooled low (Holm) | PPG ≥110 | PPG pooled high | PPG NI Holm |"); emit("|---|" + "---|" * 12)
from scipy.stats import ttest_1samp
vonly = lambda d: {k: v for k, v in d.items() if k[0] == "V"}
for nm, v, m in NAMES[:4]:
    XV, XM = D[nm]
    T = dict(v60=(ve(RV, lambda x: x["dev"] < 60), ve(XV, lambda x: x["dev"] < 60)), v120=(ve(RV, lambda x: x["dev"] >= 120), ve(XV, lambda x: x["dev"] >= 120)), v7=(ve(RV, lambda x: x["sc"] == "v7"), ve(XV, lambda x: x["sc"] == "v7")), low=(lo_dev(RV, RM), lo_dev(XV, XM)), high=(hi_dev(RV, RM), hi_dev(XV, XM)))
    res = {k: cmp(*x) for k, x in T.items()}; hs = dict(zip(res, holm([r["p"] for r in res.values()]))); nih = holm([ni_p(*T["v120"]), ni_p(*T["high"])])
    a, b = T["high"]; ks = sorted(set(a) & set(b), key=str); d = np.array([a[i] - b[i] for i in ks]); pm = float(ttest_1samp(d - 2, 0, alternative="less").pvalue)
    W = dict(w60=(vonly(lo_wave(RV, RM)), vonly(lo_wave(XV, XM))), w110=(vonly(hi_wave(RV, RM)), vonly(hi_wave(XV, XM))), wl=(lo_wave(RV, RM), lo_wave(XV, XM)), wh=(hi_wave(RV, RM), hi_wave(XV, XM)))
    wr = {k: cmp(*x) for k, x in W.items()}; whs = dict(zip(wr, holm([r["p"] for r in wr.values()]))); wni = holm([ni_p(*W["w110"]), ni_p(*W["wh"])])
    f = lambda r: f"{r['base']:.2f} → {r['cell']:.2f}"
    emit(f"| {nm} | {f(res['v60'])} (Holm {hs['v60']:.3f}) | {f(res['low'])} ({hs['low']:.4f}) | {f(res['v120'])} | {f(res['high'])} | {nih[0]:.4f} / {nih[1]:.4f} | {d.mean():+.2f} | {pm:.3f} | {f(wr['w60'])} | {f(wr['wl'])} ({whs['wl']:.4f}) | {f(wr['w110'])} | {f(wr['wh'])} | {wni[0]:.4f} / {wni[1]:.4f} |")
    J.setdefault("guarantee", {})[nm] = dict(mean_gap=float(d.mean()), mean_ni_p=pm, device=dict(res, holm=hs, ni_holm=nih), wave=dict(wr, holm=whs, ni_holm=wni))
emit("\n## E. MMPD subsets against existing methods (common labels; rank non-inferiority, one-sided)")
emit("| Existing method | MMPD <110 reference | SpecPhys | Δ | p (two-sided) | NI margin 0.3 p | MMPD ≥110 reference | SpecPhys | Δ | p (two-sided) | NI margin 2 p |"); emit("|---|" + "---|" * 10)
_me = lambda M, f: {k: v["e"] for k, v in M.items() if f(v["gt"])}
for nm, v, m in NAMES[:4]:
    XM = D[nm][1]; la, lb = _me(RM, lambda g: g < 110), _me(XM, lambda g: True); ha, hb = _me(RM, lambda g: g >= 110), _me(XM, lambda g: True)
    rl, rh = cmp(la, lb), cmp(ha, hb); pl, ph = ni_p(la, {k: lb[k] for k in la if k in lb}, 0.3), ni_p(ha, {k: hb[k] for k in ha if k in hb}, 2.0)
    emit(f"| {nm} | {rl['base']:.2f} | {rl['cell']:.2f} | {rl['d']:+.2f} | {rl['p']:.4f} | {pl:.4f} | {rh['base']:.2f} | {rh['cell']:.2f} | {rh['d']:+.2f} | {rh['p']:.4f} | {ph:.4f} |"); J.setdefault("mmpd_subsets", {})[nm] = dict(below110=dict(rl, ni_p=pl), above110=dict(rh, ni_p=ph))
emit("\n## F. SpecPhys vs comparators on the other evaluation sets (paired per video, descriptive; same seeds where the comparator has them)")
emit("| Set | n | without projection | Δ | p | original attention | Δ | p | SpecPhys-B control | Δ | p |"); emit("|---|" + "---|" * 10)
CTRL = nm_("NF_SKREF_REP_L", CONF); OA = nm_("NF_ORIGA_L", CONF)
for ax, key in PUBAX[2:]:
    a = ax_(FN, key); r1, r2, r3 = cmp(a, ax_(CTRL, key)), cmp(a, ax_(OA, key)), cmp(a, ax_("NF_HYB_U8_L", key))
    emit(f"| {ax} | {r1['n']} | {r1['base']:.2f} | {r1['d']:+.2f} | {r1['p']:.4f} | {r2['base']:.2f} | {r2['d']:+.2f} | {r2['p']:.4f} | {r3['base']:.2f} | {r3['d']:+.2f} | {r3['p']:.4f} |"); J.setdefault("other_sets", {})[ax] = dict(cell=r1["cell"], ctrl=r1, orig=r2, base=r3)
emit("\n## G. Dataset-level zero-shot datasets (never in any zero-shot training pool; Δ and p against the public FactorizePhys, Holm over the five sets)")
emit("| Set | n | public FactorizePhys | SpecPhys-B (3 seeds) | Δ | p | Holm | SpecPhys (10 seeds) | Δ | p | Holm |"); emit("|---|" + "---|" * 10)
_zs = [("VIPL-HR", None), ("MR-NIRP still", "NIRP_STILL"), ("MR-NIRP small", "NIRP_SMALL"), ("MR-NIRP large", "NIRP_LARGE"), ("DLCN", "DLCN")]; _zr = {}
XVp = D["public FactorizePhys"][0]
for tag, names in (("SpecPhys-B", "NF_HYB_L"), ("SpecPhys", FN)):
    rows = [(n, cmp(ve(vipl(names)), ve(XVp)) if k is None else cmp(axis(names, k), axis("FactorizePhys", k))) for n, k in _zs]; hs = holm([r["p"] for _, r in rows]); _zr[tag] = {n: dict(r, holm=h) for (n, r), h in zip(rows, hs)}
for n, _ in _zs:
    a_, b_ = _zr["SpecPhys-B"][n], _zr["SpecPhys"][n]; emit(f"| {n} | {a_['n']} | {a_['base']:.2f} | {a_['cell']:.2f} | {a_['d']:+.2f} | {a_['p']:.4f} | {a_['holm']:.4f} | {b_['cell']:.2f} | {b_['d']:+.2f} | {b_['p']:.4f} | {b_['holm']:.4f} |")
J["dataset_zero_shot"] = _zr
_ro = json.load(open(f"{R}/realonly_judge.json")); _ub = json.load(open(f"{R}/ubfc_judge.json"))
emit("\n## H. Contribution of the synthetic clips (earlier controls, 3 seeds each; no cell of the final recipe without synthetic clips exists)")
emit("| Control | Set | n | without synthetic | with synthetic | Δ | p |"); emit("|---|---|---|---|---|---|---|")
J["synthetic_controls"] = {}
for lab, rr in (("VIPL-HR (device labels)", _ro["vipl"]["VIPL all [문헌GT]"]), ("MMPD 7–33", _ro["mmpd"]["MMPD all"]), ("MR-NIRP still", _ro["axes"]["NIRP_STILL"]), ("MR-NIRP small", _ro["axes"]["NIRP_SMALL"]), ("MR-NIRP large", _ro["axes"]["NIRP_LARGE"]), ("DLCN", _ro["axes"]["DLCN"])):
    emit(f"| PhysFormer corpus: 161 real clips vs + 42 synthetic | {lab} | {rr['n']} | {rr['mae_a']:.2f} | {rr['mae_b']:.2f} | {rr['delta']:+.2f} | {rr['p']:.4f} |"); J["synthetic_controls"]["corpus:" + lab] = rr
for bb, tag in (("PF", "PhysFormer"), ("FP", "FactorizePhys")):
    for k, lab in (("VIPL", "VIPL-HR (device labels)"), ("MMPD", "MMPD 7–33"), ("NIRP_STILL", "MR-NIRP still"), ("NIRP_SMALL", "MR-NIRP small"), ("NIRP_LARGE", "MR-NIRP large")):
        rr = _ub[f"{bb}:U0->U1:{k}"]; emit(f"| {tag} from UBFC 34 clips: without vs + 42 synthetic | {lab} | {rr['n']} | {rr['a']:.2f} | {rr['b']:.2f} | {rr['delta']:+.2f} | {rr['p']:.4f} |"); J["synthetic_controls"][f"{tag}:{lab}"] = rr
open(f"{R}/final_config_tables.md", "w").write("\n".join(md) + "\n"); json.dump(J, open(f"{R}/final_config_tables.json", "w"), indent=1, ensure_ascii=False, default=float)
